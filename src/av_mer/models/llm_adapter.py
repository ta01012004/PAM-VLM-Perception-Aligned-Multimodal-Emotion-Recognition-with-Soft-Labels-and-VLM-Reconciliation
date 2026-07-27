"""Optional Qwen2.5 + LoRA reconciler head, driven by fusion soft-prompt tokens.

The base LLM is frozen (optionally 4-bit); only LoRA adapters + the projector
train. The classifier remains the metric source; this branch adds a generated
``<emotion>label</emotion>`` + rationale for reconciliation and zero-shot use.

Heavy deps (transformers, peft) are imported lazily so unit tests that only
touch the projector/parser never trigger a model download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class LLMAdapterConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    num_prefix_tokens: int = 4
    max_new_tokens: int = 48
    freeze_base: bool = True


class LLMReconciler(nn.Module):
    """Wrap a causal LM with LoRA and accept projector prefix embeddings.

    The projector output ``[B, P, H]`` is concatenated in front of the token
    embeddings of the text prompt, so the LLM conditions on the fusion evidence.
    """

    def __init__(self, config: LLMAdapterConfig) -> None:
        super().__init__()
        self.config = config
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(config.model_id)
        if config.freeze_base:
            base.requires_grad_(False)
        lora = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_targets),
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora)
        self.hidden_size = base.config.hidden_size

    def _embed_tokens(self, input_ids: Tensor) -> Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def forward(
        self,
        prefix_embeds: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> Any:
        """Teacher-forced forward for LoRA training (LM loss on the text target)."""

        token_embeds = self._embed_tokens(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        prefix_mask = torch.ones(
            prefix_embeds.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        if labels is not None:
            # Prefix positions are not text targets -> mask them out with -100.
            prefix_labels = torch.full(
                prefix_embeds.shape[:2], -100, dtype=labels.dtype, device=labels.device
            )
            labels = torch.cat([prefix_labels, labels], dim=1)
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        prefix_embeds: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> list[str]:
        """Greedy-decode the reconciler output text for each sample."""

        token_embeds = self._embed_tokens(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        prefix_mask = torch.ones(
            prefix_embeds.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        generated = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def sequence_nll(
        self,
        prefix_embeds: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Return mean target-token NLL for every sequence in a batch."""

        token_embeds = self._embed_tokens(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        prefix_mask = torch.ones(
            prefix_embeds.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        prefix_labels = torch.full(
            prefix_embeds.shape[:2], -100, dtype=labels.dtype, device=labels.device
        )
        full_labels = torch.cat([prefix_labels, labels], dim=1)
        logits = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
        ).logits
        shifted_logits = logits[:, :-1].contiguous()
        shifted_labels = full_labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shifted_labels.shape)
        valid = shifted_labels.ne(-100)
        return (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def set_lora_trainable(self, trainable: bool) -> None:
        """Enable or freeze every LoRA parameter without touching the base LM."""

        for name, parameter in self.model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(trainable)

    def adapter_state_dict(self) -> dict[str, Tensor]:
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(self.model)

    def load_adapter_state_dict(self, state: dict[str, Tensor]) -> None:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(self.model, state)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

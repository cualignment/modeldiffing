import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
import torch.nn as nn
import torch

class Agent:
    def __init__(self, model_name, device):
        self.name = model_name
        self.device = device
        
        self.actor = AutoModelForCausalLM.from_pretrained(
            self.name
        )
        self.freeze_actor_except_lm_head()

        self.critic = AutoModel.from_pretrained(
            self.name
        )

        self.reference = AutoModelForCausalLM.from_pretrained(
            self.name
        )

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.reference.to(device)

        hidden_size = self.critic.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(
            device=self.device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=True, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.actor.resize_token_embeddings(len(self.tokenizer))
            self.critic.resize_token_embeddings(len(self.tokenizer))

        self.actor.eval()
        self.critic.eval()

    def freeze_actor_except_lm_head(self):
        for name, param in self.actor.named_parameters():
            # Only keep gradients for the final lm_head layer
            if "lm_head" not in name:
                param.requires_grad = False


    def generate(self, prompts: list[str], tokenize: bool = False, max_new_tokens: int = 256, do_sample: bool = True, temperature: float = 0.17, top_k: int = 50, top_p: float = 0.9, num_return_sequences: int = 1) -> list[str]:
        # batch-tokenize (now using a real pad token)
        if tokenize:
            input = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        else:
            inputs = prompts
        
        # move to device
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        # generate with sampling, explicitly telling generate what our pad token is
        with torch.no_grad():
            out_ids = self.actor.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,  # now a distinct pad
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids    = out_ids[:, prompt_len:]          # [B, max_new_tokens]
        pad_id     = self.tokenizer.pad_token_id

        all_new_ids = []                              # ← use a different name
        for seq in gen_ids:
            # extract just the non-pad token IDs for this sequence
            seq_ids = seq[seq != pad_id].tolist()
            all_new_ids.append(seq_ids)

        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True), all_new_ids
    
    def get_hidden_states(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.critic(**inputs)
            hidden_states = outputs.last_hidden_state
        return hidden_states

    def get_values(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states is float16; cast to float32 for the head
        h = hidden_states.float()                    # now [B, T, H] in float32
        return self.value_head(h).squeeze(-1)        # outputs float32


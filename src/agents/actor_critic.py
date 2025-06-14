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
            self.name,
            torch_dtype=torch.float16,
        )

        self.critic = AutoModel.from_pretrained(
            self.name,
            torch_dtype=torch.float16,
        )

        self.actor.to(self.device)
        self.critic.to(self.device)
        
        hidden_size = self.critic.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=True, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.actor.resize_token_embeddings(len(self.tokenizer))
            self.critic.resize_token_embeddings(len(self.tokenizer))

        self.actor.eval()
        self.critic.eval()

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

        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True), out_ids.shape[-1] - inputs["input_ids"].shape[-1]


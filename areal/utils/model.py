import os

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


VALID_VISION_MODELS = [
    "qwen2_vl",
    "qwen2_5_vl",
]
# This registry is used to check if a model is a vision model that we have checked it works with AReaL.
# As different vision models vary in their image processing, special tokens and keys, etc.
# We will add models to this registry as we test them.
# If you want to add a new vision model, please make sure it works with AReaL.


def is_qwen2_vl_model(model_type):
    return model_type in ["qwen2_vl", "qwen2_5_vl"]

def is_rm_model(model_type):
    return model_type in ["reward_model", "critic"]

# Copied from trl
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

class RewardModelHead(nn.Linear):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = nn.functional.linear(hidden_states, self.weight, self.bias)
        return output.float()

class ModelWithRewardModelHead(nn.Module):
    def __init__(self, base_model: PreTrainedModel, reward_model_head: RewardModelHead):
        super().__init__()
        self.base_model = base_model
        self.reward_model_head = reward_model_head
        self.config = base_model.config

        if hasattr(base_model, "_no_split_modules"):
            self._no_split_modules = base_model._no_split_modules

    def forward(self, *args, **kwargs):
        output = self.base_model(*args, **kwargs)
        if hasattr(output, 'hidden_states'):
            reward_logits = self.reward_model_head(output.hidden_states[-1])
        else:
            raise ValueError("Model output does not have hidden_states or last_hidden_state")
        return CausalLMOutputWithPast(
            logits=reward_logits,
            past_key_values=getattr(output, 'past_key_values', None),
            hidden_states=getattr(output, 'hidden_states', None),
            attentions=getattr(output, 'attentions', None),
        )

    def save_pretrained(self, save_directory, **kwargs):
        self.base_model.save_pretrained(save_directory, **kwargs)
        reward_head_path = os.path.join(save_directory, "reward_head.pt")
        torch.save(self.reward_model_head.state_dict(), reward_head_path)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            f.write(self.config.to_json_string())

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            **{k: v for k, v in kwargs.items() if k != 'reward_head'}
        )
        reward_head = RewardModelHead(
            hidden_size=base_model.config.hidden_size,
            output_dim=1,
            bias=False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=base_model.dtype
        )
        reward_head_path = os.path.join(pretrained_model_name_or_path, "reward_head.pt")
        if os.path.exists(reward_head_path):
            reward_head.load_state_dict(torch.load(reward_head_path))

        return cls(base_model, reward_head)

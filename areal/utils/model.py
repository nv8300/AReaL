import torch
import torch.nn as nn


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
        return output

class ModelWithRewardModelHead(nn.Module):
    def __init__(self, base_model: nn.Module, reward_model_head: RewardModelHead):
        super().__init__()
        self.base_model = base_model
        self.reward_model_head = reward_model_head
        
    def forward(self, *args, **kwargs):
        output = self.base_model(*args, **kwargs)
        if hasattr(output, 'hidden_states'):
            output = self.reward_model_head(output.hidden_states[-1])
        elif hasattr(output, 'last_hidden_state'):
            output = self.reward_model_head(output.last_hidden_state)
        else:
            raise ValueError("Model output does not have hidden_states or last_hidden_state")
        return output

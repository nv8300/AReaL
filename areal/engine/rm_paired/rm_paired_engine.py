import torch
import torch.utils.data
from tensordict import TensorDict

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.engine.fsdp_engine import FSDPEngine
from areal.utils import stats_tracker
from areal.utils import logging


logger = logging.getLogger(__name__)


class RMPairedEngine:
    def __init__(self, engine: TrainEngine):
        self.engine = engine

    def train_rm_paired(self, data: TensorDict):
        self.engine.train()
        return self.engine.train_batch(
            input_=data,
            loss_fn=compute_rm_paired_loss,
            loss_weight_fn=lambda x: 1,
        )
        
    def evaluate_rm_paired(self, data: TensorDict):
        self.engine.eval()
        return self.engine.eval_batch(
            input_=data,
            loss_fn=compute_rm_paired_loss,
            loss_weight_fn=lambda x: 1,
        )

class FSDPRMPairedEngine(FSDPEngine):
    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        self.rm_paired_engine = RMPairedEngine(self)

    def train_rm_paired(self, data: TensorDict):
        return self.rm_paired_engine.train_rm_paired(data)
    
    def evaluate_rm_paired(self, data: TensorDict):
        return self.rm_paired_engine.evaluate_rm_paired(data)
    
def compute_rm_paired_loss(logits: torch.Tensor, input_: TensorDict) -> torch.Tensor:
    packed_input_ids: torch.Tensor = input_["input_ids"]
    cu_seqlens: torch.Tensor = input_["cu_seqlens"]
    
    scores = logits[cu_seqlens[1:] - 1].squeeze(1) 

    assert scores.shape[0] % 2 == 0, len(scores.shape) # check the data size must be an even number to form a complete pair
    seqlens = scores.shape[0] // 2
    scores = scores.view(seqlens, 2)

    per_sample_loss = -(torch.nn.functional.logsigmoid(scores[:, 0] - scores[:, 1]))
    loss = per_sample_loss.sum().to(device=logits.device)

    ## Loggin stats
    stats_tracker.denominator(
        n_seqs=torch.ones(
            seqlens, dtype=torch.bool, device=logits.device
        ),
    )
    stats_tracker.stat(loss=per_sample_loss.detach(), denominator="n_seqs")

    return loss

    

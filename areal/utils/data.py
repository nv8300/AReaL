# Pad/unpad operations are modified from flash-attention under BSD-3 license.
# Copyright (c) 2023, Tri Dao.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict

from areal.api.cli_args import MicroBatchSpec
from areal.utils import datapack, logging

logger = logging.getLogger("data utils")


def reorder_list(xs: List, indices: List[int]) -> List:
    assert len(set(indices)) == len(xs)
    return [xs[i] for i in indices]


def dict_map(x: Dict, fn: Callable) -> Dict:
    return {k: fn(v) for k, v in x.items()}


def dict_of_list2list_of_dict(
    dict_of_lists: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    if not dict_of_lists:
        return []
    keys = list(dict_of_lists.keys())
    length = len(dict_of_lists[keys[0]])
    for key, value_list in dict_of_lists.items():
        if len(value_list) != length:
            raise ValueError(
                f"All lists must have the same length. Key '{key}' has length {len(value_list)}, expected {length}"
            )
    return [{key: dict_of_lists[key][i] for key in keys} for i in range(length)]


def list_of_dict2dict_of_list(
    list_of_dicts: List[Dict[str, Any]],
) -> Dict[str, List[Any]]:
    if not list_of_dicts:
        return {}
    keys = list(list_of_dicts[0].keys())
    for i, dict_item in enumerate(list_of_dicts):
        if set(dict_item.keys()) != set(keys):
            raise ValueError(
                f"All dictionaries must have the same keys. Dictionary at index {i} has keys {set(dict_item.keys())}, expected {set(keys)}"
            )
    return {key: [dict_item[key] for dict_item in list_of_dicts] for key in keys}


def pad_sequences_to_tensors(
    sequence_list: List[TensorDict], pad_value: float = 0.0
) -> TensorDict:
    if not sequence_list:
        return TensorDict()
    skip_keys = {"pixel_values", "image_grid_thw"}
    max_length = max(
        len(seq)
        for item in sequence_list
        for key, seq in item.items()
        if key not in skip_keys
    )
    result = {}
    for key in sequence_list[0].keys():
        padded = []
        if key in skip_keys:
            result[key] = [sequence_list[i][key] for i in range(len(sequence_list))]
            continue
        for item in sequence_list:
            x = item[key]
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            padded_x = torch.nn.functional.pad(
                x, (0, max_length - len(item[key])), value=pad_value
            )
            padded.append(padded_x)
        result[key] = torch.stack(padded)
    attention_mask = [
        [1] * len(next(iter(item[key] for key in item.keys() if key not in skip_keys)))
        + [0]
        * (
            max_length
            - len(next(iter(item[key] for key in item.keys() if key not in skip_keys)))
        )
        for item in sequence_list
    ]
    result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
    return TensorDict(result, batch_size=[result["attention_mask"].shape[0]])


def unpad_input(
    hidden_states, attention_mask
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        rearrange(hidden_states, "b s ... -> (b s) ...")[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    output = hidden_states.new_zeros(batch * seqlen)
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def concat_padded_tensors(
    tensor_dicts: List[TensorDict], pad_value: float = 0.0
) -> TensorDict:
    """Concatenate and pad tensors from multiple padded tensor dictionaries."""
    if not tensor_dicts:
        return TensorDict()

    batch_sizes = [tuple(d.batch_size) for d in tensor_dicts]
    new_batch_size = [sum(x[0] for x in batch_sizes), *batch_sizes[0][1:]]

    # Find max sequence length across all dictionaries
    assert all("attention_mask" in td for td in tensor_dicts)
    max_length = max([x["attention_mask"].shape[1] for x in tensor_dicts])
    result = {}

    has_any_multi_modal = any("multi_modal_input" in td for td in tensor_dicts)

    merged_multi_modal = None

    if has_any_multi_modal:
        merged_multi_modal = []

        # Merge multi-modal data maintaining per-dp correspondence
        for tensor_dict in tensor_dicts:
            td_batch_size = tensor_dict.batch_size[0]

            if "multi_modal_input" in tensor_dict:
                # Has multi_modal_input - extend the lists
                multi_modal = tensor_dict["multi_modal_input"]
            else:
                multi_modal = [{} for _ in range(td_batch_size)]

            merged_multi_modal.extend(multi_modal)

        result["multi_modal_input"] = merged_multi_modal

    # Process each key
    for key in tensor_dicts[0].keys():
        tensors_to_concat = []
        if key == "multi_modal_input":
            continue
        for tensor_dict in tensor_dicts:
            tensor = tensor_dict[key]
            # Skip 1D tensors like rewards
            if len(tensor.shape) == 1:
                tensors_to_concat.append(tensor)
                continue
            current_length = tensor.shape[1]
            if current_length < max_length:
                # Pad tensor to max_length
                pad_width = max_length - current_length
                if key == "attention_mask":
                    # Pad attention mask with 0s
                    padding = torch.zeros(
                        (tensor.shape[0], pad_width),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )

                else:
                    # Pad feature tensors with pad_value
                    padding = torch.full(
                        (tensor.shape[0], pad_width),
                        pad_value,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )

                tensor = torch.cat([tensor, padding], dim=1)
            tensors_to_concat.append(tensor)

        result[key] = torch.cat(tensors_to_concat, dim=0)
    return TensorDict(result, batch_size=new_batch_size)


def to_device(data: Dict[str, torch.Tensor | Any], device) -> Dict[str, torch.Tensor]:
    """Move tensors in a dictionary to the specified device."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def unpack_sequence(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    lens: Optional[List[int]] = None,
    dim: int = 0,
):
    """Unpack a sequence tensor into a list of tensors based on cumulative sequence lengths."""
    if lens is not None:
        return torch.split(x, lens, dim=dim)
    if cu_seqlens is not None:
        return torch.split(
            x, (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist(), dim=dim
        )
    raise ValueError("Either cu_seqlens or input_lens must be provided.")


def allocate_balanced_mbs(mb_spec: MicroBatchSpec, lens: List[int]) -> List[List[int]]:
    assert mb_spec.max_tokens_per_mb is not None
    group_indices = datapack.ffd_allocate(
        lens, mb_spec.max_tokens_per_mb, min_groups=mb_spec.n_mbs
    )
    group_indices = sorted([sorted(g) for g in group_indices])
    return group_indices


def allocate_balanced_mbs_synced(
    mb_spec: MicroBatchSpec,
    lens: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> List[List[int]]:
    group_indices = allocate_balanced_mbs(mb_spec, lens)
    if not dist.is_initialized():
        return group_indices
    all_n_mbs = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(all_n_mbs, len(group_indices), group=group)
    if all(mbs == len(group_indices) for mbs in all_n_mbs):
        return group_indices
    return allocate_balanced_mbs_synced(
        MicroBatchSpec.new(mb_spec, n_mbs=max(all_n_mbs)), lens, group=group
    )


def pack_tensor_dict(data: TensorDict):
    """Pack a tensordict of shape [B, S, ...] into [total_length, ...], leaving other keys unchanged.

    Args:
        data (Dict[str, Any]): Dictionary containing tensors to be packed. Should contain key "attention_mask" with shape [B, S].

    Returns:
        Dict[str, Any]: Dictionary with packed tensors. The "attention_mask" key will be replaced by "cu_seqlens" with shape [B+1].
    """

    assert "attention_mask" in data, "Input data must contain 'attention_mask' key."
    attention_mask = data["attention_mask"]
    assert attention_mask.ndim == 2, "Attention mask must be a 2D tensor."
    bs = attention_mask.shape[0]
    seq_len = attention_mask.shape[1]

    # Calculate cumulative sequence lengths
    lens = attention_mask.sum(dim=1, dtype=torch.int32)
    max_seqlen = lens.max().item()
    cu_seqlens = torch.cumsum(lens, dim=0)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    total_length = int(cu_seqlens[-1].item())
    # Pack tensors
    packed_data = {}
    for key, value in data.items():
        if key == "attention_mask":
            packed_data["cu_seqlens"] = cu_seqlens
            packed_data["max_seqlen"] = max_seqlen
            continue
        # tensor and of shape [B, S, ...]
        if (
            torch.is_tensor(value)
            and value.ndim >= 2
            and value.shape[0] == bs
            and value.shape[1] == seq_len
        ):
            packed_tensor = torch.empty(
                (total_length, *value.shape[2:]), dtype=value.dtype, device=value.device
            )
            # Fill the packed tensor with values from the original tensor
            for i in range(bs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                packed_tensor[start:end] = value[i][: end - start]
            packed_data[key] = packed_tensor
        else:
            packed_data[key] = value

    return TensorDict(**packed_data)


def pad_and_stack_tensors_along_first_dim(tensor_list: List[torch.Tensor]):
    max_length = max(tensor.shape[0] for tensor in tensor_list)
    n_dim = tensor_list[0].ndim
    assert all(
        tensor.ndim == n_dim for tensor in tensor_list
    ), "All tensors must have the same number of dimensions."

    padded_tensors = []
    for tensor in tensor_list:
        pad_mode = (0,) * (2 * (n_dim - 1)) + (0, max_length - tensor.shape[0])
        padded_tensor = F.pad(tensor, pad_mode, value=0.0)
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors, dim=0)


def tensor_container_to(
    d: Dict[str, Any] | torch.Tensor | List[torch.Tensor], *args, **kwargs
):
    """Apply `t.to(*args, **kwargs)` to all tensors in the dictionary.
    Support nested dictionaries.
    """
    new_dict = {}
    if torch.is_tensor(d):
        return d.to(*args, **kwargs)
    elif isinstance(d, list):
        return [
            tensor_container_to(*args, **kwargs) if torch.is_tensor(v) else v for v in d
        ]
    elif isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict) or isinstance(value, list):
                new_dict[key] = tensor_container_to(value, *args, **kwargs)
            elif torch.is_tensor(value):
                new_dict[key] = value.to(*args, **kwargs)
            else:
                new_dict[key] = value
        return new_dict
    else:
        raise ValueError(f"Unsupported type: {type(d)}")


@dataclass
class MicroBatchList:
    data: TensorDict
    mb_spec: MicroBatchSpec
    mbs: List[TensorDict]
    forward_indices: List[int]
    backward_indices: List[int]
    group_lens: List[int]
    padded_mbs: Optional[List[TensorDict]] = None
    # Batch-level padding information
    padding_lengths: Optional[List[int]] = None
    padded_to_lengths: Optional[List[int]] = None
    # sequence-level padding information
    align_to_lengths: Optional[List[int]] = None
    old_cu_seqlens_list: Optional[List[torch.Tensor]] = None

    def to(self, *args, **kwargs):
        mbs = [
            (
                mb.to(*args, **kwargs)
                if isinstance(mb, TensorDict)
                else tensor_container_to(mb, *args, **kwargs)
            )
            for mb in self.mbs
        ]
        data = (
            self.data.to(*args, **kwargs)
            if isinstance(self.data, TensorDict)
            else tensor_container_to(self.data, *args, **kwargs)
        )
        padded_mbs = None
        if self.padded_mbs is not None:
            padded_mbs = [
                (
                    mb.to(*args, **kwargs)
                    if isinstance(mb, TensorDict)
                    else tensor_container_to(mb, *args, **kwargs)
                )
                for mb in self.padded_mbs
            ]
        old_cu_seqlens_list = None
        if self.old_cu_seqlens_list is not None:
            old_cu_seqlens_list = [
                t.to(*args, **kwargs) for t in self.old_cu_seqlens_list
            ]
        return MicroBatchList(
            data=data,
            mb_spec=self.mb_spec,
            mbs=mbs,
            forward_indices=self.forward_indices,
            backward_indices=self.backward_indices,
            group_lens=self.group_lens,
            padded_mbs=padded_mbs,
            padding_lengths=self.padding_lengths,
            padded_to_lengths=self.padded_to_lengths,
            old_cu_seqlens_list=old_cu_seqlens_list,
            align_to_lengths=self.align_to_lengths,
        )


DEFAULT_MAX_TOKENS_PER_GPU = int(1e12)


def split_padded_tensor_dict_into_mb_list(
    data: TensorDict,
    mb_spec: MicroBatchSpec,
    group: Optional[dist.ProcessGroup] = None,
) -> MicroBatchList:
    """Split a padded tensordict into micro-batches based on the attention mask.

    Args:
        data (TensorDict): Dictionary containing padded tensors.
        mb_spec (MicroBatchSpec): Specification for micro-batch splitting.
        group (Optional[dist.ProcessGroup]): Process group for distributed synchronization.

    Returns:
        MicroBatchList: A structure containing the split micro-batches and metadata.
    """
    # TODO: should align sequences first and then split, needs refactor
    assert (
        "attention_mask" in data
    ), "Input data must be padded and contain 'attention_mask' key."
    if mb_spec.max_tokens_per_gpu is None:
        mb_spec = MicroBatchSpec.new(
            mb_spec, max_tokens_per_gpu=DEFAULT_MAX_TOKENS_PER_GPU
        )
    bs = data["attention_mask"].shape[0]
    max_seqlen = data["attention_mask"].shape[1]
    input_lens = data["attention_mask"].sum(1).long().cpu().numpy()

    # check tensor shape, split only 1d tensors with length "total_lens"
    to_split = {}
    not_to_split = {}
    for key, value in data.items():
        if key == "multi_modal_input":
            continue
        if key == "position_ids" or (
            torch.is_tensor(value) and value.numel() == bs * max_seqlen
        ):
            # NOTE: qwen2.5-vl position_ids.numel() == bs * max_seqlen * 3
            to_split[key] = value
        else:
            not_to_split[key] = value

    # split
    group_indices = allocate_balanced_mbs_synced(mb_spec, input_lens, group=group)
    splitted_lens = [
        [input_lens[i] for i in group_index] for group_index in group_indices
    ]
    group_n_seqs = [len(x) for x in splitted_lens]
    group_lens = [sum(x) for x in splitted_lens]

    forward_indices = datapack.flat2d(group_indices)
    backward_indices = np.zeros(bs, dtype=np.int64)
    backward_indices[forward_indices] = np.arange(bs)

    def _split(tensor):
        """Split and pad a tensor based on forward indices and lens."""
        # Unpack the sequence
        unpacked = [tensor[i] for i in range(bs)]
        # Reorder according to forward indices
        reordered = reorder_list(unpacked, forward_indices)
        reordered = torch.stack(reordered)
        # Unpack again according to split lens
        splitted = []
        offset = 0
        for _n_seqs in group_n_seqs:
            splitted.append(reordered[offset : offset + _n_seqs])
            offset += _n_seqs
        return splitted

    to_split = dict_map(to_split, lambda x: _split(x))

    if "multi_modal_input" in data:
        multi_modal_input = data["multi_modal_input"]

        # Prepare the pixel_values and image_grid_thw for each group
        multi_modal_input_split = []

        for group_index in group_indices:
            group_pixel_multi_modal_input = [multi_modal_input[i] for i in group_index]
            # Stack pixel_values for each group (assuming pixel_values is a list of tensors)
            multi_modal_input_split.append(group_pixel_multi_modal_input)
        # Pack the split pixel_values and image_grid_thw back into the data
        to_split["multi_modal_input"] = multi_modal_input_split
    mbs = dict_of_list2list_of_dict(to_split)

    results = []
    # organize splitted micro batches
    assert len(mbs) == len(splitted_lens), (len(mbs), len(splitted_lens))
    for i, (mb, lens) in enumerate(zip(mbs, splitted_lens)):
        results.append(TensorDict(**mb, **not_to_split))

    return MicroBatchList(
        data=data,
        mb_spec=mb_spec,
        mbs=results,
        forward_indices=forward_indices,
        backward_indices=backward_indices.tolist(),
        group_lens=group_lens,
    )


N_TOKENS_PER_PAGE = 256


def pad_packed_tensor_dict(
    data: TensorDict,
    pad_to_length: int,
    pad_value: float = 0.0,
    align_sequences: bool = False,
    align_to_multiple_of: Optional[int] = None,
) -> Tuple[TensorDict, int, torch.Tensor, int]:
    """Pad a packed tensor dict to a specified length.
    This function assumes that the input data contains "cu_seqlens" and "max_seqlen" key,
    and all other tensors of shape [total_length, ] will be padded to `pad_to_length`.
    This function will pad a new sequence filled with `pad_value` to the end of each tensor,
    and update the "cu_seqlens" and "max_seqlen" keys accordingly.

    Args:
        data (TensorDict): Dictionary containing tensors to be packed.
        pad_to_length (int): The length to pad the tensors to. All tensors

    Returns:
        TensorDict: Dictionary with padded tensors and modified "cu_seqlens" and
            "max_seqlen".
        int: The pad length.
    """
    assert "cu_seqlens" in data, "Input data must contain 'cu_seqlens' key."
    assert "max_seqlen" in data, "Input data must contain 'max_seqlen' key."
    cu_seqlens = data["cu_seqlens"]
    max_seqlen = data["max_seqlen"]
    old_cu_seqlens = cu_seqlens.clone()
    total_length = data["cu_seqlens"][-1].item()
    # First pad sequences
    sequence_padded_data = {}
    align_to_length = None
    if align_sequences:
        assert (
            align_to_multiple_of is not None
        ), "align_to_multiple_of must be specified when align_sequences is True."
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_size = input_lens.shape[0]
        # Align sequences to an integer multiple of align_to_multiple_of
        pad_size = (
            align_to_multiple_of - input_lens % align_to_multiple_of
        ) % align_to_multiple_of
        input_lens_padded = input_lens + pad_size
        cu_seqlens_padded = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=cu_seqlens.device
        )
        cu_seqlens_padded[1:] = torch.cumsum(input_lens_padded, dim=0)
        max_seqlens_padded = input_lens_padded.max().item()
        padded_shape = (input_lens_padded.sum().item(),)
        for key, value in data.items():
            if key == "cu_seqlens":
                sequence_padded_data["cu_seqlens"] = cu_seqlens_padded
            elif key == "max_seqlen":
                sequence_padded_data["max_seqlen"] = max_seqlens_padded
            elif key == "position_ids":
                if len(value.shape) == 2 and value.shape[1] == 3:
                    # [total_seq_len, channel] for qwen2.5 vl, channel==3 for t,h,w
                    new_value = torch.zeros(
                        (padded_shape[0], 3), dtype=value.dtype, device=value.device
                    )
                    for i in range(batch_size):
                        new_start = cu_seqlens_padded[i]
                        new_end = cu_seqlens_padded[i + 1]
                        old_start = cu_seqlens[i]
                        old_end = cu_seqlens[i + 1]
                        length = old_end - old_start
                        pad_length = new_end - new_start - length
                        new_value[new_start : new_start + length] = value[
                            old_start:old_end
                        ]
                        new_value[new_start + length : new_end] = (
                            torch.arange(
                                pad_length, dtype=torch.long, device=value.device
                            )
                            .unsqueeze(1)
                            .expand(-1, 3)
                        )
                else:
                    new_value = torch.zeros(
                        padded_shape, dtype=value.dtype, device=value.device
                    )
                    for i in range(batch_size):
                        new_start = cu_seqlens_padded[i]
                        new_end = cu_seqlens_padded[i + 1]
                        new_value[new_start:new_end] = torch.arange(
                            new_end - new_start, dtype=value.dtype, device=value.device
                        )
                sequence_padded_data[key] = new_value
            elif torch.is_tensor(value) and value.numel() == total_length:
                new_value = torch.full(
                    padded_shape,
                    fill_value=pad_value,
                    dtype=value.dtype,
                    device=value.device,
                )
                for i in range(batch_size):
                    new_start = cu_seqlens_padded[i]
                    start = cu_seqlens[i]
                    end = cu_seqlens[i + 1]
                    length = end - start
                    new_value[new_start : new_start + length] = value[start:end]
                sequence_padded_data[key] = new_value
            else:
                sequence_padded_data[key] = value

        data = TensorDict(sequence_padded_data, batch_size=data.batch_size)
        align_to_length = cu_seqlens_padded[-1].item()
        # ensure pad_to_length is a integer multiple of both align_to_multiple_of and N_TOKENS_PER_PAGE
        lcm = np.lcm(align_to_multiple_of, N_TOKENS_PER_PAGE).item()
        pad_to_length = (pad_to_length + lcm - 1) // lcm * lcm

        cu_seqlens = data["cu_seqlens"]
        max_seqlen = data["max_seqlen"]
        total_length = data["cu_seqlens"][-1].item()
        if pad_to_length < total_length:
            # NOTE: In some occasion where sequence lengths, sequence padding will make total length
            # exceed expected `pad_to_length`. This happens more often when sequence lengths are small.
            # In this case, we increase pad_to_length.
            pad_to_length = (total_length + lcm - 1) // lcm * lcm

    # Pad batch
    pad_length = pad_to_length - total_length
    assert (
        pad_length >= 0
    ), f"pad_to_length {pad_to_length} must be greater than or equal to total length {total_length}."
    new_cu_seqlens = F.pad(cu_seqlens, (0, 1), value=pad_to_length)
    new_max_seqlen = max(max_seqlen, pad_length)
    padded_data = {}
    for key, value in data.items():
        if key == "cu_seqlens":
            padded_data[key] = new_cu_seqlens
        elif key == "max_seqlen":
            padded_data[key] = new_max_seqlen
        elif key == "position_ids":
            # [total_seqlen, channel] for qwen2.5 vl, channel==3 for t,h,w
            if len(value.shape) == 2 and value.shape[1] == 3:
                pad = (
                    torch.arange(pad_length, dtype=torch.long, device=value.device)
                    .unsqueeze(1)
                    .expand(-1, 3)
                )
                padded_tensor = torch.cat([value, pad])
            else:
                pad = torch.arange(pad_length, dtype=torch.long, device=value.device)
                padded_tensor = torch.cat([value, pad])
            padded_data[key] = padded_tensor
        elif torch.is_tensor(value) and value.numel() == total_length:
            # Pad the tensor to the new total length
            padded_tensor = torch.nn.functional.pad(
                value, (0, pad_length), value=pad_value
            )
            padded_data[key] = padded_tensor
        else:
            padded_data[key] = value
    return (
        TensorDict(padded_data, batch_size=data.batch_size),
        pad_length,
        old_cu_seqlens,
        align_to_length,
    )


def pad_mb_list(
    mb_list: MicroBatchList,
    pad_value: float = 0.0,
    pad_to_maximum: bool = False,
    align_sequences: bool = False,
    align_to_multiple_of: Optional[int] = None,
) -> MicroBatchList:
    """Pad the micro-batch list to the maximum length or to a specific size to:
        1. Reduce memory fragmentation.
        2. Align sequences to an integer multiple of `align_to_multiple_of`
        to be equally sliced into context and sequence parallel ranks.

    Args:
        mb_list (MicroBatchList): The micro-batch list to pad.
        pad_value (float, optional): The value to pad the tensors with. Defaults to 0.0.
        pad_to_maximum (bool, optional): Whether to pad to the maximum length specified in `mb_spec`. Defaults to False.
        align_sequences (bool, optional): Whether to align sequences to an integer multiple of `align_to_multiple_of`. Defaults to False.
        align_to_multiple_of (int, optional): The size to align sequences to. Defaults to None.

    Returns:
        MicroBatchList: The padded micro-batch list.
    """
    if align_sequences:
        assert (
            align_to_multiple_of is not None
        ), "align_to_multiple_of must be specified when align_sequences is True."
    padded_mb_inputs, pad_lengths = [], []
    pad_to_lengths = []
    old_cu_seqlens_list = []
    align_to_lengths = []
    if pad_to_maximum and (
        mb_list.mb_spec.max_tokens_per_gpu is None
        or mb_list.mb_spec.max_tokens_per_gpu == DEFAULT_MAX_TOKENS_PER_GPU
    ):
        logger.warning(
            f"Unable to pad to maximum because max_tokens_per_gpu is not properly set."
        )
        pad_to_maximum = False
    for mb, l in zip(mb_list.mbs, mb_list.group_lens):
        if pad_to_maximum:
            pad_to_length = mb_list.mb_spec.max_tokens_per_mb
        else:
            # NOTE: GPU page size is 2MB
            # Take hidden size 4096 with bf16 dtype as an example,
            # the batch size of a page is 256
            pad_to_length = (
                (int(l) + N_TOKENS_PER_PAGE - 1)
                // N_TOKENS_PER_PAGE
                * N_TOKENS_PER_PAGE
            )
        padded_mb, pad_len, old_cu_seqlens, align_to_length = pad_packed_tensor_dict(
            mb,
            pad_to_length,
            pad_value=pad_value,
            align_sequences=align_sequences,
            align_to_multiple_of=align_to_multiple_of,
        )
        padded_mb_inputs.append(padded_mb)
        pad_lengths.append(pad_len)
        pad_to_lengths.append(pad_to_length)
        old_cu_seqlens_list.append(old_cu_seqlens)
        align_to_lengths.append(align_to_length)
    mb_list.padded_mbs = padded_mb_inputs
    mb_list.padding_lengths = pad_lengths
    mb_list.padded_to_lengths = pad_to_lengths
    if align_sequences:
        mb_list.old_cu_seqlens_list = old_cu_seqlens_list
        mb_list.align_to_lengths = align_to_lengths
    return mb_list


def unpad_logits(
    logits: torch.Tensor,
    padding_length: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    old_cu_seqlens: Optional[torch.Tensor] = None,
):
    # TODO: when using megatron, logits are in fp32,
    # create new logits in bucket to reduce peak memory usage
    # First unpad batch
    if padding_length > 0:
        logits = logits[:-padding_length]

    # Then unpad according to old_cu_seqlens
    if old_cu_seqlens is not None:
        new_logits = torch.empty(
            (old_cu_seqlens[-1].item(), *logits.shape[1:]),
            dtype=logits.dtype,
            device=logits.device,
        )
        batch_size = old_cu_seqlens.shape[0] - 1
        for i in range(batch_size):
            old_start = old_cu_seqlens[i].item()
            old_end = old_cu_seqlens[i + 1].item()
            start = cu_seqlens[i].item()
            length = old_end - old_start
            new_logits[old_start:old_end] = logits[start : start + length]

    return new_logits


def unsqueeze_packed_tensor_dict(data: TensorDict) -> TensorDict:
    assert "cu_seqlens" in data, "Input data must contain 'cu_seqlens' key."
    assert "max_seqlen" in data, "Input data must contain 'max_seqlen' key."

    total_length = data["cu_seqlens"][-1].item()
    new_data = {}
    for key, value in data.items():
        if key == "position_ids" or (
            key
            not in [
                "cu_seqlens",
                "max_seqlen",
            ]
            and torch.is_tensor(value)
            and value.numel() == total_length
        ):
            new_data[key] = value.unsqueeze(dim=0)
        else:
            new_data[key] = value
    return TensorDict(new_data, batch_size=data.batch_size)


def unsqueeze_mb_list(
    mb_list: MicroBatchList,
) -> MicroBatchList:
    """Unsqueeze the packed tensordict in the micro-batch list."""
    new_padded_mbs = []
    for i, mb in enumerate(mb_list.mbs):
        if mb_list.padded_mbs is not None:
            new_padded_mbs.append(unsqueeze_packed_tensor_dict(mb_list.padded_mbs[i]))
    mb_list.padded_mbs = new_padded_mbs if mb_list.padded_mbs is not None else None
    return mb_list


def amend_position_ids(data: TensorDict) -> TensorDict:
    assert "attention_mask" in data, "Input data must contain 'attention_mask' key."

    attn_mask = data["attention_mask"]
    bs, seqlen = attn_mask.shape[:2]
    position_ids = (
        torch.arange(0, seqlen, dtype=torch.long, device=attn_mask.device)
        .unsqueeze(0)
        .expand(bs, -1)
    )
    position_ids.masked_fill(~attn_mask.bool(), 0)
    data["position_ids"] = position_ids
    return data


def broadcast_tensor(tensor, src_rank=0, group=None, device=None):
    """
    Broadcast a tensor from source rank to all other ranks in the process group.

    Args:
        tensor: Tensor on source rank, None on non-source ranks
        src_rank: The rank that holds the tensor to broadcast (default: 0)
        group: The process group to use for broadcasting (default: None, uses the default group)
        device: The device of the output tensor.

    Returns:
        Tensor: The broadcasted tensor on all ranks
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")

    current_rank = dist.get_rank()

    # On source rank, prepare the tensor for broadcasting
    if current_rank == src_rank:
        if tensor is None:
            raise ValueError(f"Tensor cannot be None on source rank {src_rank}")
        assert tensor.device == device
        # Prepare metadata as Python objects
        metadata = {"shape": list(tensor.shape), "dtype": tensor.dtype}

        # Broadcast metadata using broadcast_object_list
        metadata_list = [metadata]
        dist.broadcast_object_list(metadata_list, src=src_rank, group=group)

        # Broadcast the actual tensor
        dist.broadcast(tensor, src=src_rank, group=group)

        return tensor

    else:
        # On non-source ranks, receive metadata
        metadata_list = [None]
        dist.broadcast_object_list(metadata_list, src=src_rank, group=group)

        metadata = metadata_list[0]
        tensor_shape = metadata["shape"]
        dtype = metadata["dtype"]

        # Create tensor with the received shape and dtype
        tensor = torch.empty(tensor_shape, dtype=dtype, device=device)

        # Receive the actual tensor data
        dist.broadcast(tensor, src=src_rank, group=group)

        return tensor


def _unpad_unflatten(x, shape):
    assert len(x.shape) == 1
    pad_size = x.numel() - np.prod(shape)
    assert pad_size >= 0, pad_size
    return x[: x.numel() - pad_size].view(*shape)


def _flatten_pad_to_max_numel(x, shapes):
    pad_size = max(np.prod(shape) for shape in shapes) - x.numel()
    assert pad_size >= 0, pad_size
    return torch.nn.functional.pad(x.view(-1), (0, pad_size), value=0)


def all_gather_tensor_container(data, group=None) -> List:
    if torch.is_tensor(data):

        local_shape = list(data.shape)
        shapes = [None for _ in range(dist.get_world_size(group))]
        dist.all_gather_object(shapes, local_shape, group=group)

        y = _flatten_pad_to_max_numel(data, shapes)

        ys = [torch.empty_like(y) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(ys, y, group=group)

        return [_unpad_unflatten(y, shape) for y, shape in zip(ys, shapes)]

    if isinstance(data, list):
        data = [all_gather_tensor_container(d) for d in data]
        return list(zip(*data))

    if isinstance(data, (dict, TensorDict)):
        results = {k: all_gather_tensor_container(v) for k, v in data.items()}
        results = [
            {k: v[i] for k, v in results.items()}
            for i in range(dist.get_world_size(group))
        ]
        if isinstance(data, TensorDict):
            results = [
                TensorDict(r, batch_size=[r["attention_mask"].shape[0]])
                for r in results
            ]
        return results

    results = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(results, data, group=group)
    return results

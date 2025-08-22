import pickle
import subprocess

import pytest
import torch
from tensordict import TensorDict
from torch.testing import assert_close

from areal.utils.data import concat_padded_tensors
from areal.utils.network import find_free_ports


def assert_tensor_container_close(x1, x2):
    assert type(x1) == type(x2), (type(x1), type(x2))
    if torch.is_tensor(x1):
        assert_close(x1, x2)
        return
    if isinstance(x1, list):
        assert len(x1) == len(x2), (len(x1), len(x2))
        [assert_tensor_container_close(xx1, xx2) for xx1, xx2 in zip(x1, x2)]
        return
    if isinstance(x1, (dict, TensorDict)):
        assert x1.keys() == x2.keys(), (x1.keys(), x2.keys())
        for k in x1.keys():
            assert_tensor_container_close(x1[k], x2[k])
        return
    assert x1 == x2


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("granularity", [1, 2, 4])
def test_redistribute(world_size, granularity, tmp_path):
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={world_size}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "areal/tests/torchrun/redistribute.py",
                f"--dump-path={str(tmp_path)}",
                f"--granularity={granularity}",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr.decode()}")

    redistributed_data = []
    for i in range(world_size):
        with open(tmp_path / f"redistributed{i}.pkl", "rb") as f:
            redistributed_data.append(pickle.load(f))

    for x in redistributed_data[1:]:
        assert_tensor_container_close(x.all_data, redistributed_data[0].all_data)
    for x in redistributed_data[1:]:
        assert_tensor_container_close(
            x.group_indices, redistributed_data[0].group_indices
        )

    all_data = redistributed_data[0].all_data
    group_indices = redistributed_data[0].group_indices
    for x, indices in zip(redistributed_data, group_indices):
        data = concat_padded_tensors([all_data[i] for i in indices])
        assert_tensor_container_close(x.data, data)

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from areal.utils.network import find_free_ports, gethostip


def get_placement_group_master_ip_and_port(placement_group: PlacementGroup):
    def _master_ip_and_port():
        host_ip = gethostip()
        port = find_free_ports(1, (10000, 60000))[0]
        return host_ip, port

    future = ray.remote(
        num_cpus=1,
        num_gpus=0,
        memory=10 * 1024 * 1024,  # Convert MB to bytes
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=0,
        ),
    )(_master_ip_and_port).remote()
    return ray.get(future)

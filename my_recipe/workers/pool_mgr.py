from verl.single_controller.ray import RayResourcePool
from verl.trainer.ppo.ray_trainer import ResourcePoolManager


class myResourcePoolManager(ResourcePoolManager):
    def create_resource_pool(self, max_colocate_count: int = 2):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=max_colocate_count,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

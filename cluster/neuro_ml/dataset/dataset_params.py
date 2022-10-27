from dataclasses import dataclass
from enum import Enum


class SimulationEnum(Enum):
    mikkel = "mikkel"
    rust = "rust"
    rust_dense_separated_cluster = "rust_dense_separated_cluster"
    python_dense = "python_dense"
    python_dense_separated_cluster = "python_dense_separated_cluster"
    python_dense_no_inhibatory_params = "python_no_inhibatory_params"
    inner_products = "inner_products"


@dataclass
class DatasetParams:
    network_type : str
    n_clusters: int
    cluster_sizes: list[int]
    n_neurons: int
    n_timesteps: int
    timestep_bin_length: int
    number_of_files: int
    output_dim: int

    @property
    def foldername(self):
        return f"{self.network_type}/cluster_sizes_{self.cluster_sizes}_n_steps_{self.n_timesteps}" 

    
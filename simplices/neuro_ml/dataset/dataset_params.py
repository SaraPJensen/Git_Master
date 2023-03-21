from dataclasses import dataclass
from enum import Enum
import os


@dataclass
class DatasetParams:
    network_type : str
    n_clusters: int
    cluster_sizes: list[int]
    n_neurons: int
    n_neurons_remaining: int
    neurons_remove : int
    n_timesteps: int
    timestep_bin_length: int
    number_of_files: int
    output_dim: int
    advanced: bool 



    @property
    def foldername(self):
        #return f"{self.network_type}/cluster_sizes_{self.cluster_sizes}_n_steps_{self.n_timesteps}" 
        return f"neurons_{self.n_neurons}"

    @property
    def save_folder(self):
        #return f"{self.network_type}/cluster_sizes_{self.cluster_sizes}_n_steps_{self.n_timesteps}"
        folder = f"neurons_{self.n_neurons}"

        # if not os.path.isdir(folder):
        #     os.mkdir(folder)

        return folder




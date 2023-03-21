from dataclasses import dataclass
from enum import Enum


@dataclass
class DatasetParams:
    network_type : str
    n_clusters: int
    cluster_sizes: list[int]
    n_neurons: int
    n_neurons_remaining: int
    neurons_remove: int
    n_timesteps: int
    number_of_files: int
    weight_threshold: float
    simplex_threshold: int
    scaled_loss: bool


    @property
    def foldername(self):
        return f"{self.network_type}/cluster_sizes_{self.cluster_sizes}_n_steps_{self.n_timesteps}" 

    
    @property
    def save_folder(self):
        if self.scaled_loss:
            return f"{self.network_type}/cluster_sizes_{self.cluster_sizes}_n_steps_{self.n_timesteps}_simplex_threshold_{self.simplex_threshold}_scaled_loss"
        
        else:
            return f"{self.network_type}/cluster_sizes_{self.cluster_sizes}_n_steps_{self.n_timesteps}_simplex_threshold_{self.simplex_threshold}"




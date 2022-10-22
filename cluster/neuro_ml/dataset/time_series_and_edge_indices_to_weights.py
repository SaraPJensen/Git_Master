from neuro_ml.dataset.abstract import AbstractDataset

#Used for edge regressor, egde_classifier, gcn and graph_transformer

class TimeSeriesAndEdgeIndicesToWeightsDataset(AbstractDataset):  #Inherits the functions and definitions from the AbstractDataset class
    IS_GEOMETRIC = True

    def __init__(
        self,
        filenames,
        dataset_params,
        is_classifier,
    ) -> None:
        super().__init__(
            filenames,
            dataset_params,
            is_classifier,
        )

        self._create_fully_connected_edge_index(dataset_params.n_neurons)  #Creates self.edge_index

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "inputs": {"X": self.X[idx], "edge_index": self.edge_index[idx]},
            "y": self.y[idx],
        }

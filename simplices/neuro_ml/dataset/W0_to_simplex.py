from neuro_ml.dataset.simplex_abstract import SimplexAbstract 

class W0_to_simplex_Dataset(SimplexAbstract):
    IS_GEOMETRIC = True

    def __init__(self, filenames, dataset_params):

        super().__init__(filenames, dataset_params)

    def __len__(self):
        return len(self.W0)

    def __getitem__(self, idx):
        return {"W0": self.W0[idx], "edge_index": self.edge_index[idx], "y": self.y[idx]}
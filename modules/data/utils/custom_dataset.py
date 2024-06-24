import torch_geometric


class CustomDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, data_list, data_dir, data_name="Custom", transform=None):
        self.data_list = data_list
        self.data_name = data_name
        super().__init__(data_dir, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        self.save(self.data_list, self.processed_paths[0])

    def __repr__(self):
        return f"{self.data_name}({len(self)})"

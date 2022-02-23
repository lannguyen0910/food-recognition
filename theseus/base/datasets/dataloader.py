from torch.utils.data import DataLoader

class DataLoaderWithCollator(DataLoader):
    def __init__(self, dataset, **kwargs) -> None:
        self.dataset = dataset
        super().__init__(
          dataset=dataset, 
          collate_fn=dataset.collate_fn, 
          **kwargs)

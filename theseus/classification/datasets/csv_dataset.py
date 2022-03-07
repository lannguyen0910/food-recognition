import pandas as pd
from typing import List, Optional

from theseus.utilities.loggers.observer import LoggerObserver
from .dataset import ClassificationDataset

LOGGER = LoggerObserver.getLogger('main')

class CSVDataset(ClassificationDataset):
    r"""CSVDataset multi-labels classification dataset

    Reads in .csv file with structure below:
        filename | label
        -------- | -----

    image_dir: `str`
        path to directory contains images
    csv_path: `str`
        path to csv file
    txt_classnames: `str`
        path to txt file contains classnames
    transform: Optional[List]
        transformatin functions
    test: bool
        whether the dataset is used for training or test
        
    """

    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        txt_classnames: str,
        transform: Optional[List] = None,
        **kwargs
    ):
        super(CSVDataset, self).__init__(**kwargs)
        self.image_dir = image_dir
        self.txt_classnames = txt_classnames
        self.csv_path = csv_path
        self.transform = transform
        self._load_data()

    def _load_data(self):
        """
        Read data from csv and load into memory
        """
       
        with open(self.txt_classnames, 'r') as f:
            self.classnames = f.read().splitlines()
        
        # Mapping between classnames and indices
        for idx, classname in enumerate(self.classnames):
            self.classes_idx[classname] = idx
        self.num_classes = len(self.classnames)

        # Load csv
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            image_name, label = row
            self.fns.append([image_name, label])

    def _calculate_classes_dist(self):
        """
        Calculate distribution of classes
        """
        LOGGER.text("Calculating class distribution...", LoggerObserver.DEBUG)
        self.classes_dist = []

        # Load csv
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            _, label = row
            self.classes_dist.append(self.classes_idx[label])
        return self.classes_dist
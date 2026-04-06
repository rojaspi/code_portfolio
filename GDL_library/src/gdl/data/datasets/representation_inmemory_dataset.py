import torch
from torch_geometric.data import InMemoryDataset

from ..representations import RepresentationBuilder
from .dataset_base import DatasetBases


class RepresentationInMemoryDataset(DatasetBases, InMemoryDataset):
    """
    Class to create representations datasets. If data is previusly processed, this class opens
    the dataset file. If data is not previusly processed, the class processes each representation
    first and saves it a a single file.
    """
    def __init__(self, raw_data_dir, root, dataset_name, builder: RepresentationBuilder, label_dict: dict = {}):
        """
        Initialize the RepresentationDataset object
        
        :param raw_data_dir: Folder where all the structures are stored.
        :param root: Folder where the processed is or will be stored.
        :param dataset_name: 
        :param builder: The RepresentationBuilder previusly configured with representation options
        :param label_dict: A dictionary using the structure name (eg: xxx.pdb) as key, and label as value
        """
        self.raw_data_dir = raw_data_dir
        self.dataset_name = dataset_name
        self.builder = builder
        self.labels = label_dict
        
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only= False)


    @property
    def processed_file_names(self):
        """
        Creates the name of the single file storing the dataset
        """
        return [f'{self.dataset_name}_{self.builder.representation}.pt']


    def process(self):
        """
        Processes each structure and saves them all in just one .pt file
        """
        data_list = []
        failed_items = []

        for data, file_path in self._iter_raw_data():
            if data is None:
                failed_items.append(file_path)
                continue
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        #self.save_metadata(len(data_list))
        if failed_items:
            print(f"Failed items: {failed_items}")

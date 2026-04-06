import os
import torch
from torch_geometric.data import Dataset
import os.path as osp

from ..representations import RepresentationBuilder
from .dataset_base import DatasetBases
   


class RepresentationDataset(DatasetBases, Dataset):
    """
    Class to create representations datasets. If data is previusly processed, this class opens each entry
    in the dataset when called. If data is not previusly processed, the class processes each representation
    first, and saves it as separate files.
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

    

    @property
    def processed_file_names(self):
        """
        Lists the name of each processed file
        """
        exclude_files = {"pre_filter.pt", "pre_transform.pt"}
        if self.raw_data_dir == None:
            return [f for f in os.listdir(self.processed_dir)
                if f.endswith('.pt') and f not in exclude_files]

        names = []
        for idx, _ in enumerate(os.listdir(self.raw_data_dir)):
            names.append(f'data_{idx}.pt')

        return names
    

    def len(self):
        """
        Dataset length
        """
        return len(self.processed_file_names)


    def get(self, idx):
        """
        Accesses each entry in the dataset by index
        """
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data
    

    def process(self):
        """
        Processes each structure and saves them in multiple .pt files
        """
        failed_items = []

        for data, file_path in self._iter_raw_data():
            if data is None:
                failed_items.append(file_path)
                continue

            torch.save(data, file_path)

        #self.save_metadata(len(os.listdir(self.processed_dir)))
        if failed_items:
            print(f"Failed items: {failed_items}")
    
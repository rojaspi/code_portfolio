import os
import sys
import json
import torch
import warnings
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data, Dataset
import os.path as osp

from ..representations import data_representation
from .. representations import featurizer


class RepresentationBases():
    def process_representation(self, path, label = None):
        """
        Processes each file given a representation
        """
        if self.distance == None:
            distance = 8
        else:
            distance = self.distance

        if self.edge_features == None:
            edge_features = "distance"
        else:
            edge_features = self.edge_features

        match self.representation.lower():
            case "graph":
                repr = data_representation.GraphRepresentation(structure = path, featurizer= self.featurizer,
                                                               label=label, cutoff_distance=distance, edges_method=edge_features)
            case "point_cloud" | "point cloud":
                repr = data_representation.PointCloudRepresentation(structure = path, featurizer= self.featurizer,
                                                                    label=label)
            case "voxel" | "grid":
                repr = data_representation.Grid3DRepresentation(structure = path, featurizer= self.featurizer,
                                                                label=label, voxel_size= distance)
            case _:
                sys.exit("Invalid representation")

        return repr.representation
    

    def save_metadata(self, data_size = "unkown"):
        """
        Saves dataset metadata
        """
        metadata = {
            "dataset_name": self.dataset_name,
            "representation": self.representation,
            "feature_list": self.featurizer.feature_list,
            "residue_center": self.featurizer.residue_center,
            "distance": self.distance,
            "num_graphs": data_size,
            "data_root": self.root,
        }

        with open(osp.join(self.root, f'{self.dataset_name}_{self.representation}_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
    
    
    @classmethod
    def load_from_json(cls, metadata_path):
        """
        Initializes a dataset using the .json metadata file created during data processing.
        """
        with open(metadata_path, 'r') as file:
            metadata_dict = json.load(file)

        return cls(
            raw_data_dir=None,
            root=metadata_dict["data_root"],
            dataset_name=metadata_dict["dataset_name"],
            representation = metadata_dict["representation"],
            #residue_center = metadata_dict["residue_center"],
            #feature_list = metadata_dict["feature_list"],
            distance = metadata_dict["distance"]
        )
    


class RepresentationDataset(Dataset, RepresentationBases):
    """
    
    """
    def __init__(self, raw_data_dir, root, dataset_name, representation:str,
                 edge_features:str = None,
                 featurizer = featurizer.ResidueFeaturizer(),
                 distance:float = None, label_dict: dict = None):
        """
        Initialize the RepresentationDataset object

        Args:
            raw_data_dir
            root
            dataset_name
            representation
            residue_center
            feature_list
            distance
            label_dict
        """
        self.raw_data_dir = raw_data_dir
        self.dataset_name = dataset_name
        self.representation = representation
        self.edge_features = edge_features
        self.featurizer = featurizer
        self.distance = distance
        self.labels = label_dict
        
        super().__init__(root=root)

        if self.representation.lower() in ["point_cloud", "point cloud"] and self.distance is not None:
            warnings.warn(
                "Distance parameter has no influence over point cloud representations",
                UserWarning
            )
        if self.representation.lower() in ["point_cloud", "point cloud", "voxel", "grid"] and self.edge_features is not None:
            warnings.warn(
                "Edge features parameter has no influence over voxel/grid and point cloud representations",
                UserWarning
            )


    @property
    def raw_file_names(self):
        """
        Get the names of every file in the directory and uses them as raw data.
        """
        return os.listdir(self.raw_data_dir)


    @property
    def processed_file_names(self):
        """
        Creates the dataset name
        """
        exclude_files = {"pre_filter.pt", "pre_transform.pt"}
        if self.raw_data_dir == None:
            return [f for f in os.listdir(self.processed_dir)
                if f.endswith('.pt') and f not in exclude_files]

        names = []
        idx = 0
        for raw_file in os.listdir(self.raw_data_dir):
            names.append(f'data_{idx}.pt')
            idx+=1

        return names


    def len(self):
        return len(self.processed_file_names)


    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data


    def download(self):
        """
        Overrides download to bypass data downloading.
        """
        pass


    def process(self):
        """
        """
        failed_items = []

        has_labels = False
        # Buscar mejor forma
        if self.labels != None:
            has_labels = True
        
        idx = 0

        for raw_file in tqdm(os.listdir(self.raw_data_dir)):

            #Saltamos los ya procesados
            if os.path.exists(os.path.join(self.processed_dir, f'data_{idx}.pt')):
                #print(idx, raw_file)
                idx += 1
                continue  # Ya procesado

            file_path = osp.join(self.raw_data_dir, raw_file)

            # Obtenemos cada uno de los grafos
            if has_labels:
                #Tengo que corregir esto, lanza error cuando no existe la label
                label = self.labels.get(raw_file, None)
                if label == None:
                    print("File "+ raw_file + " skipped due to missing label. Please retry.")
                    idx += 1
                    continue
            
            try:
                data = self.process_representation(file_path, label=label)
                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            
            except Exception as e:
                print(f"Error procesando {file_path}: {e}")
                failed_items.append((file_path, str(e)))
            
            idx += 1

        print(failed_items)
        self.save_metadata(self.len())



class RepresentationInMemoryDataset(InMemoryDataset, RepresentationBases):
    """
    
    """
    def __init__(self, raw_data_dir, root, dataset_name, representation:str,
                 edge_features:str = None,
                 featurizer = featurizer.ResidueFeaturizer(),
                 distance:float = None, label_dict: dict = None):
        """
        Initialize the RepresentationDataset object

        Args:
            raw_data_dir
            root
            dataset_name
            representation
            residue_center
            distance
            label_dict
        """
        self.raw_data_dir = raw_data_dir
        self.dataset_name = dataset_name
        self.representation = representation
        self.edge_features = edge_features
        self.featurizer = featurizer
        self.distance = distance
        self.labels = label_dict
        
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only= False)

        
        if self.representation.lower() in ["point_cloud", "point cloud"] and self.distance is not None:
            warnings.warn(
                "Distance parameter has no influence over point cloud representations",
                UserWarning
            )
        if self.representation.lower() in ["point_cloud", "point cloud", "voxel", "grid"] and self.edge_features is not None:
            warnings.warn(
                "Edge features parameter has no influence over voxel/grid and point cloud representations",
                UserWarning
            )


    @property
    def raw_file_names(self):
        """
        Get the names of every file in the directory and uses them as raw data.
        """
        return os.listdir(self.raw_data_dir)


    @property
    def processed_file_names(self):
        """
        Creates the dataset name
        """
        return [f'{self.dataset_name}_{self.representation}.pt']


    def download(self):
        """
        Overrides download to bypass data downloading.
        """
        pass


    def process(self):
        """
        """
        data_list = []
        failed_items = []

        has_labels = False
        if self.labels != {}:
            has_labels = True
            
        for raw_file in tqdm(os.listdir(self.raw_data_dir)):

            file_path = osp.join(self.raw_data_dir, raw_file)
            
            # Obtenemos cada uno de los grafos
            if has_labels:
                label = self.labels[raw_file]
                if label == None:
                    print("File "+ raw_file + " skipped due to missing label.")
                    continue
            data = self.process_representation(file_path, label=label)
            try:
                data = self.process_representation(file_path, label=label)
                data_list.append(data)
            
            except Exception as e:
                print(f"Error procesando {file_path}: {e}")
                failed_items.append((file_path, str(e)))

        # Guardamos el dataset completo
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        self.save_metadata(len(data_list))
import os
import json
from tqdm import tqdm
import os.path as osp


class DatasetBases():
    @property
    def raw_file_names(self):
        """
        Get the names of every file in the directory and uses them as raw data.
        """
        return os.listdir(self.raw_data_dir)
    

    def download(self):
        """
        Overrides download to bypass data downloading.
        """
        pass
    
    # I need to correct data_root to work using its own location
    def save_metadata(self, data_size = "unkown"):
        """
        Writes the dataset metadata on the dataset root
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


    def _iter_raw_data(self):
        labels = self.labels or {}
        has_labels = bool(labels)

        os.makedirs(self.processed_dir, exist_ok=True)

        for idx, raw_file in enumerate(tqdm(os.listdir(self.raw_data_dir))):
            file_path = os.path.join(self.raw_data_dir, raw_file)
            target_path = os.path.join(self.processed_dir, f"data_{idx}.pt")

            if os.path.exists(target_path):
                continue
            
            label = labels.get(raw_file) if has_labels else None
            if has_labels and label is None:
                print(f"File {raw_file} skipped due to missing label.")
                continue

            try:
                data = self.builder.process_representation(file_path, label=label)
                yield data, target_path
            except Exception as e:
                print(f"Error procesando {file_path}: {e}")
                yield None, raw_file
    
    
    @classmethod
    def load_from_json(cls, metadata_path):
        """
        Initializes a dataset using the .json metadata file created during data processing.
        """
        directory = os.path.dirname(metadata_path)
        with open(metadata_path, 'r') as file:
            metadata_dict = json.load(file)

        return cls(
            raw_data_dir=None,
            root=directory,
            dataset_name=metadata_dict["dataset_name"],
            representation = metadata_dict["representation"],
            #residue_center = metadata_dict["residue_center"],
            #feature_list = metadata_dict["feature_list"],
            distance = metadata_dict["distance"]
        )

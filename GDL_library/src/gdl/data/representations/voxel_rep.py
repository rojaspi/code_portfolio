from .base_rep import BaseRepresentation

import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from ..featurization.base_feat import Featurizer


class Grid3DRepresentation(BaseRepresentation):
    """
    This class is used to generate 3D Grids from structure files. Each voxel in the 3d grid represents
    all contents in a selected distance. As an example in sapce of 10 armstrong, a voxel size of 2 would create 5 voxels.
    These voxels are populated by aggregating the features of all residues or atoms inside the voxel.
    """
    def __init__(self, structure, featurizer: Featurizer, voxel_size = 10.0, label=None):
        """
        Initializes 3D grid representation.
        
        :param structure: path to structure
        :param featurizer: The featurizer previusly configured with the features to extract
        :type featurizer: Featurizer
        :param voxel_size: Size of the voxels created in the 3D grid
        :param label:  dictionary using the structure name (eg: xxx.pdb) as key, and label as value
        """
        super().__init__(structure, featurizer = featurizer, label = label)
        self.features, self.coords, self.bonds, self.bonds_features = self.featurizer.process_structure(self.structure, False)
        self.voxel_size = voxel_size


    def _compute_representation(self):
        """
        Computes the 3d Grid. Checks wich point lands in each voxel, and then aggregates their features.

        :return: A representation as a data object. X corresponds to the numpy grid, and y the label.
        :rtype: torch_geometric.data.Data
        """
        coords_copy = self.coords
        # En esta parte creamos el grid
        min_vals = np.min(coords_copy, axis=0)
        max_vals = np.max(coords_copy, axis=0)

        x_len, y_len, z_len = np.ceil((max_vals - min_vals) / self.voxel_size).astype(int)
        
        # Line to concatenate features
        #node_features = np.concatenate((coords_copy, self.residue_encoding), axis = 1)
        node_features = self.features
        f_size = node_features.shape[1]
        
        grid = np.zeros((x_len, y_len, z_len, f_size))

        # Centraremos cada coordenada en 0
        coords_copy -= min_vals

        for (id, xyz) in enumerate(coords_copy):
            x, y, z = (xyz // self.voxel_size).astype(int)
            
            if f_size == 1:
                grid[x,y,z] += [1]
            else:
                #Here we can use different aggregations <-------------------------------------------------
                grid[x,y,z] += node_features[id]

        return Data(x = grid, y =self.label)


    def visualize_representation(self):
        """
        Generates a visualization of the representation.
        """
        mask = np.any(self.representation.x != 0, axis=3)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.voxels(mask, facecolors='cyan', edgecolors='r', alpha=0.5)
        plt.show()
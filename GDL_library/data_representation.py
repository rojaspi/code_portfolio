from . import feature_extraction
from .featurizer import Featurizer

from abc import ABC, abstractmethod

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


class BaseRepresentation:
    """
    This class is used to define shared methods and parameters between different representations.
    """
    def __init__(self, structure, featurizer: Featurizer, label = None):
        """
        Initializes BaseRepresentation class
        
        :param self:
        :param structure: Path to structure file
        :param featurizer: Needs featurizer to define residue or all atom granularity and compute features
        :type featurizer: Featurizer
        :param label: value or class of the representation
        """
        self.structure = structure
        self.label = label
        self.featurizer = featurizer
    

    @abstractmethod
    def __compute_representation(self):
        """
        Shared name to compute each representation type
        """
        pass


    @abstractmethod
    def visualize_representation(self):
        """
        Shared name to visualize each representation
        """
        pass



class GraphRepresentation(BaseRepresentation):
    """
    This class is used to generate graphs from structure files. Edges are created by the featurizer (residue or atomic bonds)
    or by distance using a cutoff distance. A edge is created if two nodes are close enough, dictated by the cutoof distance.
    """
    def __init__(self, structure, featurizer: Featurizer, edges_method = "distance", cutoff_distance = 10.0, label=None):
        """
        Initializes GraphRepresentation class
        
        :param structure: path to structure
        :param featurizer: The featurizer previusly configured with the features to extract
        :type featurizer: Featurizer
        :param edges_method: How the edges will be computed.
                                - distance: for distance graphs
                                - featurizer: to use featurizer bond features
                                - mixed: to use both
        :param cutoff_distance: Cutoff distance for distance graph
        :param label:  dictionary using the structure name (eg: xxx.pdb) as key, and label as value
        """
        self.cutoff = cutoff_distance
        self.edges_method = edges_method
        self.process_bonds = self.__check_process_bonds()
        super().__init__(structure, featurizer = featurizer, label = label)
        self.features, self.coords, self.bonds, self.bonds_features = self.featurizer.process_structure(self.structure, self.process_bonds)

        #Graph
        self.representation = self.__compute_representation()


    def __get_edges(self):
        """
        Uses the edges method to select how the edges will be computed
        """
        match self.edges_method:
            case "distance":
                edges, edge_features = self.__get_distance_edges()
            case "featurizer":
                if self.bonds is not None and self.bonds_features is not None:
                    edges, edge_features = self.bonds, self.bonds_features
                else:
                    edges, edge_features = self.__get_distance_edges()
            case "mixed":
                edges, edge_features = self.__get_mixed_edges()
            case _:
                edges, edge_features = self.__get_distance_edges()


        return edges, edge_features
    

    def __get_distance_edges(self):
        """
        Computes edges using cutoff distance
        """
        contacts = []
        distances = []
        for i in range(len(self.coords)-1):
            for j in range(i+1, len(self.coords)):
                #Here we can use different distances <------------------------------------------------------
                dist_xyz = self.coords[i] - self.coords[j]
                distance = np.sqrt(dist_xyz[0]**2 + dist_xyz[1]**2 + dist_xyz[2]**2)
                if distance < self.cutoff:
                    contacts.append([i, j])
                    distances.append(distance)
        return contacts, distances
    
    #Not implemmented
    def __get_mixed_edges(self):
        """
        Computes edges mixing cutoff distance and featurizer bonds
        """
        contacts, distances = self.__get_distance_edges()
        return contacts, distances
    
    
    def __check_process_bonds(self):
        """
        Tells if it necessary for the featurizer to compute the bonds depending on edges method.
        """
        if self.edges_method in ["mixed", "featurizer"]:
            return True
        else:
            return False

    
    
    def __compute_representation(self):
        """
        Computes the full representation, using node, node features, edges and edges features.

        :return: A representation as a data object
        :rtype: torch_geometric.data.Data
        """
        node_features = self.features
        node_features = torch.tensor(node_features, dtype=torch.float)

        edges, edge_features = self.__get_edges()
        edges = torch.tensor(edges).t()
        edge_features = torch.tensor(edge_features, dtype=torch.float)

        graph = Data(x=node_features, edge_index=edges, edge_attr=edge_features, y = self.label)

        return graph
    

    def visualize_representation(self, axis = "xy"):
        """
        Generates a visualization of the representation.
        
        :param axis: Defines the 2D face from wich the representation is visualized:
                    - XY
                    - XZ
                    - YZ
        """
        match sorted(axis.lower()):
            case ["x", "y"]:
                pos = [(n[0], n[1]) for n in self.coords]
            case ["x", "z"]:
                pos = [(n[0], n[2]) for n in self.coords]
            case ["y", "z"]:
                pos = [(n[1], n[2]) for n in self.coords]
            case _:
                pos = [(n[0], n[1]) for n in self.coords]
            
        G = to_networkx(self.representation, to_undirected=True)
        
        for i, feature in enumerate(self.representation.x):
            G.nodes[i]['feature'] = feature.numpy()
        
        plt.figure(figsize=(40, 40))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.show()



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

        #Grid
        self.representation = self.__compute_representation()


    def __compute_representation(self):
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






class PointCloudRepresentation(BaseRepresentation):
    """
    This class is used to generate Point Clouds from structure files. Points are defined as
    coordinates and a set of features generated by the featurizer.
    """
    def __init__(self, structure, featurizer: Featurizer, label=None):
        """
        Initializes Point Cloud representation.
        
        :param structure: path to structure
        :param featurizer: The featurizer previusly configured with the features to extract
        :type featurizer: FeaturizerS
        :param label:  dictionary using the structure name (eg: xxx.pdb) as key, and label as value
        """
        super().__init__(structure, featurizer = featurizer, label = label)
        self.features, self.coords, self.bonds, self.bonds_features = self.featurizer.process_structure(self.structure, False)

        #Grid
        self.representation = self.__compute_representation()


    def __compute_representation(self):
        """
        Computes the point cloud.

        :return: A representation as a data object.
        :rtype: torch_geometric.data.Data
        """
        # Line to concatenate features
        feature_matrix = self.features
        point_cloud =  Data(pos=feature_matrix[:, 0:3], x=feature_matrix[:, 3:], y = self.label)

        return point_cloud
    

    def visualize_representation(self):
        """
        Generates a visualization of the representation.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = self.representation.pos[:,0], self.representation.pos[:,1], self.representation.pos[:,2]
        ax.scatter(x, y, z, c='cyan', marker='o', edgecolors='r', alpha=0.5)

        plt.show()
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
    def __init__(self, structure, featurizer: Featurizer, label = None):
        self.structure = structure
        self.label = label
        self.featurizer = featurizer


    def get_features_matrix(self):
        return self.features
    

    @abstractmethod
    def __compute_representation(self):
        pass


    @abstractmethod
    def visualize_representation(self):
        pass



'''
Distance graph
'''
class GraphRepresentation(BaseRepresentation):

    def __init__(self, structure, featurizer: Featurizer, edges_method = "distance", cutoff_distance = 10.0, label=None):
        self.cutoff = cutoff_distance
        self.edges_method = edges_method
        self.process_bonds = self.__check_process_bonds()
        super().__init__(structure, featurizer = featurizer, label = label)
        self.features, self.coords, self.bonds, self.bonds_features = self.featurizer.process_structure(self.structure, self.process_bonds)

        #Graph
        self.representation = self.__compute_representation()


    def __get_edges(self):
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
        contacts, distances = self.__get_distance_edges()
        # print(contacts)
        # print(distances)
        # print(self.bonds)
        # print(self.bonds_features)
        return contacts, distances
    
    
    def __check_process_bonds(self):
        if self.edges_method in ["mixed", "featurizer"]:
            return True
        else:
            return False

    
    
    def __compute_representation(self):
        node_features = self.get_features_matrix()
        node_features = torch.tensor(node_features, dtype=torch.float)

        edges, edge_features = self.__get_edges()
        edges = torch.tensor(edges).t()
        edge_features = torch.tensor(edge_features, dtype=torch.float)

        graph = Data(x=node_features, edge_index=edges, edge_attr=edge_features, y = self.label)

        return graph
    

    def visualize_representation(self, axis = "xy"):
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





'''
Grid
'''
class Grid3DRepresentation(BaseRepresentation):
    def __init__(self, structure, featurizer: Featurizer, voxel_size = 10.0, label=None):
        super().__init__(structure, featurizer = featurizer, label = label)
        self.features, self.coords, self.bonds, self.bonds_features = self.featurizer.process_structure(self.structure, False)
        self.voxel_size = voxel_size

        #Grid
        self.representation = self.__compute_representation()


    def __compute_representation(self):
        coords_copy = self.coords
        # En esta parte creamos el grid
        min_vals = np.min(coords_copy, axis=0)
        max_vals = np.max(coords_copy, axis=0)

        x_len, y_len, z_len = np.ceil((max_vals - min_vals) / self.voxel_size).astype(int)
        
        # Line to concatenate features
        #node_features = np.concatenate((coords_copy, self.residue_encoding), axis = 1)
        node_features = self.get_features_matrix()
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
        mask = np.any(self.representation != 0, axis=3)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.voxels(mask, facecolors='cyan', edgecolors='r', alpha=0.5)
        plt.show()





'''
Point cloud
'''
class PointCloudRepresentation(BaseRepresentation):
    def __init__(self, structure, featurizer: Featurizer, label=None):
        super().__init__(structure, featurizer = featurizer, label = label)
        self.features, self.coords, self.bonds, self.bonds_features = self.featurizer.process_structure(self.structure, False)

        #Grid
        self.representation = self.__compute_representation()


    def __compute_representation(self):
        # Line to concatenate features
        feature_matrix = self.get_features_matrix()
        point_cloud =  Data(pos=feature_matrix[:, 0:3], x=feature_matrix[:, 3:], y = self.label)

        return point_cloud
    

    def visualize_representation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = self.representation.pos[0], self.representation.pos[1], self.representation.pos[2]
        ax.scatter(x, y, z, c='cyan', marker='o', edgecolors='r', alpha=0.5)

        plt.show()
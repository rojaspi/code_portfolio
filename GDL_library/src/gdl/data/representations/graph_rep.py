from .base_rep import BaseRepresentation
from abc import ABC, abstractmethod

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from ..featurization.base_feat import Featurizer


class EdgeBuilder(ABC):
    @abstractmethod
    def build_edges(self, coords):
        pass


class DistanceEdgeBuilder(EdgeBuilder):
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def build_edges(self, data):
        """
        Creates edges using cutoff distance
        """
        edges = []
        distances = []
        for i in range(len(data.coords)-1):
            for j in range(i+1, len(data.coords)):
                #Here we can use different distances <------------------------------------------------------
                dist_xyz = data.coords[i] - data.coords[j]
                distance = np.sqrt(dist_xyz[0]**2 + dist_xyz[1]**2 + dist_xyz[2]**2)
                if distance < self.cutoff:
                    edges.append([i, j])
                    distances.append(distance)
        return edges, distances


class BondEdgesBuilder(EdgeBuilder):

    def build_edges(self, data):
        if data.bonds is None:
            raise ValueError("Featurizer did not provide bonds")

        return data.bonds, data.bond_features
    

#Not Implemented
class MixedEdges(EdgeBuilder):

    def __init__(self, distance_builder: EdgeBuilder):
        self.distance_builder = distance_builder

    def build_edges(self, data):
        dist_edges, dist_feat = self.distance_builder.build_edges(data)

        if data.bonds is None:
            return dist_edges, dist_feat

        # combinar ambos

        return dist_edges, dist_feat
    



class GraphRepresentation(BaseRepresentation):
    """
    This class is used to generate graphs from structure files. Edges are created by the featurizer (residue or atomic bonds)
    or by distance using a cutoff distance. A edge is created if two nodes are close enough, dictated by the cutoof distance.
    """
    def __init__(self, structure, featurizer: Featurizer, edge_builder: EdgeBuilder, label=None):
        """
        Initializes GraphRepresentation class
        
        :param structure: path to structure
        :param featurizer: The featurizer previusly configured with the features to extract
        :type featurizer: Featurizer
        :param label:  dictionary using the structure name (eg: xxx.pdb) as key, and label as value
        """

        super().__init__(structure, featurizer = featurizer, label = label)
        self.edge_builder = edge_builder
    

    
    def _compute_representation(self):
        """
        Computes the full representation, using node, node features, edges and edges features.
        """

        data = self.featurizer.process_structure(self.structure)
        node_features = torch.tensor(data.node_features, dtype=torch.float)
        edges, edge_features = self.edge_builder.build_edges(data)

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
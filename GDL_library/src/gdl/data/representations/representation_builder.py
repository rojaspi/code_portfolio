import sys
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..featurization.base_feat import Featurizer

from ..representations import GraphRepresentation, Grid3DRepresentation, PointCloudRepresentation, DistanceEdgeBuilder
from ..featurization import ResidueFeaturizer, AllAtomFeaturizer

class RepresentationBuilder():
    def __init__(self, representation:str, featurizer: "Featurizer", distance=None, edges_method=None):
        """
        :param representation: Type of representation for the structures:
                                - graph
                                - voxel/grid
                                - point cloud
        :param edges_method: How the edges will be computed.
                                - distance: for distance graphs
                                - featurizer: to use featurizer bond features
                                - mixed: to use both
        :param featurizer: The featurizer previusly configured with the features to extract
        :param distance: Cutoff distance for distance graph
        """
        self.representation = representation
        self.featurizer = featurizer or ResidueFeaturizer()
        self.distance = distance or 8
        self.edges_method = edges_method or "distance"

        if self.representation.lower() in ["point_cloud", "point cloud"] and self.distance is not None:
            warnings.warn(
                "Distance parameter has no influence over point cloud representations",
                UserWarning
            )
        if self.representation.lower() in ["point_cloud", "point cloud", "voxel", "grid"] and self.edges_method is not None:
            warnings.warn(
                "Edge features parameter has no influence over voxel/grid and point cloud representations",
                UserWarning
            )


    def process_representation(self, path, label = None):
        """
        Processes each file given a representation
        
        :param path: path to structure
        :param label: label for the structure

        :return: A representation as a data object
        :rtype: torch_geometric.data.Data
        """
        match self.representation.lower():
            case "graph":
                # Need to add more methods depending on edge methods
                edge_builder = DistanceEdgeBuilder(self.distance)
                repr = GraphRepresentation(structure = path, featurizer= self.featurizer,
                                           edge_builder= edge_builder, label=label).representation
            case "point_cloud" | "point cloud":
                repr = PointCloudRepresentation(structure = path, featurizer= self.featurizer,
                                                label=label).representation
            case "voxel" | "grid":
                repr = Grid3DRepresentation(structure = path, featurizer= self.featurizer,
                                            label=label, voxel_size= self.distance).representation
            case _:
                sys.exit("Invalid representation")

        return repr
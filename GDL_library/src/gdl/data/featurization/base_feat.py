from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class FeatureData:
    node_features: any
    coords: any
    bonds: any
    bond_features: any


class Featurizer(ABC):
    """
    This class is used to define shared methods between different featurizers.
    """
    @abstractmethod
    def process_structure(self) -> FeatureData:
        pass

    @abstractmethod
    def _get_bonds(self):
        pass
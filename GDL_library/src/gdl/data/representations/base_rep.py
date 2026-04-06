from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data

if TYPE_CHECKING:
    from ..featurization.base_feat import Featurizer


class BaseRepresentation(ABC):
    """
    This class is used to define shared methods and parameters between different representations.
    """
    def __init__(self, structure, featurizer: "Featurizer", label = None):
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

        self._representation = None
    

    @property
    def representation(self):
        if self._representation is None:
            self._representation = self._compute_representation()
        return self._representation
    

    @abstractmethod
    def _compute_representation(self):
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

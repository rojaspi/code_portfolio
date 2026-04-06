#from . import feature_extraction
from .base_feat import Featurizer, FeatureData

import numpy as np
from rdkit import Chem


class AllAtomFeaturizer(Featurizer):
    """
    Atom level featurizer. Used to compute atom and atom bonds features.
    """
    def __init__(self, feature_list = [""]):
        """
        Initializes all atom featurizer class.
        
        :param self: Description
        :param feature_list: List of atom features to compute.
        """
        self.residue_center = "all-atom"
        self.feature_list = [item.lower() for item in feature_list]


    def process_structure(self, structure_file, process_bonds: bool = False):
        """
        This method processes the structure. Returns the atoms positions,
            atom features and bonds and bond features if process_bonds is active.
        
        :param self:
        :param structure_file: Path to structure file
        :param process_bonds: Used to determine if bonds and bond features between atoms should be returned.

        :return: A tuple containing:
            - node_features
            - coords: Coordenates of each atom
            - bonds: Pairs of connected atoms
            - bonds_features
        """
        mol = Chem.MolFromPDBFile(structure_file, sanitize=False)
        conformer = mol.GetConformer()

        atoms, coords, atom_map, masses  = self._get_coords_and_atoms(mol, conformer)

        context = {
            "coords": coords,
            "masses": masses,
        }

        node_features = self._compute_features(context=context)
        bonds, bonds_features = self._get_bonds(mol, process_bonds)

        return FeatureData(node_features, coords, bonds, bonds_features)



    def _get_bonds(self, mol, process_bonds:bool = False):
        """
        Get bonds and bond features for every bond between atoms in the protein.
        
        :param self:
        :param mol: molecule or protein as rdkit chem molecule
        :param process_bonds: Used to determine if bonds and bond features between atoms should be returned.
        :type process_bonds: bool

        :return: Tuple containing: 
            - bonds: Pairs of connected atoms
            - bonds_features
        """
        if process_bonds == False:
            #print("Bond processing off")
            return [None], None

        bonds = []
        bonds_features = []

        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            feature = [bond.GetBondType().numerator, int(bond.GetIsAromatic())]

            bonds.append([begin_idx, end_idx])
            bonds_features.append(feature)

        return bonds, bonds_features


    def _get_coords_and_atoms(self, mol, conformer):
        """
        Gets every atom type, position and mass
        
        :param self: Description
        :param mol:  molecule or protein as rdkit chem molecule
        :param conformer: the conformer of the molecule in rdkit chem

        :return: a tuple containing:
            - each atom as a str representing its "symbol"
            - coordinates of each atom
            - residue map: atom number in structure file
            - mass of each atom
        """
        atoms = []
        atoms_coordinates = []
        residue_map = []
        atoms_mass = []

        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            pos = conformer.GetAtomPosition(atom_idx)

            atoms.append(atom.GetSymbol())
            atoms_coordinates.append([pos.x, pos.y, pos.z])
            atoms_mass.append(atom.GetMass())

            residue_map.append(atom.GetPDBResidueInfo().GetResidueNumber())

        return np.array(atoms).reshape(-1, 1), np.array(atoms_coordinates), residue_map, np.array(atoms_mass).reshape(-1, 1)
    
    # Needs more features
    def _compute_features(self, context):
        """
        Computes and concatenates all features listed in the feature list.
        
        :param self: Description
        :param context: A dictionary with the necessary elements to compute features:
                    context = {
                        "coords": coords,
                        "masses": masses,
                        }
        """
        feature_matrix = context["coords"]
        masses = context["masses"]
        feature_matrix = np.concatenate((feature_matrix, masses), axis = 1)

        return feature_matrix


    # def _get_residue_index(self):
    #     residues = torch.tensor(self.residue_map, dtype=int)
    #     unique_residues, inverse_indices = torch.unique(residues, sorted=True, return_inverse=True)
    #     return inverse_indices


    # def _generate_graph(self):
    #     edges = torch.tensor(self.bonds).t()
    #     node_features = np.concatenate((self.coords, self.masses), axis = 1)
    #     node_features = torch.tensor(node_features, dtype=torch.float)
    #     edge_features = torch.tensor(self.bonds_features, dtype=torch.float)
    #     d = Data(x=node_features, edge_index=edges, edge_attr=edge_features)
    #     return d
from . import feature_extraction

from abc import ABC, abstractmethod

import numpy as np
import mdtraj as md
from mdtraj import load


import networkx as nx
import matplotlib.pyplot as plt
import torch


from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import aggr


class Featurizer():
        
    @abstractmethod
    def process_structure(self):
        pass

    @abstractmethod
    def __get_bonds(self):
        pass


class ResidueFeaturizer(Featurizer):
    def __init__(self, residue_center = "CA", feature_list = ["Blossum"]):
        self.residue_center = residue_center.lower()
        self.feature_list = [item.lower() for item in feature_list]


    def process_structure(self, structure_file, process_bonds = False):
        traj = self.__get_topology(structure_file)
        topo = traj.topology
        seq = self.__get_sequence(topo=topo)
        coords = self.__get_coords(traj, topo)

        context = {
            "sequence": seq,
            "trajectory": traj,
            "topology": topo,
            "coords": coords
        }

        node_features = self.__compute_features(context=context)
        bonds, bonds_features = self.__get_bonds(process_bonds)

        return node_features, coords, bonds, bonds_features

    #Not implementes
    def __get_bonds(self, process_bonds):
        if process_bonds == False:
            #print("Bond processing off")
            return None, None

        # Need to implement bond processing
        return None, None


    def __compute_features(self, context):
        feature_matrix = context["coords"]

        for feat in self.feature_list:
            key = self.NORMALIZED.get(feat)
            
            if key and key in self.FEATURE_MAP:
                entry = self.FEATURE_MAP[key]
                func = entry["func"]
                args = [context.get(p) for p in entry["params"]]

                result = func(*args)
                feature_matrix = np.concatenate((feature_matrix, result), axis=1)

        return feature_matrix


    def __get_topology(self, structure_file):
        traj = load(structure_file)
        traj =  traj.atom_slice(traj.topology.select('protein'))
    
        if traj.topology.n_chains > 1:
            top = traj.topology
            new_top = md.Topology()
            new_chain = new_top.add_chain()
            residue_map = {}

            for res in top.residues:
                # Reuse residue sequence index to merge
                if res.resSeq not in residue_map:
                    new_res = new_top.add_residue(res.name, new_chain, resSeq=res.resSeq)
                    residue_map[res.resSeq] = new_res
                else:
                    new_res = residue_map[res.resSeq]

                for atom in res.atoms:
                    new_top.add_atom(atom.name, atom.element, new_res)

            traj.topology = new_top

        return traj
    

    def __get_sequence(self, topo):
        return ''.join(residue.code for residue in topo.residues if residue.is_protein)
    

    def __get_coords(self, traj, top):
        match self.residue_center:
            case "ca":
                return self.__get_alpha_coords(traj, top)
            case "mass":
                return self.__get_center_mass_coords(traj, top)
            case "mean":
                return self.__get_mean_center_coords(traj, top)
            case _:
                return self.__get_alpha_coords(traj, top)


    def __get_alpha_coords(self, traj, top):
        ca_coords = []
        for residue in top.residues:
            if not residue.is_protein:
                continue
            for atom in residue.atoms:
                if atom.name == 'CA':
                    coord = traj.xyz[0][atom.index] * 10
                    ca_coords.append(coord)
                    break
            else:
                ## Some residues has no CA
                coord = traj.xyz[0][0] * 10
                ca_coords.append(coord)
        return np.array(ca_coords)
    

    def __get_center_mass_coords(self, traj, top):
        center_mass = []
        for residue in top.residues:
            if not residue.is_protein:
                continue
            atom_indices = [atom.index for atom in residue.atoms]
            masses = np.array([atom.element.mass for atom in residue.atoms])[:, np.newaxis]
            coords = traj.xyz[0][atom_indices]

            mass_sum = np.sum(masses)
            com = np.sum(coords * masses, axis=0) / mass_sum
            center_mass.append(com * 10)

        return np.array(center_mass)


    def __get_mean_center_coords(self, traj, top):
        centroids = []
        for residue in top.residues:
            if not residue.is_protein:
                continue
            atom_indices = [atom.index for atom in residue.atoms]
            coords = traj.xyz[0][atom_indices]
            centroid = np.mean(coords, axis=0) * 10
            centroids.append(centroid)
        return np.array(centroids)



    NORMALIZED = {
        "one hot": "one_hot",
        "ohe": "one_hot",
        "one hot encoding": "one_hot",
        "blossum": "blossum",
        "isoelectric point": "isoelectric_point",
        "hydrophobicity": "hydrophobicity",
        "polarity": "polarity",
        "dihedral": "dihedral",
        "volume": "volume",
        "secondary structure": "secondary_structure",
        "accessible surface": "accessible_surface"
    }


    FEATURE_MAP = {
        "one_hot": {
            "func": feature_extraction.get_one_hot_features,
            "params": ["sequence"]
        },
        "blossum": {
            "func": feature_extraction.get_blosum_features,
            "params": ["sequence"]
        },
        "isoelectric_point": {
            "func": feature_extraction.get_isoelectric_point_features,
            "params": ["sequence"]
        },
        "hydrophobicity": {
            "func": feature_extraction.get_hydrophobicity_kd_features,
            "params": ["sequence"]
        },
        "polarity": {
            "func": feature_extraction.get_polarity_features,
            "params": ["sequence"]
        },
        "volume": {
            "func": feature_extraction.get_sidechain_volume_features,
            "params": ["sequence"]
        },
        "dihedral": {
            "func": feature_extraction.get_dh,
            "params": ["trajectory"]
        },
        "secondary_structure": {
            "func": feature_extraction.get_secondary_structure,
            "params": ["trajectory"]
        },
        "accessible_surface": {
            "func": feature_extraction.get_accessible_surface_area,
            "params": ["trajectory", "sequence"]
        }
    }

#Incomplete
class AllAtomFeaturizer(Featurizer):
    def __init__(self, feature_list = [""]):
        #Main information
        self.residue_center = "all-atom"
        self.feature_list = [item.lower() for item in feature_list]


    def process_structure(self, structure_file, process_bonds = False):
        mol = Chem.MolFromPDBFile(structure_file, sanitize=False)
        conformer = mol.GetConformer()

        atoms, coords, residue_map, masses  = self.__get_coords_and_atoms(mol, conformer)

        context = {
            "coords": coords,
            "masses": masses,
        }

        node_features = self.__compute_features(context=context)
        bonds, bonds_features = self.__get_bonds(mol, process_bonds)

        return node_features, coords, bonds, bonds_features



    def __get_bonds(self, mol, process_bonds = False):
        if process_bonds == False:
            #print("Bond processing off")
            return [None], None

        edges = []
        edge_features = []

        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bonds_feature = [bond.GetBondType().numerator, int(bond.GetIsAromatic())]

            edges.append([begin_idx, end_idx])
            edge_features.append(bonds_feature)

        return edges, edge_features


    def __get_coords_and_atoms(self, mol, conformer):
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
    

    def __compute_features(self, context):
        feature_matrix = context["coords"]
        masses = context["masses"]
        feature_matrix = np.concatenate((feature_matrix, masses), axis = 1)

        return feature_matrix


    # def __get_residue_index(self):
    #     residues = torch.tensor(self.residue_map, dtype=int)
    #     unique_residues, inverse_indices = torch.unique(residues, sorted=True, return_inverse=True)
    #     return inverse_indices


    # def __generate_graph(self):
    #     edges = torch.tensor(self.bonds).t()
    #     node_features = np.concatenate((self.coords, self.masses), axis = 1)
    #     node_features = torch.tensor(node_features, dtype=torch.float)
    #     edge_features = torch.tensor(self.bonds_features, dtype=torch.float)
    #     d = Data(x=node_features, edge_index=edges, edge_attr=edge_features)
    #     return d
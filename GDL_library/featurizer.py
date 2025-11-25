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


VALID_ATOM_NAMES = {
    "N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "OG", "SG", "ND1", "NE2", "CD1", "CD2",
    "CG1", "CG2", "OE1", "OE2", "OD1", "OD2", "NE", "NH1", "NH2", "OG1", "SD", "H", "HA", "HB", "HG",
    "HD", "HE", "HH", "HZ", "HG1", "HG2", "HG3", "HD1", "HD2", "HD3", "HE1", "HE2", "HE3", "HH11", "HH12",
    "HH21", "HH22", "HZ1", "HZ2", "HZ3"
}

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

        
    # FEATURES!!
    # This function is from:
    # https://github.com/feiglab/ProteinStructureEmbedding/blob/main/src/dataset.py and edited by copilot
    def __get_dh(self):
        """
        Gets dihedral features (sine, cosine, mask) for all residues.
        Ensures consistent size across all features.
        """
        
        residues = [r for r in self.traj.topology.residues if r.is_protein]
        num_residues = len(residues)

        # Map atoms to residue indices
        a2r = {}
        for i, r in enumerate(residues):
            for a in r.atoms:
                a2r[a.index] = i

        def process_dihedral(dihedral_func, default_value=-2*np.pi):
            data = dihedral_func(self.traj)
            values = data[1][0]
            atoms = data[0]

            angles = np.full(num_residues, default_value)
            mask = np.zeros(num_residues)

            for i, angle in enumerate(values):
                atom_index = atoms[i][0]
                if atom_index in a2r:
                    res_index = a2r[atom_index]
                    if res_index < num_residues:
                        angles[res_index] = angle
                        mask[res_index] = 1

            sin_vals = np.sin(angles) * mask
            cos_vals = np.cos(angles) * mask
            return mask, sin_vals, cos_vals

        # Phi and Psi
        phi_mask, phi_sin, phi_cos = process_dihedral(md.compute_phi)
        psi_mask, psi_sin, psi_cos = process_dihedral(md.compute_psi)

        # Chi1, Chi2, Chi3
        chi1_mask, chi1_sin, chi1_cos = process_dihedral(md.compute_chi1, default_value=10.0)
        chi2_mask, chi2_sin, chi2_cos = process_dihedral(md.compute_chi2, default_value=10.0)
        chi3_mask, chi3_sin, chi3_cos = process_dihedral(md.compute_chi3, default_value=10.0)

        # Stack all features
        features = np.array([
            psi_mask, psi_sin, psi_cos,
            phi_mask, phi_sin, phi_cos,
            chi1_mask, chi1_sin, chi1_cos,
            chi2_mask, chi2_sin, chi2_cos,
            chi3_mask, chi3_sin, chi3_cos
        ]).transpose()

        return features


    def __get_secondary_structure(self):
        try:
            ss_raw = md.compute_dssp(self.traj, simplified=False)[0]
        except Exception as e:
            print("DSSP falló, aplicando filtrado de átomos no válidos")
            valid_indices = [atom.index for atom in self.traj.topology.atoms if atom.name in VALID_ATOM_NAMES]
            traj_filtered = self.traj.atom_slice(valid_indices)
            ss_raw = md.compute_dssp(traj_filtered, simplified=False)[0]


        ss_2_int = {"H":0, "B":1, "E":2, "G":3, "I":4, "T":5, "S":6, " ":7}

        ss_clean = np.array([s if s != "NA" else " " for s in ss_raw])

        ss_int = np.vectorize(ss_2_int.get)(ss_clean)
        num_classes = len(ss_2_int)
        one_hot_encoded = np.zeros((ss_int.size, num_classes), dtype=int)
        one_hot_encoded[np.arange(ss_int.size), ss_int] = 1

        return one_hot_encoded
    

    def __get_accessible_surface_area(self):
        try:
            sa = md.shrake_rupley(self.traj, mode="residue")[0]
        except Exception as e:
            print("DSSP falló, aplicando filtrado de átomos no válidos")
            valid_indices = [atom.index for atom in self.traj.topology.atoms if atom.name in VALID_ATOM_NAMES]
            traj_filtered = self.traj.atom_slice(valid_indices)
            sa = md.shrake_rupley(traj_filtered, mode="residue")[0]

        sa = sa[:len(self.sequence)].reshape(-1, 1)

        return sa


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
            "func": lambda self, topology, coords: self.__get_dh(topology, coords),
            "params": ["topology", "coords"]
        },
        "secondary_structure": {
            "func": lambda self, topology: self.__get_secondary_structure(topology),
            "params": ["topology"]
        },
        "accessible_surface": {
            "func": lambda self, topology, coords: self.__get_accessible_surface_area(topology, coords),
            "params": ["topology", "coords"]
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
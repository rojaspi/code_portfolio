import numpy as np
import mdtraj as md
import torch
#import esm
VALID_ATOM_NAMES = {
    "N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "OG", "SG", "ND1", "NE2", "CD1", "CD2",
    "CG1", "CG2", "OE1", "OE2", "OD1", "OD2", "NE", "NH1", "NH2", "OG1", "SD", "H", "HA", "HB", "HG",
    "HD", "HE", "HH", "HZ", "HG1", "HG2", "HG3", "HD1", "HD2", "HD3", "HE1", "HE2", "HE3", "HH11", "HH12",
    "HH21", "HH22", "HZ1", "HZ2", "HZ3"
}


_amino_acid_number_dict = {
    'A':  0, 'R':  1, 'N':  2, 'D':  3, 'C':  4, 'Q':  5, 'E':  6, 'G':  7, 'H':  8, 'I':  9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}


_blosum62 = {
    'A': {'A':  4, 'R': -1, 'N': -2, 'D': -2, 'C':  0, 'Q': -1, 'E': -1, 'G':  0, 'H': -2, 'I': -1,
          'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S':  1, 'T':  0, 'W': -3, 'Y': -2, 'V':  0},
    'R': {'A': -1, 'R':  5, 'N':  0, 'D': -2, 'C': -3, 'Q':  1, 'E':  0, 'G': -2, 'H':  0, 'I': -3,
          'L': -2, 'K':  2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
    'N': {'A': -2, 'R':  0, 'N':  6, 'D':  1, 'C': -3, 'Q':  0, 'E':  0, 'G':  0, 'H':  1, 'I': -3,
          'L': -3, 'K':  0, 'M': -2, 'F': -3, 'P': -2, 'S':  1, 'T':  0, 'W': -4, 'Y': -2, 'V': -3},
    'D': {'A': -2, 'R': -2, 'N':  1, 'D':  6, 'C': -3, 'Q':  0, 'E':  2, 'G': -1, 'H': -1, 'I': -3,
          'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S':  0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
    'C': {'A':  0, 'R': -3, 'N': -3, 'D': -3, 'C':  9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1,
          'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
    'Q': {'A': -1, 'R':  1, 'N':  0, 'D':  0, 'C': -3, 'Q':  5, 'E':  2, 'G': -2, 'H':  0, 'I': -3,
          'L': -2, 'K':  1, 'M':  0, 'F': -3, 'P': -1, 'S':  0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
    'E': {'A': -1, 'R':  0, 'N':  0, 'D':  2, 'C': -4, 'Q':  2, 'E':  5, 'G': -2, 'H':  0, 'I': -3,
          'L': -3, 'K':  1, 'M': -2, 'F': -3, 'P': -1, 'S':  0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'G': {'A':  0, 'R': -2, 'N':  0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G':  6, 'H': -2, 'I': -4,
          'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S':  0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
    'H': {'A': -2, 'R':  0, 'N':  1, 'D': -1, 'C': -3, 'Q':  0, 'E':  0, 'G': -2, 'H':  8, 'I': -3,
          'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y':  2, 'V': -3},
    'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I':  4,
          'L':  2, 'K': -3, 'M':  1, 'F':  0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V':  3},
    'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I':  2,
          'L':  4, 'K': -2, 'M':  2, 'F':  0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V':  1},
    'K': {'A': -1, 'R':  2, 'N':  0, 'D': -1, 'C': -3, 'Q':  1, 'E':  1, 'G': -2, 'H': -1, 'I': -3,
          'L': -2, 'K':  5, 'M': -1, 'F': -3, 'P': -1, 'S':  0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q':  0, 'E': -2, 'G': -3, 'H': -2, 'I':  1,
          'L':  2, 'K': -1, 'M':  5, 'F':  0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V':  1},
    'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I':  0,
          'L':  0, 'K': -3, 'M':  0, 'F':  6, 'P': -4, 'S': -2, 'T': -2, 'W':  1, 'Y':  3, 'V': -1},
    'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3,
          'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P':  7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
    'S': {'A':  1, 'R': -1, 'N':  1, 'D':  0, 'C': -1, 'Q':  0, 'E':  0, 'G':  0, 'H': -1, 'I': -2,
          'L': -2, 'K':  0, 'M': -1, 'F': -2, 'P': -1, 'S':  4, 'T':  1, 'W': -3, 'Y': -2, 'V': -2},
    'T': {'A':  0, 'R': -1, 'N':  0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1,
          'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S':  1, 'T':  5, 'W': -2, 'Y': -2, 'V':  0},
    'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3,
          'L': -2, 'K': -3, 'M': -1, 'F':  1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y':  2, 'V': -3},
    'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H':  2, 'I': -1,
          'L': -1, 'K': -2, 'M': -1, 'F':  3, 'P': -3, 'S': -2, 'T': -2, 'W':  2, 'Y':  7, 'V': -1},
    'V': {'A':  0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I':  3,
          'L':  1, 'K': -2, 'M':  1, 'F': -1, 'P': -2, 'S': -2, 'T':  0, 'W': -3, 'Y': -1, 'V':  4}
}


# Kyte–Doolittle Hydropathy Values for Amino Acids
hydrophobicity_kd = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,  'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8,  'K': -3.9, 'M': 1.9,  'F': 2.8,  'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}


# Punto isoelectrico. No se agrega carga por ser calculada a base del punto isoelectrico
isoelectric_point = {
    'A': 6.00, 'R': 10.76, 'N': 5.41, 'D': 2.77, 'C': 5.07, 'Q': 5.65, 'E': 3.22, 'G': 5.97, 'H': 7.59, 'I': 6.02,
    'L': 5.98, 'K': 9.74,  'M': 5.74, 'F': 5.48, 'P': 6.30, 'S': 5.68, 'T': 5.60, 'W': 5.89, 'Y': 5.66, 'V': 5.96
}


# 1: Alifatic Non Polar 2: Positively charged Polar 3: Non Charged Polar 4: Negatively charged Polar 5: Aromatic non Polar
polarity = {
    'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 3, 'Q': 3, 'E': 4, 'G': 1, 'H': 2, 'I': 1,
    'L': 1, 'K': 2, 'M': 1, 'F': 5, 'P': 1, 'S': 3, 'T': 3, 'W': 5, 'Y': 3, 'V': 1
}


# We can add total residue volume too
# Ref: Zamyatnin, A.A., Protein volume in solution, Prog. Biophys. Mol. Biol., 24:107-123 (1972), PMID: 4566650
# In A^3:
sidechain_volume = {
      'A': 31,  'R': 124, 'N': 56,  'D': 54,  'C': 55,   'Q': 85, 'E': 83, 'G': 3,   'H': 96,  'I': 111,
      'L': 111, 'K': 119, 'M': 105, 'F': 132, 'P': 32.5, 'S': 32, 'T': 61, 'W': 170, 'Y': 136, 'V': 84
}




#-----------------------------------------------------------------------------------------------------------
# Crear "features" con one hot encoding
def get_one_hot_features(seq):
# Inicializamos una matriz de ceros de tamaño 20 x N (20 filas, N columnas) con N el largo máximo de las secuencias 
    encoding_matrix = np.zeros((20, len(seq)))
        
    # Iteramos sobre la secuencia de aminoácidos y asignamos 1 en la posición correspondiente
    for i, amino in enumerate(seq):
        idx = _amino_acid_number_dict.get(amino)  # Obtenemos el índice del aminoácido
        if idx is not None:  # Si el aminoácido está en nuestro diccionario
            encoding_matrix[idx, i] = 1.0
        else:
            print("Non canonic AA: ", amino, " at pos ", i)
    
    return encoding_matrix.T



#------------------------------------------------------------------------------------
# Crear "features" con blosum62
def get_blosum_features(seq):
    # Inicializamos una matriz de ceros de tamaño 20 x N (20 filas, N columnas) con N el largo máximo de las secuencias 
    encoding_matrix = []
    # Iteramos sobre la secuencia de aminoácidos y asignamos el vector de la matriz Blosum62
    for i, amino in enumerate(seq): 
        # Si el aminoácido está en nuestro diccionario       
        if amino in _blosum62: 
            # Obtenemos la lista a partir de blosum
            bls = list(_blosum62[amino].values())
            encoding_matrix.append(bls)
        else:
            print("Non canonic AA: ", amino, " at pos ", i)
            listofzeros = [0] * 20
            encoding_matrix.append(listofzeros)

    
    return np.array(encoding_matrix)


#------------------------------------------------------------------------------------
# Crear "features" con isoelectric_point
def get_isoelectric_point_features(seq):
    encoding_matrix = []

    # Iteramos sobre la secuencia de aminoácidos
    for i, amino in enumerate(seq): 
        score = isoelectric_point.get(amino, 7)
        encoding_matrix.append(score)

    return np.array(encoding_matrix).reshape(-1, 1)


#------------------------------------------------------------------------------------
# Crear "features" con hydrophobicity_kd
def get_hydrophobicity_kd_features(seq):
    encoding_matrix = []

    # Iteramos sobre la secuencia de aminoácidos y asignamos el vector de la matriz Blosum62
    for i, amino in enumerate(seq): 
        score = hydrophobicity_kd.get(amino, 0)
        encoding_matrix.append(score)

    return np.array(encoding_matrix).reshape(-1, 1)


#-----------------------------------------------------------------------------------------------------------
# Crear "features" con polarity
def get_polarity_features(seq): 
    encoding_matrix = np.zeros((5, len(seq)))
        
    # Iteramos sobre la secuencia de aminoácidos y asignamos 1 en la posición correspondiente
    for i, amino in enumerate(seq):
        idx = polarity.get(amino)
        if idx is not None:  # Si el aminoácido está en nuestro diccionario
            encoding_matrix[idx-1, i] = 1.0
    
    return encoding_matrix.T


#------------------------------------------------------------------------------------
# Crear "features" con hydrophobicity_kd
def get_sidechain_volume_features(seq):
    encoding_matrix = []

    for i, amino in enumerate(seq): 
        score = sidechain_volume.get(amino, 91.6) # 91.6 is the mean of all volumes
        encoding_matrix.append(score)

    return np.array(encoding_matrix).reshape(-1, 1)





# FEATURES!!
# This function is from:
# https://github.com/feiglab/ProteinStructureEmbedding/blob/main/src/dataset.py and edited by copilot
def get_dh(traj: md.Trajectory):
    """
    Gets dihedral features (sine, cosine, mask) for all residues.
        Ensures consistent size across all features.
    
    :param traj: mdtraj trajectory of a protein.
    :type traj: md.Trajectory
    
    """
    
    residues = [r for r in traj.topology.residues if r.is_protein]
    num_residues = len(residues)

    # Map atoms to residue indices
    a2r = {}
    for i, r in enumerate(residues):
        for a in r.atoms:
            a2r[a.index] = i

    def process_dihedral(dihedral_func, default_value=-2*np.pi):
        data = dihedral_func(traj)
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


def get_secondary_structure(traj: md.Trajectory):
    """
    Gets the type of secondary structure to which each residue belongs.
    
    :param traj: mdtraj trajectory of a protein.
    :type traj: md.Trajectory
    """
    try:
        ss_raw = md.compute_dssp(traj, simplified=False)[0]
    except Exception as e:
        print("DSSP falló, aplicando filtrado de átomos no válidos")
        valid_indices = [atom.index for atom in traj.topology.atoms if atom.name in VALID_ATOM_NAMES]
        traj_filtered = traj.atom_slice(valid_indices)
        ss_raw = md.compute_dssp(traj_filtered, simplified=False)[0]


    ss_2_int = {"H":0, "B":1, "E":2, "G":3, "I":4, "T":5, "S":6, " ":7}

    ss_clean = np.array([s if s != "NA" else " " for s in ss_raw])

    ss_int = np.vectorize(ss_2_int.get)(ss_clean)
    num_classes = len(ss_2_int)
    one_hot_encoded = np.zeros((ss_int.size, num_classes), dtype=int)
    one_hot_encoded[np.arange(ss_int.size), ss_int] = 1

    return one_hot_encoded


def get_accessible_surface_area(traj: md.Trajectory, sequence):
    """
    Gets accessible surface area for each residue.
    
    :param traj: mdtraj trajectory of a protein.
    :type traj: md.Trajectory
    :param sequence: Sequence of the protein.
    """
    try:
        sa = md.shrake_rupley(traj, mode="residue")[0]
    except Exception as e:
        print("DSSP falló, aplicando filtrado de átomos no válidos")
        valid_indices = [atom.index for atom in traj.topology.atoms if atom.name in VALID_ATOM_NAMES]
        traj_filtered = traj.atom_slice(valid_indices)
        sa = md.shrake_rupley(traj_filtered, mode="residue")[0]

    sa = sa[:len(sequence)].reshape(-1, 1)

    return sa


# #----------------------------------------------------------------------------------------
# # Generar features con ESMb1
# # Preparar modelo esmb1
# def _prep_esmb1():
#     model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
#     batch_converter = alphabet.get_batch_converter()

#     return model, batch_converter, alphabet



# #Preparar datos para ESM
# def _prepare_esmb1_data(batch_converter, alphabet, data):
#     batch_labels, batch_strs, batch_tokens = batch_converter(data)
#     batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

#     return batch_tokens



# # Obtener las features en base al modelo y los tokens
# def _esmb1_representations(model, batch_tokens):
#     with torch.no_grad():
#         results = model(batch_tokens, repr_layers=[33], return_contacts=True)
#     representations = np.transpose(np.array(results["representations"][33][0, 1:-1].reshape(-1,1280)))

#     return representations



# # Función maestra para solo 1 secuencia
# def get_esmb1_features(protein_name, sequence):
#     if len(sequence) > 1:
#         sequence = "".join(sequence)
#     data = [(protein_name, sequence)]
#     model, batch_converter, alphabet = _prep_esmb1()
#     batch_tokens = _prepare_esmb1_data(batch_converter, alphabet, data)
#     features = _esmb1_representations(model, batch_tokens)
#     return features

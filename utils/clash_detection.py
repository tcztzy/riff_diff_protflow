import os
import json
from functools import lru_cache
import itertools
from Bio.PDB import *
import pandas as pd

from protflow.utils.utils import vdw_radii
from protflow.utils.biopython_tools import load_structure_from_pdbfile

@lru_cache(maxsize=10000000)
def clash_detection_LRU(entity1, entity2, bb_multiplier:float, sc_multiplier:float, vdw:dict):
    '''
    checks for clashes by comparing VanderWaals radii. If clashes with ligand should be detected, set ligand to true. Ligand chain must be added as second entity.
    bb_only: only detect backbone clashes between to proteins or a protein and a ligand.
    clash_detection_vdw_multiplier: multiply Van der Waals radii with this value to set clash detection limits higher/lower
    database: path to database directory
    '''

    def calculate_clashes(entity1_coords, entity2_coords, entity1_vdw, entity2_vdw, vdw_multiplier):
        # Compute pairwise distances using broadcasting
        dgram = np.linalg.norm(entity1_coords[:, np.newaxis] - entity2_coords[np.newaxis, :], axis=-1)

        # calculate distance cutoff for each atom pair, considering VdW radii
        distance_cutoff = entity1_vdw[:, np.newaxis] + entity2_vdw[np.newaxis, :]

        # multiply distance cutoffs with set parameter
        distance_cutoff = distance_cutoff * vdw_multiplier

        # compare distances to distance_cutoff
        check = dgram - distance_cutoff

        if np.any(check < 0):
            return True
        else:
            return False

    backbone_atoms = ['CA', 'C', 'N', 'O']
    vdw = json.loads(vdw)

    # ugly, but should be faster because only iterating once
    entity1_bb_atoms = []
    entity1_sc_atoms = []
    for atom in entity1.get_atoms():
        if atom.element == "H":
            continue
        if atom.id in backbone_atoms:
            entity1_bb_atoms.append(atom)
        else:
            entity1_sc_atoms.append(atom)

    entity2_bb_atoms = []
    entity2_sc_atoms = []
    for atom in entity2.get_atoms():
        if atom.element == "H":
            continue
        if atom.id in backbone_atoms:
            entity2_bb_atoms.append(atom)
        else:
            entity2_sc_atoms.append(atom)

    entity1_bb_coords = np.array([atom.get_coord() for atom in entity1_bb_atoms])
    entity2_bb_coords = np.array([atom.get_coord() for atom in entity2_bb_atoms])

    entity1_bb_vdw = np.array([vdw[atom.element.lower()] for atom in entity1_bb_atoms])
    entity2_bb_vdw = np.array([vdw[atom.element.lower()] for atom in entity2_bb_atoms])

    if calculate_clashes(entity1_bb_coords, entity2_bb_coords, entity1_bb_vdw, entity2_bb_vdw, bb_multiplier) == True:
        return True

    entity1_sc_coords = np.array([atom.get_coord() for atom in entity1_sc_atoms])
    entity2_sc_coords = np.array([atom.get_coord() for atom in entity2_sc_atoms])

    entity1_sc_vdw = np.array([vdw[atom.element.lower()] for atom in entity1_sc_atoms])
    entity2_sc_vdw = np.array([vdw[atom.element.lower()] for atom in entity2_sc_atoms])

    if calculate_clashes(entity1_sc_coords, entity2_sc_coords, entity1_sc_vdw, entity2_sc_vdw, sc_multiplier) == True:
        return True
    else:
        return False

def main(args):

    df = pd.read_pickle(args.pkl)
    # has to be converted because dict is not hashable
    vdw = json.dumps(vdw_radii())
    ensemble_out = []
    structdict = {}

    structs = set(df['poses'].to_list())
    for i in structs:
        structdict[i] = load_structure_from_pdbfile(i, all_models=True)

    for ens_num, ensemble in df.groupby('ensemble_num'):
        for comb in itertools.combinations([row for index, row in ensemble.iterrows()], 2):
            entity_1 = structdict[comb[0]['poses']][comb[0]['model_num']][comb[0]['chain_id']]
            entity_2 = structdict[comb[1]['poses']][comb[1]['model_num']][comb[1]['chain_id']]
            check = clash_detection_LRU(entity_1, entity_2, args.bb_multiplier, args.sc_multiplier, vdw)
            if check == True:
                break
        ensemble['clash_check'] = check
        ensemble_out.append(ensemble)
    clash_detection_LRU.cache_clear()

    output_dir = os.path.join(args.working_dir, 'scores')
    os.makedirs(output_dir, exist_ok=True)
    out_pkl = os.path.join(output_dir, f'ensembles_{args.output_prefix}.pkl')

    ensemble_out = pd.concat(ensemble_out)
    ensemble_out.to_pickle(out_pkl)

    return ensemble_out



if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--database_dir", type=str, default="/home/tripp/riffdiff2/riff_diff/database/", help="Path to folder containing rotamer libraries, fragment library, etc.")
    argparser.add_argument("--working_dir", type=str, required=True, help="Path to working directory")
    argparser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output files.")
    argparser.add_argument("--pkl", type=str, required=True, help="Input json files.")

    # stuff you might want to adjust
    argparser.add_argument("--bb_multiplier", type=float, default=1.5, help="Multiplier for VanderWaals radii for clash detection inbetween backbone fragments. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")
    argparser.add_argument("--sc_multiplier", type=float, default=0.75, help="Multiplier for VanderWaals radii for clash detection between fragment backbones and ligand. Set None if no ligand is present. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")

    args = argparser.parse_args()

    main(args)



import os
import json
from functools import lru_cache
import itertools
from Bio.PDB import *
import pandas as pd

from protflow.utils.utils import vdw_radii
from protflow.utils.biopython_tools import load_structure_from_pdbfile

@lru_cache(maxsize=10000000)
def clash_detection_LRU(entity1, entity2, bb_multiplier:float, sc_multiplier:float, vdw_radii:dict):
    '''
    checks for clashes by comparing VanderWaals radii. If clashes with ligand should be detected, set ligand to true. Ligand chain must be added as second entity.
    bb_only: only detect backbone clashes between to proteins or a protein and a ligand.
    clash_detection_vdw_multiplier: multiply Van der Waals radii with this value to set clash detection limits higher/lower
    database: path to database directory
    '''
    backbone_atoms = ['CA', 'C', 'N', 'O', 'H']
    entity1_atoms = (atom for atom in entity1.get_atoms())
    entity2_atoms = (atom for atom in entity2.get_atoms())

    vdw_radii = json.loads(vdw_radii)

    for atom_combination in itertools.product(entity1_atoms, entity2_atoms):
        distance = atom_combination[0] - atom_combination[1]
        element1 = atom_combination[0].element
        element2 = atom_combination[1].element
        if atom_combination[0].name in backbone_atoms and atom_combination[1].name in backbone_atoms:
            vdw_multiplier = bb_multiplier
        else:
            vdw_multiplier = sc_multiplier
        clash_detection_limit = vdw_multiplier * (vdw_radii[str(element1).lower()] + vdw_radii[str(element2).lower()])
        if distance < clash_detection_limit:
            return True

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

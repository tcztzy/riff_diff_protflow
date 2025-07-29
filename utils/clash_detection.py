import os
import json
import numpy as np
import pandas as pd
from Bio.PDB import *
from protflow.utils.utils import vdw_radii
from protflow.utils.biopython_tools import load_structure_from_pdbfile


def clash_detection(entity1, entity2, bb_multiplier: float, sc_multiplier: float, vdw: dict):
    def calculate_clashes(coords1, coords2, vdw1, vdw2, mult1, mult2):
        dgram = np.linalg.norm(coords1[:, np.newaxis] - coords2[np.newaxis, :], axis=-1)
        cutoff = vdw1[:, np.newaxis] * mult1 + vdw2[np.newaxis, :] * mult2
        return np.any(dgram < cutoff)

    def split_atoms(entity, bb_ids):
        bb_atoms, sc_atoms = [], []
        for atom in entity.get_atoms():
            if atom.element == "H":
                continue
            (bb_atoms if atom.id in bb_ids else sc_atoms).append(atom)
        return bb_atoms, sc_atoms

    backbone_atoms = ['CA', 'C', 'N', 'O']
    entity1_bb, entity1_sc = split_atoms(entity1, backbone_atoms)
    entity2_bb, entity2_sc = split_atoms(entity2, backbone_atoms)

    vdw1_bb = np.array([vdw[atom.element.lower()] for atom in entity1_bb])
    vdw2_bb = np.array([vdw[atom.element.lower()] for atom in entity2_bb])

    coords1_bb = np.array([atom.get_coord() for atom in entity1_bb])
    coords2_bb = np.array([atom.get_coord() for atom in entity2_bb])

    if not entity1_bb or not entity2_bb:
        raise ValueError("Backbone atoms missing in one of the inputs!")

    if coords1_bb.size == 0 or coords2_bb.size == 0:
        raise ValueError("Empty backbone coordinate array!")

    # evaluate bb-bb clashes
    if calculate_clashes(coords1_bb, coords2_bb, vdw1_bb, vdw2_bb, bb_multiplier, bb_multiplier) == True:
        return (True, 1, 0, 0)

    vdw1_sc = np.array([vdw[atom.element.lower()] for atom in entity1_sc])
    vdw2_sc = np.array([vdw[atom.element.lower()] for atom in entity2_sc])

    coords1_sc = np.array([atom.get_coord() for atom in entity1_sc])
    coords2_sc = np.array([atom.get_coord() for atom in entity2_sc])

    # evaluate bb-sc and sc-sc clashes
    if calculate_clashes(coords1_bb, coords2_sc, vdw1_bb, vdw2_sc, bb_multiplier, sc_multiplier) == True:
        return (True, 0, 1, 0)
    elif calculate_clashes(coords1_sc, coords2_bb, vdw1_sc, vdw2_bb, sc_multiplier, bb_multiplier) == True:
        return (True, 0, 1, 0)
    elif calculate_clashes(coords1_sc, coords2_sc, vdw1_sc, vdw2_sc, sc_multiplier, sc_multiplier) == True:
        return (True, 0, 0, 1)
    else:
        return (False, 0, 0, 0)


def main(args):
    # Load pose files (JSON with model/chain info)
    df1 = pd.read_json(args.pose1)
    df2 = pd.read_json(args.pose2)
    vdw = vdw_radii()

    # Load all relevant structures
    structdict = {}
    for pose in set(df1['poses']).union(df2['poses']):
        structdict[pose] = load_structure_from_pdbfile(pose, all_models=True)

    results = []

    for i, row1 in df1.iterrows():
        ent1 = structdict[row1['poses']][row1['model_num']][row1['chain_id']]
        for j, row2 in df2.iterrows():
            ent2 = structdict[row2['poses']][row2['model_num']][row2['chain_id']]
            clash, bb_bb_clash, bb_sc_clash, sc_sc_clash = clash_detection(ent1, ent2, args.bb_multiplier, args.sc_multiplier, vdw)
            results.append({
                'pose1_index': i,
                'pose2_index': j,
                'pose1_path': row1['poses'],
                'pose2_path': row2['poses'],
                'model1': row1['model_num'],
                'model2': row2['model_num'],
                'clash': clash,
                'bb_bb_clash': bb_bb_clash,
                'bb_sc_clash': bb_sc_clash,
                'sc_sc_clash': sc_sc_clash
            })

    df_out = pd.DataFrame(results)
    os.makedirs(args.working_dir, exist_ok=True)
    out_path = os.path.join(args.working_dir, f"{args.output_prefix}.json")
    df_out.to_json(out_path, orient='records')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--pose1", type=str, required=True, help="First input JSON file")
    parser.add_argument("--pose2", type=str, required=True, help="Second input JSON file")
    parser.add_argument("--working_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output file")

    parser.add_argument("--bb_multiplier", type=float, default=1.5, help="VDW multiplier for backbone-backbone clashes")
    parser.add_argument("--sc_multiplier", type=float, default=0.75, help="VDW multiplier for sidechain clashes")

    args = parser.parse_args()
    main(args)

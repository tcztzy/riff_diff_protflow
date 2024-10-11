#!/home/tripp/mambaforge/envs/protflow_new/bin/python

import os
import sys
import itertools
import copy
import logging
from pathlib import Path
import time
import Bio.PDB
from Bio.PDB import Structure, Model, Chain
import pandas as pd
import numpy as np
import shutil
import glob

# import protflow
from protflow.utils.biopython_tools import load_structure_from_pdbfile, save_structure_to_pdbfile
from protflow.jobstarters import SbatchArrayJobstarter, LocalJobStarter
from protflow.utils.plotting import violinplot_multiple_cols, violinplot_multiple_cols_dfs
from protflow.utils.utils import vdw_radii
from protflow.utils.openbabel_tools import openbabel_fileconverter
from protflow.poses import description_from_path
from protflow.residues import from_dict
from protflow.config import PROTFLOW_ENV



def identify_rotamer_by_bfactor_probability(entity):
    '''
    returns the residue number where bfactor > 0, since this is where the rotamer probability was saved
    '''
    residue = None
    for atom in entity.get_atoms():
        if atom.bfactor > 0:
            residue = atom.get_parent()
            break
    if not residue:
        raise RuntimeError('Could not find any rotamer in chain. Maybe rotamer probability was set to 0?')
    resnum = residue.id[1]
    return resnum

def distance_detection(entity1, entity2, bb_only:bool=True, ligand:bool=False, clash_detection_vdw_multiplier:float=1.0, database:str='database', resnum:int=None, covalent_bonds:str=None):
    '''
    checks for clashes by comparing VanderWaals radii. If clashes with ligand should be detected, set ligand to true. Ligand chain must be added as second entity.
    bb_only: only detect backbone clashes between to proteins or a protein and a ligand.
    clash_detection_vdw_multiplier: multiply Van der Waals radii with this value to set clash detection limits higher/lower
    database: path to database directory
    '''
    backbone_atoms = ['CA', 'C', 'N', 'O', 'H']
    if bb_only == True and ligand == False:
        entity1_atoms = (atom for atom in entity1.get_atoms() if atom.name in backbone_atoms)
        entity2_atoms = (atom for atom in entity2.get_atoms() if atom.name in backbone_atoms)
    elif bb_only == True and ligand == True:
        entity1_atoms = (atom for atom in entity1.get_atoms() if atom.name in backbone_atoms)
        entity2_atoms = (atom for atom in entity2.get_atoms())
    else:
        entity1_atoms = (atom for atom in entity1.get_atoms())
        entity2_atoms = (atom for atom in entity2.get_atoms())
    for atom_combination in itertools.product(entity1_atoms, entity2_atoms):
        #skip clash detection for covalent bonds
        covalent = False
        if resnum and covalent_bonds:
            for cov_bond in covalent_bonds.split(','):
                if atom_combination[0].get_parent().id[1] == resnum and atom_combination[0].name == cov_bond.split(':')[0] and atom_combination[1].name == cov_bond.split(':')[1]:
                    covalent = True
        if covalent == True:
            continue
        distance = atom_combination[0] - atom_combination[1]
        element1 = atom_combination[0].element
        element2 = atom_combination[1].element
        clash_detection_limit = clash_detection_vdw_multiplier * (vdw_radii()[str(element1).lower()] + vdw_radii()[str(element2).lower()])

        if distance < clash_detection_limit:
            return True
    return False

def extract_backbone_coordinates(residue):
    bb_atoms = [atom for atom in residue.get_atoms() if atom.id in ['N', 'CA', 'C', 'O']]
    coord_dict = {}
    for atom in bb_atoms:
        coord_dict[atom.id] = tuple(round(float(coord), 3) for coord in atom.get_coord())
    return coord_dict

def extract_chi_angles(residue):
    '''
    residue has to be converted to internal coords first! (on chain/model/structure level)
    '''
    chi1 = float('nan')
    chi2 = float('nan')
    chi3 = float('nan')
    chi4 = float('nan')
    resname = residue.get_resname()
    if resname in AAs_up_to_chi1() + AAs_up_to_chi2() + AAs_up_to_chi3() + AAs_up_to_chi4():
        chi1 = round(residue.internal_coord.get_angle("chi1"), 1)
    if resname in AAs_up_to_chi2() + AAs_up_to_chi3() + AAs_up_to_chi4():
        chi2 = round(residue.internal_coord.get_angle("chi2"), 1)
    if resname in AAs_up_to_chi3() + AAs_up_to_chi4():
        chi3 = round(residue.internal_coord.get_angle("chi3"), 1)
    if resname in AAs_up_to_chi4():
        chi4 = round(residue.internal_coord.get_angle("chi4"), 1)
    return {"chi1": chi1, "chi2": chi2, "chi3": chi3, "chi4": chi4}


def AAs_up_to_chi1():
    AAs = ['CYS', 'SER', 'THR', 'VAL']
    return AAs

def AAs_up_to_chi2():
    AAs = ['ASP', 'ASN', 'HIS', 'ILE', 'LEU', 'PHE', 'PRO', 'TRP', 'TYR']
    return AAs

def AAs_up_to_chi3():
    AAs = ['GLN', 'GLU', 'MET']
    return AAs

def AAs_up_to_chi4():
    AAs = ['ARG', 'LYS']
    return AAs

def log_and_print(string: str):
    logging.info(string)
    print(string)
    return string

def normalize_col(df:pd.DataFrame, col:str, scale:bool=False, output_col_name:str=None) -> pd.DataFrame:
    ''''''
    median = df[col].median()
    std = df[col].std()
    if not output_col_name:
        output_col_name = f"{col}_normalized"
    if df[col].nunique() == 1:
        df[output_col_name] = 0
        return df
    df[output_col_name] = (df[col] - median) / std
    if scale == True:
        df = scale_col(df=df, col=output_col_name, inplace=True)
    return df

def scale_col(df:pd.DataFrame, col:str, inplace=False) -> pd.DataFrame:
    #scale column to values between 0 and 1
    factor = df[col].max() - df[col].min()
    df[f"{col}_scaled"] = df[col] / factor
    df[f"{col}_scaled"] = df[f"{col}_scaled"] + (1 - df[f"{col}_scaled"].max())
    if inplace == True:
        df[col] = df[f"{col}_scaled"]
        df.drop(f"{col}_scaled", axis=1, inplace=True)
    return df


def combine_normalized_scores(df: pd.DataFrame, name:str, scoreterms:list, weights:list, normalize:bool=False, scale:bool=False):
    if not len(scoreterms) == len(weights):
        raise RuntimeError(f"Number of scoreterms ({len(scoreterms)}) and weights ({len(weights)}) must be equal!")
    df[name] = sum([df[col]*weight for col, weight in zip(scoreterms, weights)]) / sum(weights)
    df[name] = df[name] / df[name].max()
    if normalize == True:
        df = normalize_col(df, name, False)
        df.drop(name, axis=1, inplace=True)
        df.rename(columns={f'{name}_normalized': name}, inplace=True)
    if scale == True:
        df = scale_col(df, name, True)
    return df

def run_masterv2(poses, output_dir, chains, master_dir, database, jobstarter, rmsdCut:float, topN:int=None, minN:int=None, gapLen:int=None, outType:str='match'):

    """
    for each input ensembles, identifies how many times this ensemble occurs in the the database (= there is a substructure below the rmsd cutoff). if no rmsd cutoff is provided, will automatically set one according to ensemble size)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/pds_files", exist_ok=True)
    os.makedirs(f"{output_dir}/matches", exist_ok=True)

    scorefile = os.path.join(output_dir, f'master_scorefile.json')
    if os.path.isfile(scorefile):
        log_and_print(f"Found existing scorefile at {scorefile}. Skipping step.")
        return pd.read_json(scorefile)

    sbatch_options = ["-c1", f'-e {output_dir}/create_pds.err -o {output_dir}/create_pds.out']

    cmds = [f"{master_dir}createPDS --type query --pdb {pose} --pds {output_dir}/pds_files/{Path(pose).stem}.pds" for pose in poses]
    top_cmds = [cmds[x:x+200] for x in range(0, len(cmds), 200)]
    top_cmds = ["; ".join(cmds) for cmds in top_cmds]

    jobstarter.start(cmds=top_cmds, jobname="create_pds", wait=True, output_path=output_dir)

    pdsfiles = [f"{output_dir}/pds_files/{Path(pose).stem}.pds" for pose in poses]

    cmds = [f"{master_dir}/master --query {pds} --targetList {database} --outType {outType} --rmsdCut {rmsdCut} --matchOut {output_dir}/matches/{Path(pds).stem}.match" for pds in pdsfiles]
    top_cmds = [cmds[x:x+5] for x in range(0, len(cmds), 5)]
    top_cmds = ["; ".join(cmds) for cmds in top_cmds]

    matchfiles = [f"{output_dir}/matches/{Path(pds).stem}.match" for pds in pdsfiles]

    if topN:
        cmds = [cmd + f" --max {topN}" for cmd in cmds]
    if minN:
        cmds = [cmd + f" --min {minN}" for cmd in cmds]
    if gapLen:
        cmds = [cmd + f" --gapConst {gapLen}" for cmd in cmds]

    jobstarter.start(cmds=top_cmds, sbatch_options=sbatch_options, jobname="MASTER", output_path=output_dir)

    combinations = [''.join(perm) for perm in itertools.permutations(chains)]
    match_dict = {}
    out_df = []
    for match in matchfiles:
        for i in combinations:
            match_dict[i] = {'path_num_matches': 0, 'rmsds': []}
            ensemble_rmsds = []
        with open(match, 'r') as m:
            lines = m.readlines()
            for line in lines:
                order = assign_chain_letters(line)
                if not order == None:
                    match_dict[order]['path_num_matches'] += 1
                    rmsd = float(line.split()[0])
                    match_dict[order]['rmsds'].append(rmsd)
                    ensemble_rmsds.append(rmsd)

        ensemble_matches = sum([match_dict[i]['path_num_matches'] for i in combinations])
        if ensemble_matches > 0:
            mean_ensemble_rmsd = sum(ensemble_rmsds) / len(ensemble_rmsds)
            min_ensemble_rmsd = min(ensemble_rmsds)
        else:
            mean_ensemble_rmsd = None
            min_ensemble_rmsd = None

        df = pd.DataFrame({'description': [int(Path(match).stem.split("_")[-1]) for i in combinations], 'path': combinations, 'ensemble_num_matches': [ensemble_matches for i in combinations], 'path_num_matches': [match_dict[i]['path_num_matches'] for i in combinations], 'mean_match_rmsd': [mean_ensemble_rmsd for i in combinations], 'min_match_rmsd': [min_ensemble_rmsd for i in combinations]})
        out_df.append(df)

    out_df = pd.concat(out_df).reset_index(drop=True)
    out_df.to_json(scorefile)
    return out_df

def add_terminal_coordinates_to_df(df, Ntermres, Ctermres):
    bb_atoms = ['N', 'CA', 'C', 'O']

    for atom in bb_atoms:
        df[f'Nterm_{atom}_x'] = round(float(Ntermres[atom].get_coord()[0]), 3)
        df[f'Nterm_{atom}_y'] = round(float(Ntermres[atom].get_coord()[1]), 3)
        df[f'Nterm_{atom}_z'] = round(float(Ntermres[atom].get_coord()[2]), 3)
        df[f'Cterm_{atom}_x'] = round(float(Ctermres[atom].get_coord()[0]), 3)
        df[f'Cterm_{atom}_y'] = round(float(Ctermres[atom].get_coord()[1]), 3)
        df[f'Cterm_{atom}_z'] = round(float(Ctermres[atom].get_coord()[2]), 3)
    return df

def create_bbcoord_dict(series):
    bb_atoms = ['N', 'CA', 'C', 'O']
    bb_dict = {'Nterm': {}, 'Cterm': {}}
    for atom in bb_atoms:
        bb_dict['Nterm'][atom] = (series[f'Nterm_{atom}_x'], series[f'Nterm_{atom}_y'], series[f'Nterm_{atom}_z'])
        bb_dict['Cterm'][atom] = (series[f'Cterm_{atom}_x'], series[f'Cterm_{atom}_y'], series[f'Cterm_{atom}_z'])
    return bb_dict


def run_clash_detection(combinations, num_combs, directory, bb_multiplier, sc_multiplier, database, script_path, jobstarter):
    '''
    combinations: iterator object that contains pd.Series
    max_num: maximum number of ensembles per slurm task
    directory: output directory
    bb_multiplier: multiplier for clash detection only considering backbone clashes
    sc_multiplier: multiplier for clash detection, considering sc-sc and bb-sc clashes
    database: directory of riffdiff database
    script_path: path to clash_detection.py script
    '''
    ens_json_dir = os.path.join(directory, 'ensemble_pkls')
    scores_json_dir = os.path.join(directory, 'scores')
    os.makedirs(ens_json_dir, exist_ok=True)
    out_pkl = os.path.join(directory, 'clash_detection_scores.pkl')
    if os.path.isfile(out_pkl):
        log_and_print(f'Found existing scorefile at {out_pkl}. Skipping step!')

        out_df = pd.read_pickle(out_pkl)
        return out_df

    max_num = int(num_combs / jobstarter.max_cores)
    if max_num < 10000:
        max_num = 10000

    ensemble_num = 0
    ensemble_nums_toplist = []
    ensemble_names = []
    score_names = []
    ensembles_toplist = []
    ensembles_list = []

    for comb in combinations:
        if ensemble_num % max_num == 0:
            ensembles_list = []
            ensembles_toplist.append(ensembles_list)
            ensemble_nums = []
            ensemble_nums_toplist.append(ensemble_nums)
        for series in comb:
            ensemble_nums.append(ensemble_num)
            ensembles_list.append(series)
        ensemble_num += 1


    in_df = []
    count = 0
    log_and_print(f'Writing pickles...')
    for ensembles_list, ensemble_nums in zip(ensembles_toplist, ensemble_nums_toplist):
        df = pd.DataFrame(ensembles_list).reset_index(drop=True)
        df['ensemble_num'] = ensemble_nums
        in_name = os.path.join(ens_json_dir, f'ensembles_{count}.pkl')
        out_name = os.path.join(scores_json_dir, f'ensembles_{count}.pkl')
        ensemble_names.append(in_name)
        score_names.append(out_name)
        df[['ensemble_num', 'poses', 'model_num', 'chain_id']].to_pickle(in_name)
        in_df.append(df)
        count += 1
    log_and_print(f'Done writing pickles!')

    in_df = pd.concat(in_df).reset_index(drop=True)

    cmds = [f"{os.path.join(PROTFLOW_ENV, "python")} {script_path} --pkl {json} --working_dir {directory} --bb_multiplier {bb_multiplier} --sc_multiplier {sc_multiplier} --output_prefix {str(index)} --database_dir {database}" for index, json in enumerate(ensemble_names)]

    log_and_print(f'Distributing clash detection to cluster...')

    jobstarter.start(cmds=cmds, jobname="clash_detection", wait=True, output_path=directory)

    log_and_print(f'Reading in clash pickles...')
    out_df = []
    for file in score_names:
        out_df.append(pd.read_pickle(file))


    #delete input pkls because this folder probably takes a lot of space
    shutil.rmtree(ens_json_dir)
    shutil.rmtree(scores_json_dir)

    out_df = pd.concat(out_df)
    log_and_print(f'Merging with original dataframe...')
    out_df = out_df.merge(in_df, on=['ensemble_num', 'poses', 'model_num', 'chain_id']).reset_index(drop=True)
    log_and_print(f'Writing output pickle...')
    out_df.to_pickle(out_pkl)
    log_and_print(f'Clash check completed.')

    return out_df

def auto_determine_rmsd_cutoff(ensemble_size):
    '''
    calcutes rmsd cutoff based on ensemble size. equation is set so that ensemble size of 10 residues returns cutoff of 1.4 A and ensemble size of 26 residues returns cutoff of 2 A. rmsd cutoff cannot be higher than 2 A (to prevent excessive runtimes)
    parameters determined for 4 disjointed fragments of equal length, most likely lower cutoffs can be used when using 3 or less disjointed fragments
    '''
    rmsd_cutoff = 0.0375 * ensemble_size + 1.025
    if rmsd_cutoff > 2:
        rmsd_cutoff = 2
    return round(rmsd_cutoff, 2)

def assign_chain_letters(line):
    line_sep = line.split()
    cleaned_list = [int(s.replace('[', '').replace(',', '').replace(']', '')) for s in line_sep[2:]]
    sorted_list = sorted(cleaned_list)
    new_list = [sorted_list.index(value) for value in cleaned_list]

    result = ''.join([chr(65 + index) for index in new_list])
    return result

def create_covalent_bonds(df:pd.DataFrame):
    covalent_bonds = []
    for index, series in df.iterrows():
        if not series['covalent_bond'] == None:
            covalent_bond = series['covalent_bond'].split(':')
            covalent_bond = f"{series['rotamer_pos']}{series['chain_id']}_{series['catres_identities']}_{covalent_bond[0]}:1Z_{series['ligand_name']}_{covalent_bond[1]}"
            covalent_bonds.append(covalent_bond)
    if len(covalent_bonds) >= 1:
        covalent_bonds = ",".join(covalent_bonds)
    else:
        covalent_bonds = ""
    return covalent_bonds

def create_motif_contig(chain_str, fragsize_str, path_order, sep):
    chains = chain_str.split(sep)
    fragsizes = fragsize_str.split(sep)
    contig = [f"{chain}1-{length}" for chain, length in zip(chains, fragsizes)]
    contig = f"{sep}".join(sorted(contig, key=lambda x: path_order.index(x[0])))
    return contig

def create_pdbs(df:pd.DataFrame, output_dir, ligand, channel_path, preserve_channel_coordinates):
    filenames = []
    for i, row in df.iterrows():
        paths = row["poses"].split(",")
        models = row["model_num"].split(",")
        chains = row["chain_id"].split(",")
        frag_chains = [load_structure_from_pdbfile(path, all_models=True)[int(model)][chain] for path, model, chain in zip(paths, models, chains)]
        filename = ["".join([chain, model]) for chain, model in zip(chains, models)]
        filename = "-".join(sorted(filename, key=lambda x: row['path_order'].index(x[0]))) + ".pdb"
        filename = os.path.abspath(os.path.join(output_dir, filename))
        struct = Structure.Structure('out')
        struct.add(model := Model.Model(0))
        model.add(chain := Chain.Chain("Z"))
        for frag in frag_chains:
            model.add(frag)
        for lig in ligand:
            chain.add(lig)
        if channel_path and preserve_channel_coordinates:
            model.add(load_structure_from_pdbfile(channel_path)["Q"])
        elif channel_path:
            model = add_polyala_to_pose(model, polyala_path=channel_path, polyala_chain="Q", ligand_chain='Z')
        save_structure_to_pdbfile(struct, filename)
        filenames.append(filename)
    return filenames


def chain_alphabet():
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']


def import_fragment_json_files(input_dir:str):
    #import json files:
    input_jsons = glob.glob(os.path.join(input_dir, "*.json"))
    input_jsons = sorted(list(set(input_jsons)))

    inputs = []
    column_names = ['model_num', 'rotamer_pos', 'AAs', 'backbone_score', 'fragment_score', 'rotamer_probability', 'covalent_bond', 'ligand_chain', 'poses', 'poses_description']
    for file in input_jsons:
        df = pd.read_json(file)
        if not all(column in df.columns for column in column_names):
            logging.error(f'{file} is not a correct fragment json file!')
            raise RuntimeError(f'{file} is not a correct fragment json file!')
        inputs.append(df)
    inputs = pd.concat(inputs).reset_index(drop=True)
    return inputs


def add_polyala_to_pose(pose: Structure, polyala_path:str, polyala_chain:str="Q", ligand_chain:str="Z", ignore_atoms:"list[str]"=["H"]) -> Structure:
    '''
    
    '''
    # load polyala:
    polyala = load_structure_from_pdbfile(polyala_path)

    # determine polyala chain name (only first chain will be considered)
    pa_chain = [x.id for x in polyala.get_chains()][0]

    pa_atoms = [atom for atom in polyala.get_atoms() if atom.name not in ignore_atoms]


    frag_protein_atoms, frag_ligand_atoms = get_protein_and_ligand_atoms(pose, ligand_chain=ligand_chain, ignore_atoms=ignore_atoms)

    # calculate vector between fragment and ligand centroids
    frag_protein_centroid = np.mean(frag_protein_atoms, axis=0)
    frag_ligand_centroid = np.mean(frag_ligand_atoms, axis=0)
    vector_fragment = frag_ligand_centroid - frag_protein_centroid

    # calculate vector between CA of first and last residue of polyala
    polyala_ca = [atom.get_coord() for atom in pa_atoms if atom.id == "CA"]
    ca1, ca2 = polyala_ca[0], polyala_ca[-1]
    vector_polyala = ca2 - ca1

    # calculate rotation between vectors
    R = Bio.PDB.rotmat(Bio.PDB.Vector(vector_polyala), Bio.PDB.Vector(vector_fragment))

    # rotate polyala and translate into motif
    polyala_rotated = apply_rotation_to_pose(polyala, ca1, R)
    polyala_translated = apply_translation_to_pose(polyala_rotated, frag_ligand_centroid - ca1)

    # change chain id of polyala and add into pose:
    if polyala_chain in [chain.id for chain in pose.get_chains()]: raise KeyError(f"Chain {polyala_chain} already found in pose. Try other chain name!")
    if pa_chain != polyala_chain: polyala_translated[pa_chain].id = polyala_chain
    pose.add(polyala_translated[polyala_chain])
    return pose

def apply_rotation_to_pose(pose: Structure, origin: "list[float]", R: "list[list[float]]") -> Structure:
    ''''''
    for chain in pose:
        for residue in chain:
            for atom in residue:
                atom.coord = np.dot(R, atom.coord - origin) + origin
    return pose

def apply_translation_to_pose(pose: Structure, vector: "list[float]") -> Structure:
    ''''''
    for chain in pose:
        for residue in chain:
            for atom in residue:
                atom.coord += vector
    return pose

def get_protein_and_ligand_atoms(pose: Structure, ligand_chain, bb_atoms=["CA", "C", "N", "O"], ignore_atoms=["H"]) -> "tuple[list]":
    '''AAA'''
    if type(ligand_chain) == type("str"):
        # get all CA coords of protein:
        protein_atoms = np.array([atom.get_coord() for atom in get_protein_atoms(pose, ligand_chain) if atom.id in bb_atoms])

        # get Ligand Heavyatoms:
        ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms() if atom.id not in ignore_atoms])

    elif type(ligand_chain) == Bio.PDB.Chain.Chain:
        # get all CA coords of protein:
        protein_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.id == "CA"])
        
        # get Ligand Heavyatoms:
        ligand_atoms = np.array([atom.get_coord() for atom in ligand_chain.get_atoms() if atom.id not in ignore_atoms])
    else: raise TypeError(f"Expected 'ligand' to be of type str or Bio.PDB.Chain.Chain, but got {type(ligand_chain)} instead.")
    return protein_atoms, ligand_atoms

def get_protein_atoms(pose: Structure, ligand_chain:str=None, atms:list=None) -> list:
    '''Selects atoms from a pose object. If ligand_chain is given, excludes all atoms in ligand_chain'''
    # define chains of pose
    chains = [x.id for x in pose.get_chains()]
    if ligand_chain: chains.remove(ligand_chain)

    # select specified atoms
    pose_atoms = [atom for chain in chains for atom in pose[chain].get_atoms()]
    if atms: pose_atoms = [atom for atom in pose_atoms if atom.id in atms]

    return pose_atoms

def sort_dataframe_groups_by_column(df:pd.DataFrame, group_col:str, sort_col:str, method="mean", ascending:bool=True, filter_top_n:int=None) -> pd.DataFrame:
    # group by group column and calculate mean values
    df_sorted = df.groupby(group_col, sort=False).agg({sort_col: method}).sort_values(sort_col, ascending=ascending)
    # filter 
    if filter_top_n:
        df_sorted = df_sorted.head(filter_top_n)
    # merge back with original dataframe
    df = df_sorted.loc[:, []].merge(df, left_index=True, right_on=group_col)
    # reset index
    df.reset_index(drop=True, inplace=True)
    return df

def concat_columns(group):
    return ','.join(group.astype(str))

def split_str_to_dict(key_str, value_str, sep):
    return dict(zip(key_str.split(sep), [list(i) for i in value_str.split(sep)]))

def create_motif_residues(chain_str, fragsize_str, sep:str=","):
    motif_residues = [[i for i in range(1, int(fragsize)+1)] for fragsize in fragsize_str.split(sep)]
    return dict(zip(chain_str.split(sep), motif_residues))


def main(args):
    '''
    Combines every model from each input pdb with every model from other input pdbs. Input pdbs must only contain chain A and (optional) a ligand in chain Z.
    Checks if any combination contains clashes, and removes them.
    Writes the coordinates of N, CA, C of the central atom as well as the rotamer probability and other infos to a json file.
    '''

    # create output directory
    working_dir = os.path.join(args.working_dir, f"{args.output_prefix}_motif_library_assembly" if args.output_prefix else "motif_library_assembly")
    os.makedirs(working_dir, exist_ok=True)

    # set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join(working_dir, "assembly.log"))
    cmd = ''
    for key, value in vars(args).items():
        cmd += f'--{key} {value} '
    cmd = f'{sys.argv[0]} {cmd}'
    logging.info(cmd)

    ##################### SANITY CHECKS & IMPORT ####################


    riff_diff_dir = os.path.abspath(args.riff_diff_dir)
    database_dir = os.path.join(riff_diff_dir, "database")
    utils_dir = os.path.join(riff_diff_dir, "utils")
    channels_dir = os.path.join(database_dir, "channel_placeholder")

    # check if output already exists
    out_json = os.path.join(args.working_dir, f'{args.output_prefix}_selected_paths.json' if args.output_prefix else "selected_paths.json")
    if os.path.exists(out_json):
        logging.error(f'Output file already exists at {out_json}!')
        raise RuntimeError(f'Output file already exists at {out_json}!')

    # detect if channel should be added to assemblies
    if args.no_channel_placeholder:
        channel_path = None
        logging.info("Running without channel placeholder.")
    elif args.channel_path:
        if os.path.isfile(args.channel_path):
            channel = load_structure_from_pdbfile(args.channel_path)[args.channel_chain]
            channel.detach_parent()
            channel.id = "Q"
            for index, residue in enumerate(channel.get_residues()):
                residue.id = (residue.id[0], index + 1, residue.id[2])
            channel_size = index + 1
            channel_path = os.path.join(working_dir, "channel_placeholder.pdb")
            save_structure_to_pdbfile(channel, channel_path)
        else:
            logging.error(f"Could not find a PDB file at {channel_path} to add as channel placeholder!")
            raise RuntimeError(f"Could not find a PDB file at {channel_path} to add as channel placeholder!")
    else:
        channel_path = os.path.join(channels_dir, "helix_cone_long.pdb")

    if args.jobstarter == "SbatchArray": jobstarter = SbatchArrayJobstarter(max_cores=args.cpus)
    elif args.jobstarter == "Local": jobstarter = LocalJobStarter(max_cores=args.cpus)
    else: 
        logging.error(f"Jobstarter must be either 'SbatchArray' or 'Local', not {args.jobstarter}!")
        raise KeyError(f"Jobstarter must be either 'SbatchArray' or 'Local', not {args.jobstarter}!")

    in_df = import_fragment_json_files(args.input_dir)

    ################## CLASH DETECTION ##########################

    clash_dir = os.path.join(working_dir, 'clash_check')
    os.makedirs(clash_dir, exist_ok=True)

    grouped_df = in_df.groupby('poses', sort=False)
    counter = 0
    chains = []
    df_list = []
    structdict = {}
    ensemble_size = grouped_df.mean(numeric_only=True)['frag_length'].sum()

    for pose, pose_df in grouped_df:
        channel_clashes = 0
        log_and_print(f'Working on {pose}...')
        pose_df['input_poses'] = pose_df['poses']
        pose_df['chain_id'] = chain_alphabet()[counter]
        struct = load_structure_from_pdbfile(pose, all_models=True)
        model_dfs = []
        for index, series in pose_df.iterrows():
            chain = struct[series['model_num']]['A']
            if channel_path and args.preserve_channel_coordinates == True:
                if distance_detection(chain, load_structure_from_pdbfile(channel_path)[args.channel_chain], True, False, args.channel_clash_detection_vdw_multiplier, database=database_dir) == True:
                    channel_clashes += 1
                    continue
            chain.id = chain_alphabet()[counter]
            model_dfs.append(series)
        if channel_path and args.preserve_channel_coordinates == True:
            log_and_print(f'Removed {channel_clashes} models that were clashing with channel chain found in {channel_path}.')
        if len(model_dfs) == 0:
            raise RuntimeError(f'Could not find any models that are not clashing with channel chain for {pose}. Adjust clash detection parameters or move channel!')
        pose_df = pd.DataFrame(model_dfs)
        structdict[struct.id] = struct
        filename = os.path.join(clash_dir, f'{struct.id}_rechained.pdb')
        struct.id = filename
        save_structure_to_pdbfile(pose=struct, save_path=filename, multimodel=True)
        pose_df['poses'] = os.path.abspath(filename)
        chains.append(chain_alphabet()[counter])
        counter += 1
        #TODO: no idea if this is still required
        if 'covalent_bond' in pose_df.columns:
            pose_df['covalent_bond'] = pose_df['covalent_bond'].replace(np.nan, None)
        else:
            pose_df['covalent_bond'] = None
        df_list.append(pose_df)

    ligands = [lig for lig in struct[0]['Z'].get_residues()]
    for lig in ligands:
        lig.id = ("H", lig.id[1], lig.id[2])

    grouped_df = pd.concat(df_list).groupby('poses', sort=False)

    # generate every possible combination of input models
    num_models = [len(df.index) for group, df in grouped_df]
    num_combs = 1
    for i in num_models: num_combs *= i
    log_and_print(f'Generating {num_combs} possible combinations...')

    init = time.time()

    combinations = itertools.product(*[[row for index, row in pose_df.iterrows()] for _, pose_df in grouped_df])
    log_and_print(f'Performing pairwise clash detection...')
    ensemble_dfs = run_clash_detection(combinations=combinations, num_combs=num_combs, directory=clash_dir, bb_multiplier=args.fragment_backbone_clash_detection_vdw_multiplier, sc_multiplier=args.fragment_fragment_clash_detection_vdw_multiplier, database=database_dir, script_path=os.path.join(utils_dir, "clash_detection.py"), jobstarter=jobstarter)

    #calculate scores
    score_df = ensemble_dfs.groupby('ensemble_num', sort=False).mean(numeric_only=True)
    score_df = normalize_col(score_df, 'fragment_score', scale=True, output_col_name='ensemble_score')
    ensemble_dfs = ensemble_dfs.merge(score_df['ensemble_score'], left_on='ensemble_num', right_index=True).sort_values('ensemble_num').reset_index(drop=True)

    #remove all clashing ensembles
    log_and_print(f'Filtering clashing ensembles...')
    post_clash = ensemble_dfs[ensemble_dfs['clash_check'] == False].reset_index(drop=True)


    log_and_print(f'Completed clash check in {round(time.time() - init, 0)} s.')
    if len(post_clash.index) == 0:
        log_and_print(f'No ensembles found! Try adjusting VdW multipliers or pick different fragments!')
        raise RuntimeError(f'No ensembles found! Try adjusting VdW multipliers or pick different fragments!')

    passed = int(len(post_clash.index) / len(chains))
    log_and_print(f'Found {passed} non-clashing ensembles.')

    # sort ensembles by score
    log_and_print("Sorting ensembles by score...")
    post_clash = sort_dataframe_groups_by_column(df=post_clash, group_col="ensemble_num", sort_col="ensemble_score", ascending=False)
    post_clash["ensemble_num"] = post_clash.groupby("ensemble_num", sort=False).ngroup() + 1 
    log_and_print("Sorting completed.")

    # plot data
    plotpath = os.path.join(working_dir, "clash_filter.png")
    log_and_print(f"Plotting data at {plotpath}.")
    dfs = [post_clash.groupby('ensemble_num', sort=False).mean(numeric_only=True), score_df]
    df_names = ['filtered', 'unfiltered']
    cols = ['backbone_probability', 'rotamer_probability', 'phi_psi_occurrence']
    col_names = ['mean backbone probability', 'mean rotamer probability', 'mean phi psi occurrence']
    y_labels = ['probability', 'probability', 'probability', 'probability']
    violinplot_multiple_cols_dfs(dfs=dfs, df_names=df_names, cols=cols, titles=col_names, y_labels=y_labels, out_path=plotpath, show_fig=False)

    if args.run_master:

        ######################## RUN MASTER ###############################

        master_input = post_clash[post_clash["ensemble_num"] < args.max_master_input]
        dfs = [master_input, post_clash]
        df_names = ['master input', 'non-clashing ensembles']
        cols = ['ensemble_score', 'fragment_score', 'backbone_score', 'rotamer_score']
        col_names = ['ensemble score', 'fragment score', 'backbone score', 'rotamer score']
        y_labels = ['score [AU]', 'score [AU]', 'score [AU]', 'score [AU]']
        dims = [(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)]
        plotpath = os.path.join(working_dir, "master_input_filter.png")
        violinplot_multiple_cols_dfs(dfs, df_names=df_names, cols=cols, titles=col_names, y_labels=y_labels, dims=dims, out_path=plotpath)


        #### check for matches in database for top ensembles ####

        log_and_print(f'Writing input ensembles to disk...')
        master_dir = os.path.join(working_dir, 'master')
        os.makedirs(master_dir, exist_ok=True)
        os.makedirs(f"{master_dir}/pdbs/", exist_ok=True)
        filenames = []
        for index, ensemble in master_input.groupby('ensemble_num', sort=False):
            ens = Structure.Structure(f'ensemble_{index}.pdb')
            ens.add(Model.Model(0))
            for num, row in ensemble.iterrows():
                pose = row['poses_description']
                ens_chain = structdict[pose][row['model_num']][row['chain_id']]
                ens_chain.id = row['chain_id']
                ens[0].add(ens_chain)
            filename = os.path.join(f"{master_dir}/pdbs/", ens.id)
            if not os.path.isfile(filename):
                save_structure_to_pdbfile(ens, filename, multimodel=True)
            filenames.append(filename)

        if args.master_rmsd_cutoff == "auto":
            rmsd_cutoff = auto_determine_rmsd_cutoff(ensemble_size)
        else:
            rmsd_cutoff = args.master_rmsd_cutoff

        log_and_print(f'Checking for matches in database below {rmsd_cutoff} A.')

        df = run_masterv2(poses=filenames, output_dir=f"{master_dir}/output", chains= chains, rmsdCut=rmsd_cutoff, master_dir=args.master_dir, database=args.master_db, jobstarter=jobstarter)
        #df['combined_matches'] = df['ensemble_num_matches'] + df['path_num_matches']
        df = normalize_col(df, 'path_num_matches', True)
        df = normalize_col(df, 'ensemble_num_matches', True)
        df = combine_normalized_scores(df, 'match_score', ['path_num_matches_normalized', 'ensemble_num_matches_normalized'], [args.path_match_weight, args.ensemble_match_weight], False, False)
        post_match = master_input.merge(df, left_on='ensemble_num', right_on='description').drop('description', axis=1)
        post_match = combine_normalized_scores(post_match, 'path_score', ['ensemble_score', 'match_score'], [args.fragment_score_weight, args.match_score_weight], False, False)
        violinplot_multiple_cols(post_match, cols=['match_score', 'path_num_matches', 'ensemble_num_matches'], titles=['match score', f'path matches\n< {rmsd_cutoff}', f'ensemble matches\n< {rmsd_cutoff}'], y_labels=['AU', '#', '#'], dims=[(-0.05, 1.05), (post_match['path_num_matches'].max() * -0.05, post_match['path_num_matches'].max() * 1.05 ), (post_match['ensemble_num_matches'].max() * -0.05, post_match['ensemble_num_matches'].max() * 1.05 )], out_path=os.path.join(working_dir, f"{args.output_prefix}_master_matches_<_{rmsd_cutoff}.png"), show_fig=False)


        path_df = post_match.copy()
        if args.match_cutoff:
            passed = int(len(path_df.index) / len(chains))
            path_df = path_df[path_df['ensemble_num_matches'] >= args.match_cutoff]
            filtered = int(len(path_df.index) / len(chains))
            log_and_print(f'Removed {passed - filtered} paths with less than {args.match_cutoff} ensemble matches below {rmsd_cutoff} A. Remaining paths: {filtered}.')
            dfs = [path_df, post_match]
            df_names = ['filtered', 'unfiltered']
            cols = ['path_score', 'match_score', 'ensemble_score', 'fragment_score', 'backbone_score', 'rotamer_score']
            col_names = ['path score', 'master score', 'ensemble score', 'fragment score', 'backbone score', 'rotamer score']
            y_labels = ['score [AU]', 'score [AU]', 'score [AU]', 'score [AU]', 'score [AU]', 'score [AU]']
            dims = [(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)]
            plotpath = os.path.join(working_dir, f"{args.output_prefix}_num_matches_<_{args.match_cutoff}_filter.png")
            violinplot_multiple_cols(path_df, cols=['match_score', 'path_num_matches', 'ensemble_num_matches', 'mean_match_rmsd', 'min_match_rmsd'], titles=['match score', f'path\nmatches < {rmsd_cutoff}', f'ensemble\nmatches < {rmsd_cutoff}', 'ensemble\nmean match rmsd', 'ensemble\nminimum match rmsd'], y_labels=['AU', '#', '#', 'A', 'A'], dims=[(-0.05, 1.05), (path_df['path_num_matches'].min() - (path_df['path_num_matches'].max() - path_df['path_num_matches'].min()) * 0.05, (path_df['path_num_matches'].max() + (path_df['path_num_matches'].max() - path_df['path_num_matches'].min()) * 0.05)), (args.match_cutoff - (path_df['path_num_matches'].max() - args.match_cutoff) * 0.05, (path_df['ensemble_num_matches'].max() + (path_df['ensemble_num_matches'].max() - args.match_cutoff) * 0.05)), (rmsd_cutoff * -0.05, rmsd_cutoff * 1.05), (rmsd_cutoff * -0.05, rmsd_cutoff * 1.05)], out_path=os.path.join(working_dir, f"{args.output_prefix}_master_matches_<_{rmsd_cutoff}_matchfilter.png"), show_fig=False)
            violinplot_multiple_cols_dfs(dfs, df_names=df_names, cols=cols, titles=col_names, y_labels=y_labels, dims=dims, out_path=plotpath, show_fig=False)

    else:
        log_and_print("Creating paths out of ensembles...")
        post_clash['match_score'] = 0
        post_clash['path_score'] = post_clash['ensemble_score']
        post_clash['path_num_matches'] = 0
        # filter for top ensembles to speed things up, since paths within an ensemble have the same score if not running master
        paths = ["".join(perm) for perm in itertools.permutations(chains)]
        post_clash = sort_dataframe_groups_by_column(df=post_clash, group_col="ensemble_num", sort_col="path_score", ascending=False, filter_top_n=1 + int(args.max_out / len(paths)))
        dfs = [post_clash.assign(path_name=post_clash['ensemble_num'].astype(str)+"_" + p) for p in paths]
        path_df = pd.concat(dfs, ignore_index=True)
        print(len(path_df.index))
        log_and_print("Done creating paths.")

    pdb_dir = os.path.join(working_dir, "motif_library")
    os.makedirs(pdb_dir, exist_ok=True)

    if args.max_paths_per_ensemble:
        df_list = []
        for _, df in path_df.groupby('ensemble_num', sort=False):
            df = sort_dataframe_groups_by_column(df, group_col="path_name", sort_col="path_score", ascending=False, filter_top_n=args.max_paths_per_ensemble)
            df_list.append(df)
        path_df = pd.concat(df_list)

    print(len(path_df))
    # select top n paths
    log_and_print(f"Selecting top {args.max_out} paths...")
    top_path_df = sort_dataframe_groups_by_column(df=path_df, group_col="path_name", sort_col="path_score", ascending=False, filter_top_n=args.max_out)
    print(len(top_path_df))

    log_and_print(f"Found {int(len(top_path_df.index)/len(chains))} paths.")

    # create path dataframe
    log_and_print(f"Creating path dataframe...")
    aggregate = {'poses': concat_columns,
                 'chain_id': concat_columns,
                 'model_num': concat_columns,
                 'rotamer_pos':concat_columns, 
                 'frag_length': concat_columns, 
                 'path_score': 'mean', 
                 'backbone_probability': [("backbone_probability", concat_columns), ("backbone_probability_mean", "mean")],
                 'rotamer_probability': [("rotamer_probability", concat_columns), ("rotamer_probability_mean", "mean")],
                 'phi_psi_occurrence': [("phi_psi_occurrence", concat_columns), ("phi_psi_occurrence_mean", "mean")]}

    selected_paths = top_path_df.groupby('path_name', sort=False).agg(aggregate).reset_index(names=["path_name"])
    selected_paths.columns = ['path_name', 'poses', 'chain_id', 'model_num', 'rotamer_pos', 'frag_length',
                               'path_score', 'backbone_probability', 'backbone_probability_mean', 'rotamer_probability',
                               'rotamer_probability_mean','phi_psi_occurrence', 'phi_psi_occurrence_mean']
    
    # create residue selections
    selected_paths["fixed_residues"] = selected_paths.apply(lambda row: split_str_to_dict(row['chain_id'], row['rotamer_pos'], sep=","), axis=1)
    selected_paths["fixed_residues"] = selected_paths.apply(lambda row: from_dict(row["fixed_residues"]), axis=1)
    selected_paths["motif_residues"] = selected_paths.apply(lambda row: create_motif_residues(row['chain_id'], row['frag_length'], sep=","), axis=1)
    selected_paths["motif_residues"] = selected_paths.apply(lambda row: from_dict(row["motif_residues"]), axis=1)
    selected_paths["ligand_motif"] = [from_dict({"Z":[i+1 for i, _ in enumerate(ligands)]}) for id in selected_paths.index]

    selected_paths["path_order"] = selected_paths['path_name'].str.split('_').str[-1]
    selected_paths["motif_contigs"] = selected_paths.apply(lambda row: create_motif_contig(row['chain_id'], row['frag_length'], row['path_order'], sep=","), axis=1)
    if channel_path:
        selected_paths["channel_contig"] = selected_paths.apply(lambda row: create_motif_contig("Q", str(channel_size), "Q", sep=","), axis=1)

    # combine multiple ligands into one for rfdiffusion
    ligand = copy.deepcopy(ligands)
    for lig in ligand:
        lig.resname = "LIG"

    lib_path = os.path.join(working_dir, "motif_library")
    log_and_print(f"Writing motif library .pdbs to {lib_path}")
    os.makedirs(lib_path, exist_ok=True)
    selected_paths["poses"] = create_pdbs(selected_paths, lib_path, ligand, channel_path, args.preserve_channel_coordinates)
    selected_paths["input_poses"] = selected_paths["poses"]
    selected_paths["poses_description"] = selected_paths.apply(lambda row: description_from_path(row['poses']), axis=1)


    log_and_print(f"Writing data to {out_json}")

    ligand_dir = os.path.join(working_dir, 'ligand')
    os.makedirs(ligand_dir, exist_ok=True)
    params_paths = []
    ligand_paths = []
    for index, ligand in enumerate(ligands):
        ligand_pdbfile = save_structure_to_pdbfile(ligand, lig_path:=os.path.abspath(os.path.join(ligand_dir, f"LG{index+1}.pdb")))
        lig_name = ligand.get_resname()
        ligand_paths.append(lig_path)
        if len(list(ligand.get_atoms())) > 2:
            # store ligand as .mol file for rosetta .molfile-to-params.py
            log_and_print(f"Running 'molfile_to_params.py' to generate params file for Rosetta.")
            lig_molfile = openbabel_fileconverter(input_file=lig_path, output_file=lig_path.replace(".pdb", ".mol2"), input_format="pdb", output_format=".mol2")
            cmd = f"{os.path.join(PROTFLOW_ENV, "python")} {os.path.join(utils_dir, 'molfile_to_params.py')} -n {lig_name} -p {ligand_dir}/LG{index+1} {lig_molfile} --keep-names --clobber --chain=Z"
            LocalJobStarter().start(cmds=[cmd], jobname="moltoparams", output_path=ligand_dir)
            params_paths.append(lig_path.replace(".pdb", ".params"))
        else:
            log_and_print(f"Ligand at {ligand_pdbfile} contains less than 3 atoms. No Rosetta Params file can be written for it.")

    if params_paths:
        selected_paths["params_path"] = ",".join(params_paths)
    if ligand_paths:
        selected_paths["ligand_path"] = ",".join(ligand_paths)

    # write output json
    selected_paths.to_json(out_json)

    violinplot_multiple_cols(selected_paths, cols=['backbone_probability_mean', 'phi_psi_occurrence_mean', 'rotamer_probability_mean'], titles=['mean backbone\nprobability', 'mean phi/psi\nprobability', 'mean rotamer\nprobability'], y_labels=['probability', 'probability', 'probability'], out_path=os.path.join(working_dir, "selected_paths_info.png"), show_fig=False)

    log_and_print(f'Done!')

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--riff_diff_dir", default=".", type=str, help="Path to the riff_diff directory. This is workaround and will hopefully be resolved later.")

    # mandatory input
    argparser.add_argument("--input_dir", type=str, required=True, help="Prefix for all json files that should be combined (including path, e.g. './output/mo6_'). Alternative to --json_files")
    argparser.add_argument("--output_prefix", type=str, default=None, help="Prefix for output.")
    argparser.add_argument("--working_dir", type=str, required=True, help="Path to working directory. Has to contain the input pdb files, otherwise run_ensemble_evaluator.py will not work!")

    # stuff you might want to adjust
    argparser.add_argument("--max_paths_per_ensemble", type=int, default=None, help="Maximum number of paths per ensemble (=same fragments but in different order)")
    argparser.add_argument("--channel_clash_detection_vdw_multiplier", type=float, default=0.9, help="Multiplier for VanderWaals radii for clash detection between backbone fragments and channel placeholder. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")
    argparser.add_argument("--fragment_backbone_clash_detection_vdw_multiplier", type=float, default=1.0, help="Multiplier for VanderWaals radii for clash detection inbetween backbone fragments. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")
    argparser.add_argument("--backbone_ligand_clash_detection_vdw_multiplier", type=float, default=1.0, help="Multiplier for VanderWaals radii for clash detection between fragment backbones and ligand. Set None if no ligand is present. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")
    argparser.add_argument("--rotamer_ligand_clash_detection_vdw_multiplier", type=float, default=0.75, help="Multiplier for VanderWaals radii for clash detection between rotamer sidechain and ligand. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")
    argparser.add_argument("--fragment_fragment_clash_detection_vdw_multiplier", type=float, default=0.85, help="Multiplier for VanderWaals radii for clash detection inbetween fragments (including sidechains!). Effectively detects clashes between rotamer of one fragment and the other fragment (including the other rotamer) if multiplier is lower than <fragment_backbone_clash_detection_vdw_multiplier>. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")
    argparser.add_argument("--fragment_score_weight", type=float, default=1, help="Maximum number of cpus to run on")
    argparser.add_argument("--match_cutoff", type=int, default=1, help="Remove all ensembles that have less matches than <match_cutoff> below <master_rmsd_cutoff>")
    argparser.add_argument("--max_out", type=int, default=1000, help="Maximum number of output paths")
    
    # channel placeholder
    argparser.add_argument("--channel_path", type=str, default=None, help="Use a custom channel placeholder. Must be the path to a .pdb file.")
    argparser.add_argument("--channel_chain", type=str, default="Q", help="Chain of the custom channel placeholder (if using a custom channel specified with <channel_path>)")
    argparser.add_argument("--preserve_channel_coordinates", action="store_true", help="Copies channel from channel reference pdb without superimposing on moitf-substrate centroid axis. Useful when channel is present in catalytic array.")
    argparser.add_argument("--no_channel_placeholder", action="store_true", help="Do not add a channel placeholder during diffusion.")

    # master2
    argparser.add_argument("--run_master", action='store_true', help="Score motifs based on similarity to motifs found in natural enzymes. Time consuming!")
    argparser.add_argument("--master_rmsd_cutoff", default='auto', help="Detects how many structures have segments with RMSD below this cutoff for each ensemble. Higher cutoff increases runtime tremendously!")
    argparser.add_argument("--master_dir", type=str, default=None, help="Path to master2 executable (e.g. /path/to/MASTER-v2-masterlib/bin/)")
    argparser.add_argument("--master_db", type=str, default=None, help="Path to Master database (e.g. /path/to/MASTER-v2-masterlib/master_db/list)")
    argparser.add_argument("--max_master_input", type=int, default=20000, help="Maximum number of ensembles that should be fed into master, sorted by fragment score")
    argparser.add_argument("--path_match_weight", type=float, default=0.5, help="Weight of the path-specific number of matches for calculating match score")
    argparser.add_argument("--ensemble_match_weight", type=float, default=1, help="Weight of the number of matches for all paths within the ensemble for calculating match score")
    argparser.add_argument("--match_score_weight", type=float, default=0.5, help="Weight of match score when calculating path score out of ensemble score and match score")

    
    argparser.add_argument("--ligand_chain", type=str, default="Z", help="Name of ligand chain.")

    argparser.add_argument("--jobstarter", type=str, default="SbatchArray", help="Only relevant if <pick_frags_from_db> is set. Defines if jobs run locally or distributed on a cluster using a protflow jobstarter. Must be one of ['SbatchArray', 'Local'].")
    argparser.add_argument("--cpus", type=int, default=320, help="Only relevant if <pick_frags_from_db> is set. Defines how many cpus should be used for distributed computing.")

    args = argparser.parse_args()

    main(args)

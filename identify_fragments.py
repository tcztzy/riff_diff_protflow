import logging
import os
import sys
import copy

# import dependencies
import Bio
from Bio.PDB import Structure, Model, Chain, Residue, Atom
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

# protflow
from protflow.utils.biopython_tools import load_structure_from_pdbfile, save_structure_to_pdbfile
from protflow.utils.utils import vdw_radii
from protflow.jobstarters import SbatchArrayJobstarter, LocalJobStarter
from protflow.utils.plotting import violinplot_multiple_cols
from protflow.config import PROTFLOW_ENV


def split_pdb_numbering(pdbnum):
    resnum = ""
    chain = ""
    for char in pdbnum:
        if char.isdigit():
            resnum += char
        else:
            chain += char
    resnum = int(resnum)
    if not chain:
        chain = "A"
    return [resnum, chain]

def tip_symmetric_residues():
    symres = ["ARG", "ASP", "GLU", "LEU", "PHE", "TYR", "VAL"]
    return symres

def return_residue_rotamer_library(library_folder:str, residue_identity:str):
    '''
    finds the correct library for a given amino acid and drops not needed chi angles
    '''
    prefix = residue_identity.lower()
    rotlib = pd.read_csv(os.path.join(library_folder, f'{prefix}.bbdep.rotamers.lib'))
    if residue_identity in AAs_up_to_chi3():
        rotlib.drop(['chi4', 'chi4sig'], axis=1, inplace=True)
    elif residue_identity in AAs_up_to_chi2():
        rotlib.drop(['chi3', 'chi3sig', 'chi4', 'chi4sig'], axis=1, inplace=True)
    elif residue_identity in AAs_up_to_chi1():
        rotlib.drop(['chi2', 'chi2sig', 'chi3', 'chi3sig', 'chi4', 'chi4sig'], axis=1, inplace=True)

    return rotlib

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

def num_chis_for_residue_id(res_id):
    if res_id in AAs_up_to_chi4():
        return 4
    if res_id in AAs_up_to_chi3():
        return 3
    elif res_id in AAs_up_to_chi2():
        return 2
    elif res_id in AAs_up_to_chi1():
        return 1
    else:
        return 0


def rama_plot(df, x_col, y_col, color_col, size_col, save_path=None):
    df_list = []
    for phi_psi, df in df.groupby([x_col, y_col]):
        top = df.sort_values(color_col, ascending=False).head(1)
        df_list.append(top)
    df = pd.concat(df_list)
    df = df[df[size_col] > 0]
    df[size_col] = df[size_col] * 100
    fig, ax = plt.subplots()
    norm_color = plt.Normalize(0, df[color_col].max())
    cmap = plt.cm.Blues
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    #norm_size = plt.Normalize(0, df[size_col].max())
    ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap=cmap, s=df[size_col], norm=norm_color)
    fig.colorbar(sm, label="probability", ax=ax)
    ax.set_xlabel("phi [degrees]")
    ax.set_ylabel("psi [degrees]")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    fig.gca().set_aspect('equal', adjustable='box')
    if save_path:
        plt.savefig(save_path, dpi=300)

def identify_backbone_angles_suitable_for_rotamer(residue_identity:str, rotlib:pd.DataFrame, output_dir: str, output_prefix:str=None, limit_sec_struct:str=None, occurrence_cutoff:int=5, max_output:int=None, rotamer_diff_to_best=0.05, rotamer_chi_binsize=None, rotamer_phipsi_binsize=None, prob_cutoff=None):
    '''
    finds phi/psi angles most common for a given set of chi angles from a rotamer library
    chiX_bin multiplies the chiXsigma that is used to check if a rotamer fits to the given set of chi angles --> increasing leads to higher number of hits, decreasing leads to rotamers that more closely resemble input chis. Default=1
    if score_cutoff is provided, returns only phi/psi angles above score_cutoff
    if fraction is provided, returns only the top rotamer fraction ranked by score
    if max_output is provided, returns only max_output phi/psi combinations
    '''

    filename = os.path.join(output_dir, output_prefix + f'{residue_identity}_rama_pre_filtering')
    #utils.create_output_dir_change_filename(output_dir, output_prefix + f'{residue_identity}_rama_pre_filtering')

    rama_plot(rotlib, 'phi', 'psi', 'probability', 'phi_psi_occurrence', filename)

    if limit_sec_struct:
        rotlib = filter_rotamers_by_sec_struct(rotlib, limit_sec_struct)

    if occurrence_cutoff:
        rotlib = rotlib.loc[rotlib['phi_psi_occurrence'] > occurrence_cutoff / 100]
        if rotlib.empty:
            log_and_print(f"No rotamers passed occurrence cutoff of {occurrence_cutoff}")
            logging.error(f"No rotamers passed occurrence cutoff of {occurrence_cutoff}")

    if prob_cutoff:
        rotlib = rotlib[rotlib['probability'] >= prob_cutoff]
        if rotlib.empty:
            log_and_print(f"No rotamers passed probability cutoff of {prob_cutoff}")
            logging.error(f"No rotamers passed probability cutoff of {prob_cutoff}")

    rotlib = rotlib.sort_values('rotamer_score', ascending=False)

    if rotamer_chi_binsize and rotamer_phipsi_binsize:
        rotlib = filter_rotlib_for_rotamer_diversity(rotlib, rotamer_chi_binsize, rotamer_phipsi_binsize)

    if max_output:
        rotlib = rotlib.head(max_output)

    if rotamer_diff_to_best:
        rotlib = rotlib[rotlib['probability'] >= rotlib['probability'].max() * (1 - rotamer_diff_to_best)]

    if rotlib.empty:
        raise RuntimeError('Could not find any rotamers that fit. Try setting different filter values!')

    rotlib.reset_index(drop=True, inplace=True)

    filename = os.path.join(output_dir, output_prefix + f'{residue_identity}_rama_post_filtering')
    rama_plot(rotlib, 'phi', 'psi', 'probability', 'phi_psi_occurrence', filename)

    return rotlib

def angle_difference(angle1, angle2):

    return min([abs(angle1 - angle2), abs(angle1 - angle2 + 360), abs(angle1 - angle2 - 360)])

def filter_rotlib_for_rotamer_diversity(rotlib:pd.DataFrame, rotamer_chi_binsize:float, rotamer_phipsi_binsize:float=None):
    accepted_rotamers = []
    if not rotamer_phipsi_binsize: rotamer_phipsi_binsize = 361
    chi_columns = [column for column in rotlib.columns if column.startswith('chi') and not column.endswith('sig')]

    for index, row in rotlib.iterrows():
        accept_list = []
        if len(accepted_rotamers) == 0:
            accepted_rotamers.append(row)
            continue
        for accepted_rot in accepted_rotamers:
            column_accept_list = []
            phipsi_difference = sum([angle_difference(row['phi'], accepted_rot['phi']), angle_difference(row['psi'], accepted_rot['psi'])])
            for column in chi_columns:
                #only accept rotamers that are different from already accepted ones
                if angle_difference(row[column], accepted_rot[column]) >= rotamer_chi_binsize / 2:
                    column_accept_list.append(True)
                else:
                    # if no rotamer_phipsi_binsize was defined or phi/psi angles are similar to accepted one, kick it out
                    if not rotamer_phipsi_binsize or phipsi_difference < rotamer_phipsi_binsize:
                        column_accept_list.append(False)
                    #still accept rotamers that are similar if their backbone angles are different enough
                    else:
                        column_accept_list.append(True)
            if True in column_accept_list:
                accept_list.append(True)
            else:
                accept_list.append(False)
        if set(accept_list) == {True}:
            accepted_rotamers.append(row)
    rotlib = pd.DataFrame(accepted_rotamers)
    return rotlib


def filter_rotamers_by_sec_struct(rotlib:pd.DataFrame, secondary_structure:str):
    filtered_list = []
    sec_structs = [*secondary_structure]
    #phi and psi angle range was determined from fragment library
    if "-" in sec_structs:
        phi_range = [x for x in range(-170, -39, 10)] + [x for x in range(60, 81, 10)]
        psi_range = [x for x in range(-180, -159, 10)] + [x for x in range(-40, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "B" in sec_structs:
        phi_range = [x for x in range(-170, -49, 10)]
        psi_range = [x for x in range(-180, -169, 10)] + [x for x in range(80, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "E" in sec_structs:
        phi_range = [x for x in range(-170, -59, 10)]
        psi_range = [x for x in range(-180, -169, 10)] + [x for x in range(90, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "G" in sec_structs:
        phi_range = [x for x in range(-130, -39, 10)] + [x for x in range(50, 71, 10)]
        psi_range = [x for x in range(-50, 41, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(copy.deepcopy(filtered))
    if "H" in sec_structs:
        phi_range = [x for x in range(-100, -39, 10)]
        psi_range = [x for x in range(-60, 1, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "I" in sec_structs:
        phi_range = [x for x in range(-140, -49, 10)]
        psi_range = [x for x in range(-80, 1, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "S" in sec_structs:
        phi_range = [x for x in range(-170, -49, 10)] + [x for x in range(50, 111, 10)]
        psi_range = [x for x in range(-180, -149, 10)] + [x for x in range(-60, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "T" in sec_structs:
        phi_range = [x for x in range(-130, -40, 10)] + [x for x in range(40, 111, 10)]
        psi_range = [x for x in range(-60, 61, 10)] + [x for x in range(120, 151, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    rotlib = pd.concat(filtered_list)
    if rotlib.empty:
        log_and_print(f"No rotamers passed secondary structure filtering for secondary structure {secondary_structure}.")
        logging.error(f"No rotamers passed secondary structure filtering for secondary structure {secondary_structure}.")
    return rotlib

def import_fragment_library(library_path:str):
    '''
    reads in a fragment library
    '''
    library = pd.read_pickle(library_path)
    #library.drop(library.columns[[0]], axis=1, inplace=True)
    return library

def is_unique(df_column):
    '''
    determines if all values in column are the same. quicker than nunique according to some guy on stackoverflow
    '''
    a = df_column.to_numpy()
    return (a[0] == a).all()

def check_for_chainbreaks(df, columname, fragsize):
    '''
    returns true if dataframe column is consistently numbered
    '''
    if df[columname].diff().sum() + 1 == fragsize:
        return True
    else:
        return False

def filter_frags_df_by_secondary_structure_content(frags_df, frag_sec_struct_fraction):

    frags_df_list = []
    for frag_num, df in frags_df.groupby('frag_num', sort=False):
        for sec_struct in frag_sec_struct_fraction:
            if df['ss'].str.contains(sec_struct).sum() / len(df.index) >= frag_sec_struct_fraction[sec_struct]:
                frags_df_list.append(df)
                break
    if len(frags_df_list) > 0:
        frags_df = pd.concat(frags_df_list)
        return frags_df
    else:
        return pd.DataFrame()

def filter_frags_df_by_score(frags_df, score_cutoff, scoretype, mode):

    frags_df_list = []
    for frag_num, df in frags_df.groupby('frag_num', sort=False):
        #only accepts fragments where mean value is above threshold
        if mode == 'mean_max_cutoff':
            if df[scoretype].mean() < score_cutoff:
                frags_df_list.append(df)
        if mode == 'mean_min_cutoff':
            if df[scoretype].mean() > score_cutoff:
                frags_df_list.append(df)
        #only accepts fragments if none of the residues is above threshold
        elif mode == 'max_cutoff':
            if df[scoretype].max() < score_cutoff:
                frags_df_list.append(df)
        elif mode == 'min_cutoff':
            if df[scoretype].min() > score_cutoff:
                frags_df_list.append(df)

    if len(frags_df_list) > 0:
        frags_df = pd.concat(frags_df_list)
        return frags_df
    else:
        return pd.DataFrame()

def add_frag_to_structure(frag, structure):
    frag_num = len([model for model in structure.get_models()])
    model = Model.Model(frag_num)
    model.add(frag)
    structure.add(model)

def check_fragment(frag, frag_list, frag_df, df_list, ligand, channel, vdw_radii, rotamer_position, covalent_bond, rmsd_cutoff, backbone_ligand_clash_detection_vdw_multiplier, rotamer_ligand_clash_detection_vdw_multiplier, channel_fragment_clash_detection_vdw_multiplier, num_channel_clash, num_bb_clash, num_sc_clash, num_rmsd_fails):
    frag_df['frag_num'] = len(frag_list)
    clash_check = False
    clash_atoms = []
    if channel:
        clash_check, clashing_atoms = distance_detection(frag, channel, vdw_radii, True, False, channel_fragment_clash_detection_vdw_multiplier, rotamer_position, None)
        if clash_check == True:
            num_channel_clash += 1
    if ligand and clash_check == False:
        #check for backbone clashes
        clash_check, clashing_atoms = distance_detection(frag, ligand, vdw_radii, True, True, backbone_ligand_clash_detection_vdw_multiplier, rotamer_position, covalent_bond)
        if clash_check == True:
            num_bb_clash += 1
        if clash_check == False:
            #check for rotamer clashes
            clash_check, clashing_atoms = distance_detection(frag[rotamer_position], ligand, vdw_radii, False, True, rotamer_ligand_clash_detection_vdw_multiplier, rotamer_position, covalent_bond, True)
            if clash_check == True:
                num_sc_clash += 1
                clash_atoms.append(clashing_atoms)
    #add the first encountered fragment without rmsd checking
    if clash_check == False and len(frag_list) == 0:
        frag_list.append(frag)
        df_list.append(frag_df)
        return frag_list, df_list, num_channel_clash, num_bb_clash, num_sc_clash, num_rmsd_fails, list(set(clash_atoms))
    #calculate rmsds for all already accepted fragments
    if clash_check == False and len(frag_list) > 0:
        rmsdlist = [calculate_rmsd_bb(picked_frag, frag) for picked_frag in frag_list]
        #if the lowest rmsd compared to all other fragments is higher than the set cutoff, add it to the filtered dataframe
        if min(rmsdlist) >= rmsd_cutoff:
            frag_list.append(frag)
            df_list.append(frag_df)
        else:
            num_rmsd_fails += 1

    return frag_list, df_list, num_channel_clash, num_bb_clash, num_sc_clash, num_rmsd_fails, list(set(clash_atoms))


def distance_detection(entity1, entity2, vdw_radii:dict, bb_only:bool=True, ligand:bool=False, clash_detection_vdw_multiplier:float=1.0, resnum:int=None, covalent_bond:str=None, ignore_func_groups:bool=True):
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
        #exclude backbone atoms because they have been checked in previous step
        entity1_atoms = (atom for atom in entity1.get_atoms() if not atom.name in backbone_atoms)
        entity2_atoms = (atom for atom in entity2.get_atoms())

    for atom_combination in itertools.product(entity1_atoms, entity2_atoms):
        #skip clash detection for covalent bonds
        covalent = False
        if ignore_func_groups == True and atom_combination[0].name in atoms_of_functional_groups():
            covalent = True
        if resnum and covalent_bond:
            for cov_bond in covalent_bond.split(','):
                if atom_combination[0].get_parent().id[1] == resnum and atom_combination[0].name == cov_bond.split(':')[0] and atom_combination[1].name == cov_bond.split(':')[1]:
                    covalent = True
        if covalent == True:
            continue
        distance = atom_combination[0] - atom_combination[1]
        element1 = atom_combination[0].element
        element2 = atom_combination[1].element
        clash_detection_limit = clash_detection_vdw_multiplier * (vdw_radii[str(element1).lower()] + vdw_radii[str(element2).lower()])
        if distance < clash_detection_limit:
            return True, (atom_combination[0].name, atom_combination[1].name)
    return False, None

def atoms_of_functional_groups():
    return ["NH1", "NH2", "OD1", "OD2", "ND2", "NE", "SG", "OE1", "OE2", "NE2", "ND1", "NZ", "SD", "OG", "OG1", "NE1", "OH"]


def sort_frags_df_by_score(frags_df, backbone_score_weight, rotamer_score_weight, frag_length):

    # calculate ratio of identical backbone fragments
    frags_df['backbone_count'] = frags_df.groupby('frag_identifier')['frag_num'].transform('nunique')
    frags_df['backbone_probability'] = frags_df['backbone_count'] / frags_df['frag_num'].nunique()

    df_list = []
    frag_num = 0
    for identifier, unique_df in frags_df.groupby('frag_identifier', sort=False):
        rotamer_pos = unique_df.iloc[0, unique_df.columns.get_loc('rotamer_pos')]
        rotamer_id = unique_df.iloc[0, unique_df.columns.get_loc('rotamer_id')]
        phis = []
        psis = []
        omegas = []
        indices = [[index + pos for index in range(0, len(unique_df.index), frag_length)] for pos in range(0, frag_length)]
        for col in ['phi', 'psi', 'omega']:
            unique_df.loc[unique_df[col] <= -175, col] += 360
        for index, index_list in enumerate(indices):
            df = unique_df.iloc[index_list]                
            phis.append(df['phi'].median())
            psis.append(df['psi'].median())
            omegas.append(df['omega'].median())
        
        unique_df.iloc[0:frag_length, unique_df.columns.get_loc('AA')] = 'GLY'
        unique_df.iloc[rotamer_pos - 1, unique_df.columns.get_loc('AA')] = rotamer_id
        unique_df.iloc[0:frag_length, unique_df.columns.get_loc('phi')] = phis
        unique_df.iloc[0:frag_length, unique_df.columns.get_loc('psi')] = psis
        unique_df.iloc[0:frag_length, unique_df.columns.get_loc('omega')] = omegas
        unique_df.iloc[0:frag_length, unique_df.columns.get_loc('frag_num')] = frag_num
        
        frag_num += 1
        df_list.append(unique_df.iloc[0:frag_length])

    
    frags_df = pd.concat(df_list)
    frags_df.drop(['pdb', 'ss', 'frag_identifier'], axis=1, inplace=True)
    
    #sort frags by fragment score
    grouped = frags_df[['frag_num', 'backbone_probability', 'rotamer_score']].groupby('frag_num', sort=False).mean(numeric_only=True)
    grouped['log_backbone_probability'] = np.log(grouped['backbone_probability'])
    grouped = normalize_col(grouped, 'log_backbone_probability', True, "backbone_score")
    grouped = combine_normalized_scores(grouped, 'fragment_score', ['backbone_score', 'rotamer_score'], [backbone_score_weight, rotamer_score_weight], False, False)
    grouped = grouped[['log_backbone_probability', 'backbone_score', 'fragment_score']].sort_values('fragment_score', ascending=False)
    frags_df = grouped.merge(frags_df, left_index=True, right_on="frag_num").reset_index(drop=True)
    return frags_df

def calculate_rmsd_bb(entity1, entity2):
    '''
    calculates rmsd of 2 structures considering CA atoms. does no superposition!
    '''
    bb_atoms = ["CA"]
    entity1_atoms = [atom for atom in entity1.get_atoms() if atom.id in bb_atoms]
    entity2_atoms = [atom for atom in entity2.get_atoms() if atom.id in bb_atoms]

    rmsd = math.sqrt(sum([(atom1 - atom2) ** 2 for atom1, atom2 in zip(entity1_atoms, entity2_atoms)]) / len(entity1_atoms))

    return rmsd

def create_fragment_from_df(df:pd.DataFrame):
    '''
    creates a biopython chain from dataframe containing angles and coordinates
    '''
    chain = Chain.Chain('A')
    df.reset_index(drop=True, inplace=True)

    serial_num = 1
    for index, row in df.iterrows():
        res = Residue.Residue((' ', index + 1, ' '), row['AA'], ' ')
        for atom in ["N", "CA", "C", "O"]:
            coords = np.array([row[f'{atom}_x'], row[f'{atom}_y'], row[f'{atom}_z']])
            bfactor = 0 if math.isnan(row['probability']) else round(row['probability'] * 100, 2)
            bb_atom = Atom.Atom(name=atom, coord=coords, bfactor=bfactor, occupancy=1.0, altloc=' ', fullname=f' {atom} ', serial_number=serial_num, element=atom[0])
            serial_num += 1
            res.add(bb_atom)
        chain.add(res)

    chain.atom_to_internal_coordinates()
    for index, row in df.iterrows():
        chain[index + 1].internal_coord.set_angle('phi', row['phi'])
        chain[index + 1].internal_coord.set_angle('psi', row['psi'])
        chain[index + 1].internal_coord.set_angle('omega', row['omega'])
    chain.internal_to_atom_coordinates()



    return chain

def check_if_angle_in_bin(df, phi, psi, phi_psi_bin):

    df['phi_difference'] = df.apply(lambda row: angle_difference(row['phi'], phi), axis=1)
    df['psi_difference'] = df.apply(lambda row: angle_difference(row['psi'], psi), axis=1)

    df = df[(df['psi_difference'] < phi_psi_bin / 2) & (df['phi_difference'] < phi_psi_bin / 2)]
    df = df.drop(['phi_difference', 'psi_difference'], axis=1)
    return df


def identify_positions_for_rotamer_insertion(fraglib_path, rotlib, rot_sec_struct, phi_psi_bin, directory, script_path, prefix, chi_std_multiplier, jobstarter) -> pd.DataFrame:

    os.makedirs(directory, exist_ok=True)
    out_pkl = os.path.join(directory, f"{prefix}_rotamer_positions_collected.pkl")
    if os.path.isfile(out_pkl):
        log_and_print(f"Found existing scorefile at {out_pkl}. Skipping step.")
        return pd.read_pickle(out_pkl)

    in_filenames = []
    out_filenames = []
    for index, row in rotlib.iterrows():
        in_file = os.path.join(directory, f"{prefix}_rotamer_{index}.json")
        out_file = os.path.join(directory, f"{prefix}_rotamer_positions_{index}.pkl")
        row.to_json(in_file)
        in_filenames.append(in_file)
        out_filenames.append(out_file)

    #sbatch_options = ["-c1", f'-e {directory}/{prefix}_identify_rotamer_positions.err -o {directory}/{prefix}_identify_rotamer_positions.out']

    cmds = [f"{os.path.join(PROTFLOW_ENV, "python")} {script_path} --input_json {in_file} --fraglib {fraglib_path} --output_pickle {out_file} --phi_psi_bin {phi_psi_bin} --chi_std_multiplier {chi_std_multiplier}" for in_file, out_file in zip(in_filenames, out_filenames)]
    if rot_sec_struct:
        cmds = [cmd + f" --rot_sec_struct {rot_sec_struct}" for cmd in cmds]
    
    jobstarter.start(cmds=cmds, jobname="position_identification", output_path=directory)
    #sbatch_array_jobstarter(cmds=cmds, sbatch_options=sbatch_options, jobname="id_rot_pos", max_array_size=320, wait=True, remove_cmdfile=False, cmdfile_dir=directory)

    rotamer_positions = []
    for index, out_file in enumerate(out_filenames):
        df = pd.read_pickle(out_file)
        df['rotamer_index'] = index
        os.remove(out_file)
        log_and_print(f"Found {len(df.index)} positions for rotamer {index}")
        rotamer_positions.append(df)

    rotamer_positions = pd.concat(rotamer_positions)
    # TODO: Find out why this is necessary, there should not be any duplicates in theory
    rotamer_positions = rotamer_positions.loc[~rotamer_positions.index.duplicated(keep='first')]
    rotamer_positions.to_pickle(out_pkl)
    
    return rotamer_positions
    

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


def create_frag_identifier(df, pos):
    dihedrals = ['phi', 'psi', 'omega']
    index = df['rotamer_index'].values[0]
    # Round the values and replace -180 with 180 using vectorized operations
    for col in dihedrals:
        rounded_col = f'{col}_rounded'
        df[rounded_col] = df[col].round(-1)
        df[rounded_col] = df[rounded_col].replace(-180, 180)  # Replace -180 with 180
    
    # Create the identifier using vectorized string operations
    identifier = (df['phi_rounded'].astype(str) + '_' + 
                  df['psi_rounded'].astype(str) + '_' + 
                  df['omega_rounded'].astype(str)).str.cat(sep='_') + f"_{index}_{pos}"
    return identifier

def has_consecutive_numbers(df:pd.DataFrame, fragsize):
    if df['position'].diff().sum() + 1 == fragsize: return True
    else: return False

def trim_indices(arr, upper_limit, lower_limit, n):
    # remove n indices as long as there are indices above upper_limit or below lower limit
    while arr[-1] > upper_limit:
        arr = arr[:-n] 
    while arr[0] < lower_limit:
        arr = arr[n:]
    return arr


def extract_fragments(rotamer_positions_df: pd.DataFrame, fraglib: pd.DataFrame, frag_pos_to_replace: list, fragsize: int):
    '''
    frag_pos_to_replace: the position in the fragment the future rotamer should be inserted. central position recommended.
    residue_identity: only accept fragments with the correct residue identity at that position (recommended)
    rotamer_secondary_structure: accepts a string describing secondary structure (B: isolated beta bridge residue, E: strand, G: 3-10 helix, H: alpha helix, I: pi helix, T: turn, S: bend, -: none (not in the sense of no filter --> use None instead!)). e.g. provide EH if central atom in fragment should be a helix or strand.
    '''

    #choose fragments from fragment library that contain the positions selected above
    fragnum = 0
    frag_dict = {}
    rotamer_positions_df["temp_index_for_merge"] = rotamer_positions_df.index
    for pos in frag_pos_to_replace:
        rotamer_positions_df["temp_pos_for_merge"] = pos
        
        # define start and end index for each position
        index_starts = rotamer_positions_df.index - pos + 1
        index_ends = index_starts + fragsize

        # create range between start and end
        all_values = []
        for start, end in zip(index_starts, index_ends):
            all_values.extend(range(start, end))  # Append the range to the list
        indices = np.array(all_values)

        # check if indices are below 0 or above fraglib size
        indices = trim_indices(indices, fraglib.index.max(), 0, fragsize)

        # extract all indices
        df = fraglib.loc[indices]
        df['temp_index_for_merge'] = df.index
        df.reset_index(drop=True, inplace=True)

        # group by fragsize
        group_key = df.index // fragsize
        # check if each group has a unique pdb (to prevent taking frags with res from different pdbs) and consecutive numbers (to prevent using frags with chainbreaks)
        valid_groups = df.groupby(group_key).filter(lambda x: x['pdb'].nunique() == 1 and has_consecutive_numbers(x, fragsize))
        if valid_groups.empty:
            log_and_print(f"No fragments passed for position {pos}!")
            frag_dict[pos] = pd.DataFrame()
            continue

        valid_groups.reset_index(drop=True, inplace=True)

        # Create a new identifier column based on the valid groups
        group_key = valid_groups.index // fragsize
        valid_group_ids = valid_groups.groupby(group_key).ngroup() + 1  # Create unique group identifiers

        # Assign the identifier to the filtered DataFrame
        valid_groups['frag_num'] = valid_group_ids + fragnum

        # number residues within fragments from 1 to fragsize
        valid_groups['residue_numbers'] = valid_groups.groupby(group_key).cumcount() + 1

        # merge with rotamer_positions df on index column and correct position (to make sure not merging with rotamer at another position!)
        valid_groups_merge = valid_groups.copy()
        preserve_cols = ['temp_index_for_merge', 'temp_pos_for_merge', 'AA', 'probability', 'phi_psi_occurrence', 'rotamer_score', 'rotamer_index']
        valid_groups_merge = valid_groups_merge[['temp_index_for_merge', 'residue_numbers', 'frag_num']].merge(rotamer_positions_df[preserve_cols], left_on=['temp_index_for_merge', 'residue_numbers'], right_on=['temp_index_for_merge', 'temp_pos_for_merge'])

        # modify df for downstream processing
        valid_groups_merge.rename({'AA': 'rotamer_id'}, inplace=True, axis=1)
        valid_groups_merge.drop(['temp_index_for_merge', 'residue_numbers'], axis=1, inplace=True)
        valid_groups_merge['phi_psi_occurrence'] = valid_groups_merge['phi_psi_occurrence'] * 100

        # merge back with all residues from fragments
        frags_df = valid_groups.merge(valid_groups_merge, on="frag_num")

        # set non-rotamer positions to 0
        frags_df.loc[frags_df['residue_numbers'] != pos, 'probability'] = None
        frags_df.loc[frags_df['residue_numbers'] != pos, 'phi_psi_occurrence'] = None

        # define rotamer position
        frags_df['rotamer_pos'] = pos

        # assign identifier based on phi/psi/omega angles, rotamer id and rotamer position
        frags_l = []
        for _, df in frags_df.groupby("frag_num", sort=False):
            df["frag_identifier"] = create_frag_identifier(df,pos)
            frags_l.append(df)
        frags_df = pd.concat(frags_l)

        # add fragnum so that fragment numbering is continous for next position
        fragnum = fragnum + frags_df['frag_num'].max()

        # drop identifier column
        frags_df.drop(['temp_index_for_merge', 'temp_pos_for_merge'], axis=1, inplace=True)

        frag_dict[pos] = frags_df

    return frag_dict

def is_unique(s):
    '''
    determines if all values in column are the same. quicker than nunique according to some guy on stackoverflow
    '''
    a = s.to_numpy()
    return (a[0] == a).all()

def attach_rotamer_to_fragments(df, frag, AA_alphabet):
    rotamer_on_fragments = Structure.Structure("rotonfrags")
    rotamer = identify_rotamer_position_by_probability(df)
    columns = ['chi1', 'chi2', 'chi3', 'chi4']
    chi_angles = [None if math.isnan(rotamer[chi]) else rotamer[chi] for chi in columns]
    rot_pos = int(rotamer['rotamer_pos'])
    to_mutate = frag[rot_pos]
    resid = to_mutate.id
    backbone_angles = extract_backbone_angles(frag, rot_pos)
    backbone_bondlengths = extract_backbone_bondlengths(frag, rot_pos)
    res = generate_rotamer(AA_alphabet, rotamer['AA'], resid, backbone_angles["phi"], backbone_angles["psi"], backbone_angles["omega"], backbone_angles["carb_angle"], backbone_angles["tau"], backbone_bondlengths["N_CA"], backbone_bondlengths["CA_C"], backbone_bondlengths["C_O"], chi_angles[0], chi_angles[1], chi_angles[2], chi_angles[3], rotamer['probability'])
    delattr(res, 'internal_coord')
    rotamer_on_fragments = attach_rotamer_to_backbone(frag, to_mutate, res)

    return rotamer_on_fragments

def attach_rotamer_to_backbone(fragment, fragment_residue, rotamer):
    fragment.detach_child(fragment_residue.id)
    to_mutate_atoms = []
    res_atoms = []
    for atom in ["N", "CA", "C"]:
        to_mutate_atoms.append(fragment_residue[atom])
        res_atoms.append(rotamer[atom])
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(to_mutate_atoms, res_atoms)
    sup.rotran
    sup.apply(rotamer)

    fragment.insert(rotamer.id[1]-1, rotamer)

    return fragment


def extract_backbone_angles(chain, resnum:int):
    '''
    takes a biopython chain and extracts phi/psi/omega angles of specified residue
    '''
    #convert to internal coordinates, read phi/psi angles
    #chain = copy.deepcopy(chain)
    chain.internal_coord
    chain.atom_to_internal_coordinates()
    phi = chain[resnum].internal_coord.get_angle("phi")
    psi = chain[resnum].internal_coord.get_angle("psi")
    omega = chain[resnum].internal_coord.get_angle("omg")
    carb_angle = round(chain[resnum].internal_coord.get_angle("N:CA:C:O"), 1)
    tau = round(chain[resnum].internal_coord.get_angle("tau"), 1)
    if not phi == None:
        phi = round(phi, 1)
    if not psi == None:
        psi = round(psi, 1)
    if not omega == None:
        omega = round(omega, 1)
    return {"phi": phi, "psi": psi, "omega": omega, "carb_angle": carb_angle, "tau": tau}

def extract_backbone_bondlengths(chain, resnum:int):
    '''
    takes a biopython chain and extracts phi/psi/omega angles of specified residue
    '''
    #convert to internal coordinates, read phi/psi angles
    #chain = copy.deepcopy(chain)
    chain.internal_coord
    chain.atom_to_internal_coordinates()
    N_CA = round(chain[resnum].internal_coord.get_length("N:CA"), 3)
    CA_C = round(chain[resnum].internal_coord.get_length("CA:C"), 3)
    C_O = round(chain[resnum].internal_coord.get_length("C:O"), 3)
    return {"N_CA": N_CA, "CA_C": CA_C, "C_O": C_O}



def generate_rotamer(AAalphabet_structure, residue_identity:str, res_id, phi:float=None, psi:float=None, omega:float=None, carb_angle:float=None, tau:float=None, N_CA_length:float=None, CA_C_length:float=None, C_O_length:float=None, chi1:float=None, chi2:float=None, chi3:float=None, chi4:float=None, rot_probability:float=None):
    '''
    builds a rotamer from residue identity, phi/psi/omega/chi angles
    '''
    alphabet = copy.deepcopy(AAalphabet_structure)
    for res in alphabet[0]["A"]:
        if res.get_resname() == residue_identity:
            #set internal coordinates
            alphabet[0]["A"].atom_to_internal_coordinates()
            #change angles to specified value
            if tau:
                res.internal_coord.set_angle("tau", tau)
            if carb_angle:
                res.internal_coord.bond_set("N:CA:C:O", carb_angle)
            if phi:
                res.internal_coord.set_angle("phi", phi)
            if psi:
                res.internal_coord.set_angle("psi", psi)
            if omega:
                res.internal_coord.set_angle("omega", omega)
            if N_CA_length:
                res.internal_coord.set_length("N:CA", N_CA_length)
            if CA_C_length:
                res.internal_coord.set_length("CA:C", CA_C_length)
            if C_O_length:
                res.internal_coord.set_length("C:O", C_O_length)

            max_chis = num_chis_for_residue_id(residue_identity)

            if max_chis > 0:
                res.internal_coord.bond_set("chi1", chi1)
            if max_chis > 1:
                res.internal_coord.bond_set("chi2", chi2)
            if max_chis > 2:
                res.internal_coord.bond_set("chi3", chi3)
            if max_chis > 3:
                res.internal_coord.set_angle("chi4", chi4)
            alphabet[0]["A"].internal_to_atom_coordinates()
            #change residue number to the one that is replaced (detaching is necessary because otherwise 2 res with same resid would exist in alphabet)
            alphabet[0]["A"].detach_child(res.id)
            res.id = res_id
            if rot_probability:
                for atom in res.get_atoms():
                    atom.bfactor = rot_probability * 100

            return res

def identify_rotamer_position_by_probability(df):
    #drop all rows that do not contain the rotamer, return a series --> might cause issues downstream if more than one rotamer on a fragment
    rotamer_pos = df.dropna(subset = ['probability']).squeeze()
    return(rotamer_pos)

def align_to_sidechain(entity, entity_residue_to_align, sidechain, flip_symmetric:bool=True):
    '''
    aligns an input structure (bb_fragment_structure, resnum_to_align) to a sidechain residue (sc_structure, resnum_to_alignto)
    '''
    sc_residue_identity = sidechain.get_resname()

    #superimpose structures based on specified atoms
    bbf_atoms = atoms_for_func_group_alignment(entity_residue_to_align)
    sc_atoms = atoms_for_func_group_alignment(sidechain)
    if flip_symmetric == True and sc_residue_identity in tip_symmetric_residues():
        order = [1, 0, 2]
        sc_atoms = [sc_atoms[i] for i in order]
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(sc_atoms, bbf_atoms)
    sup.rotran
    sup.apply(entity)

    return entity

def identify_his_central_atom(histidine, ligand):
    HIS_NE2 = histidine["NE2"]
    HIS_ND1 = histidine["ND1"]
    lig_atoms =  [atom for atom in ligand.get_atoms()]
    NE2_distance = min([HIS_NE2 - atom for atom in lig_atoms])
    ND1_distance = min([HIS_ND1 - atom for atom in lig_atoms])
    if NE2_distance < ND1_distance:
        his_central_atom = "NE2"
    else:
        his_central_atom = "ND1"
    return his_central_atom

def rotate_histidine_fragment(entity, degrees, theozyme_residue, his_central_atom, ligand):
    if his_central_atom == "auto":
        if not ligand:
            raise RuntimeError(f"Ligand is required if using <flip_histidines='auto'>!")
        his_central_atom = identify_his_central_atom(theozyme_residue, ligand)
    if his_central_atom == "NE2":
        half = theozyme_residue["ND1"].coord + 0.5 * (theozyme_residue["CG"].coord - theozyme_residue["ND1"].coord)
    elif his_central_atom == "ND1":
        half = theozyme_residue["NE2"].coord + 0.5 * (theozyme_residue["CD2"].coord - theozyme_residue["NE2"].coord)
    entity = rotate_entity_around_axis(entity, theozyme_residue[his_central_atom].coord, half, degrees)
    
    return entity

def rotate_phenylalanine_fragment(entity, degrees, theozyme_residue):

    center = theozyme_residue["CZ"].coord - 0.5 * (theozyme_residue["CZ"].coord - theozyme_residue["CG"].coord)
    vector_A = theozyme_residue["CG"].coord - theozyme_residue["CZ"].coord
    vector_B = theozyme_residue["CE1"].coord - theozyme_residue["CZ"].coord
    N = np.cross(vector_A, vector_B)
    N = N / np.linalg.norm(N)
    point = center + N

    entity = rotate_entity_around_axis(entity, center, point, degrees)
    
    return entity

def rotate_entity_around_axis(entity, coords_1, coords_2, angle):
    rotation_axis = coords_1 - coords_2
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    angle_radians = np.radians(angle)
    for atom in entity.get_atoms():
        array = atom.coord - coords_1
        rotated_array = rotate_array_around_vector(array, rotation_axis, angle_radians)
        atom.coord = rotated_array + coords_1
    return entity

def rotation_matrix(V, X):
    K = np.array([[0, -V[2], V[1]],
                  [V[2], 0, -V[0]],
                  [-V[1], V[0], 0]])
    I = np.identity(3)
    R = I + np.sin(X)*K + (1-np.cos(X))*np.dot(K,K)
    return R

def rotate_array_around_vector(array, axis, angle):
    """
    Rotate a 3D NumPy array around a specified axis by a given angle.

    Parameters:
        array: A 3D NumPy array to be rotated.
        axis: A 3D NumPy array representing the rotation axis.
        angle: The angle (in radians) of rotation.

    Returns:
        The rotated 3D NumPy array.
    """
    rotation_matrix_3x3 = rotation_matrix(axis, angle)
    rotated_array = np.dot(rotation_matrix_3x3, array.T).T
    return rotated_array


def atoms_for_func_group_alignment(residue):
    '''
    return the atoms used for superposition via functional groups
    '''
    sc_residue_identity = residue.get_resname()

    if not sc_residue_identity in func_groups():
        raise RuntimeError(f'Unknown residue with name {sc_residue_identity}!')
    else:
        return [residue[atom] for atom in func_groups()[sc_residue_identity]]
    
def func_groups():
    func_groups = {
        "ALA": ["CB", "CA", "N"],
        "ARG": ["NH1", "NH2", "CZ"],
        "ASP": ["OD1", "OD2", "CG"],
        "ASN": ["OD1", "ND2", "CG"],
        "CYS": ["SG", "CB", "CA"],
        "GLU": ["OE1", "OE2", "CD"],
        "GLN": ["OE1", "NE2", "CD"],
        "GLY": ["CA", "N", "C"],
        "HIS": ["ND1", "NE2", "CG"],
        "ILE": ["CD1", "CG1", "CB"],
        "LEU": ["CD1", "CD2", "CG"],
        "LYS": ["NZ", "CE", "CD"],
        "MET": ["CE", "SD", "CG"],
        "PHE": ["CD1", "CD2", "CZ"],
        "PRO": ["CD", "CG", "CB"],
        "SER": ["OG", "CB", "CA"],
        "THR": ["OG1", "CG2", "CB"],
        "TRP": ["NE1", "CZ3", "CG"],
        "TYR": ["CE1", "CE2", "OH"],
        "VAL": ["CG1", "CG2", "CB"]
        }
    return func_groups


def clean_input_backbone(entity: Structure):
    chains = [chain for chain in entity.get_chains()]
    models = [model for model in entity.get_models()]
    if len(chains) > 1 or len(models):
        logging.error("Input backbone fragment pdb must only contain a single chain and a single model!")
    for model in models:
        model.id = 0
    for chain in chains:
        chain.id = "A"
    for index, residue in enumerate(entity.get_residues()):
        residue.id = (residue.id[0], index + 1, residue.id[2])
    for atom in entity.get_atoms():
        atom.bfactor = 0
    return entity[0]["A"]

def identify_residues_with_equivalent_func_groups(residue):
    '''
    checks if residues with same functional groups exist, returns a list of these residues
    '''
    resname = residue.get_resname()
    if resname in ['ASP', 'GLU']:
        return ['ASP', 'GLU']
    elif resname in ['ASN', 'GLN']:
        return ['ASN', 'GLN']
    elif resname in ['VAL', 'ILE']:
        return ['VAL', 'LEU']
    else:
        return [resname]
    
def rotamers_for_backbone(resnames, rotlib_path, phi, psi, rot_prob_cutoff:float=0.05, prob_diff_to_best:float=0.5, max_rotamers:int=70, max_stdev:float=2, level:int=2):
    rotlib_list = []
    for res in resnames:
        if res in ["ALA", "GLY"]:
            #TODO: assign proper scores for log prob and occurrence
            rotlib = pd.DataFrame([[res, phi, psi, float("nan"), 1, float("nan"), 0, 0]], columns=["AA", "phi", "psi", "count", "probability", "phi_psi_occurrence", "log_prob", "log_occurrence"])
            rotlib_list.append(rotlib)
        else:
            rotlib = return_residue_rotamer_library(rotlib_path, res)
            rotlib_list.append(identify_rotamers_suitable_for_backbone(phi, psi, rotlib, rot_prob_cutoff, prob_diff_to_best, max_rotamers, max_stdev, level))
    if len(rotlib_list) > 1:
        filtered_rotlib = pd.concat(rotlib_list)
        filtered_rotlib = filtered_rotlib.sort_values("probability", ascending=False)
        filtered_rotlib.reset_index(drop=True, inplace=True)
        return filtered_rotlib
    else:
        return rotlib_list[0]
    
def identify_rotamers_suitable_for_backbone(phi:float, psi:float, rotlib:pd.DataFrame, prob_cutoff:float=None, prob_diff_to_best:float=None, max_rotamers:int=None, max_stdev:float=2, level:int=3):
    '''
    identifies suitable rotamers by filtering for phi/psi angles
    if fraction is given, returns only the top rotamer fraction ranked by probability (otherwise returns all rotamers)
    if prob_cutoff is given, returns only rotamers more common than prob_cutoff
    '''
    rotlib.rename(columns={'identity': 'AA'}, inplace=True)
    #round dihedrals to the next tens place
    if not phi == None:
        phi = round(phi, -1)
    if not psi == None:
        psi = round(psi, -1)
    #extract all rows containing specified phi/psi angles from library
    if phi and psi:
        log_and_print(f"Searching for rotamers in phi/psi bin {phi}/{psi}.")
        rotlib = rotlib.loc[(rotlib['phi'] == phi) & (rotlib['psi'] == psi)].reset_index(drop=True)
    elif not phi or not psi:
        if not phi:
            rotlib = rotlib[rotlib['psi'] == psi].reset_index(drop=True)
        elif not psi:
            rotlib = rotlib[rotlib['phi'] == phi].reset_index(drop=True)
        rotlib = rotlib.loc[rotlib['phi_psi_occurrence'] >= 1]
        rotlib = rotlib.drop_duplicates(subset=['phi', 'psi'], keep='first')
        rotlib.sort_values("probability", ascending=False)
        rotlib = rotlib.head(5)
    #filter top rotamers
    rotlib = rotlib.sort_values("probability", ascending=False)
    if prob_cutoff:
        rotlib = rotlib.loc[rotlib['probability'] > prob_cutoff]
    if prob_diff_to_best:
        rotlib = rotlib[rotlib['probability'] >= rotlib['probability'].max() * (1 - prob_diff_to_best)]
    if level > 0:
        rotlib = diversify_chi_angles(rotlib, max_stdev, level)
        #filter again, since diversify_chi_angles produces rotamers with lower probability
        if prob_cutoff:
            rotlib = rotlib.loc[rotlib['probability'] > prob_cutoff]
    if prob_diff_to_best:
        rotlib = rotlib[rotlib['probability'] >= rotlib['probability'].max() * (1 - prob_diff_to_best)]
    if max_rotamers:
        rotlib = rotlib.head(max_rotamers)
    return rotlib


def diversify_chi_angles(rotlib, max_stdev:float=2, level:int=3):
    '''
    adds additional chi angles based on standard deviation.
    max_stdev: defines how far to stray from mean based on stdev. chi_new = chi_orig +- stdev * max_stdev
    level: defines how many chis should be sampled within max_stdev. if level = 1, mean, mean + stdev*max_stdev, mean - stdev*max_stdev will be returned. if level = 2, mean, mean + 1/2 stdev*max_stdev, mean + stdev*max_stdev, mean - 1/2 stdev*max_stdev, mean - stdev*max_stdev will be returned
    '''
    #check which chi angles exist in rotamer library
    columns = list(rotlib.columns)
    columns = [column for column in columns if column.startswith('chi') and not 'sig' in column]
    #generate deviation parameters
    devs = [max_stdev * i / level for i in range(-level, level +1)]
    #calculate chi angles
    for chi_angle in columns:
        new_chis_list = []
        for dev in devs:
            new_chis = alter_chi(rotlib, chi_angle, f'{chi_angle}sig', dev)
            new_chis_list.append(new_chis)
        rotlib = pd.concat(new_chis_list)
        rotlib.drop([f'{chi_angle}sig'], axis=1, inplace=True)
        rotlib[chi_angle] = round(rotlib[chi_angle], 1)
    rotlib.sort_values('probability', inplace=True, ascending=False)
    rotlib.reset_index(drop=True, inplace=True)
    return rotlib

def create_df_from_fragment(backbone, fragment_name):

    pdbnames = [fragment_name for res in backbone.get_residues()]
    resnames = [res.resname for res in backbone.get_residues()]
    pos_list = [res.id[1] for res in backbone.get_residues()]
    problist = [float("nan") for res in backbone.get_residues()]
    phi_angles = [extract_backbone_angles(backbone, res.id[1])['phi'] for res in backbone.get_residues()]
    psi_angles = [extract_backbone_angles(backbone, res.id[1])['psi'] for res in backbone.get_residues()]
    omega_angles = [extract_backbone_angles(backbone, res.id[1])['omega'] for res in backbone.get_residues()]
    CA_x_coords_list = [(round(res["CA"].get_coord()[0], 3)) for res in backbone.get_residues()]
    CA_y_coords_list = [(round(res["CA"].get_coord()[1], 3)) for res in backbone.get_residues()]
    CA_z_coords_list = [(round(res["CA"].get_coord()[2], 3)) for res in backbone.get_residues()]
    C_x_coords_list = [(round(res["C"].get_coord()[0], 3)) for res in backbone.get_residues()]
    C_y_coords_list = [(round(res["C"].get_coord()[1], 3)) for res in backbone.get_residues()]
    C_z_coords_list = [(round(res["C"].get_coord()[2], 3)) for res in backbone.get_residues()]
    N_x_coords_list = [(round(res["N"].get_coord()[0], 3)) for res in backbone.get_residues()]
    N_y_coords_list = [(round(res["N"].get_coord()[1], 3)) for res in backbone.get_residues()]
    N_z_coords_list = [(round(res["N"].get_coord()[2], 3)) for res in backbone.get_residues()]
    O_x_coords_list = [(round(res["O"].get_coord()[0], 3)) for res in backbone.get_residues()]
    O_y_coords_list = [(round(res["O"].get_coord()[1], 3)) for res in backbone.get_residues()]
    O_z_coords_list = [(round(res["O"].get_coord()[2], 3)) for res in backbone.get_residues()]

    df = pd.DataFrame(list(zip(pdbnames, resnames, pos_list, phi_angles, psi_angles, omega_angles, CA_x_coords_list, CA_y_coords_list, CA_z_coords_list, C_x_coords_list, C_y_coords_list, C_z_coords_list, N_x_coords_list, N_y_coords_list, N_z_coords_list, O_x_coords_list, O_y_coords_list, O_z_coords_list, problist)), columns=["pdb", "AA", "position", "phi", "psi", "omega", "CA_x", "CA_y", "CA_z", "C_x", "C_y", "C_z", "N_x", "N_y", "N_z", "O_x", "O_y", "O_z", "probability"])
    df[["chi1", "chi2", "chi3", "chi4"]] = float("nan")
    return df

def normal_dist_density(x):
    '''
    calculates y value for normal distribution from distance from mean TODO: check if it actually makes sense to do it this way
    '''
    y = math.e **(-(x)**2 / 2)
    return y


def alter_chi(rotlib, chi_column, chi_sig_column, dev):
    '''
    calculate deviations from input chi angle for rotamer library
    '''
    new_chis = copy.deepcopy(rotlib)
    new_chis[chi_column] = new_chis[chi_column] + new_chis[chi_sig_column] * dev
    new_chis['probability'] = new_chis['probability'] * normal_dist_density(dev)
    new_chis['log_prob'] = np.log(new_chis['probability'])
    return new_chis

def exchange_covalent(covalent_bond):
    atom = covalent_bond.split(":")[0]
    exchange_dict = {"OE1": "OD1", "OE2": "OD2", "CD1": "CG1", "CD2": "CG2", "NE2": "ND2", "OD1": "OE1", "OD2": "OE2", "CG1": "CD1", "CG2": "CD2", "ND2": "NE2"}
    try:
        atom = exchange_dict[atom]
    except:
        atom = atom
    return atom + ":" + covalent_bond.split(":")[1]

def flip_covalent(covalent_bond, residue):
    atom = covalent_bond.split(":")[0]
    exchange_dict = {
        "GLU": {"OE1": "OE2", "OE2": "OE1"},
        "ASP": {"OD1": "OD2", "OD2": "OD1"},
        "VAL": {"CD1": "CD2", "CD2": "CD1"},
        "LEU": {"CG1": "CG2", "CG2": "CG1"},
        "ARG": {"NH1": "NH2", "NH2": "NH1"}     
        }
    try:
        atom = exchange_dict[residue][atom]
    except:
        atom = atom
    return atom + ":" + covalent_bond.split(":")[1]

def log_and_print(string: str): 
    logging.info(string)
    print(string)
    return string

def combine_normalized_scores(df: pd.DataFrame, name:str, scoreterms:list, weights:list, normalize:bool=False, scale:bool=False):
    if not len(scoreterms) == len(weights):
        raise RuntimeError(f"Number of scoreterms ({len(scoreterms)}) and weights ({len(weights)}) must be equal!")
    df[name] = sum([df[col]*weight for col, weight in zip(scoreterms, weights)]) / sum(weights)
    if df[name].nunique() == 1:
        df[name] = 0
        return df
    
    if normalize == True:
        df = normalize_col(df, name, False)
        df.drop(name, axis=1, inplace=True)
        df.rename(columns={f'{name}_normalized': name}, inplace=True)
    if scale == True:
        df[name] = df[name] / df[name].max()
        df = scale_col(df, name, True)
    return df

def define_rotamer_positions(fragsize):
    rotamer_positions = int(fragsize / 2)
    if rotamer_positions * 2 == fragsize:
        rotamer_positions = [rotamer_positions, rotamer_positions + 1]
    else:
        rotamer_positions = [rotamer_positions + 1]
    return rotamer_positions


def main(args):

    start = time.time()

    working_dir = os.path.join(args.working_dir, f"{args.output_prefix}_fragments" if args.output_prefix else "fragments")
    os.makedirs(working_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join(working_dir, f"fragment_picker_{args.theozyme_resnums}.log"))
    cmd = ''
    for key, value in vars(args).items():
        cmd += f'--{key} {value} '
    cmd = f'{sys.argv[0]} {cmd}'
    logging.info(cmd)

    #import and prepare stuff
    riff_diff_dir = os.path.abspath(args.riff_diff_dir)
    database_dir = os.path.join(riff_diff_dir, "database")
    utils_dir = os.path.join(riff_diff_dir, "utils")
    rotlib_dir = os.path.join(database_dir, "bb_dep_rotlibs")

    theozyme = load_structure_from_pdbfile(args.theozyme_pdb, all_models=True)
    AA_alphabet = load_structure_from_pdbfile(os.path.join(database_dir, 'AA_alphabet.pdb'), all_models=True)
    fragment_path = os.path.join(database_dir, "backbone_frags", "7helix.pdb") if not args.fragment_pdb else args.fragment_pdb

    if args.rotamer_position == "auto":
        if args.pick_frags_from_db:
            frag_pos_to_replace = define_rotamer_positions(args.fragsize)
        else:
            backbone = load_structure_from_pdbfile(fragment_path, all_models=True)
            backbone = clean_input_backbone(backbone)
            frag_pos_to_replace = [res.id[1] for res in backbone.get_residues()]
            frag_pos_to_replace = frag_pos_to_replace[1:-1]
    elif len(args.rotamer_position) > 1:
        frag_pos_to_replace = [int(i) for i in args.rotamer_position]
    else:
        frag_pos_to_replace = [int(args.rotamer_position[0])]

    #sanity check command line input
    if args.frag_sec_struct_fraction:
        sec_structs = args.frag_sec_struct_fraction.split(',')
        sec_dict = {}
        for i in sec_structs:
            sec, frac = i.split(':')
            frac = float(frac)
            if frac > 1 or frac < 0:
                logging.error(f'Fraction for secondary structure {sec} must be a value between 0 and 1, but it is {frac}!')
                raise ValueError(f'Fraction for secondary structure {sec} must be a value between 0 and 1, but it is {frac}!')
            if (args.fragsize - frac * args.fragsize) < 1 and sec != args.rot_sec_struct and args.rot_sec_struct != None:
                logging.error(f"If limiting all fragment residues to secondary structure {sec}, it is not possible that the rotamer has secondary structure {args.rot_sec_struct}!")
                raise KeyError(f"If limiting all fragment residues to secondary structure {sec}, it is not possible that the rotamer has secondary structure {args.rot_sec_struct}!")
            elif (args.fragsize - frac * args.fragsize) < 1 and args.rot_sec_struct == None and len(sec_structs) == 1:
                log_and_print(f"Setting <rot_sec_struct> to {sec} because all residues in fragment have to have secondary structure {sec}!")
                args.rot_sec_struct = sec
            sec_dict[sec] = float(frac)
    else:
        sec_dict = None

    if args.ligands:
        ligand = Chain.Chain('Z')
        ligand_ids = args.ligands.split(',')
        for index, lig_id in enumerate(ligand_ids):
            resnum, chain = split_pdb_numbering(lig_id)
            if not chain in [chain.id for chain in theozyme.get_chains()] or not resnum in [res.id[1] for res in theozyme[0][chain].get_residues()]:
                logging.error(f'No ligand found in chain {chain} with residue number {resnum}. Please make sure the theozyme pdb is correctly formatted.')
                raise KeyError(f'No ligand found in chain {chain} with residue number {resnum}. Please make sure the theozyme pdb is correctly formatted.')
            else:
                # No idea why, but lig = theozyme[0][chain][resnum] does not work (I think because of heteroatoms)
                for res in theozyme[0][chain].get_residues():
                    if res.id[1] == resnum:
                        lig = theozyme[0][chain][res.id]
                logging.info(f"Found ligand in chain {chain} with residue number {resnum}.")
                lig.detach_parent()
                lig.id = (lig.id[0], index+1, lig.id[2])
                ligand.add(lig)

    if args.channel_chain:
        if not args.channel_chain in [chain.id for chain in theozyme.get_chains()]:
            raise RuntimeError(f'No channel placeholder found in chain {args.channel_chain}. Please make sure the theozyme pdb is correctly formatted.')
        channel = copy.deepcopy(theozyme[0][args.channel_chain])
        channel.detach_parent()
        channel.id = "Q"
        logging.info(f"Found channel placeholder in chain {args.channel_chain}.")
    else:
        channel = None
        logging.info(f"No channel placeholder chain provided. Channel placeholder will be added automatically in following steps.")

    # create output folders
    rotinfo_dir = os.path.join(working_dir, "rotamer_info")
    os.makedirs(rotinfo_dir, exist_ok=True)
    fraginfo_dir = os.path.join(working_dir, "fragment_info")
    os.makedirs(fraginfo_dir, exist_ok=True)

    for resname in args.theozyme_resnums.split(","):
        resnum, chain = split_pdb_numbering(resname)
        theozyme_residue = theozyme[0][chain][resnum]

        if args.covalent_bond:
            if not args.ligands:
                logging.warning("WARNING: Covalent bonds are only useful if ligand is present!")
            for cov_bond in args.covalent_bond.split(','):
                if not cov_bond.split(':')[0] in [atom.name for atom in theozyme_residue.get_atoms()]:
                    raise KeyError(f"Could not find atom {cov_bond.split(':')[0]} from covalent bond {cov_bond} in residue {resname}!")
                if not cov_bond.split(':')[1] in [atom.name for atom in ligand.get_atoms()]:
                    raise KeyError(f"Could not find atom {cov_bond.split(':')[1]} from covalent bond {cov_bond} in ligand {args.ligands}!")
                
        if args.add_equivalent_func_groups:
            residue_identities = identify_residues_with_equivalent_func_groups(theozyme_residue)
            logging.info(f"Added residues with equivalent functional groups: {residue_identities}")
        else:
            residue_identities = [theozyme_residue.get_resname()]
                
        if not args.pick_frags_from_db:

            #################################### BACKBONE ROTAMER FINDER ####################################
            backbone_df = create_df_from_fragment(backbone, os.path.basename(fragment_path))
            backbone_residues = [res.id[1] for res in backbone.get_residues()]
            rotlibs = []
            log_and_print(f"Identifying rotamers...")
            for pos in frag_pos_to_replace:
                if not pos in backbone_residues:
                    logging.error(f'Positions for rotamer insertion {frag_pos_to_replace} do not match up with backbone fragment {backbone_residues}')
                    raise ValueError(f'Positions for rotamer insertion {frag_pos_to_replace} do not match up with backbone fragment {backbone_residues}')
                backbone_angles = extract_backbone_angles(backbone, pos)
                log_and_print(f"Position {pos} phi/psi angles: {backbone_angles['phi']} / {backbone_angles['psi']}.")
                rotlib = rotamers_for_backbone(residue_identities, rotlib_dir, backbone_angles["phi"], backbone_angles["psi"], args.prob_cutoff, args.rotamer_diff_to_best, 100, 2, 2)
                rotlib["rotamer_position"] = pos
                log_and_print(f"Found {len(rotlib.index)} rotamers for position {pos}.")
                rotlibs.append(rotlib)
            rotlib = pd.concat(rotlibs).reset_index(drop=True)
            rotlib = normalize_col(rotlib, 'log_prob', scale=False)
            rotlib = normalize_col(rotlib, 'log_occurrence', scale=False)
            rotlib = combine_normalized_scores(rotlib, 'rotamer_score', ['log_prob_normalized', 'log_occurrence_normalized'], [args.prob_weight, args.occurrence_weight], False, True)
            rotlib = rotlib.sort_values('rotamer_score', ascending=False).reset_index(drop=True)
            #print(rotlib)

            log_and_print(f"Found {len(rotlib.index)} rotamers in total.")
            rotlibcsv = os.path.join(rotinfo_dir, f'rotamers_{resname}_combined.csv')
            log_and_print(f"Writing phi/psi combinations to {rotlibcsv}.")
            rotlib.to_csv(rotlibcsv)

            frag_dict = {}
            for pos, rotlib in rotlib.groupby('rotamer_position'):
                pos_frags = []
                for index, row in rotlib.iterrows():
                    df = copy.deepcopy(backbone_df)
                    df.loc[pos - 1, [column for column in rotlib.columns if column.startswith("chi") or column in ["probability", "phi_psi_occurrence", "AA"]]] = [row[column] for column in rotlib.columns if column.startswith("chi") or column in ["probability", "phi_psi_occurrence", "AA"]]
                    df['frag_num'] = index
                    df['rotamer_pos'] = pos
                    df['rotamer_score'] = row['rotamer_score']
                    df['fragment_score'] = df['rotamer_score']
                    df['backbone_score'] = 0
                    df['backbone_probability'] = 0
                    pos_frags.append(df)
                log_and_print(f"Created {len(pos_frags)} fragments for position {pos}.")
                frag_dict[pos] = pd.concat(pos_frags)

        else:

            #################################### FRAGMENT FINDER ####################################
            
            fraglib_path = os.path.join(database_dir, 'fraglib_noscore.pkl')
            log_and_print(f"Importing fragment library from {fraglib_path}")
            
            fraglib = import_fragment_library(fraglib_path)

            rotlibs = []

            for residue_identity in residue_identities:
                #find rotamer library for given amino acid
                log_and_print(f"Importing backbone dependent rotamer library for residue {residue_identity} from {database_dir}")
                rotlib = return_residue_rotamer_library(rotlib_dir, residue_identity)
                rotlib = normalize_col(rotlib, 'log_prob', scale=False)
                rotlib = normalize_col(rotlib, 'log_occurrence', scale=False)
                rotlib = combine_normalized_scores(rotlib, 'rotamer_score', ['log_prob_normalized', 'log_occurrence_normalized'], [args.prob_weight, args.occurrence_weight], False, True)
                log_and_print(f"Identifying most probable rotamers for residue {residue_identity}")
                rotlib = identify_backbone_angles_suitable_for_rotamer(residue_identity, rotlib, rotinfo_dir, f'{resname}_', args.rot_sec_struct, args.phipsi_occurrence_cutoff, int(args.max_phi_psis / len(residue_identities)), args.rotamer_diff_to_best, args.rotamer_chi_binsize, args.rotamer_phipsi_binsize, args.prob_cutoff)
                log_and_print(f"Found {len(rotlib.index)} phi/psi/chi combinations.")
                rotlibs.append(rotlib)

            rotlib = pd.concat(rotlibs).sort_values("rotamer_score", ascending=False).reset_index(drop=True)
            rotlib = normalize_col(rotlib, 'log_prob', scale=False)
            rotlib = normalize_col(rotlib, 'log_occurrence', scale=False)
            rotlib = combine_normalized_scores(rotlib, 'rotamer_score', ['log_prob_normalized', 'log_occurrence_normalized'], [args.prob_weight, args.occurrence_weight], False, True)
            rotlib = rotlib.sort_values('rotamer_score', ascending=False).reset_index(drop=True)
            rotlibcsv = os.path.join(rotinfo_dir, f'rotamers_{resname}_combined.csv')
            log_and_print(f"Writing phi/psi combinations to {rotlibcsv}.")
            rotlib.to_csv(rotlibcsv)

            # setting up jobstarters
            if args.jobstarter == "SbatchArray": jobstarter = SbatchArrayJobstarter(max_cores=args.cpus)
            elif args.jobstarter == "Local": jobstarter = LocalJobStarter(max_cores=args.cpus)
            else: raise KeyError("Jobstarter must be either 'SbatchArray' or 'Local'!")

            log_and_print(f"Identifying positions for rotamer insertion...")
            rotamer_positions = identify_positions_for_rotamer_insertion(fraglib_path, rotlib, args.rot_sec_struct, args.phi_psi_bin, os.path.join(working_dir, "rotamer_positions"), os.path.join(utils_dir, "identify_positions_for_rotamer_insertion.py"), resname, args.chi_std_multiplier, jobstarter=jobstarter)
            log_and_print(f"Found {len(rotamer_positions.index)} fitting positions.")
            log_and_print(f"Extracting fragments from rotamer positions...")
            frag_dict = extract_fragments(rotamer_positions, fraglib, frag_pos_to_replace, args.fragsize)
            frag_num = int(sum([len(frag_dict[pos].index) for pos in frag_dict]) / args.fragsize)
            log_and_print(f'Found {frag_num} fragments.')

            #filter fragments
            for pos in frag_dict:
                frag_nums = int(len(frag_dict[pos].index) / args.fragsize)
                log_and_print(f'Found {frag_nums} fragments for position {pos}.')
                if frag_nums == 0:
                    frag_dict.pop(pos)
                    log_and_print(f"Could not find fragments for position {pos}.")
                    continue
                if sec_dict:
                    frag_dict[pos] = filter_frags_df_by_secondary_structure_content(frag_dict[pos], sec_dict)
                    log_and_print(f"{int(len(frag_dict[pos]) / args.fragsize)} fragments passed secondary structure filtering with filter {args.frag_sec_struct_fraction} for position {pos}.")
                if frag_dict[pos].empty:
                    frag_dict.pop(pos)
                    log_and_print(f"Could not find fragments for position {pos}.")

            if len(frag_dict) == 0:
                raise RuntimeError('Could not find any fragments that fit criteria! Try adjusting filter values!')
        

            combined = pd.concat([frag_dict[pos] for pos in frag_dict])
            log_and_print(f"Averaging and sorting fragments by fragment score with weights (backbone: {args.backbone_score_weight}, rotamer: {args.rotamer_score_weight}).")
            combined = sort_frags_df_by_score(combined, args.backbone_score_weight, args.rotamer_score_weight, args.fragsize)

            for pos, df in combined.groupby('rotamer_pos', sort=True):
                frag_dict[pos] = df
                log_and_print(f"Created {int(len(frag_dict[pos].index) / args.fragsize)} unique fragments for position {pos}.")
            combined = combined.groupby('frag_num', sort=False).mean(numeric_only=True)

            # visualize information about fragments
            violinplot_multiple_cols(dataframe=combined, cols=['fragment_score', 'backbone_score', 'rotamer_score'], titles=['fragment score', 'backbone score', 'rotamer score'], y_labels=['AU', 'AU', 'AU'], dims=[(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)], out_path=os.path.join(fraginfo_dir, f"{resname}_pre_clash_filter.png"), show_fig=False)
            del combined

        #################################### CREATE FRAGS, ATTACH ROTAMERS, FILTER ####################################

        residual_to_max = 0
        fragments = Structure.Structure('fragments')
        frags_table = []
        frags_info = []
        frag_num = 0

        for pos in frag_dict:
            num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails = 0, 0, 0, 0
            log_and_print(f'Creating fragment structures, attaching rotamer, superpositioning with theozyme residue, calculating rmsd to all accepted fragments with cutoff {args.rmsd_cutoff} A for position {pos}.')
            picked_frags = []
            frag_dfs = []
            #calculate maximum number of fragments per position, add missing fragments from previous position to maximum
            max_frags = int(args.max_frags / len(frag_dict)) + residual_to_max

            sc_clashes = []
            #loop over fragment dataframe, create fragments
            for frag_index, frag_df in frag_dict[pos].groupby('frag_num', sort=False):
                if len(picked_frags) < max_frags:
                    frag = create_fragment_from_df(frag_df)
                    frag = attach_rotamer_to_fragments(frag_df, frag, AA_alphabet)
                    frag = align_to_sidechain(frag, frag[pos], theozyme_residue, False)
                    for res in frag.get_residues():
                        try:
                            delattr(res, 'internal_coord')
                        except:
                            continue
                    delattr(frag, 'internal_coord')
                    picked_frags, frag_dfs, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails, sc_clashing_atoms = check_fragment(frag, picked_frags, frag_df, frag_dfs, ligand, channel, vdw_radii(), pos, args.covalent_bond, args.rmsd_cutoff, args.backbone_ligand_clash_detection_vdw_multiplier, args.rotamer_ligand_clash_detection_vdw_multiplier, args.channel_fragment_clash_detection_vdw_multiplier, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails)
                    sc_clashes = sc_clashes + sc_clashing_atoms
                    frag_df['flipped'] = False
                    frag_df['rotated degrees'] = 0
                    #flip rotamer and fragment if theozyme residue is tip symmetric or a histidine
                    if not args.not_flip_symmetric and theozyme_residue.get_resname() in tip_symmetric_residues()  and len(picked_frags) < max_frags:
                        flipped_frag = copy.deepcopy(frag)
                        flipped_frag_df = frag_df.copy()
                        flipped_frag_df['flipped'] = True
                        flipped_frag = align_to_sidechain(flipped_frag, frag[pos], theozyme_residue, True)
                        picked_frags, frag_dfs, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails, sc_clashing_atoms = check_fragment(flipped_frag, picked_frags, flipped_frag_df, frag_dfs, ligand, channel, vdw_radii(), pos, args.covalent_bond, args.rmsd_cutoff, args.backbone_ligand_clash_detection_vdw_multiplier, args.rotamer_ligand_clash_detection_vdw_multiplier, args.channel_fragment_clash_detection_vdw_multiplier, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails)
                        sc_clashes = sc_clashes + sc_clashing_atoms
                    if args.rotate_histidine and theozyme_residue.get_resname() == "HIS":
                        for deg in range(args.rotate_histidines_deg, 360, args.rotate_histidines_deg):
                            if len(picked_frags) >= max_frags:
                                break
                            rot_frag = copy.deepcopy(frag)
                            rot_frag_df = frag_df.copy()
                            rot_frag_df['rotated degrees'] = deg
                            rot_frag = rotate_histidine_fragment(rot_frag, deg, theozyme_residue, args.his_central_atom, ligand)
                            picked_frags, frag_dfs, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails, sc_clashing_atoms = check_fragment(rot_frag, picked_frags, rot_frag_df, frag_dfs, ligand, channel, vdw_radii(), pos, args.covalent_bond, args.rmsd_cutoff, args.backbone_ligand_clash_detection_vdw_multiplier, args.rotamer_ligand_clash_detection_vdw_multiplier, args.channel_fragment_clash_detection_vdw_multiplier, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails)
                            sc_clashes = sc_clashes + sc_clashing_atoms
                    if args.rotate_phenylalanine and theozyme_residue.get_resname() == "PHE":
                        for deg in range(args.rotate_phenylalanines_deg, 360, args.rotate_phenylalanines_deg):
                            if len(picked_frags) >= max_frags:
                                break
                            rot_frag = copy.deepcopy(frag)
                            rot_frag_df = frag_df.copy()
                            rot_frag_df['rotated degrees'] = deg
                            rot_frag = rotate_phenylalanine_fragment(rot_frag, deg, theozyme_residue)
                            picked_frags, frag_dfs, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails, sc_clashing_atoms = check_fragment(rot_frag, picked_frags, rot_frag_df, frag_dfs, ligand, channel, vdw_radii(), pos, args.covalent_bond, args.rmsd_cutoff, args.backbone_ligand_clash_detection_vdw_multiplier, args.rotamer_ligand_clash_detection_vdw_multiplier, args.channel_fragment_clash_detection_vdw_multiplier, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails)
                            sc_clashes = sc_clashes + sc_clashing_atoms
                else:
                    break
            
            log_and_print(f"Discarded {num_channel_clash} fragments that show clashes between backbone and channel placeholder with VdW multiplier {args.channel_fragment_clash_detection_vdw_multiplier}")
            log_and_print(f"Discarded {num_bb_clash} fragments that show clashes between backbone and ligand with VdW multiplier {args.backbone_ligand_clash_detection_vdw_multiplier}")
            log_and_print(f"Discarded {num_sc_clash} fragments that show clashes between sidechain and ligand with VdW multiplier {args.rotamer_ligand_clash_detection_vdw_multiplier}")
            log_and_print(f"Atoms involved in sidechain-ligand clashes: {list(set(sc_clashes))}")
            log_and_print(f"Discarded {rmsd_fails} fragments that did not pass RMSD cutoff of {args.rmsd_cutoff} to all other picked fragments")


            log_and_print(f"Found {len(picked_frags)} fragments for position {pos} of a maximum of {max_frags}.")
            residual_to_max = max_frags - len(picked_frags)
            for frag, df in zip(picked_frags, frag_dfs):
                
                rot = df.iloc[pos-1].squeeze() #identify_rotamer_position_by_probability(df)
                covalent_bonds = args.covalent_bond
                if covalent_bonds and args.add_equivalent_func_groups and theozyme_residue.get_resname() != rot['AA']:
                    covalent_bonds = ",".join([exchange_covalent(covalent_bond) for covalent_bond in covalent_bonds.split(",")])
                if covalent_bonds and rot['flipped'] == True:
                    covalent_bonds = ",".join([flip_covalent(covalent_bond, rot["AA"]) for covalent_bond in covalent_bonds.split(",")])
                row = pd.Series({'model_num': frag_num, 'rotamer_pos': pos, 'AAs': df['AA'].to_list(), 'frag_length': len(df.index), 'backbone_score': df['backbone_score'].mean(), 'fragment_score': df['fragment_score'].mean(), 'rotamer_probability': rot['probability'], 'phi_psi_occurrence': rot['phi_psi_occurrence'], 'backbone_probability': df['backbone_probability'].mean(), 'covalent_bond': covalent_bonds, 'rotamer_score': df['rotamer_score'].mean()})
                model = Model.Model(frag_num)
                model.add(frag)
                if ligand:
                    model.add(ligand)
                    row['ligand_chain'] = ligand.id
                if channel:
                    model.add(channel)
                    row['channel_chain'] = channel.id
                fragments.add(model)
                df['frag_num'] = frag_num
                frags_table.append(df)
                frags_info.append(row)
                frag_num += 1
            del(picked_frags)

        log_and_print(f'Found {len(frags_info)} fragments that passed all filters.')

        #write fragment info to disk
        frags_table = pd.concat(frags_table)
        frags_table_path = os.path.join(fraginfo_dir, f'fragments_{resname}.csv')
        log_and_print(f'Writing fragment details to {frags_table_path}.')
        frags_table.to_csv(frags_table_path)

        #write multimodel fragment pdb to disk
        filename_pdb = os.path.join(working_dir, f'{resname}.pdb')
        log_and_print(f'Writing multimodel fragment pdb to {filename_pdb}.')
        save_structure_to_pdbfile(fragments, filename_pdb, multimodel=True)
        #utils.write_multimodel_structure_to_pdb(fragments, filename_pdb)

        #write output json to disk
        frags_info = pd.DataFrame(frags_info)
        frags_info['poses'] = os.path.abspath(filename_pdb)
        frags_info['poses_description'] = f'{resname}'
        filename_json = os.path.join(working_dir, f'{resname}.json')
        log_and_print(f'Writing output json to {filename_json}.')
        frags_info.to_json(filename_json)

        if args.pick_frags_from_db:
            combined = frags_table.groupby('frag_num', sort=False).mean(numeric_only=True)
            violinplot_multiple_cols(combined, cols=['fragment_score', 'backbone_score', 'rotamer_score'], titles=['fragment score', 'backbone score', 'rotamer score'], y_labels=['AU', 'AU', 'AU'], dims=[(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)], out_path=os.path.join(fraginfo_dir, f"{resname}_post_filter.png"), show_fig=False)
        log_and_print(f"Done in {round(time.time() - start, 1)} seconds!")


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # mandatory input
    argparser.add_argument("--riff_diff_dir", default=".", type=str, help="Path to the riff_diff directory. This is workaround and will hopefully be resolved later.")
    argparser.add_argument("--theozyme_pdb", type=str, required=True, help="Path to pdbfile containing theozyme, must contain all residues in chain A numbered from 1 to n, ligand must be in chain Z (if there is one).")
    argparser.add_argument("--theozyme_resnums", required=True, help="Comma-separated list of residue numbers with chain information (e.g. A25,A38,B188) in theozyme pdb to find fragments for.")
    argparser.add_argument("--working_dir", type=str, required=True, help="Output directory")
    argparser.add_argument("--output_prefix", type=str, default=None, help="Prefix for all output files")
    argparser.add_argument("--ligands", type=str, required=True, help="Comma-separated list of ligands in the format X188,Z1.")
    
    # important parameters
    argparser.add_argument("--fragment_pdb", type=str, default=None, help="Path to backbone fragment pdb. If not set, an idealized 7-residue helix fragment is used.")
    argparser.add_argument("--pick_frags_from_db", action="store_true", help="Select backbone fragments from database instead of providing a specific backbone manually. WARNING: This is much more time consuming and is currently not recommended!")
    argparser.add_argument("--channel_chain", type=str, default=None, help="If a channel placeholder from the input theozyme should be used, provide the chain name here (important for clash detection!)")
    argparser.add_argument("--rotamer_position", default="auto", nargs='+', help="Position in fragment the rotamer should be inserted, can either be int or a list containing first and last position (e.g. 2,6 if rotamer should be inserted at every position from 2 to 6). Recommended not to include N- and C-terminus! If auto, rotamer is inserted at every position when using backbone finder and in the central location when using fragment finder.")
    argparser.add_argument("--rmsd_cutoff", type=float, default=1.0, help="Set minimum RMSD of output fragments. Increase to get more diverse fragments, but high values might lead to very long runtime or few fragments!")
    argparser.add_argument("--prob_cutoff", type=float, default=0.05, help="Do not return any phi/psi combinations with chi angle probabilities below this value")
    argparser.add_argument("--add_equivalent_func_groups", action="store_true", help="use ASP/GLU, GLN/ASN and VAL/ILE interchangeably")

    # stuff you might want to adjust
    argparser.add_argument("--max_frags", type=int, default=100, help="Maximum number of frags that should be returned.")
    argparser.add_argument("--covalent_bond", type=str, default=None, help="Add covalent bond(s) between rotamer and ligand in the form 'RotAtomA:LigAtomA,RotAtomB:LigAtomB'. Atom names should follow PDB numbering schemes, e.g. 'NZ:C3' for a covalent bond between a Lysine nitrogen and the third carbon atom of the ligand.")
    argparser.add_argument("--rotamer_ligand_clash_detection_vdw_multiplier", type=float, default=0.75, help="Multiplier for VanderWaals radii for clash detection between rotamer and ligand. Clash is detected if a distance between atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")
    argparser.add_argument("--backbone_ligand_clash_detection_vdw_multiplier", type=float, default=1.0, help="Multiplier for VanderWaals radii for clash detection between fragment backbone and ligand. Clash is detected if a distance between atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")
    argparser.add_argument("--channel_fragment_clash_detection_vdw_multiplier", type=float, default=1.0, help="Multiplier for VanderWaals radii for clash detection between fragment backbone and channel placeholder. Clash is detected if a distance between atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")

    # options if running in fragment picking mode (<pick_frags_from_db> is set)
    argparser.add_argument("--fragsize", type=int, default=7, help="Size of output fragments. Only used if <pick_frags_from_db> is set.")
    argparser.add_argument("--rot_sec_struct", type=str, default=None, help="Limit fragments to secondary structure at rotamer position. Provide string of one-letter code of dssp secondary structure elements (B, E, G, H, I, T, S, -), e.g. 'HE' if rotamer should be in helices or beta strands.")
    argparser.add_argument("--frag_sec_struct_fraction", type=str, default=None, help="Limit to fragments containing at least fraction of residues with the provided secondary structure. If fragment should have at least 50 percent helical residues OR 60 percent beta-sheet, pass 'H:0.5,E:0.6'")
    argparser.add_argument("--phipsi_occurrence_cutoff", type=float, default=0.5, help="Limit how common the phi/psi combination of a certain rotamer has to be. Value is in percent")
    argparser.add_argument("--jobstarter", type=str, default="SbatchArray", help="Only relevant if <pick_frags_from_db> is set. Defines if jobs run locally or distributed on a cluster using a protflow jobstarter. Must be one of ['SbatchArray', 'Local'].")
    argparser.add_argument("--cpus", type=int, default=320, help="Only relevant if <pick_frags_from_db> is set. Defines how many cpus should be used for distributed computing.")
    argparser.add_argument("--rotamer_chi_binsize", type=float, default=None, help="Filter for diversifying found rotamers. Lower numbers mean more similar rotamers will be found. Similar rotamers will still be accepted if their backbone angles are different. Recommended value: 15")
    argparser.add_argument("--rotamer_phipsi_binsize", type=float, default=None, help="Filter for diversifying found rotamers. Lower numbers mean similar rotamers from more similar backbone angles will be accepted. Recommended value: 50")

    # stuff you probably don't want to touch
    argparser.add_argument("--phi_psi_bin", type=float, default=9.9, help="Binsize used to identify if fragment fits to phi/psi combination. Should not be above 10!")
    argparser.add_argument("--max_phi_psis", type=int, default=15, help="maximum number of phi/psi combination that should be returned. Can be increased if not enough fragments are found downstream (e.g. because secondary structure filter was used, and there are not enough phi/psi combinations in the output that fit to the specified secondary structure.")
    argparser.add_argument("--rotamer_diff_to_best", type=float, default=0.7, help="Accept rotamers that have a probability not lower than this percentage of the most probable accepted rotamer. 1 means all rotamers will be accepted.")
    argparser.add_argument("--his_central_atom", type=str, default="auto", help="Only important if rotamer is HIS and <rotate_histidine> is True, sets the name of the atom that should not be flipped. If auto, the histidine nitrogen closest to the ligand is the coordinating atom. Can be manually set to NE2 or ND1")
    argparser.add_argument("--not_flip_symmetric", action="store_true", help="Do not flip tip symmetric residues (ARG, ASP, GLU, LEU, PHE, TYR, VAL).")
    argparser.add_argument("--rotate_histidine", action="store_true", help="Rotate the orientation of histidine residues in <rotate_his_deg> steps to generate more fragment orientations")
    argparser.add_argument("--rotate_histidines_deg", type=int, default=30, help="Rotate fragments with histidines as catalytic residues around central atom around 360 degrees in <rotate_histidine_deg> steps.")
    argparser.add_argument("--rotate_phenylalanine", action="store_true", help="Rotate the orientation of phenylalanine residues in <rotate_phenylalanines_deg> steps to generate more fragment orientations")
    argparser.add_argument("--rotate_phenylalanines_deg", type=int, default=60, help="Rotate fragments with phenylalanines as catalytic residues around center in <rotate_phenylalanines_deg> steps.")
    argparser.add_argument("--prob_weight", type=float, default=2, help="Weight for rotamer probability importance when picking rotamers.")
    argparser.add_argument("--occurrence_weight", type=float, default=1, help="Weight for phi/psi-occurrence importance when picking rotamers.")
    argparser.add_argument("--backbone_score_weight", type=float, default=1, help="Weight for importance of fragment backbone score (boltzman score of number of occurrences of similar fragments in the database) when sorting fragments.")
    argparser.add_argument("--rotamer_score_weight", type=float, default=1, help="Weight for importance of rotamer score (combined score of probability and occurrence) when sorting fragments.")
    argparser.add_argument("--chi_std_multiplier", type=float, default=2, help="Multiplier for chi angle standard deviation to check if rotamer in database fits to desired rotamer.")

    args = argparser.parse_args()


    main(args)

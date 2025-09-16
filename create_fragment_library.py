import copy
import itertools
import logging
import math
import os
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TypedDict

import Bio
import Bio.PDB
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from Bio.PDB import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.internal_coords import IC_Residue
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from protflow.jobstarters import JobStarter, LocalJobStarter, SbatchArrayJobstarter
from protflow.poses import description_from_path
from protflow.residues import ResidueSelection, from_dict
from protflow.utils.biopython_tools import (
    save_structure_to_pdbfile,
)
from protflow.utils.openbabel_tools import openbabel_fileconverter
from protflow.utils.plotting import violinplot_multiple_cols
from protflow.utils.utils import vdw_radii
from sklearn.preprocessing import minmax_scale, scale

from config import Arguments, CovalentBond
from config import Residue as ResidueModel

TIP_SYMMETRIC_RESIDUES = ("ARG", "ASP", "GLU", "LEU", "PHE", "TYR", "VAL")
"""3-letter code for residues with symmetric functional groups"""

RESIDUE_ID_TO_NUM_CHI = {
    "CYS": 1,
    "SER": 1,
    "THR": 1,
    "VAL": 1,
    "ASP": 2,
    "ASN": 2,
    "HIS": 2,
    "ILE": 2,
    "LEU": 2,
    "PHE": 2,
    "PRO": 2,
    "TRP": 2,
    "TYR": 2,
    "GLN": 3,
    "GLU": 3,
    "MET": 3,
    "ARG": 4,
    "LYS": 4,
}

ATOMS_OF_FUNCTIONAL_GROUPS = (
    "NH1",
    "NH2",
    "OD1",
    "OD2",
    "ND2",
    "NE",
    "SG",
    "OE1",
    "OE2",
    "NE2",
    "ND1",
    "NZ",
    "SD",
    "OG",
    "OG1",
    "NE1",
    "OH",
)
"""PDB names of functional group atoms for each amino acid"""

FUNC_GROUPS = {
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
    "VAL": ["CG1", "CG2", "CB"],
}
"""dictionary for functional group atoms for each residue"""

EQUIVALENT_RESIDUE_GROUPS = [
    ["ASP", "GLU"],  # Acidic
    ["ASN", "GLN"],  # Amides
    ["VAL", "ILE", "LEU"],  # Aliphatic, branched-chain (BCAAs)
    # More groups can be easily added here in the future, e.g.:
    # ["PHE", "TYR", "TRP"], # Aromatic
]

RESIDUE_GROUP_MAP = {
    residue: group for group in EQUIVALENT_RESIDUE_GROUPS for residue in group
}


class CheckDict(TypedDict):
    selected_frags: list
    selected_frag_dfs: list
    sc_clashes: list
    channel_clashes: int
    bb_clashes: int
    rmsd_fails: int


def load_pdb(path_to_pdb: str | Path) -> Structure:
    """
    Load a structure from a PDB file using BioPython's PDBParser.

    This function parses a PDB file and returns a structure object. It allows
    the option to load all models from the PDB file or a specific model.

    Parameters:
        path_to_pdb (str):
            Path to the PDB file to be parsed.

    Returns:
        Bio.PDB.Structure:
            The parsed structure object from the PDB file.

    Raises:
        FileNotFoundError:
            If the specified PDB file does not exist.
        ValueError:
            If the specified model index is out of range for the PDB file.

    Example:
        >>> load_pdb("examples/inputs/5an7.pdb")
        <Structure id=5an7>
        >>> from pathlib import Path
        >>> load_pdb(Path.cwd() / "database" / "AA_alphabet.pdb")
        <Structure id=AA_alphabet>
        >>> load_pdb("not_exist.pdb")
        Traceback (most recent call last):
        FileNotFoundError: PDB file not_exist.pdb not found!
        >>> load_pdb("examples/inputs/in.json")
        Traceback (most recent call last):
        ValueError: File must be .pdb file. File: examples/inputs/in.json
    """
    # sanity
    path_to_pdb = Path(path_to_pdb)
    if not path_to_pdb.is_file():
        raise FileNotFoundError(f"PDB file {path_to_pdb} not found!")
    if path_to_pdb.suffix != ".pdb":
        raise ValueError(f"File must be .pdb file. File: {path_to_pdb}")
    # load poses
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    return pdb_parser.get_structure(path_to_pdb.stem, path_to_pdb)


def read_bbdep(bbdep_dir: Path) -> pd.DataFrame:
    dfs = []
    for lib in bbdep_dir.glob("*.bbdep.rotamers.lib"):
        df = pd.read_csv(lib).rename(columns={"identity": "AA"}).drop(columns=["count"])
        aa = df["AA"].iloc[0]
        num_chi = RESIDUE_ID_TO_NUM_CHI.get(aa, 0)
        if num_chi < 4:
            df[["chi4", "chi4sig"]] = float("nan")
        if num_chi < 3:
            df[["chi3", "chi3sig"]] = float("nan")
        if num_chi < 2:
            df[["chi2", "chi2sig"]] = float("nan")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def rama_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    size_col: str,
    save_path=None,
):
    """
    Ramachandran plot
    """
    df_list = []
    for _, df in df.groupby([x_col, y_col]):
        top = df.sort_values(color_col, ascending=False).head(1)
        df_list.append(top)
    df = pd.concat(df_list)
    df = df[df[size_col] > 0]
    df[size_col] = df[size_col] * 100
    fig, ax = plt.subplots()
    norm_color = plt.Normalize(0, df[color_col].max())
    cmap = "Blues"
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    # norm_size = plt.Normalize(0, df[size_col].max())
    ax.scatter(
        df[x_col],
        df[y_col],
        c=df[color_col],
        cmap=cmap,
        s=df[size_col],
        norm=norm_color,
    )
    fig.colorbar(sm, label="probability", ax=ax)
    ax.set_xlabel("phi [degrees]")
    ax.set_ylabel("psi [degrees]")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    fig.gca().set_aspect("equal", adjustable="box")
    if save_path:
        plt.savefig(save_path, dpi=300)


def identify_backbone_angles_suitable_for_rotamer(
    residue_identity: str,
    rotlib: pd.DataFrame,
    output_dir: Path,
    output_prefix: str,
    limit_sec_struct: str | None = None,
    occurrence_cutoff: int = 5,
    max_output: int | None = None,
    rotamer_diff_to_best=0.05,
    rotamer_chi_binsize=None,
    rotamer_phipsi_binsize=None,
    prob_cutoff=None,
):
    """
    finds phi/psi angles most common for a given set of chi angles from a rotamer library
    chiX_bin multiplies the chiXsigma that is used to check if a rotamer fits to the given set of chi angles --> increasing leads to higher number of hits, decreasing leads to rotamers that more closely resemble input chis. Default=1
    if score_cutoff is provided, returns only phi/psi angles above score_cutoff
    if fraction is provided, returns only the top rotamer fraction ranked by score
    if max_output is provided, returns only max_output phi/psi combinations
    """

    filename = output_dir / (output_prefix + f"{residue_identity}_rama_pre_filtering")

    rama_plot(rotlib, "phi", "psi", "probability", "phi_psi_occurrence", filename)

    if limit_sec_struct:
        rotlib = filter_rotamers_by_sec_struct(rotlib, limit_sec_struct)

    if occurrence_cutoff:
        rotlib = rotlib.loc[rotlib["phi_psi_occurrence"] > occurrence_cutoff / 100]
        if rotlib.empty:
            logging.error(
                f"No rotamers passed occurrence cutoff of {occurrence_cutoff}"
            )

    if prob_cutoff:
        rotlib = rotlib[rotlib["probability"] >= prob_cutoff]
        if rotlib.empty:
            logging.error(f"No rotamers passed probability cutoff of {prob_cutoff}")

    rotlib = rotlib.sort_values("rotamer_score", ascending=False)

    rotlib = filter_rotlib_for_rotamer_diversity(
        rotlib, rotamer_chi_binsize, rotamer_phipsi_binsize
    )

    if rotamer_diff_to_best:
        rotlib = rotlib[
            rotlib["probability"]
            >= rotlib["probability"].max() * (1 - rotamer_diff_to_best)
        ]

    rotlib = rotlib.sort_values("rotamer_score", ascending=False)

    if max_output:
        rotlib = rotlib.head(max_output)

    if rotlib.empty:
        raise RuntimeError(
            "Could not find any rotamers that fit. Try setting different filter values!"
        )

    rotlib.reset_index(drop=True, inplace=True)

    filename = output_dir / (output_prefix + f"{residue_identity}_rama_post_filtering")
    rama_plot(rotlib, "phi", "psi", "probability", "phi_psi_occurrence", filename)

    return rotlib


def angle_difference(angle1, angle2):
    """
    calculate difference between angles (considers negative values and +360)
    """
    return min(
        [abs(angle1 - angle2), abs(angle1 - angle2 + 360), abs(angle1 - angle2 - 360)]
    )


def filter_rotlib_for_rotamer_diversity(
    rotlib: pd.DataFrame,
    rotamer_chi_binsize: float | None = None,
    rotamer_phipsi_binsize: float | None = None,
):
    """
    filters rotamer library for more diversity in rotamers
    """
    accepted_rotamers: list[pd.Series] = []
    chi_columns = [
        column
        for column in rotlib.columns
        if column.startswith("chi") and not column.endswith("sig")
    ]

    for _, row in rotlib.iterrows():
        rotamer_accept_list = []
        if len(accepted_rotamers) == 0:
            accepted_rotamers.append(row)
            continue
        for accepted_rot in accepted_rotamers:
            angle_accept_list = []
            if rotamer_phipsi_binsize:  # check for backbone angle difference
                phi_difference = angle_difference(row["phi"], accepted_rot["phi"])
                psi_difference = angle_difference(row["psi"], accepted_rot["psi"])
                if sum([phi_difference, psi_difference]) >= rotamer_phipsi_binsize:
                    angle_accept_list.append(True)
                else:
                    angle_accept_list.append(False)
            if rotamer_chi_binsize:
                for column in chi_columns:
                    # only accept rotamers that are different from already accepted ones
                    if (
                        angle_difference(row[column], accepted_rot[column])
                        >= rotamer_chi_binsize
                    ):
                        angle_accept_list.append(True)
                    else:
                        # if chi angles are similar to accepted one, set False
                        angle_accept_list.append(False)
            if (
                not rotamer_chi_binsize and not rotamer_phipsi_binsize
            ):  # set true if no filter was set
                angle_accept_list.append(True)
            if True in angle_accept_list:
                rotamer_accept_list.append(
                    True
                )  # if any angle was different enough, accept it
            else:
                rotamer_accept_list.append(
                    False
                )  # if no angle was different enough, discard it
        if set(rotamer_accept_list) == {
            True
        }:  # check if difference to all accepted rotamers was ok
            accepted_rotamers.append(row)
    rotlib = pd.DataFrame(accepted_rotamers)
    return rotlib


def filter_rotamers_by_sec_struct(rotlib: pd.DataFrame, secondary_structure: str):
    """
    filters rotamers according to secondary structure (via phi/psi angles that are typical for a given secondary structure)
    """
    filtered_list = []
    sec_structs = [*secondary_structure]
    # phi and psi angle range was determined from fragment library
    if "-" in sec_structs:
        phi_range = list(range(-170, -39, 10)) + list(range(60, 81, 10))
        psi_range = list(range(-180, -159, 10)) + list(range(-40, 181, 10))
        filtered = rotlib.loc[
            (rotlib["phi"].isin(phi_range)) & (rotlib["psi"].isin(psi_range))
        ]
        filtered_list.append(filtered)
    if "B" in sec_structs:
        phi_range = list(range(-170, -49, 10))
        psi_range = list(range(-180, -169, 10)) + list(range(80, 181, 10))
        filtered = rotlib.loc[
            (rotlib["phi"].isin(phi_range)) & (rotlib["psi"].isin(psi_range))
        ]
        filtered_list.append(filtered)
    if "E" in sec_structs:
        phi_range = list(range(-170, -59, 10))
        psi_range = list(range(-180, -169, 10)) + list(range(90, 181, 10))
        filtered = rotlib.loc[
            (rotlib["phi"].isin(phi_range)) & (rotlib["psi"].isin(psi_range))
        ]
        filtered_list.append(filtered)
    if "G" in sec_structs:
        phi_range = list(range(-130, -39, 10)) + list(range(50, 71, 10))
        psi_range = list(range(-50, 41, 10))
        filtered = rotlib.loc[
            (rotlib["phi"].isin(phi_range)) & (rotlib["psi"].isin(psi_range))
        ]
        filtered_list.append(copy.deepcopy(filtered))
    if "H" in sec_structs:
        phi_range = list(range(-100, -39, 10))
        psi_range = list(range(-60, 1, 10))
        filtered = rotlib.loc[
            (rotlib["phi"].isin(phi_range)) & (rotlib["psi"].isin(psi_range))
        ]
        filtered_list.append(filtered)
    if "I" in sec_structs:
        phi_range = list(range(-140, -49, 10))
        psi_range = list(range(-80, 1, 10))
        filtered = rotlib.loc[
            (rotlib["phi"].isin(phi_range)) & (rotlib["psi"].isin(psi_range))
        ]
        filtered_list.append(filtered)
    if "S" in sec_structs:
        phi_range = list(range(-170, -49, 10)) + list(range(50, 111, 10))
        psi_range = list(range(-180, -149, 10)) + list(range(-60, 181, 10))
        filtered = rotlib.loc[
            (rotlib["phi"].isin(phi_range)) & (rotlib["psi"].isin(psi_range))
        ]
        filtered_list.append(filtered)
    if "T" in sec_structs:
        phi_range = list(range(-130, -40, 10)) + list(range(40, 111, 10))
        psi_range = list(range(-60, 61, 10)) + list(range(120, 151, 10))
        filtered = rotlib.loc[
            (rotlib["phi"].isin(phi_range)) & (rotlib["psi"].isin(psi_range))
        ]
        filtered_list.append(filtered)
    rotlib = pd.concat(filtered_list)
    if rotlib.empty:
        logging.error(
            f"No rotamers passed secondary structure filtering for secondary structure {secondary_structure}."
        )
    return rotlib


def filter_frags_df_by_secondary_structure_content(frags_df, frag_sec_struct_fraction):
    """
    filter fragment dataframe to specified fraction of secondary structure content
    """
    frags_df_list = []
    for _, df in frags_df.groupby("frag_num", sort=False):
        for sec_struct, fraction in frag_sec_struct_fraction.items():
            if (
                df["ss"].str.contains(sec_struct, regex=False).sum() / len(df.index)
                >= fraction
            ):
                frags_df_list.append(df)
                break
    if len(frags_df_list) > 0:
        frags_df = pd.concat(frags_df_list)
        return frags_df
    return pd.DataFrame()


def check_fragment(
    frag,
    check_dict,
    frag_df,
    ligand,
    channel,
    rotamer_position,
    rmsd_cutoff,
    bb_lig_clash_vdw_multiplier,
    rot_lig_clash_vdw_multiplier,
    channel_frag_clash_vdw_multiplier,
):
    """
    checks fragment for clashes and RMSD to all other accepted fragments
    """
    frag_bb_atoms = [
        atom for atom in frag.get_atoms() if atom.id in ["N", "CA", "C", "O"]
    ]
    ligand_atoms = [atom for atom in ligand.get_atoms() if not atom.element == "H"]
    # select all sidechain heavy atoms, but ignore functional groups
    frag_sc_atoms = [
        atom
        for atom in frag[rotamer_position].get_atoms()
        if not atom.element == "H"
        and atom.id not in ATOMS_OF_FUNCTIONAL_GROUPS
        and atom.id not in ["N", "CA", "C", "O"]
    ]

    # check for channel clashes
    if channel:
        channel_bb_atoms = [
            atom for atom in channel.get_atoms() if atom.id in ["N", "CA", "C", "O"]
        ]
        channel_clashing_atoms = clash_detection(
            frag_bb_atoms, channel_bb_atoms, channel_frag_clash_vdw_multiplier
        )
        if channel_clashing_atoms:
            check_dict["channel_clashes"] += 1
            return check_dict

    # check for fragment-bb ligand clashes
    frag_ligand_clashes = clash_detection(
        frag_bb_atoms, ligand_atoms, bb_lig_clash_vdw_multiplier
    )
    if frag_ligand_clashes:
        check_dict["bb_clashes"] += 1
        return check_dict

    # check for rotamer ligand clashes
    sc_ligand_clashes = clash_detection(
        frag_sc_atoms, ligand_atoms, rot_lig_clash_vdw_multiplier
    )
    if sc_ligand_clashes:
        clash_ids = [
            f"{clash[0].get_parent().get_resname()} {clash[0].id} - {clash[1].get_parent().get_resname()} {clash[1].id}"
            for clash in sc_ligand_clashes
        ]
        check_dict["sc_clashes"] = check_dict["sc_clashes"] + clash_ids
        return check_dict

    # check rmsd to all other fragments
    if len(check_dict["selected_frags"]) == 0:
        check_dict["selected_frags"].append(frag)
        check_dict["selected_frag_dfs"].append(frag_df)
    else:
        rmsdlist = [
            calculate_rmsd_bb(picked_frag, frag)
            for picked_frag in check_dict["selected_frags"]
        ]
        if min(rmsdlist) >= rmsd_cutoff:
            check_dict["selected_frags"].append(frag)
            check_dict["selected_frag_dfs"].append(frag_df)
        else:
            check_dict["rmsd_fails"] += 1

    return check_dict


def clash_detection(entity1, entity2, vdw_multiplier):
    """
    checks for clashes by comparing VanderWaals radii. If clashes with ligand should be detected, set ligand to true. Ligand chain must be added as second entity.
    bb_only: only detect backbone clashes between to proteins or a protein and a ligand.
    clash_detection_vdw_multiplier: multiply Van der Waals radii with this value to set clash detection limits higher/lower
    database: path to database directory
    """

    entity1_coords = np.array([atom.get_coord() for atom in entity1])
    entity2_coords = np.array([atom.get_coord() for atom in entity2])

    entity1_vdw = np.array([vdw_radii()[atom.element.lower()] for atom in entity1])
    entity2_vdw = np.array([vdw_radii()[atom.element.lower()] for atom in entity2])

    if np.any(np.isnan(entity1_vdw)) or np.any(np.isnan(entity2_vdw)):
        raise RuntimeError(
            "Could not find Van der Waals radii for all elements in ligand. Check protflow.utils.vdw_radii and add it, if applicable!"
        )

    # Compute pairwise distances using broadcasting
    dgram = np.linalg.norm(
        entity1_coords[:, np.newaxis] - entity2_coords[np.newaxis, :], axis=-1
    )

    # calculate distance cutoff for each atom pair, considering VdW radii
    distance_cutoff = entity1_vdw[:, np.newaxis] + entity2_vdw[np.newaxis, :]

    # multiply distance cutoffs with set parameter
    distance_cutoff = distance_cutoff * vdw_multiplier

    # compare distances to distance_cutoff
    check = dgram - distance_cutoff

    # Find the indices where the distances are smaller than the cutoff (clashing)
    clashing_pairs = np.argwhere(check < 0)

    # check if any clashes have occurred
    if len(clashing_pairs) >= 1:
        clashing_atoms = [[entity1[i], entity2[j]] for i, j in clashing_pairs]
        return clashing_atoms
    return None


def sort_frags_df_by_score(
    frags_df, backbone_score_weight, rotamer_score_weight, frag_length
):
    """
    creates a single fragment out of fragments with same identifier, sorts fragments according to a composite score (comprised of backbone and rotamer score)
    """
    # calculate number of fragments
    total_fragments = frags_df["frag_num"].nunique()

    # correct angles for better median calculations
    for col in ["phi", "psi", "omega"]:
        frags_df.loc[frags_df[col] <= -175, col] += 360

    cols = frags_df.columns
    AA_i = cols.get_loc("AA")
    phi_i = cols.get_loc("phi")
    psi_i = cols.get_loc("psi")
    omega_i = cols.get_loc("omega")
    frag_i = cols.get_loc("frag_num")
    rotpos_i = cols.get_loc("rotamer_pos")
    rotid_i = cols.get_loc("rotamer_id")

    df_list = []
    frag_num = 0
    # assume frag_num is defined before the loop
    for _, unique_df in frags_df.groupby("frag_identifier", sort=False):
        # read rotamer info
        rotamer_pos = int(unique_df.iat[0, rotpos_i])
        rotamer_id = unique_df.iat[0, rotid_i]

        # backbone count (fast path if guaranteed layout: len(unique_df)//frag_length)
        backbone_count = len(unique_df) // frag_length

        # compute medians for each residue position
        # (your existing lists phis/psis/omegas work; vectorized shown here)
        n = len(unique_df)
        bc = n // frag_length

        phi_v = unique_df.iloc[:, phi_i].to_numpy()
        psi_v = unique_df.iloc[:, psi_i].to_numpy()
        omega_v = unique_df.iloc[:, omega_i].to_numpy()

        phis = np.nanmedian(phi_v.reshape(frag_length, bc), axis=1)
        psis = np.nanmedian(psi_v.reshape(frag_length, bc), axis=1)
        omegas = np.nanmedian(omega_v.reshape(frag_length, bc), axis=1)

        head = slice(0, frag_length)

        # set AA for the first fragment
        unique_df.iloc[head, AA_i] = "GLY"
        unique_df.iat[rotamer_pos - 1, AA_i] = rotamer_id  # still OK (scalar)

        # write medians only into the first fragment rows
        unique_df.iloc[head, [phi_i, psi_i, omega_i]] = np.column_stack(
            (phis, psis, omegas)
        )

        # and only tag those rows
        unique_df.iloc[head, frag_i] = frag_num
        unique_df.loc[unique_df.index[head], "backbone_count"] = backbone_count

        df_list.append(unique_df.iloc[head])
        frag_num += 1

    frags_df = pd.concat(df_list, ignore_index=True)

    frags_df.drop(["pdb", "ss", "frag_identifier"], axis=1, inplace=True)
    frags_df["backbone_probability"] = frags_df["backbone_count"] / total_fragments

    # sort frags by fragment score
    grouped = (
        frags_df[["frag_num", "backbone_probability", "rotamer_score"]]
        .groupby("frag_num", sort=False)
        .mean(numeric_only=True)
    )
    grouped["log_backbone_probability"] = np.log(grouped["backbone_probability"])
    grouped["backbone_score"] = minmax_scale(scale(grouped["log_backbone_probability"]))
    grouped["fragment_score"] = np.average(
        grouped[["backbone_score", "rotamer_score"]],
        weights=[backbone_score_weight, rotamer_score_weight],
    )
    grouped = grouped[
        ["log_backbone_probability", "backbone_score", "fragment_score"]
    ].sort_values("fragment_score", ascending=False)
    frags_df = grouped.merge(
        frags_df, left_index=True, right_on="frag_num"
    ).reset_index(drop=True)
    return frags_df


def calculate_rmsd_bb(entity1, entity2):
    """
    calculates rmsd of 2 structures considering CA atoms. does no superposition!
    """
    bb_atoms = ["CA"]
    entity1_atoms = [atom for atom in entity1.get_atoms() if atom.id in bb_atoms]
    entity2_atoms = [atom for atom in entity2.get_atoms() if atom.id in bb_atoms]

    rmsd = math.sqrt(
        sum(
            [(atom1 - atom2) ** 2 for atom1, atom2 in zip(entity1_atoms, entity2_atoms)]
        )
        / len(entity1_atoms)
    )

    return rmsd


def create_fragment_from_df(df: pd.DataFrame, rotamer_position, aa_alphabet: Chain):
    """
    creates a biopython chain from dataframe containing angles and coordinates
    """
    chain = Chain("A")
    df.reset_index(drop=True, inplace=True)

    serial_num = 1
    for index, row in df.iterrows():
        res = Residue((" ", index + 1, " "), row["AA"], " ")
        for atom in ["N", "CA", "C", "O"]:
            coords = np.array([row[f"{atom}_x"], row[f"{atom}_y"], row[f"{atom}_z"]])
            bfactor = (
                0
                if math.isnan(row["probability"])
                else round(row["probability"] * 100, 2)
            )
            bb_atom = Atom.Atom(
                name=atom,
                coord=coords,
                bfactor=bfactor,
                occupancy=1.0,
                altloc=" ",
                fullname=f" {atom} ",
                serial_number=serial_num,
                element=atom[0],
            )
            serial_num += 1
            res.add(bb_atom)
        if index + 1 == rotamer_position:
            chi_angles = [
                None if math.isnan(row[chi]) else row[chi]
                for chi in ["chi1", "chi2", "chi3", "chi4"]
            ]
            rotamer_id = row["AA"]
            prob = row["probability"]
        chain.add(res)

    chain.atom_to_internal_coordinates()

    for index, row in df.iterrows():
        chain[index + 1].internal_coord.set_angle("phi", row["phi"])
        chain[index + 1].internal_coord.set_angle("psi", row["psi"])
        chain[index + 1].internal_coord.set_angle("omega", row["omega"])
    chain.internal_to_atom_coordinates()

    fragment = attach_rotamer_to_fragments(
        chain, rotamer_position, rotamer_id, chi_angles, prob, aa_alphabet
    )

    for res in chain:
        if hasattr(res, "internal_coord"):
            delattr(res, "internal_coord")

    delattr(fragment, "internal_coord")

    return fragment


def attach_rotamer_to_fragments(
    frag: Chain, rot_pos, rotamer_id, chi_angles, probability, aa_alphabet: Chain
):
    """
    generates a rotamer based on angles and attaches it to a backbone fragment
    """
    to_mutate = frag[rot_pos]
    backbone_angles = extract_backbone_angles(frag, rot_pos)
    backbone_bondlengths = extract_backbone_bondlengths(frag, rot_pos)
    res = generate_rotamer(
        aa_alphabet,
        rotamer_id,
        to_mutate.id,
        backbone_angles["phi"],
        backbone_angles["psi"],
        backbone_angles["omega"],
        backbone_angles["carb_angle"],
        backbone_angles["tau"],
        backbone_bondlengths["N_CA"],
        backbone_bondlengths["CA_C"],
        backbone_bondlengths["C_O"],
        chi_angles[0],
        chi_angles[1],
        chi_angles[2],
        chi_angles[3],
        probability,
    )
    delattr(res, "internal_coord")
    rotamer_on_fragments = attach_rotamer_to_backbone(frag, to_mutate, res)

    return rotamer_on_fragments


def attach_rotamer_to_backbone(fragment, fragment_residue, rotamer: Residue):
    """
    attaches a rotamer to a backbone fragment using Biopython
    """
    fragment.detach_child(fragment_residue.id)
    to_mutate_atoms = []
    res_atoms = []
    for atom in ["N", "CA", "C"]:
        to_mutate_atoms.append(fragment_residue[atom])
        res_atoms.append(rotamer[atom])
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(to_mutate_atoms, res_atoms)
    sup.apply(rotamer)
    rotamer.detach_child("O")
    rotamer.add(fragment_residue["O"])
    fragment.insert(rotamer.id[1] - 1, rotamer)

    return fragment


def identify_positions_for_rotamer_insertion(
    fraglib_path,
    rotlib,
    rot_sec_struct,
    phi_psi_bin,
    directory,
    script_path,
    prefix,
    chi_std_multiplier,
    jobstarter,
) -> pd.DataFrame:
    """
    identifies positions in the library that fit to the selected rotamers
    """
    os.makedirs(directory, exist_ok=True)
    out_pkl = os.path.join(directory, f"{prefix}_rotamer_positions_collected.pkl")
    if os.path.isfile(out_pkl):
        logging.info(f"Found existing scorefile at {out_pkl}. Skipping step.")
        return pd.read_pickle(out_pkl)

    in_filenames = []
    out_filenames = []
    for index, row in rotlib.iterrows():
        in_file = os.path.join(directory, f"{prefix}_rotamer_{index}.json")
        out_file = os.path.join(directory, f"{prefix}_rotamer_positions_{index}.pkl")
        row.to_json(in_file)
        in_filenames.append(in_file)
        out_filenames.append(out_file)

    cmds = [
        f"{sys.executable} {script_path} --input_json {in_file} --fraglib {fraglib_path} --output_pickle {out_file} --phi_psi_bin {phi_psi_bin} --chi_std_multiplier {chi_std_multiplier}"
        for in_file, out_file in zip(in_filenames, out_filenames)
    ]
    if rot_sec_struct:
        cmds = [cmd + f" --rot_sec_struct {rot_sec_struct}" for cmd in cmds]

    jobstarter.start(
        cmds=cmds, jobname="position_identification", output_path=directory
    )

    rotamer_positions = []
    for index, out_file in enumerate(out_filenames):
        df = pd.read_pickle(out_file)
        df["rotamer_index"] = index
        os.remove(out_file)
        logging.info(f"Found {len(df.index)} positions for rotamer {index}")
        rotamer_positions.append(df)

    rotamer_positions = pd.concat(rotamer_positions)
    # TODO: Find out why this is necessary, there should not be any duplicates in theory
    rotamer_positions = rotamer_positions.loc[
        ~rotamer_positions.index.duplicated(keep="first")
    ]
    rotamer_positions.to_pickle(out_pkl)

    return rotamer_positions


def scale_col(df: pd.DataFrame, col: str, inplace=False) -> pd.DataFrame:
    """
    scale a dataframe column to values between 0 and 1
    """
    factor = df[col].max() - df[col].min()
    df[f"{col}_scaled"] = df[col] / factor
    df[f"{col}_scaled"] = df[f"{col}_scaled"] + (1 - df[f"{col}_scaled"].max())
    if inplace:
        df[col] = df[f"{col}_scaled"]
        df.drop(f"{col}_scaled", axis=1, inplace=True)
    return df


def extract_fragments(
    rotamer_positions_df: pd.DataFrame,
    fraglib_path: str,
    frag_pos_to_replace: list,
    fragsize: int,
    working_dir: str,
    script_path: str,
    jobstarter,
):
    """
    frag_pos_to_replace: the position in the fragment the future rotamer should be inserted. central position recommended.
    residue_identity: only accept fragments with the correct residue identity at that position (recommended)
    rotamer_secondary_structure: accepts a string describing secondary structure (B: isolated beta bridge residue, E: strand, G: 3-10 helix, H: alpha helix, I: pi helix, T: turn, S: bend, -: none (not in the sense of no filter --> use None instead!)). e.g. provide EH if central atom in fragment should be a helix or strand.
    """

    def write_rotamer_extraction_cmd(
        script_path,
        rotamer_positions_path,
        fraglib_path,
        rotamer_positions,
        fragsize,
        out_path,
    ):
        cmd = f"{sys.executable} {script_path} --rotpos_path {rotamer_positions_path} --fraglib_path {fraglib_path} --rotamer_positions {','.join([str(pos) for pos in rotamer_positions])} --fragsize {fragsize} --outfile {out_path}"
        return cmd

    # choose fragments from fragment library that contain the positions selected above
    rotamer_positions_df["temp_index_for_merge"] = rotamer_positions_df.index

    os.makedirs(working_dir, exist_ok=True)

    cmds = []
    out_paths = []
    # split rotamer positions df into several parts, create fragments for each part
    for i, split_df in enumerate(
        np.array_split(
            rotamer_positions_df,
            min(jobstarter.max_cores, len(rotamer_positions_df.index)),
        )
    ):
        split_df.to_pickle(
            positions_path := os.path.join(working_dir, f"rotamer_positions_{i}.pkl")
        )
        out_paths.append(out_path := os.path.join(working_dir, f"fragments_{i}.pkl"))
        cmds.append(
            write_rotamer_extraction_cmd(
                script_path,
                positions_path,
                fraglib_path,
                frag_pos_to_replace,
                fragsize,
                out_path,
            )
        )

    jobstarter.start(
        cmds=cmds, jobname="fragment_extraction", wait=True, output_path=working_dir
    )  # distribute fragment extraction to cluster

    # combine all fragments
    num_frags = 0
    frags_dfs = []
    for out_path in out_paths:
        # read in results
        frags_df = pd.read_pickle(out_path)
        max_frags = frags_df["frag_num"].max()
        frags_df["frag_num"] = (
            frags_df["frag_num"] + num_frags
        )  # update frag_num to have continuous fragnums over all positions
        num_frags = num_frags + max_frags + 1
        frags_dfs.append(frags_df)

    frags_dfs = pd.concat(frags_dfs)

    return frags_dfs


def extract_backbone_angles(chain, resnum: int):
    """
    takes a biopython chain and extracts phi/psi/omega angles of specified residue
    """
    # convert to internal coordinates, read phi/psi angles
    # chain = copy.deepcopy(chain)
    # chain.internal_coord
    chain.atom_to_internal_coordinates()
    phi = chain[resnum].internal_coord.get_angle("phi")
    psi = chain[resnum].internal_coord.get_angle("psi")
    omega = chain[resnum].internal_coord.get_angle("omg")
    carb_angle = round(chain[resnum].internal_coord.get_angle("N:CA:C:O"), 1)
    tau = round(chain[resnum].internal_coord.get_angle("tau"), 1)
    if phi is not None:
        phi = round(phi, 1)
    if psi is not None:
        psi = round(psi, 1)
    if omega is not None:
        omega = round(omega, 1)
    return {
        "phi": phi,
        "psi": psi,
        "omega": omega,
        "carb_angle": carb_angle,
        "tau": tau,
    }


def extract_backbone_bondlengths(chain: Chain, resnum: int):
    """
    takes a biopython chain and extracts phi/psi/omega angles of specified residue
    """
    # convert to internal coordinates, read phi/psi angles
    chain.atom_to_internal_coordinates()
    n_ca = round(chain[resnum].internal_coord.get_length("N:CA"), 3)
    ca_c = round(chain[resnum].internal_coord.get_length("CA:C"), 3)
    c_o = round(chain[resnum].internal_coord.get_length("C:O"), 3)
    return {"N_CA": n_ca, "CA_C": ca_c, "C_O": c_o}


def generate_rotamer(
    aa_alphabet: Chain,
    residue_identity: str,
    res_id,
    phi: float | None = None,
    psi: float | None = None,
    omega: float | None = None,
    carb_angle: float | None = None,
    tau: float | None = None,
    N_CA_length: float | None = None,
    CA_C_length: float | None = None,
    C_O_length: float | None = None,
    chi1: float | None = None,
    chi2: float | None = None,
    chi3: float | None = None,
    chi4: float | None = None,
    rot_probability: float | None = None,
):
    """
    builds a rotamer from residue identity, phi/psi/omega/chi angles
    """
    for res in aa_alphabet:
        if res.resname == residue_identity:
            # set internal coordinates
            aa_alphabet.atom_to_internal_coordinates()
            # change angles to specified value
            ic: IC_Residue = res.internal_coord
            if tau:
                ic.set_angle("tau", tau)
            if carb_angle:
                ic.bond_set("N:CA:C:O", carb_angle)
            if phi:
                ic.set_angle("phi", phi)
            if psi:
                ic.set_angle("psi", psi)
            if omega:
                ic.set_angle("omega", omega)
            if N_CA_length:
                ic.set_length("N:CA", N_CA_length)
            if CA_C_length:
                ic.set_length("CA:C", CA_C_length)
            if C_O_length:
                ic.set_length("C:O", C_O_length)

            max_chis = RESIDUE_ID_TO_NUM_CHI.get(residue_identity, 0)

            if max_chis > 0:
                ic.bond_set("chi1", chi1)
            if max_chis > 1:
                ic.bond_set("chi2", chi2)
            if max_chis > 2:
                ic.bond_set("chi3", chi3)
            if max_chis > 3:
                ic.set_angle("chi4", chi4)
            aa_alphabet.internal_to_atom_coordinates()
            # change residue number to the one that is replaced (detaching is necessary because otherwise 2 res with same resid would exist in alphabet)
            aa_alphabet.detach_child(res.id)
            res.id = res_id
            if rot_probability:
                for atom in res:
                    atom.bfactor = rot_probability * 100

            return res
    raise KeyError(f"Could not found the residue {residue_identity}")


def align_to_sidechain(
    entity, entity_residue_to_align, sidechain: Residue, flip_symmetric: bool = True
):
    """
    aligns an input structure (bb_fragment_structure, resnum_to_align) to a sidechain residue (sc_structure, resnum_to_alignto)
    """
    sc_residue_identity = sidechain.get_resname()

    # superimpose structures based on specified atoms
    bbf_atoms = atoms_for_func_group_alignment(entity_residue_to_align)
    sc_atoms = atoms_for_func_group_alignment(sidechain)
    if flip_symmetric and sc_residue_identity in TIP_SYMMETRIC_RESIDUES:
        order = [1, 0, 2]
        sc_atoms = [sc_atoms[i] for i in order]
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(sc_atoms, bbf_atoms)
    # sup.rotran
    sup.apply(entity)

    return entity


def identify_his_central_atom(histidine, ligand):
    """
    identify the atom for histidine rotation by distance to ligand
    """
    his_NE2 = histidine["NE2"]
    his_ND1 = histidine["ND1"]
    lig_atoms = [atom for atom in ligand.get_atoms()]
    NE2_distance = min([his_NE2 - atom for atom in lig_atoms])
    ND1_distance = min([his_ND1 - atom for atom in lig_atoms])
    if NE2_distance < ND1_distance:
        his_central_atom = "NE2"
    else:
        his_central_atom = "ND1"
    return his_central_atom


def rotation_matrix(v, x):
    """
    calculate rotation matrix
    """
    k = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    i = np.identity(3)
    r = i + np.sin(x) * k + (1 - np.cos(x)) * np.dot(k, k)
    return r


def atoms_for_func_group_alignment(residue: Residue):
    """
    return the atoms used for superposition via functional groups
    """
    sc_residue_identity = residue.get_resname()

    if sc_residue_identity not in FUNC_GROUPS:
        raise RuntimeError(f"Unknown residue with name {sc_residue_identity}!")
    else:
        return [residue[atom] for atom in FUNC_GROUPS[sc_residue_identity]]


def clean_input_backbone(entity: Structure):
    """
    clean input backbone for chain names and resnums
    """
    chains = [chain for chain in entity.get_chains()]
    models = [model for model in entity.get_models()]
    if len(chains) > 1 or len(models) > 1:
        logging.error(
            "Input backbone fragment pdb must only contain a single chain and a single model!"
        )
    for model in models:
        model.id = 0
    for chain in chains:
        chain.id = "A"
    for index, residue in enumerate(entity.get_residues()):
        residue.id = (residue.id[0], index + 1, residue.id[2])
    for atom in entity.get_atoms():
        atom.bfactor = 0
    return entity[0]["A"]


def rotamers_for_backbone(
    bbdep: pd.DataFrame,
    resnames: list[str],
    phi,
    psi,
    rot_prob_cutoff: float = 0.05,
    prob_diff_to_best: float = 0.5,
):
    """
    identify rotamers that fit to a given backbone
    """
    rotlib_list = []
    for res in resnames:
        if res in ["ALA", "GLY"]:
            # TODO: assign proper scores for log prob and occurrence
            rotlib = pd.DataFrame(
                [[res, phi, psi, 1.0, 0.0, float("nan"), 0.0]],
                columns=[
                    "AA",
                    "phi",
                    "psi",
                    "probability",
                    "log_prob",
                    "phi_psi_occurrence",
                    "log_occurrence",
                ],
            )
            rotlib_list.append(rotlib)
        else:
            rotlib = bbdep[bbdep["AA"] == res]
            rotlib_list.append(
                identify_rotamers_suitable_for_backbone(
                    phi,
                    psi,
                    rotlib,
                    rot_prob_cutoff,
                    prob_diff_to_best,
                )
            )
    if len(rotlib_list) > 1:
        return (
            pd.concat(rotlib_list)
            .sort_values("probability", ascending=False)
            .reset_index(drop=True)
        )
    else:
        return rotlib_list[0]


def identify_rotamers_suitable_for_backbone(
    phi: float,
    psi: float,
    rotlib: pd.DataFrame,
    prob_cutoff: float | None = None,
    prob_diff_to_best: float | None = None,
):
    """
    identifies suitable rotamers by filtering for phi/psi angles
    if fraction is given, returns only the top rotamer fraction ranked by probability (otherwise returns all rotamers)
    if prob_cutoff is given, returns only rotamers more common than prob_cutoff
    """
    # round dihedrals to the next tens place
    if phi is not None:
        phi = round(phi, -1)
    if psi is not None:
        psi = round(psi, -1)
    # extract all rows containing specified phi/psi angles from library
    if phi and psi:
        logging.info(f"Searching for rotamers in phi/psi bin {phi}/{psi}.")
        rotlib = rotlib.loc[
            (rotlib["phi"] == phi) & (rotlib["psi"] == psi)
        ].reset_index(drop=True)
    elif not phi or not psi:
        if not phi:
            rotlib = rotlib[rotlib["psi"] == psi].reset_index(drop=True)
        elif not psi:
            rotlib = rotlib[rotlib["phi"] == phi].reset_index(drop=True)
        rotlib = rotlib.loc[rotlib["phi_psi_occurrence"] >= 1]
        rotlib = rotlib.drop_duplicates(subset=["phi", "psi"], keep="first")
        rotlib.sort_values("probability", ascending=False)
        rotlib = rotlib.head(5)
    # filter top rotamers
    rotlib = rotlib.sort_values("probability", ascending=False)
    if prob_cutoff:
        rotlib = rotlib.loc[rotlib["probability"] > prob_cutoff]
    if prob_diff_to_best:
        rotlib = rotlib[
            rotlib["probability"]
            >= rotlib["probability"].max() * (1 - prob_diff_to_best)
        ]
    rotlib = diversify_chi_angles(rotlib, 2, 2)
    # filter again, since diversify_chi_angles produces rotamers with lower probability
    if prob_cutoff:
        rotlib = rotlib.loc[rotlib["probability"] > prob_cutoff]
    if prob_diff_to_best:
        rotlib = rotlib[
            rotlib["probability"]
            >= rotlib["probability"].max() * (1 - prob_diff_to_best)
        ]
    return rotlib.head(100)


def diversify_chi_angles(
    rotlib: pd.DataFrame, max_stdev: float = 2, level: int = 3
) -> pd.DataFrame:
    """
    adds additional chi angles based on standard deviation.
    max_stdev: defines how far to stray from mean based on stdev. chi_new = chi_orig +- stdev * max_stdev
    level: defines how many chis should be sampled within max_stdev. if level = 1, mean, mean + stdev*max_stdev, mean - stdev*max_stdev will be returned. if level = 2, mean, mean + 1/2 stdev*max_stdev, mean + stdev*max_stdev, mean - 1/2 stdev*max_stdev, mean - stdev*max_stdev will be returned
    """
    # check which chi angles exist in rotamer library
    columns = [
        column
        for column in rotlib.columns
        if column.startswith("chi") and "sig" not in column
    ]
    # generate deviation parameters
    devs = [max_stdev * i / level for i in range(-level, level + 1)]
    # calculate chi angles
    for chi_angle in columns:
        new_chis_list = []
        for dev in devs:
            new_chis = alter_chi(rotlib, chi_angle, f"{chi_angle}sig", dev)
            new_chis_list.append(new_chis)
        rotlib = pd.concat(new_chis_list)
        rotlib.drop([f"{chi_angle}sig"], axis=1, inplace=True)
        rotlib[chi_angle] = round(rotlib[chi_angle], 1)
    rotlib.sort_values("probability", inplace=True, ascending=False)
    rotlib.reset_index(drop=True, inplace=True)
    return rotlib


def create_df_from_fragment(backbone, fragment_name):
    """
    extract information from fragment to create a dataframe
    """
    pdbnames = [fragment_name for res in backbone.get_residues()]
    resnames = [res.resname for res in backbone.get_residues()]
    pos_list = [res.id[1] for res in backbone.get_residues()]
    problist = [float("nan") for res in backbone.get_residues()]
    phi_angles = [
        extract_backbone_angles(backbone, res.id[1])["phi"]
        for res in backbone.get_residues()
    ]
    psi_angles = [
        extract_backbone_angles(backbone, res.id[1])["psi"]
        for res in backbone.get_residues()
    ]
    omega_angles = [
        extract_backbone_angles(backbone, res.id[1])["omega"]
        for res in backbone.get_residues()
    ]
    CA_x_coords_list = [
        (round(res["CA"].get_coord()[0], 3)) for res in backbone.get_residues()
    ]  # pylint: disable=C0103
    CA_y_coords_list = [
        (round(res["CA"].get_coord()[1], 3)) for res in backbone.get_residues()
    ]  # pylint: disable=C0103
    CA_z_coords_list = [
        (round(res["CA"].get_coord()[2], 3)) for res in backbone.get_residues()
    ]  # pylint: disable=C0103
    C_x_coords_list = [
        (round(res["C"].get_coord()[0], 3)) for res in backbone.get_residues()
    ]  # pylint: disable=C0103
    C_y_coords_list = [
        (round(res["C"].get_coord()[1], 3)) for res in backbone.get_residues()
    ]  # pylint: disable=C0103
    C_z_coords_list = [
        (round(res["C"].get_coord()[2], 3)) for res in backbone.get_residues()
    ]  # pylint: disable=C0103
    N_x_coords_list = [
        (round(res["N"].get_coord()[0], 3)) for res in backbone.get_residues()
    ]  # pylint: disable=C0103
    N_y_coords_list = [
        (round(res["N"].get_coord()[1], 3)) for res in backbone.get_residues()
    ]  # pylint: disable=C0103
    N_z_coords_list = [
        (round(res["N"].get_coord()[2], 3)) for res in backbone.get_residues()
    ]  # pylint: disable=C0103
    O_x_coords_list = [
        (round(res["O"].get_coord()[0], 3)) for res in backbone.get_residues()
    ]  # pylint: disable=C0103
    O_y_coords_list = [
        (round(res["O"].get_coord()[1], 3)) for res in backbone.get_residues()
    ]  # pylint: disable=C0103
    O_z_coords_list = [
        (round(res["O"].get_coord()[2], 3)) for res in backbone.get_residues()
    ]  # pylint: disable=C0103

    df = pd.DataFrame(
        list(
            zip(
                pdbnames,
                resnames,
                pos_list,
                phi_angles,
                psi_angles,
                omega_angles,
                CA_x_coords_list,
                CA_y_coords_list,
                CA_z_coords_list,
                C_x_coords_list,
                C_y_coords_list,
                C_z_coords_list,
                N_x_coords_list,
                N_y_coords_list,
                N_z_coords_list,
                O_x_coords_list,
                O_y_coords_list,
                O_z_coords_list,
                problist,
            )
        ),
        columns=[
            "pdb",
            "AA",
            "position",
            "phi",
            "psi",
            "omega",
            "CA_x",
            "CA_y",
            "CA_z",
            "C_x",
            "C_y",
            "C_z",
            "N_x",
            "N_y",
            "N_z",
            "O_x",
            "O_y",
            "O_z",
            "probability",
        ],
    )
    df[["chi1", "chi2", "chi3", "chi4"]] = float("nan")
    return df


def normal_dist_density(x):
    """
    calculates y value for normal distribution from distance from mean TODO: check if it actually makes sense to do it this way
    """
    y = math.e ** (-((x) ** 2) / 2)
    return y


def alter_chi(rotlib: pd.DataFrame, chi_column: str, chi_sig_column: str, dev: float):
    """
    calculate deviations from input chi angle for rotamer library
    """
    new_chis = copy.deepcopy(rotlib)
    new_chis[chi_column] = new_chis[chi_column] + new_chis[chi_sig_column] * dev
    new_chis["probability"] = new_chis["probability"] * normal_dist_density(dev)
    new_chis["log_prob"] = np.log(new_chis["probability"])
    return new_chis


def exchange_covalent(covalent_bond):
    """
    exchange atoms involved in covalent bonds when using residues with same functional groups
    """
    atom = covalent_bond[0]
    exchange_dict = {
        "OE1": "OD1",
        "OE2": "OD2",
        "CD1": "CG1",
        "CD2": "CG2",
        "NE2": "ND2",
        "OD1": "OE1",
        "OD2": "OE2",
        "CG1": "CD1",
        "CG2": "CD2",
        "ND2": "NE2",
    }
    new_atom = exchange_dict.get(atom, atom)
    covalent_bond[0] = new_atom
    return covalent_bond


def flip_covalent(covalent_bond, residue: str):
    """
    flip atoms involved in covalent bonds when flipping a rotamer
    """
    atom = covalent_bond[0]
    exchange_dict = {
        "GLU": {"OE1": "OE2", "OE2": "OE1"},
        "ASP": {"OD1": "OD2", "OD2": "OD1"},
        "VAL": {"CD1": "CD2", "CD2": "CD1"},
        "LEU": {"CG1": "CG2", "CG2": "CG1"},
        "ARG": {"NH1": "NH2", "NH2": "NH1"},
    }
    inner = exchange_dict.get(residue, {})
    new_atom = inner.get(atom, atom)

    covalent_bond[0] = new_atom
    return covalent_bond


def define_rotamer_positions(rotamer_positions, fragsize):
    """
    parse rotamer positions
    """
    if rotamer_positions == "auto":
        rotamer_positions = list(range(2, fragsize))
        return rotamer_positions
    elif isinstance(rotamer_positions, list):
        if any(int(pos) > fragsize for pos in rotamer_positions):
            raise KeyError("Rotamer positions must be smaller than fragment size!")
        return [int(pos) for pos in rotamer_positions]
    elif isinstance(rotamer_positions, int):
        if rotamer_positions > fragsize:
            raise KeyError("Rotamer positions must be smaller than fragment size!")
        return [rotamer_positions]
    else:
        raise KeyError(
            f"rotamer_positions must be 'auto', a list of int or a single int value, not {type(rotamer_positions)}!"
        )


def update_covalent_bonds(
    covalent_bonds: list | None,
    rotamer_id: str,
    rotamer_position: int,
    ligand_dict: dict,
):
    """
    update covalent bonds with rotamer position information
    """
    if not covalent_bonds:
        return None
    updated_bonds = []
    for cov_bond in covalent_bonds:
        res_atom, lig_chain, lig_resnum, lig_atom = cov_bond
        ligand = ligand_dict[lig_chain][int(lig_resnum)]
        _, lig_resnum, _ = ligand.id
        lig_id = ligand.get_resname()
        updated_bonds.append(
            f"{rotamer_position}A_{rotamer_id}-{res_atom}:{lig_resnum}Z_{lig_id}-{lig_atom}"
        )
    return ",".join(updated_bonds)


def validate_covalent_bond_info(
    covalent_bonds: list[CovalentBond],
    theozyme_residue: Residue,
    lig_dict: dict[str, dict[int, Residue]],
    resnum,
    chain,
):
    """
    parses covalent bond information
    """
    for covalent_bond in covalent_bonds:
        # check if covalent bond atoms are present in theozyme residue and ligand
        if covalent_bond.residue_atom not in [atom.name for atom in theozyme_residue]:
            raise KeyError(
                f"Could not find atom {covalent_bond.residue_atom} from covalent bond {covalent_bond} in residue {chain}{resnum}!"
            )
        if covalent_bond.ligand_atom not in [
            atom.name
            for atom in lig_dict[covalent_bond.ligand.chain][covalent_bond.ligand.index]
        ]:
            raise KeyError(
                f"Could not find atom {covalent_bond.ligand_atom} from covalent bond {covalent_bond} in ligand {covalent_bond.ligand}!"
            )


def create_frag_sec_struct_fraction_dict(
    frag_sec_struct_fraction: str, fragsize: int, rot_sec_struct: str
):
    """parse secondary structure input"""
    if frag_sec_struct_fraction:
        sec_structs = frag_sec_struct_fraction.split(",")
        sec_dict = {}
        for i in sec_structs:
            sec, fraction = i.split(":")
            frac = float(fraction)
            if frac > 1 or frac < 0:
                raise ValueError(
                    f"Fraction for secondary structure {sec} must be a value between 0 and 1, but it is {frac}!"
                )
            if (
                (fragsize - frac * fragsize) < 1
                and sec != rot_sec_struct
                and rot_sec_struct is not None
            ):
                raise KeyError(
                    f"If limiting all fragment residues to secondary structure {sec}, it is not possible that the rotamer has secondary structure {rot_sec_struct}!"
                )
            elif (
                (fragsize - frac * fragsize) < 1
                and rot_sec_struct is None
                and len(sec_structs) == 1
            ):
                logging.info(
                    f"Setting <rot_sec_struct> to {sec} because all residues in fragment have to have secondary structure {sec}!"
                )
                rot_sec_struct = sec
            sec_dict[sec] = float(frac)
    else:
        sec_dict = None

    return sec_dict, rot_sec_struct


def replace_covalent_bonds_chain(
    chain: str, covalent_bonds: str | None = None
) -> ResidueSelection:
    """
    update covalent bonds chain information after assigning new chain names
    """
    if not isinstance(covalent_bonds, str):
        return None
    new_cov_bonds = []
    for cov_bond in covalent_bonds.split(","):
        rot, lig = cov_bond.split(":")
        rechained = rot.split("_")[0][:-1] + chain + "_" + rot.split("_")[1]
        new_cov_bonds.append(":".join([rechained, lig]))
    return ",".join(new_cov_bonds)


def run_clash_detection(
    data, directory, bb_multiplier, sc_multiplier, script_path, jobstarter: JobStarter
):
    """
    run clash detection between fragments
    """

    def write_clash_detection_cmd(
        pose1, pose2, bb_multiplier, sc_multiplier, script_path, directory, prefix
    ):
        cmd = f"{sys.executable} {script_path} --pose1 {pose1} --pose2 {pose2} --working_dir {directory} --bb_multiplier {bb_multiplier} --sc_multiplier {sc_multiplier} --output_prefix {prefix}"
        return cmd

    def generate_valid_combinations(n_sets, compat_maps, set_lengths):
        valid_combos = []

        def backtrack(current_indices, depth):
            if depth == n_sets:
                valid_combos.append(tuple(current_indices))
                return

            for idx in range(set_lengths[depth]):
                # Check compatibility of this new index with all previous ones
                is_valid = True
                for prev_set in range(depth):
                    prev_idx = current_indices[prev_set]
                    # Make sure there's an entry
                    if prev_idx not in compat_maps[prev_set][depth]:
                        is_valid = False
                        break
                    if idx not in compat_maps[prev_set][depth][prev_idx]:
                        is_valid = False
                        break
                if is_valid:
                    backtrack(current_indices + [idx], depth + 1)

        backtrack([], 0)
        return valid_combos

    in_files = []
    in_dfs = []
    for pose, df in data.groupby("poses", sort=False):
        filename = os.path.join(
            directory, f"{os.path.splitext(os.path.basename(pose))[0]}.json"
        )
        df.reset_index(drop=True, inplace=True)
        df.to_json(filename)
        in_files.append(filename)
        in_dfs.append(df)

    set_lengths = [len(df.index) for df in in_dfs]
    n_sets = len(in_files)

    cmds = []
    prefixes = []
    prefix_map = {}
    for i, set1 in enumerate(in_files):  # iterative over each set
        for j in range(i + 1, n_sets):
            set2 = in_files[j]  # define second set
            prefixes.append(
                prefix
                := f"{os.path.splitext(os.path.basename(set1))[0]}_{os.path.splitext(os.path.basename(set2))[0]}"
            )  # create a unique prefix for each pair
            cmds.append(
                write_clash_detection_cmd(
                    set1,
                    set2,
                    bb_multiplier,
                    sc_multiplier,
                    script_path,
                    directory,
                    prefix,
                )
            )  # write clash detection cmds
            prefix_map[prefix] = (i, j)
    logging.info("Calculating pairwise compatibility maps...")

    jobstarter.start(
        cmds=cmds, jobname="clash_detection", wait=True, output_path=directory
    )  # distribute clash detection to cluster

    # Build compatibility maps using fast filtering (no row iteration)
    compat_maps = [[None] * n_sets for _ in range(n_sets)]

    logging.info("Importing results...")
    # import results
    clash_dfs = []
    for prefix in prefixes:
        i, j = prefix_map[prefix]
        filepath = os.path.join(directory, f"{prefix}.json")
        clash_df = pd.read_json(filepath)
        clash_dfs.append(clash_df)
        filtered_df = clash_df[~clash_df["clash"]]

        # Group each pose1_index by its non-clashing pose2_index set
        a_to_b = filtered_df.groupby("pose1_index")["pose2_index"].agg(set).to_dict()
        b_to_a = filtered_df.groupby("pose2_index")["pose1_index"].agg(set).to_dict()

        compat_maps[i][j] = a_to_b
        compat_maps[j][i] = b_to_a  # Optional bidirectional support

        # analyze number of clashes
        bb_bb_clashes = clash_df["bb_bb_clash"].sum()
        bb_sc_clashes = clash_df["bb_sc_clash"].sum()
        sc_sc_clashes = clash_df["sc_sc_clash"].sum()
        logging.info(
            f"Number of clashes for combination {prefix}:\nbackbone-backbone clashes: {bb_bb_clashes}\nbackbone-sidechain clashes: {bb_sc_clashes}\nsidechain-sidechain clashes: {sc_sc_clashes}"
        )

    clash_df = pd.concat(clash_dfs)
    bb_bb_clashes = clash_df["bb_bb_clash"].sum()
    bb_sc_clashes = clash_df["bb_sc_clash"].sum()
    sc_sc_clashes = clash_df["sc_sc_clash"].sum()
    logging.info(
        f"Total number of clashes:\nbackbone-backbone clashes: {bb_bb_clashes}\nbackbone-sidechain clashes: {bb_sc_clashes}\nsidechain-sidechain clashes: {sc_sc_clashes}"
    )
    logging.info(
        "If number of sidechain clashes is high, this is often a result of missing covalent bonds. Otherwise, <frag_frag_sc_clash_vdw_multiplier> can be reduced."
    )

    logging.info("Generating valid combinations...")
    valid_combos = generate_valid_combinations(n_sets, compat_maps, set_lengths)

    if len(valid_combos) < 1:
        logging.error(
            "No valid non-clashing combinations found. Adjust parameters like Van-der-Waals multiplier or pick different fragments!"
        )

    logging.info(f"Found {len(valid_combos)} valid combinations.")

    valid_combos_arr = np.array(valid_combos)  # shape (num_ensembles, n_sets)

    logging.info("Extracting data for each pose...")
    flattened_dfs = []
    for i in range(n_sets):
        indices = valid_combos_arr[:, i].flatten()  # indices from set i
        df = in_dfs[i].iloc[indices]
        flattened_dfs.append(df)

    logging.info("Combining data to ensemble dataframe...")
    # Combine all into final DataFrame (optimized for speed)
    ensemble_df = pd.concat(flattened_dfs, ignore_index=True)
    ensemble_df["ensemble_num"] = [i for i in range(len(valid_combos))] * n_sets
    ensemble_df.reset_index(drop=True, inplace=True)
    return ensemble_df


def sort_dataframe_groups_by_column(
    df: pd.DataFrame,
    group_col: str,
    sort_col: str,
    method="mean",
    ascending: bool = True,
    filter_top_n: int | None = None,
    randomize_ties: bool = False,
) -> pd.DataFrame:
    """group by group column and calculate mean values"""
    df_sorted = df.groupby(group_col, sort=False).agg({sort_col: method})
    if randomize_ties:
        df_sorted["temp_randomizer"] = np.random.rand(len(df_sorted))
        df_sorted.sort_values(
            [sort_col, "temp_randomizer"], ascending=ascending, inplace=True
        )
    else:
        df_sorted.sort_values(sort_col, ascending=ascending, inplace=True)
    # filter
    if filter_top_n:
        df_sorted = df_sorted.head(filter_top_n)
    # merge back with original dataframe
    df = df_sorted.loc[:, []].merge(df, left_index=True, right_on=group_col)
    # reset index
    df.reset_index(drop=True, inplace=True)
    return df


def concat_columns(group):
    """concat columns within a group to a single str"""
    non_null_elements = group.dropna().astype(str)
    return ",".join(non_null_elements) if not non_null_elements.empty else None


def split_str_to_dict(key_str, value_str, sep):
    """split a str according to sep, convert to dict"""
    return dict(zip(key_str.split(sep), [list(i) for i in value_str.split(sep)]))


def create_motif_residues(chain_str, fragsize_str, sep: str = ","):
    """create motif residue dict"""
    motif_residues = [
        [i for i in range(1, int(fragsize) + 1)] for fragsize in fragsize_str.split(sep)
    ]
    return dict(zip(chain_str.split(sep), motif_residues))


def create_motif_contig(
    chain_str: str, fragsize_str: str, path_order: str, sep: str
) -> str:
    """create motif contig for diffusion"""
    chains = chain_str.split(sep)
    fragsizes = fragsize_str.split(sep)
    contig = [f"{chain}1-{length}" for chain, length in zip(chains, fragsizes)]
    return f"{sep}".join(sorted(contig, key=lambda x: path_order.index(x[0])))


def create_pdbs(
    df: pd.DataFrame, output_dir, ligand, channel_path, preserve_channel_coordinates
) -> list[str]:
    """write output pdbs"""
    filenames = []
    for _, row in df.iterrows():
        paths = row["poses"].split(",")
        models = row["model_num"].split(",")
        chains = row["chain_id"].split(",")
        frag_chains = [
            load_pdb(path)[int(model)][chain]
            for path, model, chain in zip(paths, models, chains)
        ]
        filename_parts = [
            "".join([chain, model]) for chain, model in zip(chains, models)
        ]
        filename = (
            "-".join(
                sorted(filename_parts, key=lambda x: row["path_order"].index(x[0]))
            )
            + ".pdb"
        )
        filename = os.path.abspath(os.path.join(output_dir, filename))
        struct = Structure("out")
        struct.add(model := Model(0))
        model.add(chain := Chain("Z"))
        for frag in frag_chains:
            model.add(frag)
        for lig in ligand:
            chain.add(lig)
        if channel_path and preserve_channel_coordinates:
            model.add(load_pdb(channel_path)[0]["Q"])
        elif channel_path:
            model = add_placeholder_to_pose(
                model, channel_path=channel_path, channel_chain="Q", ligand_chain="Z"
            )
        save_structure_to_pdbfile(struct, filename)
        filenames.append(filename)
    return filenames


def fit_line_helix_centroids(points, window=4, trim_ends=0, outlier_z=2.5):
    """
    Helix-aware axis fit using sliding-window centroids (e.g., i..i+3 Cα).
    Returns:
      c : centroid on axis
      v : unit direction vector (N→C oriented)
      (s_min, s_max) : projection range of the ORIGINAL points along v
      r_est : RMS radial distance of ORIGINAL points to axis
    Args:
      points     : (N,3) array (ideally Cα atoms ordered from N to C)
      window     : sliding window length for centroids (4 is classic for α-helix)
      trim_ends  : optional number of residues to drop from each end before centroiding
      outlier_z  : remove centroid outliers farther than z * MAD from preliminary line
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < max(5, window + 1):
        # fallback to your PCA if not enough points
        return fit_line_pca(P)

    # 0) optionally trim ragged ends before centroiding
    if trim_ends > 0:
        P_used = P[trim_ends:-trim_ends]
        if P_used.shape[0] < window + 1:
            P_used = P  # not enough points to trim
    else:
        P_used = P

    # 1) sliding-window centroids (i..i+window-1)
    M = P_used.shape[0] - window + 1
    C = np.empty((M, 3), dtype=float)
    for i in range(M):
        C[i] = P_used[i : i + window].mean(axis=0)

    # 2) preliminary line from centroids via PCA
    Cc = C - C.mean(axis=0)
    _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    v0 = Vt[0]
    v0 /= np.linalg.norm(v0)
    c0 = C.mean(axis=0)

    # 3) robust pass: remove centroid outliers (radial distance from preliminary line)
    #    distance to line through c0 with direction v0
    t = (C - c0) @ v0
    C_proj = c0 + np.outer(t, v0)
    resid = np.linalg.norm(C - C_proj, axis=1)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-9  # robust scale
    keep = resid <= (med + outlier_z * 1.4826 * mad)  # 1.4826 ≈ MAD→σ for Gauss

    if keep.sum() >= max(3, window):  # refit if we actually kept enough points
        Ck = C[keep]
        Ckc = Ck - Ck.mean(axis=0)
        _, _, Vt2 = np.linalg.svd(Ckc, full_matrices=False)
        v = Vt2[0]
        v /= np.linalg.norm(v)
        c = Ck.mean(axis=0)
    else:
        v, c = v0, c0

    # 4) Orient axis N→C using original endpoints
    if ((P[-1] - P[0]) @ v) < 0:
        v = -v

    # 5) Project ORIGINAL points to get span and radius estimate
    X = P - c
    s_vals = X @ v

    s_first = float(s_vals[0])  # projection coordinate of N-terminal Cα
    p_first_proj = c + s_first * v  # 3D coords of projected first Cα

    return c, v, p_first_proj


def fit_line_pca(points):
    """
    Fit a 3D line (axis) through a set of coordinates using PCA (SVD).
    Returns:
      c      = centroid (a point on the axis)
      v      = unit direction vector along the helix axis
      (s_min, s_max) = scalar range of projections of the points along v
      r_est  = RMS radial distance of points to the fitted axis (≈ helix radius)
    """
    P = np.asarray(
        points, dtype=float
    )  # ensure input is a NumPy float array of shape (N,3)
    c = P.mean(axis=0)  # compute centroid of all points
    X = P - c  # center the data around the centroid
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # perform SVD; right-singular vectors in Vt are principal directions

    v = Vt[0]  # take the first principal component (direction of max variance)
    v = v / np.linalg.norm(v)  # normalize to unit vector

    s_vals = X @ v  # project all centered points onto axis → scalar coords

    s_first = float(s_vals[0])  # projection coordinate of N-terminal Cα
    p_first_proj = c + s_first * v  # 3D coords of projected first Cα

    return c, v, p_first_proj  # return centroid, direction, span, and estimated radius


def add_placeholder_to_pose(
    pose: Structure,
    channel_path: str,
    channel_chain: str = "Q",
    ligand_chain: str = "Z",
) -> Structure:
    """
    add channel placehoder to pose
    """
    ignore_atoms = ["H"]

    # load polyala:
    channel = load_pdb(channel_path)[0]

    channel_atoms = [
        atom for atom in channel.get_atoms() if atom.name not in ignore_atoms
    ]

    frag_protein_atoms, frag_ligand_atoms = get_protein_and_ligand_atoms(
        pose, ligand_chain=ligand_chain, ignore_atoms=ignore_atoms
    )

    # calculate vector between fragment and ligand centroids
    frag_protein_centroid = np.mean(frag_protein_atoms, axis=0)
    frag_ligand_centroid = np.mean(frag_ligand_atoms, axis=0)
    vector_fragment = frag_ligand_centroid - frag_protein_centroid

    # calculate vector between CA of first and last residue of polyala
    channel_ca = [atom.get_coord() for atom in channel_atoms if atom.id == "CA"]

    channel_centroid, channel_vector, projected_Nterm = fit_line_helix_centroids(
        channel_ca
    )

    # calculate rotation between vectors
    r = Bio.PDB.rotmat(Bio.PDB.Vector(channel_vector), Bio.PDB.Vector(vector_fragment))

    # rotate polyala and translate into motif
    polyala_rotated = apply_rotation_to_pose(channel, projected_Nterm, r)
    polyala_translated = apply_translation_to_pose(
        polyala_rotated, frag_ligand_centroid - projected_Nterm
    )

    # change chain id of polyala and add into pose:
    if channel_chain in [chain.id for chain in pose.get_chains()]:
        raise KeyError(
            f"Chain {channel_chain} already found in pose. Try other chain name!"
        )
    pose.add(polyala_translated[channel_chain])
    return pose


def apply_rotation_to_pose(
    pose: Structure, origin: "list[float]", r: "list[list[float]]"
) -> Structure:
    """apply rotation"""
    for chain in pose:
        for residue in chain:
            for atom in residue:
                atom.coord = np.dot(r, atom.coord - origin) + origin
    return pose


def apply_translation_to_pose(pose: Structure, vector: "list[float]") -> Structure:
    """apply translation"""
    for chain in pose:
        for residue in chain:
            for atom in residue:
                atom.coord += vector
    return pose


def get_protein_and_ligand_atoms(
    pose: Structure, ligand_chain, bb_atoms=None, ignore_atoms="H"
) -> tuple[npt.NDArray, npt.NDArray]:
    """AAA"""
    if isinstance(ignore_atoms, str) or not ignore_atoms:
        ignore_atoms = [ignore_atoms]

    if not bb_atoms:
        bb_atoms = ["CA", "C", "N", "O"]

    if isinstance(ligand_chain, str):
        # get all CA coords of protein:
        protein_atoms = np.array(
            [
                atom.get_coord()
                for atom in get_protein_atoms(pose, ligand_chain)
                if atom.id in bb_atoms
            ]
        )

        # get Ligand Heavyatoms:
        ligand_atoms = np.array(
            [
                atom.get_coord()
                for atom in pose[ligand_chain].get_atoms()
                if atom.id not in ignore_atoms
            ]
        )

    elif isinstance(ligand_chain, Bio.PDB.Chain.Chain):
        # get all CA coords of protein:
        protein_atoms = np.array(
            [atom.get_coord() for atom in pose.get_atoms() if atom.id == "CA"]
        )

        # get Ligand Heavyatoms:
        ligand_atoms = np.array(
            [
                atom.get_coord()
                for atom in ligand_chain.get_atoms()
                if atom.id not in ignore_atoms
            ]
        )
    else:
        raise TypeError(
            f"Expected 'ligand' to be of type str or Bio.PDB.Chain.Chain, but got {type(ligand_chain)} instead."
        )
    return protein_atoms, ligand_atoms


def get_protein_atoms(
    pose: Structure, ligand_chain: str | None = None, atms: list | None = None
) -> list:
    """Selects atoms from a pose object. If ligand_chain is given, excludes all atoms in ligand_chain"""
    # define chains of pose
    chains = [x.id for x in pose.get_chains()]
    if ligand_chain:
        chains.remove(ligand_chain)

    # select specified atoms
    pose_atoms = [atom for chain in chains for atom in pose[chain].get_atoms()]
    if atms:
        pose_atoms = [atom for atom in pose_atoms if atom.id in atms]

    return pose_atoms


def create_ligand_dict(ligands: list[ResidueModel], theozyme: Model):
    """create a dictionary containing each ligand in the theozyme"""
    ligand = Chain("Z")
    lig_dict: dict[str, dict[int, Residue]] = {}

    for i, lig_id in enumerate(ligands):
        chain_id, resnum = lig_id.chain, lig_id.index
        if chain_id not in theozyme or resnum not in [
            res.id[1] for res in theozyme[chain_id]
        ]:
            raise KeyError(
                f"No ligand found in chain {chain_id} with residue number {resnum}. Please make sure the theozyme pdb is correctly formatted."
            )
        for res in theozyme[chain_id]:
            if res.id[1] == resnum:
                lig = res
        logging.info(f"Found ligand in chain {chain_id} with residue number {resnum}.")
        lig.detach_parent()
        # set occupancy to 1 (to prevent downstream issues)
        for atom in lig.get_atoms():
            atom.occupancy = 1
        lig.id = (lig.id[0], i + 1, lig.id[2])
        ligand.add(lig)
        lig_dict[chain_id] = {resnum: lig}
    return lig_dict, ligand


def import_channel(path: Path | None, chain: str | None, database: Path) -> Chain:
    """import channel placeholder"""
    if not path:
        path = database / "channel_placeholder" / "helix_cone_long.pdb"
        chain = "Q"
    else:
        if not chain:
            raise KeyError(
                "<channel_chain> must be provided if using a custom channel placeholder!"
            )
    if not path.is_file():
        raise RuntimeError(
            f"Could not find a PDB file at {path} to add as channel placeholder!"
        )
    model = load_pdb(path)[0]
    if chain not in model:
        raise RuntimeError(
            f"No channel placeholder found in {path} on chain {chain}. Please make sure the channel pdb is correctly formatted."
        )
    channel = model[chain]
    channel.detach_parent()
    channel.id = "Q"
    for index, residue in enumerate(channel):
        residue.id = (residue.id[0], index + 1, residue.id[2])

    return channel


def build_frag_dict(
    rotlib: pd.DataFrame, backbone_df: pd.DataFrame
) -> dict[int, pd.DataFrame]:
    """create a fragment dictionary"""
    frag_dict = {}
    relevant_columns = [
        col
        for col in rotlib.columns
        if col.startswith("chi") or col in ["probability", "phi_psi_occurrence", "AA"]
    ]

    for pos, group in rotlib.groupby("rotamer_position"):
        pos_frags = []
        for idx, row in group.iterrows():
            df = pd.DataFrame(
                backbone_df.values, columns=backbone_df.columns
            )  # recreate backbone
            df.loc[pos - 1, relevant_columns] = row[relevant_columns].values
            df["frag_num"] = idx
            df["rotamer_pos"] = pos
            df["rotamer_score"] = row["rotamer_score"]
            df["fragment_score"] = row["rotamer_score"]
            df["backbone_score"] = 0
            df["backbone_probability"] = 0
            pos_frags.append(df)
        frag_dict[pos] = pd.concat(pos_frags, ignore_index=True)
        logging.info(f"Created {len(pos_frags)} fragments for position {pos}.")

    return frag_dict


def main(args: Arguments):
    start = time.time()
    fragment_dir = args.working_dir / (
        f"{args.output_prefix}_fragments" if args.output_prefix else "fragments"
    )
    fragment_dir.mkdir(exist_ok=True)

    # import and prepare stuff
    database_dir = args.riff_diff_dir / "database"
    bbdep = read_bbdep(database_dir / "bb_dep_rotlibs")
    utils_dir = args.riff_diff_dir / "utils"

    theozyme = load_pdb(args.theozyme_pdb)[0]
    aa_alphabet = load_pdb(database_dir / "AA_alphabet.pdb")[0]["A"]

    # import ligands
    lig_dict, ligand = create_ligand_dict(args.ligands, theozyme)

    # import channel
    channel = import_channel(args.custom_channel_path, args.channel_chain, database_dir)
    channel_size = len(channel)
    save_structure_to_pdbfile(
        channel, str(channel_path := fragment_dir / "channel_placeholder.pdb")
    )

    # create output folders
    (rotinfo_dir := fragment_dir / "rotamer_info").mkdir(exist_ok=True)
    (fraginfo_dir := fragment_dir / "fragment_info").mkdir(exist_ok=True)

    assembly: list[pd.DataFrame] = []
    for theozyme_resnum in args.theozyme_resnums:
        chain = theozyme_resnum.chain
        resnum = theozyme_resnum.index
        resname = str(theozyme_resnum)
        res_args = args.get_res_args(resname)

        try:
            theozyme_residue: Residue = theozyme[chain][resnum]
        except Exception as exc:
            raise KeyError(
                f"Could not find residue {resnum} on chain {chain} in theozyme {args.theozyme_pdb}!"
            ) from exc

        # import covalent bonds
        if res_args.covalent_bonds is not None:
            validate_covalent_bond_info(
                res_args.covalent_bonds, theozyme_residue, lig_dict, resnum, chain
            )
        covalent_bonds = (
            res_args.covalent_bonds
            if res_args.covalent_bonds is None
            else [
                (cb.residue_atom, cb.ligand.chain, cb.ligand.index, cb.ligand_atom)
                for cb in res_args.covalent_bonds
            ]
        )

        # define residue ids
        if res_args.add_equivalent_func_groups:
            residue_identities: list[str] = RESIDUE_GROUP_MAP.get(
                theozyme_residue.resname, [theozyme_residue.resname]
            )
        else:
            residue_identities = [theozyme_residue.resname]
        logging.info(f"Looking for rotamers for these residues: {residue_identities}")

        if not res_args.pick_frags_from_db:
            #################################### BACKBONE ROTAMER FINDER ####################################
            # import backbone fragment
            fragment_path = res_args.fragment_pdb or (
                database_dir / "backbone_frags" / "7helix.pdb"
            )
            backbone_structure = load_pdb(fragment_path)
            backbone = clean_input_backbone(backbone_structure)
            # define positions for rotamer insertion
            if res_args.rotamer_positions == "auto":
                frag_pos_to_replace = [
                    i + 1 for i, _ in enumerate(backbone.get_residues())
                ][1:-1]
            elif isinstance(res_args.rotamer_positions, list):
                frag_pos_to_replace = res_args.rotamer_positions
            elif isinstance(res_args.rotamer_positions, int):
                frag_pos_to_replace = [res_args.rotamer_positions]
            else:
                raise KeyError(
                    f"<rotamer_position> for residue {resname} must be 'auto', a list of int, or a single int!"
                )

            # extract data from backbone
            backbone_df = create_df_from_fragment(
                backbone, os.path.basename(fragment_path)
            )

            # identify rotamers
            rotlibs = []
            logging.info("Identifying rotamers...")
            for pos in frag_pos_to_replace:
                backbone_angles = extract_backbone_angles(backbone, pos)
                logging.info(
                    f"Position {pos} phi/psi angles: {backbone_angles['phi']} / {backbone_angles['psi']}."
                )
                rotlib = rotamers_for_backbone(
                    bbdep,
                    residue_identities,
                    backbone_angles["phi"],
                    backbone_angles["psi"],
                    res_args.prob_cutoff,
                    res_args.rotamer_diff_to_best,
                )
                rotlib["rotamer_position"] = pos
                logging.info(f"Found {len(rotlib.index)} rotamers for position {pos}.")
                rotlibs.append(rotlib)

            # combine rotamer libraries for each position
            rotlib = pd.concat(rotlibs).reset_index(drop=True)

            # rank rotamers
            rotlib["log_prob_normalized"] = scale(rotlib["log_prob"])
            rotlib["log_occurrence_normalized"] = scale(rotlib["log_occurrence"])
            rotlib["rotamer_score"] = minmax_scale(
                np.average(
                    rotlib[["log_prob_normalized", "log_occurrence_normalized"]],
                    weights=[res_args.prob_weight, res_args.occurrence_weight],
                )
            )
            rotlib = rotlib.sort_values("rotamer_score", ascending=False).reset_index(
                drop=True
            )

            logging.info(f"Found {len(rotlib.index)} rotamers in total.")
            rotlibcsv = os.path.join(rotinfo_dir, f"rotamers_{resname}_combined.csv")
            logging.info(f"Writing phi/psi combinations to {rotlibcsv}.")
            rotlib.to_csv(rotlibcsv)

            # create dictionary containing dataframes for all fragments
            frag_dict = build_frag_dict(rotlib, backbone_df)

        else:
            #################################### FRAGMENT FINDER ####################################
            # sanity check command line input
            sec_dict, res_args.rot_sec_struct = create_frag_sec_struct_fraction_dict(
                res_args.frag_sec_struct_fraction,
                res_args.fragsize,
                res_args.rot_sec_struct,
            )

            frag_pos_to_replace = define_rotamer_positions(
                res_args.rotamer_positions, res_args.fragsize
            )

            fraglib_path = os.path.join(database_dir, "fraglib_noscore.pkl")
            if not os.path.isfile(fraglib_path):
                raise RuntimeError(
                    f"Could not find fragment library at {fraglib_path}. Did you forget to download it?"
                )

            rotlibs = []

            for residue_identity in residue_identities:
                # find rotamer library for given amino acid
                logging.info(
                    f"Importing backbone dependent rotamer library for residue {residue_identity} from {database_dir}"
                )
                rotlib = bbdep[bbdep["AA"] == residue_identity]
                rotlib["log_prob_normalized"] = scale(rotlib["log_prob"])
                rotlib["log_occurrence_normalized"] = scale(rotlib["log_occurrence"])
                rotlib["rotamer_score"] = minmax_scale(
                    np.average(
                        rotlib[["log_prob_normalized", "log_occurrence_normalized"]],
                        weights=[res_args.prob_weight, res_args.occurrence_weight],
                    )
                )
                logging.info(
                    f"Identifying most probable rotamers for residue {residue_identity}"
                )
                rotlib = identify_backbone_angles_suitable_for_rotamer(
                    residue_identity,
                    rotlib,
                    rotinfo_dir,
                    f"{resname}_",
                    res_args.rot_sec_struct,
                    res_args.phipsi_occurrence_cutoff,
                    int(res_args.max_rotamers / len(residue_identities)),
                    res_args.rotamer_diff_to_best,
                    res_args.rotamer_chi_binsize,
                    res_args.rotamer_phipsi_binsize,
                    res_args.prob_cutoff,
                )
                logging.info(f"Found {len(rotlib.index)} phi/psi/chi combinations.")
                rotlibs.append(rotlib)

            rotlib = (
                pd.concat(rotlibs)
                .sort_values("rotamer_score", ascending=False)
                .reset_index(drop=True)
            )
            rotlib["log_prob_normalized"] = scale(rotlib["log_prob"])
            rotlib["log_occurrence_normalized"] = scale(rotlib["log_occurrence"])
            rotlib["rotamer_score"] = minmax_scale(
                np.average(
                    rotlib[["log_prob_normalized", "log_occurrence_normalized"]],
                    weights=[res_args.prob_weight, res_args.occurrence_weight],
                )
            )
            rotlib = rotlib.sort_values("rotamer_score", ascending=False).reset_index(
                drop=True
            )
            rotlibcsv = os.path.join(rotinfo_dir, f"rotamers_{resname}_combined.csv")
            logging.info(f"Writing phi/psi combinations to {rotlibcsv}.")
            rotlib.to_csv(rotlibcsv)

            # setting up jobstarters
            if args.jobstarter == "SbatchArray":
                jobstarter = SbatchArrayJobstarter(max_cores=args.cpus)
            elif args.jobstarter == "Local":
                jobstarter = LocalJobStarter(max_cores=args.cpus)
            else:
                raise KeyError("Jobstarter must be either 'SbatchArray' or 'Local'!")

            logging.info("Identifying positions for rotamer insertion...")
            rotamer_positions = identify_positions_for_rotamer_insertion(
                fraglib_path,
                rotlib,
                res_args.rot_sec_struct,
                res_args.phi_psi_bin,
                os.path.join(fragment_dir, "rotamer_positions"),
                os.path.join(utils_dir, "identify_positions_for_rotamer_insertion.py"),
                resname,
                res_args.chi_std_multiplier,
                jobstarter=jobstarter,
            )
            logging.info(f"Found {len(rotamer_positions.index)} fitting positions.")
            logging.info("Extracting fragments from rotamer positions...")
            combined = extract_fragments(
                rotamer_positions,
                fraglib_path,
                frag_pos_to_replace,
                res_args.fragsize,
                os.path.join(fragment_dir, f"{resname}_database_fragments"),
                os.path.join(utils_dir, "fragment_extraction.py"),
                jobstarter,
            )
            frag_num = int(len(combined.index) / res_args.fragsize)
            logging.info(f"Found {frag_num} fragments.")

            # filter fragments
            if frag_num == 0:
                logging.info("Could not find fragments.")
                raise RuntimeError("Could not find fragments.")
            if sec_dict:
                combined = filter_frags_df_by_secondary_structure_content(
                    combined, sec_dict
                )
                logging.info(
                    f"{int(len(combined) / res_args.fragsize)} fragments passed secondary structure filtering with filter {res_args.frag_sec_struct_fraction}."
                )
            if combined.empty:
                logging.info(
                    "Could not find any fragments that fit criteria! Try adjusting filter values!"
                )
                raise RuntimeError(
                    "Could not find any fragments that fit criteria! Try adjusting filter values!"
                )

            logging.info(
                f"Averaging and sorting fragments by fragment score with weights (backbone: {res_args.backbone_score_weight}, rotamer: {res_args.rotamer_score_weight})."
            )
            combined = sort_frags_df_by_score(
                combined,
                res_args.backbone_score_weight,
                res_args.rotamer_score_weight,
                res_args.fragsize,
            )

            frag_dict = {}
            for pos, df in combined.groupby("rotamer_pos", sort=True):
                frag_dict[pos] = df
                logging.info(
                    f"Created {int(len(df) / res_args.fragsize)} unique fragments for position {pos}."
                )
            combined = combined.groupby("frag_num", sort=False).mean(numeric_only=True)

            # visualize information about fragments
            violinplot_multiple_cols(
                dataframe=combined,
                cols=["fragment_score", "backbone_score", "rotamer_score"],
                titles=["fragment score", "backbone score", "rotamer score"],
                y_labels=["AU", "AU", "AU"],
                dims=[(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)],
                out_path=os.path.join(fraginfo_dir, f"{resname}_pre_clash_filter.png"),
                show_fig=False,
            )
            del combined

        #################################### CREATE FRAGS, ATTACH ROTAMERS, FILTER ####################################

        fragments = Structure("fragments")
        frags_table = []
        frags_info = []
        frag_num = 0

        for pos, pos_df in frag_dict.items():
            logging.info(
                f"Creating fragment structures, attaching rotamer, superpositioning with theozyme residue, calculating rmsd to all accepted fragments with cutoff {res_args.rmsd_cutoff} A for position {pos}."
            )

            pos_df["flipped"] = False
            # check if residues should be flipped to increase number of fragments
            flip = False
            if not res_args.not_flip_symmetric and any(
                x in TIP_SYMMETRIC_RESIDUES for x in residue_identities
            ):
                flip = True

            check_dict: CheckDict = {
                "selected_frags": [],
                "selected_frag_dfs": [],
                "sc_clashes": [],
                "channel_clashes": 0,
                "bb_clashes": 0,
                "rmsd_fails": 0,
            }
            for _, df in pos_df.groupby("frag_num", sort=False):
                # check if maximum number of fragments has been reached
                if len(
                    check_dict["selected_frags"]
                ) >= res_args.max_frags_per_residue / len(frag_dict):
                    break
                frag = create_fragment_from_df(df, pos, aa_alphabet)
                frag = align_to_sidechain(frag, frag[pos], theozyme_residue, False)
                check_dict = check_fragment(
                    frag=frag,
                    check_dict=check_dict,
                    frag_df=df,
                    ligand=ligand,
                    channel=channel if res_args.preserve_channel_coordinates else None,
                    rotamer_position=pos,
                    rmsd_cutoff=res_args.rmsd_cutoff,
                    bb_lig_clash_vdw_multiplier=res_args.bb_lig_clash_vdw_multiplier,
                    rot_lig_clash_vdw_multiplier=res_args.rot_lig_clash_vdw_multiplier,
                    channel_frag_clash_vdw_multiplier=res_args.channel_frag_clash_vdw_multiplier,
                )

                if flip is True:
                    flipped_frag = copy.deepcopy(frag)
                    flipped_df = df.copy()

                    flipped_frag = align_to_sidechain(
                        flipped_frag, frag[pos], theozyme_residue, True
                    )
                    flipped_df["flipped"] = True
                    check_dict = check_fragment(
                        frag=flipped_frag,
                        check_dict=check_dict,
                        frag_df=flipped_df,
                        ligand=ligand,
                        channel=channel
                        if res_args.preserve_channel_coordinates
                        else None,
                        rotamer_position=pos,
                        rmsd_cutoff=res_args.rmsd_cutoff,
                        bb_lig_clash_vdw_multiplier=res_args.bb_lig_clash_vdw_multiplier,
                        rot_lig_clash_vdw_multiplier=res_args.rot_lig_clash_vdw_multiplier,
                        channel_frag_clash_vdw_multiplier=res_args.channel_frag_clash_vdw_multiplier,
                    )

            logging.info(
                f"Discarded {check_dict['channel_clashes']} fragments that show clashes between backbone and channel placeholder with VdW multiplier {res_args.channel_frag_clash_vdw_multiplier}"
            )
            logging.info(
                f"Discarded {check_dict['bb_clashes']} fragments that show clashes between backbone and ligand with VdW multiplier {res_args.bb_lig_clash_vdw_multiplier}"
            )
            logging.info(
                f"Discarded {len(check_dict['sc_clashes'])} fragments that show clashes between sidechain and ligand with VdW multiplier {res_args.rot_lig_clash_vdw_multiplier}"
            )
            if len(check_dict["sc_clashes"]) > 0:
                logging.info(
                    f"Atoms involved in sidechain-ligand clashes: {Counter(check_dict['sc_clashes'])}"
                )
                logging.info(
                    f"You might want to try to adjust the <rot_lig_clash_vdw_multiplier> parameter (currently: {res_args.rot_lig_clash_vdw_multiplier}) for this residue: {chain}{resnum}"
                )
            logging.info(
                f"Discarded {check_dict['rmsd_fails']} fragments that did not pass RMSD cutoff of {res_args.rmsd_cutoff} to all other picked fragments"
            )
            passed_frags = len(check_dict["selected_frags"])
            if passed_frags < 1:
                logging.info(
                    f"Could not find any passing fragments for {chain}{resnum} position {pos}!"
                )
            logging.info(
                f"Found {passed_frags} fragments for position {pos} of a maximum of {res_args.max_frags_per_residue / len(frag_dict)}."
            )

            for frag, df in zip(
                check_dict["selected_frags"], check_dict["selected_frag_dfs"]
            ):
                rot = df.iloc[pos - 1].squeeze()
                if covalent_bonds and theozyme_residue.get_resname() != rot["AA"]:
                    covalent_bonds = [
                        exchange_covalent(covalent_bond)
                        for covalent_bond in covalent_bonds
                    ]
                if covalent_bonds and rot["flipped"] is True:
                    covalent_bonds = [
                        flip_covalent(covalent_bond, rot["AA"])
                        for covalent_bond in covalent_bonds
                    ]

                updated_bonds = update_covalent_bonds(
                    covalent_bonds, rot["AA"], pos, lig_dict
                )

                row = pd.Series(
                    {
                        "model_num": frag_num,
                        "rotamer_pos": pos,
                        "rotamer_id": rot["AA"],
                        "AAs": df["AA"].to_list(),
                        "frag_length": len(df.index),
                        "backbone_score": df["backbone_score"].mean(),
                        "fragment_score": df["fragment_score"].mean(),
                        "rotamer_probability": rot["probability"],
                        "phi_psi_occurrence": rot["phi_psi_occurrence"],
                        "backbone_probability": df["backbone_probability"].mean(),
                        "covalent_bonds": updated_bonds,
                        "rotamer_score": df["rotamer_score"].mean(),
                    }
                )
                model = Model(frag_num)
                model.add(frag)
                if ligand:
                    model.add(ligand)
                    row["ligand_chain"] = ligand.id
                if res_args.preserve_channel_coordinates:
                    model.add(channel)
                    row["channel_chain"] = channel.id
                fragments.add(model)
                df["frag_num"] = frag_num
                frags_table.append(df)
                frags_info.append(row)
                frag_num += 1
            del check_dict

        logging.info(f"Found {len(frags_info)} fragments that passed all filters.")

        # write fragment info to disk
        frags_table = pd.concat(frags_table)
        frags_table_path = os.path.join(fraginfo_dir, f"fragments_{resname}.csv")
        logging.info(f"Writing fragment details to {frags_table_path}.")
        frags_table.to_csv(frags_table_path)

        # write multimodel fragment pdb to disk
        filename_pdb = os.path.join(fragment_dir, f"{resname}.pdb")
        logging.info(f"Writing multimodel fragment pdb to {filename_pdb}.")
        save_structure_to_pdbfile(fragments, filename_pdb, multimodel=True)
        # utils.write_multimodel_structure_to_pdb(fragments, filename_pdb)

        # write output json to disk
        frags_info = pd.DataFrame(frags_info)
        frags_info["poses"] = os.path.abspath(filename_pdb)
        frags_info["poses_description"] = f"{resname}"
        filename_json = os.path.join(fragment_dir, f"{resname}.json")
        logging.info(f"Writing output json to {filename_json}.")
        frags_info.to_json(filename_json)

        if res_args.pick_frags_from_db:
            combined = frags_table.groupby("frag_num", sort=False).mean(
                numeric_only=True
            )
            violinplot_multiple_cols(
                combined,
                cols=["fragment_score", "backbone_score", "rotamer_score"],
                titles=["fragment score", "backbone score", "rotamer score"],
                y_labels=["AU", "AU", "AU"],
                dims=[(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)],
                out_path=os.path.join(fraginfo_dir, f"{resname}_post_filter.png"),
                show_fig=False,
            )

        assembly.append(frags_info)
        logging.info(f"Done in {round(time.time() - start, 1)} seconds!")

    working_dir = args.working_dir / (
        f"{args.output_prefix}_motif_library_assembly"
        if args.output_prefix
        else "motif_library_assembly"
    )

    # check if output already exists
    out_json = args.working_dir / (
        f"{args.output_prefix}_selected_paths.json"
        if args.output_prefix
        else "selected_paths.json"
    )
    if out_json.exists():
        raise RuntimeError(f"Output file already exists at {out_json}!")

    if args.jobstarter == "SbatchArray":
        jobstarter = SbatchArrayJobstarter(max_cores=args.cpus)
    elif args.jobstarter == "Local":
        jobstarter = LocalJobStarter(max_cores=args.cpus)
    else:
        raise KeyError(
            f"Jobstarter must be either 'SbatchArray' or 'Local', not {args.jobstarter}!"
        )

    in_df = pd.concat(assembly)

    ################## CLASH DETECTION ##########################

    logging.info("Fragment selection completed, continuing with clash checks...")

    clash_dir = os.path.join(working_dir, "clash_check")
    os.makedirs(clash_dir, exist_ok=True)

    grouped_df = in_df.groupby("poses", sort=False)
    counter = 0
    chains = []
    df_list = []
    structdict = {}

    for pose, pose_df in grouped_df:
        logging.info(f"Working on {pose}...")
        pose_df["input_poses"] = pose_df["poses"]
        pose_df["chain_id"] = list(string.ascii_uppercase)[counter]
        struct = load_pdb(pose)
        model_dfs = []
        for index, series in pose_df.iterrows():
            chain = struct[series["model_num"]]["A"]
            chain.id = list(string.ascii_uppercase)[counter]
            model_dfs.append(series)
        pose_df = pd.DataFrame(model_dfs)
        structdict[struct.id] = struct
        filename = os.path.join(clash_dir, f"{struct.id}_rechained.pdb")
        struct.id = filename
        save_structure_to_pdbfile(pose=struct, save_path=filename, multimodel=True)
        pose_df["poses"] = os.path.abspath(filename)
        chains.append(list(string.ascii_uppercase)[counter])
        counter += 1
        pose_df["covalent_bonds"] = pose_df.apply(
            lambda row: replace_covalent_bonds_chain(
                row["chain_id"], row["covalent_bonds"]
            ),
            axis=1,
        )
        df_list.append(pose_df)

    ligands = [lig for lig in struct[0]["Z"].get_residues()]
    for lig in ligands:
        lig.id = ("H", lig.id[1], lig.id[2])

    combined = pd.concat(df_list)
    grouped_df = combined.groupby("poses", sort=False)

    # generate every possible combination of input models
    num_models = [len(df.index) for _, df in grouped_df]
    num_combs = 1
    for i in num_models:
        num_combs *= i
    logging.info(f"Generating {num_combs} possible fragment combinations...")

    init = time.time()

    # combinations = itertools.product(*[[row for _, row in pose_df.iterrows()] for _, pose_df in grouped_df])
    logging.info("Performing pairwise clash detection...")
    ensemble_dfs = run_clash_detection(
        data=combined,
        directory=clash_dir,
        bb_multiplier=args.frag_frag_bb_clash_vdw_multiplier,
        sc_multiplier=args.frag_frag_sc_clash_vdw_multiplier,
        script_path=os.path.join(utils_dir, "clash_detection.py"),
        jobstarter=jobstarter,
    )

    # calculate scores
    score_df = ensemble_dfs.groupby("ensemble_num", sort=False).mean(numeric_only=True)
    score_df["ensemble_score"] = minmax_scale(scale(score_df["fragment_score"]))
    logging.info(f"Found {len(score_df.index)} non-clashing ensembles.")

    plotpath = os.path.join(working_dir, "clash_filter.png")
    logging.info(f"Plotting data at {plotpath}.")
    score_df_size = len(score_df.index)
    if score_df_size > 1000000:
        logging.info(
            f"Downsampling dataframe for plotting because dataframe size is too big ({score_df_size} rows). Plotting only 1000000 random rows."
        )
        score_df_small = score_df.sample(n=1000000)
        violinplot_multiple_cols(
            score_df_small,
            cols=[
                "ensemble_score",
                "backbone_probability",
                "rotamer_probability",
                "phi_psi_occurrence",
            ],
            titles=[
                "ensemble_score",
                "mean backbone\nprobability",
                "mean rotamer\nprobability",
                "mean phi psi\noccurrence",
            ],
            y_labels=["score", "probability", "probability", "probability"],
            out_path=plotpath,
            show_fig=False,
        )
    else:
        violinplot_multiple_cols(
            score_df,
            cols=[
                "ensemble_score",
                "backbone_probability",
                "rotamer_probability",
                "phi_psi_occurrence",
            ],
            titles=[
                "ensemble_score",
                "mean backbone\nprobability",
                "mean rotamer\nprobability",
                "mean phi psi\noccurrence",
            ],
            y_labels=["score", "probability", "probability", "probability"],
            out_path=plotpath,
            show_fig=False,
        )

    # pre-filtering to reduce df size
    score_df_top = score_df.nlargest(args.max_top_out, "ensemble_score")

    if args.max_random_out > 0:
        # drop all previously picked paths
        remaining_index = score_df.index.difference(score_df_top.index)
        score_df = score_df.loc[remaining_index]
        if not score_df.empty:
            sample_n = min(args.max_random_out, len(score_df))
            score_df = score_df.sample(n=sample_n, replace=False)
        score_df = pd.concat([score_df, score_df_top])  # combine with top ensembles
    else:
        score_df = score_df_top

    plotpath = os.path.join(working_dir, "pre_filter.png")
    logging.info(f"Plotting selected ensemble results at {plotpath}.")
    violinplot_multiple_cols(
        score_df,
        cols=[
            "ensemble_score",
            "backbone_probability",
            "rotamer_probability",
            "phi_psi_occurrence",
        ],
        titles=[
            "ensemble_score",
            "mean backbone\nprobability",
            "mean rotamer\nprobability",
            "mean phi psi\noccurrence",
        ],
        y_labels=["score", "probability", "probability", "probability"],
        out_path=plotpath,
        show_fig=False,
    )

    post_clash = (
        ensemble_dfs.merge(
            score_df["ensemble_score"], left_on="ensemble_num", right_index=True
        )
        .sort_values("ensemble_num")
        .reset_index(drop=True)
    )

    logging.info(f"Completed clash check in {round(time.time() - init, 0)} s.")
    if len(post_clash.index) == 0:
        logging.info(
            "No ensembles found! Try adjusting VdW multipliers or pick different fragments!"
        )
        raise RuntimeError(
            "No ensembles found! Try adjusting VdW multipliers or pick different fragments!"
        )

    # sort ensembles by score
    logging.info("Sorting ensembles by score...")
    post_clash = sort_dataframe_groups_by_column(
        df=post_clash,
        group_col="ensemble_num",
        sort_col="ensemble_score",
        ascending=False,
    )
    post_clash["ensemble_num"] = (
        post_clash.groupby("ensemble_num", sort=False).ngroup() + 1
    )
    logging.info("Sorting completed.")

    logging.info("Creating paths out of ensembles...")
    post_clash["path_score"] = post_clash["ensemble_score"]
    post_clash["path_num_matches"] = 0

    # filter for top ensembles to speed things up, since paths within an ensemble have the same score
    paths = ["".join(perm) for perm in itertools.permutations(chains)]
    # post_clash = sort_dataframe_groups_by_column(df=post_clash, group_col="ensemble_num", sort_col="path_score", ascending=False)
    dfs = [
        post_clash.assign(path_name=post_clash["ensemble_num"].astype(str) + "_" + p)
        for p in paths
    ]
    path_df = pd.concat(dfs, ignore_index=True)
    logging.info("Done creating paths.")

    pdb_dir = os.path.join(working_dir, "motif_library")
    os.makedirs(pdb_dir, exist_ok=True)

    if args.max_paths_per_ensemble:
        df_list = []
        for _, ensembles in path_df.groupby("ensemble_num", sort=False):
            df = sort_dataframe_groups_by_column(
                ensembles,
                group_col="path_name",
                sort_col="path_score",
                ascending=False,
                filter_top_n=args.max_paths_per_ensemble,
                randomize_ties=True,
            )
            df_list.append(df)
        path_df = pd.concat(df_list)

    # select top n paths
    logging.info(f"Selecting top {args.max_top_out} paths...")
    top_path_df = sort_dataframe_groups_by_column(
        df=path_df,
        group_col="path_name",
        sort_col="path_score",
        ascending=False,
        randomize_ties=True,
        filter_top_n=args.max_top_out,
    )

    logging.info(f"Found {int(len(top_path_df.index) / len(chains))} paths.")

    # select random paths
    if args.max_random_out > 0:
        # remove all selected paths
        logging.info(f"Selecting random {args.max_top_out} paths...")

        path_df = (
            path_df.merge(
                top_path_df[["ensemble_num", "path_name"]],
                on=["ensemble_num", "path_name"],
                how="left",
                indicator=True,
            )
            .query('_merge == "left_only"')
            .drop(columns="_merge")
        )

        path_df["randomizer_score"] = 0
        random_path_df = sort_dataframe_groups_by_column(
            df=path_df,
            group_col="path_name",
            sort_col="randomizer_score",
            ascending=False,
            randomize_ties=True,
            filter_top_n=args.max_random_out,
        )
        logging.info(
            f"Found {int(len(random_path_df.index) / len(chains))} random paths."
        )

        selected_paths = pd.concat([random_path_df, top_path_df])
    else:
        selected_paths = top_path_df

    # create path dataframe
    logging.info("Creating path dataframe...")
    aggregate = {
        "poses": concat_columns,
        "chain_id": concat_columns,
        "model_num": concat_columns,
        "rotamer_pos": concat_columns,
        "frag_length": concat_columns,
        "path_score": "mean",
        "backbone_probability": [
            ("backbone_probability", concat_columns),
            ("backbone_probability_mean", "mean"),
        ],
        "rotamer_probability": [
            ("rotamer_probability", concat_columns),
            ("rotamer_probability_mean", "mean"),
        ],
        "phi_psi_occurrence": [
            ("phi_psi_occurrence", concat_columns),
            ("phi_psi_occurrence_mean", "mean"),
        ],
        "covalent_bonds": concat_columns,
    }

    selected_paths = (
        selected_paths.groupby("path_name", sort=False)
        .agg(aggregate)
        .reset_index(names=["path_name"])
    )
    selected_paths.columns = [
        "path_name",
        "poses",
        "chain_id",
        "model_num",
        "rotamer_pos",
        "frag_length",
        "path_score",
        "backbone_probability",
        "backbone_probability_mean",
        "rotamer_probability",
        "rotamer_probability_mean",
        "phi_psi_occurrence",
        "phi_psi_occurrence_mean",
        "covalent_bonds",
    ]

    # create residue selections
    selected_paths["fixed_residues"] = selected_paths.apply(
        lambda row: split_str_to_dict(row["chain_id"], row["rotamer_pos"], sep=","),
        axis=1,
    )
    selected_paths["fixed_residues"] = selected_paths.apply(
        lambda row: from_dict(row["fixed_residues"]), axis=1
    )
    selected_paths["motif_residues"] = selected_paths.apply(
        lambda row: create_motif_residues(row["chain_id"], row["frag_length"], sep=","),
        axis=1,
    )
    selected_paths["motif_residues"] = selected_paths.apply(
        lambda row: from_dict(row["motif_residues"]), axis=1
    )
    selected_paths["ligand_motif"] = [
        from_dict({"Z": [i + 1 for i, _ in enumerate(ligands)]})
        for id in selected_paths.index
    ]

    selected_paths["path_order"] = selected_paths["path_name"].str.split("_").str[-1]
    selected_paths["motif_contigs"] = selected_paths.apply(
        lambda row: create_motif_contig(
            row["chain_id"], row["frag_length"], row["path_order"], sep=","
        ),
        axis=1,
    )
    selected_paths["channel_contig"] = selected_paths.apply(
        lambda row: create_motif_contig("Q", str(channel_size), "Q", sep=","),
        axis=1,
    )

    # combine multiple ligands into one for rfdiffusion
    ligand = copy.deepcopy(ligands)
    for lig in ligand:
        lig.resname = "LIG"

    lib_path = os.path.join(working_dir, "motif_library")
    logging.info(f"Writing motif library .pdbs to {lib_path}")
    os.makedirs(lib_path, exist_ok=True)
    selected_paths["poses"] = create_pdbs(
        selected_paths,
        lib_path,
        ligand,
        channel_path,
        args.preserve_channel_coordinates,
    )
    selected_paths["input_poses"] = selected_paths["poses"]
    selected_paths["poses_description"] = selected_paths.apply(
        lambda row: description_from_path(row["poses"]), axis=1
    )

    logging.info(f"Writing data to {out_json}")

    ligand_dir = os.path.join(working_dir, "ligand")
    os.makedirs(ligand_dir, exist_ok=True)
    params_paths = []
    ligand_paths = []
    for index, ligand in enumerate(ligands):
        save_structure_to_pdbfile(
            ligand,
            lig_path := os.path.abspath(os.path.join(ligand_dir, f"LG{index + 1}.pdb")),
        )
        lig_name = ligand.get_resname()
        ligand_paths.append(lig_path)
        if len(list(ligand.get_atoms())) > 3:
            # store ligand as .mol file for rosetta .molfile-to-params.py
            logging.info(
                "Running 'molfile_to_params.py' to generate params file for Rosetta."
            )
            lig_molfile = openbabel_fileconverter(
                input_file=lig_path,
                output_file=lig_path.replace(".pdb", ".mol2"),
                input_format="pdb",
                output_format=".mol2",
            )
            cmd = f"{sys.executable} {os.path.join(utils_dir, 'molfile_to_params.py')} -n {lig_name} -p {ligand_dir}/LG{index + 1} {lig_molfile} --keep-names --clobber --chain=Z"
            LocalJobStarter().start(
                cmds=[cmd], jobname="moltoparams", output_path=ligand_dir
            )
            params_paths.append(lig_path.replace(".pdb", ".params"))
        else:
            logging.info(
                f"Ligand at {lig_path} contains less than 4 atoms. No Rosetta Params file can be written for it."
            )

    if params_paths:
        selected_paths["params_path"] = ",".join(params_paths)
    if ligand_paths:
        selected_paths["ligand_path"] = ",".join(ligand_paths)

    # write output json
    selected_paths.to_json(out_json)

    violinplot_multiple_cols(
        selected_paths,
        cols=[
            "backbone_probability_mean",
            "phi_psi_occurrence_mean",
            "rotamer_probability_mean",
        ],
        titles=[
            "mean backbone\nprobability",
            "mean phi/psi\nprobability",
            "mean rotamer\nprobability",
        ],
        y_labels=["probability", "probability", "probability"],
        out_path=os.path.join(working_dir, "selected_paths_info.png"),
        show_fig=False,
    )

    logging.info("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config_json", type=Path)
    args = parser.parse_args()
    arguments = Arguments.model_validate_json(args.config_json.read_text())
    arguments.working_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=(
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                arguments.working_dir
                / f"motif_library_{arguments.theozyme_pdb.stem}.log"
            ),
        ),
    )
    try:
        main(arguments)
    except Exception as e:
        logging.error(e)

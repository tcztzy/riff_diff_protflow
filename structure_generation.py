#imports
import json
import logging
import os
import sys
import copy
import itertools
import shutil
import numpy as np
import pandas as pd

import protflow
import protflow.config
from protflow.jobstarters import SbatchArrayJobstarter
import protflow.poses
import protflow.residues
import protflow.tools
import protflow.tools.colabfold
import protflow.tools.esmfold
import protflow.tools.ligandmpnn
import protflow.tools.attnpacker
import protflow.metrics.rmsd
import protflow.metrics.tmscore
import protflow.tools.protein_edits
import protflow.tools.rfdiffusion
from protflow.poses import Poses
from protflow.residues import ResidueSelection
from protflow.metrics.generic_metric_runner import GenericMetric
from protflow.metrics.ligand import LigandClashes, LigandContacts
from protflow.metrics.rmsd import BackboneRMSD, MotifRMSD, MotifSeparateSuperpositionRMSD
import protflow.tools.rosetta
from protflow.utils.biopython_tools import renumber_pdb_by_residue_mapping, load_structure_from_pdbfile, save_structure_to_pdbfile
import protflow.utils.plotting as plots
from protflow.tools.residue_selectors import DistanceSelector


def write_pymol_alignment_script(df:pd.DataFrame, scoreterm: str, top_n:int, path_to_script: str, ascending=True, use_original_location=False,
                                 ref_motif_col: str = "template_motif", target_motif_col: str = "motif_residues",
                                 ref_catres_col: str = "template_fixedres", target_catres_col: str = "fixed_residues"
                                 ) -> str:
    '''
    Writes .pml script for automated pymol alignment.
    '''
    cmds = []
    for index in df.sort_values(scoreterm, ascending=ascending).head(top_n).index:
        cmd = write_align_cmds(
            input_data=df.loc[index],
            use_original_location=use_original_location,
            ref_motif_col=ref_motif_col,
            target_motif_col=target_motif_col,
            ref_catres_col=ref_catres_col,
            target_catres_col=target_catres_col
        )
        cmds.append(cmd)

    with open(path_to_script, 'w', encoding="UTF-8") as f:
        f.write("\n".join(cmds))
    return path_to_script

def write_align_cmds(input_data: pd.Series, use_original_location=False, ref_motif_col: str = "template_motif", target_motif_col: str = "motif_residues", ref_catres_col: str = "template_fixedres", target_catres_col: str = "fixed_residues"):
    '''AAA'''
    cmds = list()
    if use_original_location: 
        ref_pose = input_data["input_poses"].replace(".pdb", "")
        pose = input_data["esm_location"]
    else: 
        ref_pose = input_data["input_poses"].split("/")[-1].replace(".pdb", "")
        pose = input_data["poses_description"] + ".pdb"

    # load pose and reference
    cmds.append(f"load {pose}")
    ref_pose_name = input_data['poses_description'] + "_ref"
    cmds.append(f"load {ref_pose}.pdb, {ref_pose_name}")

    # basecolor
    cmds.append(f"color violetpurple, {input_data['poses_description']}")
    cmds.append(f"color yelloworange, {ref_pose_name}")

    # select inpaint_motif residues
    cmds.append(f"select temp_motif_res, {write_pymol_motif_selection(input_data['poses_description'], input_data[target_motif_col])}")
    cmds.append(f"select temp_ref_res, {write_pymol_motif_selection(ref_pose_name, input_data[ref_motif_col])}")

    # alignimpose inpaint_motif_residues:
    #cmds.append(f"align temp_ref_res, temp_motif_res")

    # select fixed residues, show sticks and color
    cmds.append(f"select temp_cat_res, {write_pymol_motif_selection(input_data['poses_description'], input_data[target_catres_col])}")
    cmds.append(f"select temp_refcat_res, {write_pymol_motif_selection(ref_pose_name, input_data[ref_catres_col])}")
    cmds.append(f"show sticks, temp_cat_res")
    cmds.append(f"show sticks, temp_refcat_res")
    cmds.append(f"hide sticks, hydrogens")
    cmds.append(f"color atomic, (not elem C)")

    # store scene, delete selection and disable object:
    cmds.append(f"center temp_motif_res")
    cmds.append(f"scene {input_data['poses_description']}, store")
    cmds.append(f"disable {input_data['poses_description']}")
    cmds.append(f"disable {ref_pose_name}")
    cmds.append(f"delete temp_cat_res")
    cmds.append(f"delete temp_refcat_res")
    cmds.append(f"delete temp_motif_res")
    cmds.append(f"delete temp_ref_res")
    return "\n".join(cmds)

def write_pymol_motif_selection(obj: str, motif: dict) -> str:
    '''AAA'''
    if isinstance(motif, ResidueSelection):
        motif = motif.to_dict()

    residues = [f"chain {chain} and resi {'+'.join([str(x) for x in res_ids])}" for chain, res_ids in motif.items()]
    pymol_selection = ' or '.join([f"{obj} and {resis}" for resis in residues])
    return pymol_selection


def update_and_copy_reference_frags(input_df: pd.DataFrame, ref_col:str, desc_col:str, prefix: str, out_pdb_path=None, keep_ligand_chain:str="") -> "list[str]":
    '''Updates reference fragments (input_pdbs) to the motifs that were set during diffusion.'''
    # create residue mappings {old: new} for renaming
    list_of_mappings = [protflow.tools.rfdiffusion.get_residue_mapping(ref_motif, inp_motif) for ref_motif, inp_motif in zip(input_df[f"{prefix}_con_ref_pdb_idx"].to_list(), input_df[f"{prefix}_con_hal_pdb_idx"].to_list())]

    # compile list of output filenames
    output_pdb_names_list = [f"{out_pdb_path}/{desc}.pdb" for desc in input_df[desc_col].to_list()]

    # renumber
    return [renumber_pdb_by_residue_mapping(ref_frag, res_mapping, out_pdb_path=pdb_output, keep_chain=keep_ligand_chain) for ref_frag, res_mapping, pdb_output in zip(input_df[ref_col].to_list(), list_of_mappings, output_pdb_names_list)]

def instantiate_trajectory_plotting(plot_dir, df):
    # instantiate plotting trajectories:
    esm_plddt_traj = plots.PlottingTrajectory(y_label="pLDDT", location=os.path.join(plot_dir, "trajectory_plddt.png"), title="pLDDT Trajectory", dims=(0,100))
    esm_plddt_traj.add_and_plot(df["esm_plddt"], "screening")
    esm_bb_ca_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "trajectory_esm_bb_ca.png"), title="ESMFold BB-Ca\nRMSD Trajectory", dims=(0,5))
    esm_bb_ca_rmsd_traj.add_and_plot(df["esm_backbone_rmsd"], "screening")
    esm_motif_ca_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "trajectory_esm_fixedres_ca.png"), title="ESMFold Fixed Residues\nCa RMSD Trajectory", dims=(0,5))
    esm_motif_ca_rmsd_traj.add_and_plot(df["esm_catres_bb_rmsd"], "screening")
    esm_catres_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "trajectory_esm_catres_rmsd.png"), title="ESMFold Fixed Residues\nSidechain RMSD Trajectory", dims=(0,5))
    esm_catres_rmsd_traj.add_and_plot(df["esm_catres_heavy_rmsd"], "screening")
    fastrelax_total_score_traj = plots.PlottingTrajectory(y_label="Rosetta total score [REU]", location=os.path.join(plot_dir, "trajectory_rosetta_total_score.png"), title="FastRelax Total Score Trajectory")
    postrelax_motif_ca_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "trajectory_postrelax_fixedres_rmsd.png"), title="Postrelax Fixed Residues\nCa RMSD Trajectory", dims=(0,5))
    postrelax_motif_catres_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "trajectory_postrelax_fixedres_catres.png"), title="Postrelax Fixed Residues\nSidechain RMSD Trajectory", dims=(0,5))
    delta_apo_holo_traj = plots.PlottingTrajectory(y_label="Rosetta delta total score [REU]", location=os.path.join(plot_dir, "trajectory_delta_apo_holo.png"), title="Delta Apo Holo Total Score Trajectory")
    postrelax_ligand_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "trajectory_postrelax_ligand_rmsd.png"), title="Postrelax Ligand\nRMSD Trajectory", dims=(0,5))
    sap_score_traj = plots.PlottingTrajectory(y_label="Spatial Aggregation Propensity", location=os.path.join(plot_dir, "trajectory_sap_score.png"), title="SAP Score Trajectory", dims=(0,250))
    return {'esm_plddt': esm_plddt_traj, 'esm_backbone_rmsd': esm_bb_ca_rmsd_traj, 'esm_catres_bb_rmsd': esm_motif_ca_rmsd_traj, 'esm_catres_heavy_rmsd': esm_catres_rmsd_traj, 'fastrelax_total_score': fastrelax_total_score_traj, 'postrelax_catres_heavy_rmsd': postrelax_motif_catres_rmsd_traj, 'postrelax_catres_bb_rmsd': postrelax_motif_ca_rmsd_traj, 'delta_apo_holo': delta_apo_holo_traj, 'postrelax_ligand_rmsd': postrelax_ligand_rmsd_traj, 'fastrelax_sap_score_mean': sap_score_traj}

def update_trajectory_plotting(trajectory_plots:dict, df:pd.DataFrame, cycle:int):
    for traj in trajectory_plots:
        trajectory_plots[traj].add_and_plot(df[f"cycle_{cycle}_{traj}"], f"cycle_{cycle}")
    return trajectory_plots

def add_final_data_to_trajectory_plots(df: pd.DataFrame, trajectory_plots):
    trajectory_plots['esm_plddt'].add_and_plot(df['final_AF2_plddt'], "eval (AF2)")
    trajectory_plots['esm_backbone_rmsd'].add_and_plot(df['final_AF2_backbone_rmsd'], "eval (AF2)")
    trajectory_plots['esm_catres_bb_rmsd'].add_and_plot(df['final_AF2_catres_bb_rmsd'], "eval (AF2)")
    trajectory_plots['esm_catres_heavy_rmsd'].add_and_plot(df['final_AF2_catres_heavy_rmsd'], "eval (AF2)")
    trajectory_plots['fastrelax_total_score'].add_and_plot(df['final_fastrelax_total_score'], "eval (AF2)")
    trajectory_plots['postrelax_catres_heavy_rmsd'].add_and_plot(df['final_postrelax_catres_heavy_rmsd'], "eval (AF2)")
    trajectory_plots['postrelax_catres_bb_rmsd'].add_and_plot(df['final_postrelax_catres_bb_rmsd'], "eval (AF2)")
    trajectory_plots['delta_apo_holo'].add_and_plot(df['final_delta_apo_holo'], "eval (AF2)")
    trajectory_plots['postrelax_ligand_rmsd'].add_and_plot(df['final_postrelax_ligand_rmsd'], "eval (AF2)")
    trajectory_plots['fastrelax_sap_score_mean'].add_and_plot(df['final_fastrelax_sap_score_mean'], "eval (AF2)")
    return trajectory_plots

def create_ref_results_dir(poses, dir:str, cycle:int):
    # plot outputs and write alignment script
    logging.info(f"Creating refinement output directory for refinement cycle {cycle} at {dir}...")
    os.makedirs(dir, exist_ok=True)

    logging.info(f"Plotting outputs of cycle {cycle}.")
    cols = [f"cycle_{cycle}_esm_plddt", f"cycle_{cycle}_esm_backbone_rmsd", f"cycle_{cycle}_esm_catres_heavy_rmsd", f"cycle_{cycle}_fastrelax_total_score", f"cycle_{cycle}_postrelax_ligand_rmsd", f"cycle_{cycle}_fastrelax_sap_score_mean"]
    titles = ["ESMFold pLDDT", "ESMFold BB-Ca RMSD", "ESMFold Sidechain\nRMSD", "Rosetta total_score", "Postrelax ligand RMSD", "Spatial Aggregation Propensity"]
    y_labels = ["pLDDT", "Angstrom", "Angstrom", "[REU]", "Angstrom", "AU"]
    dims = [(0,100), (0,5), (0,5), None, (0,5), None]

    # plot results
    plots.violinplot_multiple_cols(
        dataframe = poses.df,
        cols = cols,
        titles = titles,
        y_labels = y_labels,
        dims = dims,
        out_path = os.path.join(dir, f"cycle_{cycle}_results.png"),
        show_fig = False
    )

    poses.save_poses(out_path=dir)
    poses.save_poses(out_path=dir, poses_col="input_poses")
    poses.save_scores(out_path=dir)

    # write pymol alignment script?
    logging.info(f"Writing pymol alignment script for backbones after refinement cycle {cycle} at {dir}.")
    write_pymol_alignment_script(
        df = poses.df,
        scoreterm = f"cycle_{cycle}_refinement_composite_score",
        top_n = np.min([len(poses.df.index), 25]),
        path_to_script = os.path.join(dir, "align_results.pml"),
        ref_motif_col = "template_fixedres",
        ref_catres_col = "template_fixedres",
        target_catres_col = "fixed_residues",
        target_motif_col = "fixed_residues"
    )

def create_final_results_dir(poses, dir:str):
    # plot outputs and write alignment script

    os.makedirs(dir, exist_ok=True)

    logging.info(f"Plotting final outputs.")
    cols = ["final_AF2_plddt", "final_AF2_mean_plddt", "final_AF2_backbone_rmsd", "final_AF2_catres_heavy_rmsd", "final_fastrelax_total_score", "final_postrelax_catres_heavy_rmsd", "final_postrelax_catres_bb_rmsd", "final_delta_apo_holo", "final_AF2_catres_heavy_rmsd_mean", "final_postrelax_catres_heavy_rmsd_mean", "final_postrelax_ligand_rmsd", "final_postrelax_ligand_rmsd_mean", "final_fastrelax_sap_score_mean"]
    titles = ["AF2 pLDDT", "mean AF2 pLDDT", "AF2 BB-Ca RMSD", "AF2 Sidechain\nRMSD", "Rosetta total_score", "Relaxed Sidechain\nRMSD", "Relaxed BB-Ca RMSD", "Delta Apo Holo", "Mean AF2 Sidechain\nRMSD", "Mean Relaxed Sidechain\nRMSD", "Postrelax Ligand\nRMSD", "Mean Postrelax Ligand\nRMSD", "Spatial Aggregation\nPropensity"]
    y_labels = ["pLDDT", "pLDDT", "Angstrom", "Angstrom", "[REU]", "Angstrom", "Angstrom", "[REU]", "Angstrom", "Angstrom", "Angstrom", "Angstrom", "SAP score"]
    dims = [(0,100), (0,100), (0,5), (0,5), None, (0,5), (0,5), None, (0,5), (0,5), (0,5), (0,5), None]

    # plot results
    plots.violinplot_multiple_cols(
        dataframe = poses.df,
        cols = cols,
        titles = titles,
        y_labels = y_labels,
        dims = dims,
        out_path = os.path.join(dir, f"evaluation_results.png"),
        show_fig = False
    )

    plots.violinplot_multiple_cols(
        dataframe=poses.df,
        cols=["final_AF2_catres_heavy_rmsd_mean", "final_AF2_catres_bb_rmsd_mean", "final_postrelax_catres_heavy_rmsd_mean", "final_postrelax_catres_bb_rmsd_mean", "final_postrelax_ligand_rmsd_mean"],
        y_labels=["Angstrom", "Angstrom", "Angstrom", "Angstrom", "Angstrom"],
        titles=["Mean AF2\nSidechain RMSD", "Mean AF2 catres\nBB RMSD", "Mean Relaxed\nSidechain RMSD", "Mean Relaxed catres\nBB RMSD", "Mean Postrelax\nLigand RMSD"],
        out_path=os.path.join(dir, "evaluation_mean_rmsds.png"),
        show_fig=False
    )

    poses.df.sort_values("final_composite_score", ascending=True, inplace=True)
    poses.df.reset_index(drop=True, inplace=True)
    poses.save_poses(out_path=dir)
    poses.save_poses(out_path=dir, poses_col="input_poses")
    poses.save_scores(out_path=dir)

    # write pymol alignment script?
    logging.info(f"Writing pymol alignment script for backbones after evaluation at {dir}.")
    write_pymol_alignment_script(
        df = poses.df,
        scoreterm = "final_composite_score",
        top_n = np.min([25, len(poses.df.index)]),
        path_to_script = os.path.join(dir, "align_results.pml"),
        ref_motif_col = "template_fixedres",
        ref_catres_col = "template_fixedres",
        target_catres_col = "fixed_residues",
        target_motif_col = "fixed_residues"
    )

def create_variants_results_dir(poses, dir:str):
    # plot outputs and write alignment script

    os.makedirs(dir, exist_ok=True)

    logging.info(f"Plotting final outputs.")
    cols = ["variants_AF2_plddt", "variants_AF2_plddt_mean", "variants_AF2_backbone_rmsd", "variants_AF2_catres_heavy_rmsd", "variants_AF2_fastrelax_total_score", "variants_AF2_postrelax_catres_heavy_rmsd", "variants_AF2_postrelax_catres_bb_rmsd", "variants_AF2_delta_apo_holo", "variants_AF2_catres_heavy_rmsd_mean", "variants_AF2_postrelax_catres_heavy_rmsd_mean", "variants_AF2_postrelax_ligand_rmsd", "variants_AF2_postrelax_ligand_rmsd_mean", "variants_AF2_fastrelax_sap_score_mean"]
    titles = ["AF2 pLDDT", "mean AF2 pLDDT", "AF2 BB-Ca RMSD", "AF2 Sidechain\nRMSD", "Rosetta total_score", "Relaxed Sidechain\nRMSD", "Relaxed BB-Ca RMSD", "Delta Apo Holo", "Mean AF2 Sidechain\nRMSD", "Mean Relaxed Sidechain\nRMSD", "Postrelax Ligand\nRMSD", "Mean Postrelax Ligand\nRMSD", "Spatial Aggregation\nPropensity"]
    y_labels = ["pLDDT", "pLDDT", "Angstrom", "Angstrom", "[REU]", "Angstrom", "Angstrom", "[REU]", "Angstrom", "Angstrom", "Angstrom", "Angstrom", "SAP score"]
    dims = [(0,100), (0,100), (0,5), (0,5), None, (0,5), (0,5), None, (0,5), (0,5), (0,5), (0,5), None]

    # plot results
    plots.violinplot_multiple_cols(
        dataframe = poses.df,
        cols = cols,
        titles = titles,
        y_labels = y_labels,
        dims = dims,
        out_path = os.path.join(dir, f"variants_results.png"),
        show_fig = False
    )

    plots.violinplot_multiple_cols(
        dataframe=poses.df,
        cols=["variants_AF2_catres_heavy_rmsd_mean", "variants_AF2_catres_bb_rmsd_mean", "variants_AF2_postrelax_catres_heavy_rmsd_mean", "variants_AF2_postrelax_catres_bb_rmsd_mean", "variants_AF2_postrelax_ligand_rmsd_mean"],
        y_labels=["Angstrom", "Angstrom", "Angstrom", "Angstrom", "Angstrom"],
        titles=["Mean AF2\nSidechain RMSD", "Mean AF2 catres\nBB RMSD", "Mean Relaxed\nSidechain RMSD", "Mean Relaxed catres\nBB RMSD", "Mean Postrelax\nLigand RMSD"],
        out_path=os.path.join(dir, "variants_mean_rmsds.png"),
        show_fig=False
    )

    poses.save_poses(out_path=dir)
    poses.save_poses(out_path=dir, poses_col="input_poses")

    poses.df.sort_values("variants_AF2_composite_score", inplace=True)
    poses.df.reset_index(drop=True, inplace=True)
    poses.save_scores(out_path=os.path.join(dir, "results_variants.json"))

    # write pymol alignment script?
    logging.info(f"Writing pymol alignment script for backbones after evaluation at {dir}.")
    write_pymol_alignment_script(
        df = poses.df,
        scoreterm = "variants_AF2_composite_score",
        top_n = np.min([25, len(poses.df.index)]),
        path_to_script = os.path.join(dir, "align_results.pml"),
        ref_motif_col = "template_fixedres",
        ref_catres_col = "template_fixedres",
        target_catres_col = "fixed_residues",
        target_motif_col = "fixed_residues"
    )


def write_bbopt_opts(row: pd.Series, cycle: int, total_cycles: int, reference_location_col:str, motif_res_col: str, cat_res_col: str, ligand_chain: str) -> str:
    return f"-in:file:native {row[reference_location_col]} -parser:script_vars motif_res={row[motif_res_col].to_string(ordering='rosetta')} cat_res={row[cat_res_col].to_string(ordering='rosetta')} substrate_chain={ligand_chain} sd={0.8 - (0.4 * cycle/total_cycles)}"

def calculate_mean_scores(poses: Poses, scores: list, remove_layers: int=None):
    for score in scores:
        poses.calculate_mean_score(name=f"{score}_mean", score_col=score, remove_layers=remove_layers)
    return poses

def combine_screening_results(dir: str, prefixes: list, scores: list, weights: list, residue_cols: list, input_motifs: Poses):
    if len(prefixes) == 0:
        logging.error("No poses passed in any of the screening runs. Aborting!"); sys.exit(1)
    
    # set up output dir
    out_dir = os.path.join(dir, 'screening_results')
    os.makedirs(out_dir, exist_ok=True)

    # combine all screening outputs into new poses
    pose_df = []
    for prefix in prefixes:
        df = pd.read_json(os.path.join(dir, "screening", prefix, f"{prefix}_scores.json"))
        df['screen_passed_poses'] = len(df.index)
        pose_df.append(df)
    pose_df = pd.concat(pose_df).reset_index(drop=True)
    poses = Poses(poses=pose_df, work_dir=dir)

    # recalculate composite score over all screening runs
    poses.calculate_composite_score(
        name="design_composite_score",
        scoreterms=["esm_plddt", "esm_tm_TM_score_ref", "esm_catres_bb_rmsd", "esm_catres_heavy_rmsd", "esm_lig_contacts", "esm_ligand_clashes"],
        weights=[-1, -1, 4, 3, -0.5, 0.5],
        plot=True
    )

    # convert columns to residues (else, pymol script writer and refinement crash)
    for residue_col in residue_cols:
        poses.df[residue_col] = [ResidueSelection(motif, from_scorefile=True) for motif in poses.df[residue_col].to_list()]
    # calculate screening composite score
    poses.calculate_composite_score(name='screening_composite_score', scoreterms=scores, weights=weights, plot=True)

    poses.reindex_poses(prefix="reindexed_screening_poses", remove_layers=1, force_reindex=1)

    grouped_df = poses.df.groupby('screen', sort=True)
    # plot all scores
    df_names, dfs = zip(*[(name, df) for name, df in grouped_df])
    plot_scores = scores + ['design_composite_score', 'screen_decentralize_weight', 'screen_decentralize_distance', 'screen', 'screening_composite_score']
    for score in plot_scores:
        plots.violinplot_multiple_cols_dfs(dfs=dfs, df_names=df_names, cols=[score], y_labels=[score], out_path=os.path.join(out_dir, f'{score}_violin.png'), show_fig=False)

    # save poses dataframe as well
    poses.df.sort_values("design_composite_score", ascending=True, inplace=True)
    poses.df.reset_index(drop=True, inplace=True)
    poses.save_scores(out_path=os.path.join(out_dir, 'screening_results_all.json'))

    logging.info(f"Writing pymol alignment script for screening results at {out_dir}")
    write_pymol_alignment_script(
        df=poses.df,
        scoreterm = "design_composite_score",
        top_n = np.min([len(poses.df.index), 25]),
        path_to_script = os.path.join(out_dir, "align_results.pml"),
        ref_motif_col = "template_fixedres",
        ref_catres_col = "template_fixedres",
        target_catres_col = "fixed_residues",
        target_motif_col = "fixed_residues"
    )

    poses.save_poses(out_path=out_dir)
    poses.save_poses(out_path=out_dir, poses_col="input_poses")

    # save successful input motifs
    counts = poses.df["input_poses"].value_counts()
    poses.df["successful_screening_results"] = poses.df["input_poses"].map(counts)
    unique = poses.df.drop_duplicates(subset=["input_poses"], keep="first")
    successfull_motifs = input_motifs.df.merge(unique[['input_poses', 'successful_screening_results']], on="input_poses")
    successfull_motifs.sort_values("successful_screening_results", ascending=False, inplace=True)
    successfull_motifs.reset_index(drop=True, inplace=True)
    successfull_motifs.to_json(os.path.join(out_dir, "successful_input_motifs.json"))

    return poses
    
def ramp_cutoff(start_value, end_value, cycle, total_cycles):
    if total_cycles == 1:
        return end_value
    step = (end_value - start_value) / (total_cycles - 1)
    return start_value + (cycle - 1) * step
        

def split_combined_ligand(poses_dir:str, ligand_paths: list):
    logging.info('Replacing RFdiffusion combined ligand with separated ligands...')
    new_ligands = [[res for res in load_structure_from_pdbfile(new_lig).get_residues()][0] for new_lig in ligand_paths]
    poses = Poses(poses=poses_dir, glob_suffix="*.pdb")
    for pose_path in poses.df['poses'].to_list():
        pose = load_structure_from_pdbfile(pose_path)
        original_ligands = [lig for lig in pose["Z"].get_residues()]
        for old_lig in original_ligands:
            pose["Z"].detach_child(old_lig.id)
        for new_lig in new_ligands:
            pose["Z"].add(new_lig)
        save_structure_to_pdbfile(pose, pose_path)

def create_reduced_motif(fixed_res:ResidueSelection, motif_res:ResidueSelection):
    reduced_dict = {}
    fixed_res = fixed_res.to_dict()
    motif_res = motif_res.to_dict()
    for chain in fixed_res:
        res = []
        reduced_motif = []
        for residue in fixed_res[chain]:
            res.append(residue -1)
            res.append(residue)
            res.append(residue + 1)
        for i in res:
            if i in motif_res[chain]:
                reduced_motif.append(i)
        reduced_dict[chain] = reduced_motif
    return protflow.residues.from_dict(reduced_dict)

def aa_one_letter_code() -> str:
    return "ARDNCEQGHILKMFPSTWYV"

def omit_AAs(omitted_aas:str, allowed_aas:str, dir:str, name:str) -> str:
    mutations_dict = {}
    if not allowed_aas and not omitted_aas:
        return None
    if isinstance(omitted_aas, str):
        omitted_aas = omitted_aas.split(";")
        for mutation in omitted_aas:
            position, omitted_aas = mutation.split(":")
            mutations_dict[position.strip()] = omitted_aas.strip()
    if isinstance(allowed_aas, str):
        allowed_aas = allowed_aas.split(";")
        for mutation in allowed_aas:
            position, allowed_aas = mutation.split(":")
            all_aas = aa_one_letter_code()
            for aa in allowed_aas:
                all_aas = all_aas.replace(aa.upper(), "")
            mutations_dict[position.strip()] = all_aas.upper().strip()
    filename = os.path.join(dir, f"{name}_mutations.json")
    with open(filename, "w") as out:
        json.dump(mutations_dict, out)
    return f"--omit_AA_per_residue {filename}" 

def log_cmd(arguments):
    cmd = ''
    for key, value in vars(arguments).items():
        cmd += f'--{key} {value} '
    cmd = f'{sys.argv[0]} {cmd}'
    logging.info(f"{sys.argv[0]} {cmd}")

def determine_last_ref_cycle(df: pd.DataFrame) -> int:
    cols = [col for col in df.columns if col.startswith("cycle_")]
    nums = [int(col.split("_")[1]) for col in cols]
    last_cycle = max(nums)
    return last_cycle

def write_rfdiffusion_contig(frag_contigs:str, total_length:int, flanker_length:int, frag_sizes:str, flanker_type:str, channel_contig:str=None, sep=","):
    frag_contigs = [frag for frag in frag_contigs.split(sep)]
    frag_total_length = sum([int(size) for size in frag_sizes.split(sep)])
    num_linkers = len(frag_contigs) - 1
    linker_length = 5 + int((total_length - flanker_length - frag_total_length) / num_linkers)
    frag_contigs = f"/10-{linker_length}/".join(frag_contigs)
    if flanker_type == "split":
        contig = f"{flanker_length}/{frag_contigs}/{flanker_length}"
    elif flanker_type == "cterm":
        contig = f"{frag_contigs}/{flanker_length}"
    elif flanker_type == "nterm":
        contig = f"{flanker_length}/{frag_contigs}"
    else:
        logging.error(f"<flanker_type> must be one of 'split', 'cterm' or 'nterm', not {flanker_type}!")
        raise KeyError(f"<flanker_type> must be one of 'split', 'cterm' or 'nterm', not {flanker_type}!")
    if channel_contig:
        contig = f"{channel_contig}/0 {contig}"
    return f"'contigmap.contigs=[{contig}]' "

def write_rfdiffusion_inpaint_seq(motif_residues:ResidueSelection, fixed_residues:ResidueSelection, sep=","):
    inpaint_seq = motif_residues - fixed_residues
    inpaint_seq = inpaint_seq.to_string(delim="/")
    return f"'contigmap.inpaint_seq=[{inpaint_seq}]'"

def add_covalent_bonds_info(poses:Poses, prefix:str, covalent_bonds_col:str) -> Poses:
    if poses.df[covalent_bonds_col].isna().all():
        return poses
    covalent_bonds = poses.df[covalent_bonds_col].to_list()
    os.makedirs(out_dir := os.path.join(poses.work_dir, prefix), exist_ok=True)
    paths = []
    for pose, cov_bonds in zip(poses.poses_list(), covalent_bonds):
        links = "\n".join([parse_link_from_cov_bond(cov_bond) for cov_bond in cov_bonds.split(",")]) + "\n"
        if not os.path.isfile(outfile := os.path.join(out_dir, os.path.basename(pose))):
            prefix_string_to_file(pose, links, outfile)
        paths.append(outfile)
    poses.df["poses"] = paths
    return poses

def prefix_string_to_file(file_path: str, prefix: str, save_path:str=None):
    '''Adds something to the beginning of a file.'''
    with open(file_path, 'r') as f: file_str = f.read()
    with open(save_path or file_path, 'w') as f: f.write(prefix + file_str)

def parse_link_from_cov_bond(covalent_bond: str) -> str:
    '''parses covalent bond into Rosetta formated link record for PDB headers.'''
    res_data, lig_data = covalent_bond.split(':')
    res_data, res_atom = res_data.split('-')
    lig_data, lig_atom = lig_data.split('-')
    res_data, res_id = res_data.split("_")
    lig_data, lig_id = lig_data.split("_")
    res_chain = res_data[-1]
    res_num = res_data[:-1]
    lig_chain = lig_data[-1]
    lig_num = lig_data[:-1]
    return f"LINK         {res_atom:<3} {res_id:<3} {res_chain:>1}{res_num:>4}                {lig_atom:<3}  {lig_id:>3} {lig_chain:>1}{lig_num:>4}                  0.00"

def update_covalent_bonds_info(bonds:str, original_fixedres:ResidueSelection, updated_fixedres:ResidueSelection) -> str:
    if not bonds:
        return None
    original = original_fixedres.to_list(ordering="rosetta")
    new = updated_fixedres.to_list(ordering="rosetta")
    resnum = bonds.split("_")[0]
    idx = original.index(resnum)
    new_resnum = new[idx]
    covalent_bond = "_".join([new_resnum] + bonds.split("_")[1:])
    return covalent_bond

def main(args):
    '''executes everyting (duh)'''
    ################################################# SET UP #########################################################
    # logging and checking of inputs
    if not os.path.isdir(args.working_dir):
        os.makedirs(args.working_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.working_dir, "riffdiff.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    log_cmd(args)

    if not sum(bool(v) for v in [args.screen_input_json, args.ref_input_json, args.eval_input_json, args.variants_input_json]) == 1:
        raise ValueError("One (but not more) of :screen_input_json:, :ref_input_json:, :eval_input_json: and :variants_input_json: must be set!")

    if args.screen_input_json:
        backbones = protflow.poses.load_poses(args.screen_input_json)
        backbones.df["template_motif"] = backbones.df["motif_residues"]
        backbones.df["template_fixedres"] = backbones.df["fixed_residues"]
    elif args.ref_input_json:
        backbones = protflow.poses.load_poses(args.ref_input_json)
    elif args.eval_input_json:
        backbones = protflow.poses.load_poses(args.eval_input_json)
    elif args.variants_input_json:
        backbones = protflow.poses.load_poses(args.variants_input_json)

    # create residue selections
    residue_cols = ["fixed_residues", "motif_residues", "template_motif", "template_fixedres", "ligand_motif"]
    for res_col in residue_cols:
        backbones.df[res_col] = [ResidueSelection(motif, from_scorefile=True) for motif in backbones.df[res_col].to_list()]

    # check params files
    if args.params_files:
        params = args.params.split(",")
    elif "params_path" in backbones.df.columns:
        params = backbones.df["params_path"].unique()
        if len(params) > 1:
            logging.error("Poses with different params files found! Did you mix up input poses?")
            raise RuntimeError("Poses with different params files found! Did you mix up input poses?")
        params = params[0].split(",")
    else:
        params = None
    
    # check ligands
    if "ligand_path" in backbones.df.columns:
        ligand_paths = backbones.df["ligand_path"].unique()
        if len(ligand_paths) > 1:
            logging.error("Poses with different ligand paths found! Did you mix up input poses?")
            raise RuntimeError("Poses with different ligand paths found! Did you mix up input poses?")
        ligand_paths = ligand_paths[0].split(",")
    else:
        ligand_paths = None
        
    # setup jobstarters
    cpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_cpus)
    small_cpu_jobstarter = SbatchArrayJobstarter(max_cores=10)
    gpu_jobstarter = cpu_jobstarter if args.cpu_only else SbatchArrayJobstarter(max_cores=args.max_gpus, gpus=1)
    real_gpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_gpus, gpus=1)

    # set up runners
    logging.info(f"Settung up runners.")
    rfdiffusion = protflow.tools.rfdiffusion.RFdiffusion(jobstarter = gpu_jobstarter)
    chain_adder = protflow.tools.protein_edits.ChainAdder(jobstarter = small_cpu_jobstarter)
    chain_remover = protflow.tools.protein_edits.ChainRemover(jobstarter = small_cpu_jobstarter)
    bb_rmsd = BackboneRMSD(chains="A", jobstarter = small_cpu_jobstarter)
    catres_motif_bb_rmsd = MotifRMSD(ref_col = "updated_reference_frags_location", target_motif = "fixed_residues", ref_motif = "fixed_residues", atoms=["N", "CA", "C"], jobstarter=small_cpu_jobstarter)
    catres_motif_heavy_rmsd = MotifRMSD(ref_col = "updated_reference_frags_location", target_motif = "fixed_residues", ref_motif = "fixed_residues", jobstarter=small_cpu_jobstarter)
    ligand_clash = LigandClashes(ligand_chain="Z", factor=args.ligand_clash_factor, atoms=['N', 'CA', 'C', 'O'], jobstarter=small_cpu_jobstarter)
    ligand_contacts = LigandContacts(ligand_chain="Z", min_dist=0, max_dist=8, atoms=['CA'], jobstarter=small_cpu_jobstarter)
    rog_calculator = GenericMetric(module="protflow.utils.metrics", function="calc_rog_of_pdb", jobstarter=small_cpu_jobstarter)
    tm_score_calculator = protflow.metrics.tmscore.TMalign(jobstarter = small_cpu_jobstarter)
    ligand_mpnn = protflow.tools.ligandmpnn.LigandMPNN(jobstarter = gpu_jobstarter)
    rosetta = protflow.tools.rosetta.Rosetta(jobstarter = cpu_jobstarter, fail_on_missing_output_poses=True)
    esmfold = protflow.tools.esmfold.ESMFold(jobstarter = real_gpu_jobstarter)
    ligand_rmsd = MotifSeparateSuperpositionRMSD(
        ref_col="updated_reference_frags_location",
        super_target_motif="fixed_residues",
        super_ref_motif="fixed_residues",
        super_atoms=['N', 'CA', 'C'],
        rmsd_target_motif="ligand_motif",
        rmsd_ref_motif="ligand_motif",
        rmsd_atoms=None,
        rmsd_include_het_atoms=True,
        jobstarter = small_cpu_jobstarter)
    colabfold = protflow.tools.colabfold.Colabfold(jobstarter=real_gpu_jobstarter)
    if args.attnpacker_repack:
        attnpacker = protflow.tools.attnpacker.AttnPacker(jobstarter=gpu_jobstarter)


    # set up general ligandmpnn options
    ligandmpnn_options = f"--ligand_mpnn_use_side_chain_context 1 {args.ligandmpnn_options if args.ligandmpnn_options else ""}"

    # set up general rosetta options

    bb_opt_options = f"-parser:protocol {os.path.abspath(os.path.join(args.riff_diff_dir, 'utils', "fr_constrained.xml"))} -beta -ignore_zero_occupancy false"
    fr_options = f"-parser:protocol {os.path.abspath(os.path.join(protflow.config.AUXILIARY_RUNNER_SCRIPTS_DIR, 'fastrelax_sap.xml'))} -beta -ignore_zero_occupancy false"
    if params:
        fr_options = fr_options + f" -extra_res_fa {' '.join(params)}"
        bb_opt_options = bb_opt_options + f" -extra_res_fa {' '.join(params)}"

    ############################################## SCREENING ######################################################


    if args.screen_input_json:
        screen_prefix = f"{args.screen_prefix}_" if args.screen_prefix else ""
        screening_dir = os.path.join(args.working_dir, f"{screen_prefix}screening")
        os.makedirs(screening_dir, exist_ok=True)
        # load poses
        input_poses_path = os.path.join(screening_dir, 'screening_input_poses', 'screening_input_poses.json')
        backbones.set_work_dir(screening_dir)
        if os.path.isfile(input_poses_path):
            logging.info(f"Reading in previously selected poses from {input_poses_path}")
            previous_selection = Poses(poses=input_poses_path)
            backbones.df = previous_selection.df.loc[:, ["poses_description"]].merge(backbones.df, on="poses_description")
        else:
            if args.screen_input_selection == "weighted":
                logging.info(f"Selecting screening input poses randomly weighted by path score.")
                backbones.df = backbones.df.sample(n=min(args.screen_input_poses, len(backbones.df.index)), weights=backbones.df['path_score'])
            elif args.screen_input_selection == "top":
                logging.info(f"Selecting screening input poses according to path score.")
                backbones.filter_poses_by_rank(n=min(args.screen_input_poses, len(backbones.df.index)), score_col='path_score', ascending=False, prefix='screening_input', plot=True)
            elif args.screen_input_selection == "random":
                logging.info(f"Selecting screening input poses randomly.")
                backbones.df = backbones.df.sample(n=min(args.screen_input_poses, len(backbones.df.index)))
            else:
                logging.error(f"<screen_input_selection> must be one of 'weighted', 'top' or 'random', not {args.screen_input_selection}!")
                raise KeyError(f"<screen_input_selection> must be one of 'weighted', 'top' or 'random', not {args.screen_input_selection}!")
            backbones.save_poses(os.path.join(screening_dir, 'screening_input_poses'))
            backbones.save_scores(input_poses_path)
        
        logging.info(f"Selected {len(backbones.df.index)} poses as input for screening.")

        # save input backbones for later
        starting_motifs = copy.deepcopy(backbones)

        # save run name in df, makes it easier to identify where poses come from when merging results with other runs
        backbones.df["run_name"] = os.path.basename(screening_dir)

        ############################################## RFDiffusion ######################################################
    
        # setup rfdiffusion options:
        backbones.df["rfdiffusion_contigs"] = backbones.df.apply(lambda row: write_rfdiffusion_contig(row['motif_contigs'], args.total_length, args.flanker_length, row["frag_length"], args.flanking, row["channel_contig"] if "channel_contig" in backbones.df.columns else None, sep=","), axis=1)
        backbones.df["rfdiffusion_inpaint_seq"] = backbones.df.apply(lambda row: write_rfdiffusion_inpaint_seq(row['motif_residues'], row['fixed_residues'], sep=","), axis=1)
        backbones.df["rfdiffusion_pose_opts"] = backbones.df["rfdiffusion_contigs"] + backbones.df["rfdiffusion_inpaint_seq"]

        if args.recenter:
            logging.info(f"Parameter --recenter specified. Setting direction for custom recentering during diffusion towards {args.recenter}")
            if len(args.recenter.split(";")) != 3:
                raise ValueError(f"--recenter needs to be semicolon separated coordinates. E.g. --recenter=31.123;-12.123;-0.342")
            recenter = f",recenter_xyz:{args.recenter}"
        else:
            recenter = ""

        input_backbones = copy.deepcopy(backbones)
        decentralize_weights = args.screen_decentralize_weights.split(',')
        decentralize_distances = args.screen_decentralize_distances.split(',')
        settings = tuple(itertools.product(decentralize_weights, decentralize_distances))
        prefixes = [f"screen_{i+1}" for i, s in enumerate(settings)]

        for prefix, setting in zip(prefixes, settings):
            logging.info(f"Running {prefix} with settings: 'decentralize_weight: {setting[0]}, decentralize_distance: {setting[1]}")
            backbones = copy.deepcopy(input_backbones)
            backbones.df['screen_decentralize_weight'] = float(setting[0])
            backbones.df['screen_decentralize_distance'] = float(setting[1])

            # save screen number in poses
            backbones.df['screen'] = int(prefix.split('_')[1])

            # set different directory for each screen
            backbones.set_work_dir(os.path.join(screening_dir, prefix))

            # run diffusion
            diffusion_options = f"diffuser.T=50 potentials.guide_scale=5 potentials.guiding_potentials=[\\'type:substrate_contacts,weight:0\\',\\'type:custom_recenter_ROG,weight:{setting[0]},rog_weight:0,distance:{setting[1]}{recenter}\\'] potentials.guide_decay=quadratic contigmap.length={args.total_length}-{args.total_length} potentials.substrate=LIG"
            backbones = rfdiffusion.run(
                poses=backbones,
                prefix="rfdiffusion",
                num_diffusions=args.screen_num_rfdiffusions,
                options=diffusion_options,
                pose_options=backbones.df["rfdiffusion_pose_opts"].to_list(),
                update_motifs=["fixed_residues", "motif_residues"],
                fail_on_missing_output_poses=False
            )

            # remove channel chain (chain B)
            logging.info(f"Diffusion completed, removing channel chain from diffusion outputs.")
            if "channel_contig" in backbones.df.columns:
                chain_remover.run(
                    poses = backbones,
                    prefix = "channel_removed",
                    chains = "B"
                )
            else:
                backbones.df["channel_removed_location"] = backbones.df["rfdiffusion_location"]

            # create updated reference frags:
            if not os.path.isdir((updated_ref_frags_dir := os.path.join(backbones.work_dir, "updated_reference_frags"))):
                os.makedirs(updated_ref_frags_dir)

            logging.info(f"Channel chain removed, now renumbering reference fragments.")
            backbones.df["updated_reference_frags_location"] = update_and_copy_reference_frags(
                input_df = backbones.df,
                ref_col = "input_poses",
                desc_col = "poses_description",
                prefix = "rfdiffusion",
                out_pdb_path = updated_ref_frags_dir,
                keep_ligand_chain = "Z"
            )

            # replace combined ligand with original ligands (do it for single ligands as well, to change the ligand name from LIG to original name (important for covalent bonds))
            split_combined_ligand(updated_ref_frags_dir, ligand_paths)

            # update covalent bonds info
            backbones.df["covalent_bonds"] = backbones.df.apply(lambda row: update_covalent_bonds_info(row['covalent_bonds'], row["template_fixedres"], row["fixed_residues"]), axis=1)

            # calculate ROG after RFDiffusion, when channel chain is already removed:
            logging.info(f"Calculating rfdiffusion_rog and rfdiffusion_catres_rmsd")
            backbones = rog_calculator.run(poses=backbones, prefix="rfdiffusion_rog")
            backbones = catres_motif_bb_rmsd.run(
                poses = backbones,
                prefix = "rfdiffusion_catres",
            )

            # add back the ligand:
            logging.info(f"Metrics calculated, now adding Ligand chain back into backbones.")
            chain_adder.superimpose_add_chain(
                poses = backbones,
                prefix = "post_rfdiffusion_ligand",
                ref_col = "updated_reference_frags_location",
                target_motif = "fixed_residues",
                copy_chain = "Z"
            )

            # calculate ligand stats
            logging.info(f"Calculating Ligand Statistics")
            backbones = ligand_clash.run(poses=backbones, prefix="rfdiffusion_ligand")
            backbones = ligand_contacts.run(poses=backbones, prefix="rfdiffusion_lig")

            # plot rfdiffusion_stats
            results_dir = os.path.join(backbones.work_dir, "results")
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            plots.violinplot_multiple_cols(
                dataframe = backbones.df,
                cols = ["rfdiffusion_catres_rmsd", "rfdiffusion_rog_data", "rfdiffusion_lig_contacts", "rfdiffusion_ligand_clashes", "rfdiffusion_plddt"],
                titles = ["Template RMSD", "ROG", "Ligand Contacts", "Ligand Clashes", "RFdiffusion pLDDT"],
                y_labels = ["RMSD [\u00C5]", "ROG [\u00C5]", "#CA", "#Clashes", "pLDDT"],
                dims = [(0,5), None, None, None, (0.8,1)],
                out_path = os.path.join(results_dir, "rfdiffusion_statistics.png"),
                show_fig = False
            )

            if args.rfdiffusion_max_clashes:
                backbones.filter_poses_by_value(score_col="rfdiffusion_ligand_clashes", value=args.rfdiffusion_max_clashes, operator="<=", prefix="rfdiffusion_ligand_clashes", plot=True)
            if args.rfdiffusion_max_rog:
                backbones.filter_poses_by_value(score_col="rfdiffusion_rog_data", value=args.rfdiffusion_max_rog, operator="<=", prefix="rfdiffusion_rog", plot=True)
            if args.rfdiffusion_min_ligand_contacts:
                backbones.filter_poses_by_value(score_col="rfdiffusion_lig_contacts", value=args.rfdiffusion_min_ligand_contacts, operator=">=", prefix="rfdiffusion_lig_contacts", plot=True)
            
            if len(backbones.df) == 0:
                logging.warning(f"No poses passed RFdiffusion filtering steps during {prefix}")
                prefixes.remove(prefix)
                continue

            ############################################# SEQUENCE DESIGN AND ESMFOLD ########################################################
            # run LigandMPNN
            if args.screen_skip_mpnn_rlx_mpnn:
                logging.info(f"Running LigandMPNN on {len(backbones)} poses. Designing {args.screen_num_mpnn_sequences} sequences per pose.")
                backbones = ligand_mpnn.run(
                    poses = backbones,
                    prefix = "postdiffusion_ligandmpnn",
                    nseq = args.screen_num_mpnn_sequences,
                    options = ligandmpnn_options,
                    model_type = "ligand_mpnn",
                    fixed_res_col = "fixed_residues",
                    return_seq_threaded_pdbs_as_pose= False
            )

            else:
                backbones = ligand_mpnn.run(
                    poses = backbones,
                    prefix = "postdiffusion_ligandmpnn",
                    nseq = args.screen_num_seq_thread_sequences,
                    options = ligandmpnn_options,
                    model_type = "ligand_mpnn",
                    fixed_res_col = "fixed_residues",
                    return_seq_threaded_pdbs_as_pose = True
            )
                # add covalent bonds info to poses pre-relax
                backbones = add_covalent_bonds_info(poses=backbones, prefix="bbopt_cov_info", covalent_bonds_col="covalent_bonds")

                # optimize backbones 
                backbones.df[f'screen_bbopt_opts'] = [write_bbopt_opts(row=row, cycle=1, total_cycles=5, reference_location_col="updated_reference_frags_location", cat_res_col="fixed_residues", motif_res_col="motif_residues", ligand_chain="Z") for _, row in backbones.df.iterrows()]
                backbones = rosetta.run(
                    poses = backbones,
                    prefix = "bbopt",
                    rosetta_application="rosetta_scripts.default.linuxgccrelease",
                    nstruct = 1,
                    options = bb_opt_options,
                    pose_options='screen_bbopt_opts'
                )

                # filter backbones down to starting backbones
                backbones.filter_poses_by_rank(n=3, score_col=f"bbopt_total_score", remove_layers=2)

                # run ligandmpnn on relaxed poses
                backbones = ligand_mpnn.run(
                    poses = backbones,
                    prefix = "mpnn",
                    nseq = args.screen_num_mpnn_sequences,
                    model_type = "ligand_mpnn",
                    options = ligandmpnn_options,
                    fixed_res_col = "fixed_residues",
                )

                # filter backbones down to starting backbones
                backbones.filter_poses_by_rank(n=args.screen_num_mpnn_sequences, score_col=f"mpnn_overall_confidence", ascending=False, remove_layers=3)

            # predict with ESMFold
            logging.info(f"LigandMPNN finished, now predicting {len(backbones)} sequences using ESMFold.")
            backbones = esmfold.run(
                poses = backbones,
                prefix = "esm"
            )

            ################################################ METRICS ################################################################


            # calculate ROG
            backbones = rog_calculator.run(poses=backbones, prefix="esm_rog")

            # calculate RMSDs (backbone, motif, fixedres)
            logging.info(f"Prediction of {len(backbones.df.index)} sequences completed. Calculating RMSDs to rfdiffusion backbone and reference fragment.")
            backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = "esm_catres_bb")
            backbones = bb_rmsd.run(poses = backbones, ref_col="rfdiffusion_location", prefix = "esm_backbone")
            backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = "esm_catres_heavy")

            # calculate TM-Score and get sc-tm score:
            tm_score_calculator.run(
                poses = backbones,
                prefix = "esm_tm",
                ref_col = "channel_removed_location",
            )

            # filter poses:
            backbones.filter_poses_by_value(score_col="esm_plddt", value=75, operator=">=")
            backbones.filter_poses_by_value(score_col="esm_tm_TM_score_ref", value=0.8, operator=">=")
            backbones.filter_poses_by_value(score_col="esm_catres_bb_rmsd", value=1.5, operator="<=")

            ############################################# BACKBONE FILTER ########################################################

            # add back ligand and determine pocket-ness!
            logging.info(f"Adding Ligand back into the structure for ligand-based pocket prediction.")
            chain_adder.superimpose_add_chain(
                poses = backbones,
                prefix = "post_prediction_ligand",
                ref_col = "updated_reference_frags_location",
                target_motif = "fixed_residues",
                copy_chain = "Z"
            )

            backbones = ligand_clash.run(poses=backbones, prefix="esm_ligand")
            backbones = ligand_contacts.run(poses=backbones, prefix="esm_lig")



            # calculate multi-scorerterm score for the final backbone filter:
            backbones.calculate_composite_score(
                name="design_composite_score",
                scoreterms=["esm_plddt", "esm_tm_TM_score_ref", "esm_catres_bb_rmsd", "esm_catres_heavy_rmsd", "esm_lig_contacts", "esm_ligand_clashes"],
                weights=[-1, -1, 4, 3, -0.5, 0.5],
                plot=True
            )

            # filter down to rfdiffusion backbones
            backbones.filter_poses_by_rank(
                n=1,
                score_col="design_composite_score",
                prefix=f"{prefix}_backbone_filter",
                plot=True,
                remove_layers=1 if args.screen_skip_mpnn_rlx_mpnn else 3
            )

            # plot outputs
            logging.info(f"Plotting outputs.")
            cols = ["rfdiffusion_catres_rmsd", "esm_plddt", "esm_backbone_rmsd", "esm_catres_heavy_rmsd", "esm_tm_sc_tm", "esm_rog_data", "esm_lig_contacts", "esm_ligand_clashes"]
            titles = ["RFDiffusion Motif\nBackbone RMSD", "ESMFold pLDDT", "ESMFold BB-Ca RMSD", "ESMFold Sidechain\nRMSD", "SC-TM Score", "Radius of Gyration", "Ligand Contacts", "Ligand Clashes"]
            y_labels = ["Angstrom", "pLDDT", "Angstrom", "Angstrom", "TM Score", "Angstrom", "#", "#"]
            dims = [(0,8), (0,100), (0,8), (0,8), (0,1), None, None, None]

            # plot results
            plots.violinplot_multiple_cols(
                dataframe = backbones.df,
                cols = cols,
                titles = titles,
                y_labels = y_labels,
                dims = dims,
                out_path = os.path.join(results_dir, "design_results.png"),
                show_fig = False
            )

            #logging.info(f"Fraction of RFdiffusion design-successful backbones that contain active-site pocket: {pocket_containing_fraction}")
            backbones.df.sort_values("design_composite_score", ascending=True, inplace=True)
            backbones.df.reset_index(drop=True, inplace=True)
            backbones.reindex_poses(prefix="reindex", remove_layers=2 if args.screen_skip_mpnn_rlx_mpnn else 4, force_reindex=True)

            # copy filtered poses to new location
            backbones.save_poses(out_path=results_dir)
            backbones.save_poses(out_path=results_dir, poses_col="input_poses")
            backbones.save_scores(out_path=results_dir)
            backbones.save_scores()

            # write pymol alignment script?
            logging.info(f"Created results/ folder and writing pymol alignment script for best backbones at {results_dir}")
            write_pymol_alignment_script(
                df=backbones.df,
                scoreterm="design_composite_score",
                top_n=np.min([len(backbones), 25]),
                path_to_script=os.path.join(results_dir, "align_results.pml"),
                ref_motif_col = "template_fixedres",
                ref_catres_col = "template_fixedres",
                target_catres_col = "fixed_residues",
                target_motif_col = "fixed_residues"
            )

            # write successfull motifs to file, so that they can be read in again
            backbones.df.drop_duplicates(subset=["input_poses"], keep="first", inplace=True)
            successfull_motifs = starting_motifs.df.merge(backbones.df['input_poses'], on="input_poses")
            successfull_motifs.reset_index(drop=True, inplace=True)
            successfull_motifs.to_json(os.path.join(results_dir, "successful_input_motifs.json"))

        scores = ["esm_plddt", "esm_tm_TM_score_ref", "esm_catres_bb_rmsd", "esm_catres_heavy_rmsd", "esm_rog_data", "esm_lig_contacts", "esm_ligand_clashes", "screen_passed_poses"]
        weights = [-1, -1, 4, 3, 1, -1, -1, 1]
        backbones = combine_screening_results(dir=args.working_dir, prefixes=prefixes, scores=scores, weights=weights, residue_cols=residue_cols, input_motifs=starting_motifs)
        backbones.set_work_dir(args.working_dir)
        backbones.save_scores()

        if args.skip_refinement:
            logging.info(f"Skipping refinement. Run concluded, output can be found in {results_dir}")
            sys.exit(1)
        else:
            args.ref_input_json = backbones.scorefile
    
    ############################################# REFINEMENT ########################################################
    if args.ref_input_json:
        ref_prefix = f"{args.ref_prefix}_" if args.ref_prefix else ""

        refinement_dir = os.path.join(args.working_dir, f"{ref_prefix}refinement")
        os.makedirs(refinement_dir, exist_ok=True)

        backbones.set_work_dir(refinement_dir)

        if args.ref_input_poses_per_bb:
            logging.info(f"Filtering refinement input poses on per backbone level according to design_composite_score...")
            backbones.filter_poses_by_rank(n=args.ref_input_poses_per_bb, score_col=f'design_composite_score', remove_layers=1, prefix='refinement_input_bb', plot=True)
        if args.ref_input_poses:
            logging.info(f"Filtering refinement input according to design_composite_score...")
            backbones.filter_poses_by_rank(n=args.ref_input_poses, score_col=f'design_composite_score', prefix='refinement_input', plot=True)

        # use reduced motif if specified
        if args.use_reduced_motif:
            backbones.df["motif_residues"] = backbones.df.apply(lambda row: create_reduced_motif(row['fixed_residues'], row['motif_residues']), axis=1)

        # create refinement input poses dir
        refinement_input_dir = os.path.join(refinement_dir, "refinement_input_poses")
        os.makedirs(refinement_input_dir, exist_ok=True)
        backbones.save_scores(out_path=os.path.join(refinement_input_dir, "refinement_input_scores.json"), out_format="json")
        backbones.save_poses(out_path=refinement_input_dir)
        backbones.save_poses(out_path=refinement_input_dir, poses_col="input_poses")
        write_pymol_alignment_script(
            df=backbones.df,
            scoreterm="design_composite_score",
            top_n=np.min([len(backbones), 25]),
            path_to_script=os.path.join(refinement_input_dir, "align_poses.pml"),
            ref_motif_col = "template_fixedres",
            ref_catres_col = "template_fixedres",
            target_catres_col = "fixed_residues",
            target_motif_col = "fixed_residues"
        )

        logging.info(f"Plotting refinement input data.")
        cols = ["esm_plddt", "esm_backbone_rmsd", "esm_catres_bb_rmsd", "esm_catres_heavy_rmsd", "esm_tm_sc_tm", "esm_rog_data", "esm_lig_contacts", "esm_ligand_clashes", "screen", "screen_decentralize_weight", "screen_decentralize_distance"]
        titles = ["ESMFold pLDDT", "ESMFold BB-Ca RMSD", "ESMFold fixed res\nBB-Ca RMSD", "ESMFold Sidechain\nRMSD", "SC-TM Score", "Radius of Gyration", "Ligand Contacts", "Ligand Clashes", "screen number", "decentralize weight", "decentralize distance"]
        y_labels = ["pLDDT", "Angstrom", "Angstrom", "Angstrom", "TM Score", "Angstrom", "#", "#", "#", "AU", "Angstrom"]
        dims = [None for _ in cols]

        plots.violinplot_multiple_cols(
            dataframe = backbones.df,
            cols = cols,
            titles = titles,
            y_labels = y_labels,
            dims = dims,
            out_path = os.path.join(backbones.plots_dir, f"{ref_prefix}refinement_input_poses.png"),
            show_fig = False
        )
        shutil.copy(os.path.join(backbones.plots_dir, f"{ref_prefix}refinement_input_poses.png"), refinement_input_dir)

        # instantiate plotting trajectories
        trajectory_plots = instantiate_trajectory_plotting(backbones.plots_dir, backbones.df)

        for cycle in range(args.ref_start_cycle, args.ref_cycles+1):
            cycle_work_dir = os.path.join(refinement_dir, f"cycle_{cycle}")
            backbones.set_work_dir(cycle_work_dir)
            logging.info(f"Starting refinement cycle {cycle} in directory {cycle_work_dir}")

            # remove ligand and add it back to make sure theozyme orientation is used and not the relaxed output from the previous cycle!
            if cycle > 1:
                logging.info("Removing ligand from relaxed poses...")
                backbones = chain_remover.run(
                    poses = backbones,
                    prefix = f"cycle_{cycle}_remove_ligand",
                    chains = "Z"
                )

                logging.info("Adding ligand to ESMFold predictions...")
                backbones = chain_adder.superimpose_add_chain(
                poses = backbones,
                prefix = f"cycle_{cycle}_reset_ligand",
                ref_col = "updated_reference_frags_location",
                target_motif = "fixed_residues",
                copy_chain = "Z"
                )

            logging.info("Threading sequences on poses with LigandMPNN...")
            # run ligandmpnn, return pdbs as poses
            backbones = ligand_mpnn.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_seq_thread",
                nseq = args.ref_seq_thread_num_mpnn_seqs,
                model_type = "ligand_mpnn",
                options = ligandmpnn_options,
                fixed_res_col = "fixed_residues",
                return_seq_threaded_pdbs_as_pose=True
            )

            # optimize backbones 
            logging.info("Optimizing backbones with Rosetta...")
            backbones.df[f'cycle_{cycle}_bbopt_opts'] = [write_bbopt_opts(row=row, cycle=cycle, total_cycles=args.ref_cycles, reference_location_col="updated_reference_frags_location", cat_res_col="fixed_residues", motif_res_col="motif_residues", ligand_chain="Z") for _, row in backbones.df.iterrows()]

            # add covalent bonds info to poses pre-relax
            backbones = add_covalent_bonds_info(poses=backbones, prefix=f"cycle_{cycle}_bbopt_cov_info", covalent_bonds_col="covalent_bonds")

            backbones = rosetta.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_bbopt",
                rosetta_application="rosetta_scripts.default.linuxgccrelease",
                nstruct = 1,
                options = bb_opt_options,
                pose_options=f'cycle_{cycle}_bbopt_opts'
            )

            # filter backbones down to starting backbones
            logging.info("Selecting poses with lowest total score for each input backbone...")
            backbones.filter_poses_by_rank(n=1, score_col=f"cycle_{cycle}_bbopt_total_score", remove_layers=2)

            # run ligandmpnn on optimized poses
            logging.info("Generating sequences for each pose...")
            backbones = ligand_mpnn.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_mpnn",
                nseq = args.ref_num_mpnn_seqs,
                model_type = "ligand_mpnn",
                options = ligandmpnn_options,
                fixed_res_col = "fixed_residues",
            )

            # predict structures using ESMFold
            logging.info("Predicting sequences with ESMFold...")
            backbones = esmfold.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_esm",
            )

            # calculate rmsds, TMscores and clashes
            logging.info(f"Calculating post-ESMFold RMSDs...")
            backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_esm_catres_heavy")
            backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_esm_catres_bb")
            backbones = bb_rmsd.run(poses = backbones, ref_col=f"cycle_{cycle}_bbopt_location", prefix = f"cycle_{cycle}_esm_backbone")
            backbones = tm_score_calculator.run(poses = backbones, prefix = f"cycle_{cycle}_esm_tm", ref_col = f"cycle_{cycle}_bbopt_location")
            
            # calculate cutoff & filter
            logging.info(f"Applying post-ESMFold backbone filters...")
            plddt_cutoff = ramp_cutoff(args.ref_plddt_cutoff_start, args.ref_plddt_cutoff_end, cycle, args.ref_cycles)
            catres_bb_rmsd_cutoff = ramp_cutoff(args.ref_catres_bb_rmsd_cutoff_start, args.ref_catres_bb_rmsd_cutoff_end, cycle, args.ref_cycles)
            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_plddt", value=plddt_cutoff, operator=">=", prefix=f"cycle_{cycle}_esm_plddt", plot=True)
            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_tm_TM_score_ref", value=0.9, operator=">=", prefix=f"cycle_{cycle}_esm_TM_score", plot=True)
            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_catres_bb_rmsd", value=catres_bb_rmsd_cutoff, operator="<=", prefix=f"cycle_{cycle}_esm_catres_bb", plot=True)

            # repack predictions with attnpacker, if set
            if args.attnpacker_repack:
                logging.info("Repacking ESMFold output with Attnpacker...")
                backbones = attnpacker.run(
                    poses=backbones,
                    prefix=f"cycle_{cycle}_packing"
                )

            # copy description column for merging with holo relaxed structures later
            backbones.df[f'cycle_{cycle}_rlx_description'] = backbones.df['poses_description']
            apo_backbones = copy.deepcopy(backbones)

            # relax apo poses
            logging.info("Relaxing poses without ligand present...")
            apo_backbones = rosetta.run(
                poses = apo_backbones,
                prefix = f"cycle_{cycle}_fastrelax_apo",
                rosetta_application="rosetta_scripts.default.linuxgccrelease",
                nstruct = 3,
                options = fr_options
            )

            apo_backbones = catres_motif_heavy_rmsd.run(poses = apo_backbones, prefix = f"cycle_{cycle}_postrelax_apo_catres_heavy")
            apo_backbones = catres_motif_bb_rmsd.run(poses = apo_backbones, prefix = f"cycle_{cycle}_postrelax_apo_catres_bb")
            apo_backbones= calculate_mean_scores(poses=apo_backbones, scores=[f"cycle_{cycle}_postrelax_apo_catres_heavy_rmsd", f"cycle_{cycle}_postrelax_apo_catres_bb_rmsd"],  remove_layers=1)

            # filter for top relaxed apo pose and merge with original dataframe
            logging.info("Selecting top poses for each relaxed structure...")
            apo_backbones.filter_poses_by_rank(n=1, score_col=f"cycle_{cycle}_fastrelax_apo_total_score", remove_layers=1)
            preserve_cols = [f'cycle_{cycle}_rlx_description', f"cycle_{cycle}_fastrelax_apo_total_score", f"cycle_{cycle}_postrelax_apo_catres_heavy_rmsd", f"cycle_{cycle}_postrelax_apo_catres_heavy_rmsd_mean", f"cycle_{cycle}_postrelax_apo_catres_bb_rmsd", f"cycle_{cycle}_postrelax_apo_catres_bb_rmsd_mean"]
            backbones.df = backbones.df.merge(apo_backbones.df[preserve_cols], on=f'cycle_{cycle}_rlx_description')

            # add ligand to poses
            logging.info("Adding ligand to ESMFold predictions...")
            backbones = chain_adder.superimpose_add_chain(
                poses = backbones,
                prefix = f"cycle_{cycle}_ligand",
                ref_col = "updated_reference_frags_location",
                target_motif = "fixed_residues",
                copy_chain = "Z"
            )

            # add covalent bonds info to poses pre-relax
            backbones = add_covalent_bonds_info(poses=backbones, prefix=f"cycle_{cycle}_fastrelax_cov_info", covalent_bonds_col="covalent_bonds")

            # run rosetta_script to evaluate residuewise energy
            logging.info("Relaxing poses with ligand present...")
            backbones = rosetta.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_fastrelax",
                rosetta_application="rosetta_scripts.default.linuxgccrelease",
                nstruct = 3,
                options = fr_options
            )

            # calculate RMSD on relaxed poses
            logging.info(f"Calculating RMSD of catalytic residues and ligand for relaxed poses...")
            backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_postrelax_catres_heavy")
            backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_postrelax_catres_bb")
            backbones = ligand_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_postrelax_ligand")
            backbones= calculate_mean_scores(poses=backbones, scores=[f"cycle_{cycle}_postrelax_catres_heavy_rmsd", f"cycle_{cycle}_postrelax_catres_bb_rmsd", f"cycle_{cycle}_postrelax_ligand_rmsd", f"cycle_{cycle}_fastrelax_sap_score"], remove_layers=1)

            # filter backbones down to relax input backbones
            backbones.filter_poses_by_rank(n=1, score_col=f"cycle_{cycle}_fastrelax_total_score", remove_layers=1)

            # ramp cutoffs during refinement
            ligand_rmsd_cutoff = ramp_cutoff(args.ref_ligand_rmsd_start, args.ref_ligand_rmsd_end, cycle, args.ref_cycles)
            # apply filters
            logging.info("Removing poses with ligand rmsd above cutoff...")
            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_postrelax_ligand_rmsd", value=ligand_rmsd_cutoff, operator="<=", prefix=f"cycle_{cycle}_ligand_rmsd", plot=True)        

            # calculate delta apo holo score
            logging.info("Calculating delta total score between relaxed poses with and without ligand present...")
            backbones.df[f'cycle_{cycle}_delta_apo_holo'] = backbones.df[f"cycle_{cycle}_fastrelax_total_score"] - backbones.df[f"cycle_{cycle}_fastrelax_apo_total_score"]

            # calculate multi-scoreterm score for the final backbone filter:
            logging.info("Calculating composite score for refinement evaluation...")
            backbones.calculate_composite_score(
                name=f"cycle_{cycle}_refinement_composite_score",
                scoreterms=[f"cycle_{cycle}_esm_plddt", f"cycle_{cycle}_esm_tm_TM_score_ref", f"cycle_{cycle}_esm_catres_bb_rmsd", f"cycle_{cycle}_esm_catres_heavy_rmsd", f"cycle_{cycle}_delta_apo_holo", f"cycle_{cycle}_postrelax_ligand_rmsd", f"cycle_{cycle}_postrelax_catres_heavy_rmsd", f"cycle_{cycle}_fastrelax_sap_score_mean", f"cycle_{cycle}_postrelax_apo_catres_heavy_rmsd"],
                weights=[-2, -2, 8, 4, 2, 2, 2, 1, 2],
                plot=True
            )

            # define number of index layers that were added during refinement cycle (higher in subsequent cycles because reindexing adds a layer)
            layers = 4
            if cycle > 1: layers += 1
            if args.attnpacker_repack: layers += 1

            # manage screen output
            backbones.reindex_poses(prefix=f"cycle_{cycle}_reindex", remove_layers=layers, force_reindex=True)

            # copy output of final round pre-filtering
            if cycle == args.ref_cycles: refinement_results = copy.deepcopy(backbones)

            # filter down to rfdiffusion backbones
            logging.info("Filtering poses according to composite score...")
            backbones.filter_poses_by_rank(
                n=args.ref_num_cycle_poses,
                score_col=f"cycle_{cycle}_refinement_composite_score",
                prefix=f"cycle_{cycle}_refinement_composite_score",
                plot=True,
                remove_layers=1
            )

            trajectory_plots = update_trajectory_plotting(trajectory_plots=trajectory_plots, df=backbones.df, cycle=cycle)
            results_dir = os.path.join(backbones.work_dir, f"cycle_{cycle}_results")
            create_ref_results_dir(poses=backbones, dir=results_dir, cycle=cycle)

        # sort output
        backbones = refinement_results
        backbones.df.sort_values(f"cycle_{cycle}_refinement_composite_score", ascending=True, inplace=True)
        backbones.df.reset_index(drop=True, inplace=True)

        refinement_results_dir = os.path.join(args.working_dir, f"{ref_prefix}refinement_results")
        create_ref_results_dir(poses=backbones, dir=refinement_results_dir, cycle=cycle)
        backbones.save_scores(out_path=os.path.join(refinement_results_dir, f"results_{ref_prefix}_refinement"))
        backbones.set_work_dir(args.working_dir)
        backbones.save_scores()

        if args.skip_evaluation:
            logging.info(f"Skipping evaluation. Run concluded, per-backbone output can be found in {os.path.join(backbones.work_dir, f'cycle_{cycle}_results')}. Overall results can be found in {refinement_results_dir}.")
            sys.exit(1)
        else:
            args.eval_input_json = backbones.scorefile

    ########################### FINAL EVALUATION ###########################
    if args.eval_input_json:
        if args.eval_prefix: eval_prefix = f"{args.eval_prefix}_"
        else: eval_prefix = args.ref_prefix if args.ref_prefix else ""

        last_ref_cycle = determine_last_ref_cycle(backbones.df)
        logging.info(f"Last refinement cycle determined as: {last_ref_cycle}")
        
        if args.eval_drop_previous_results:
            eval_cols = [col for col in backbones.df.columns if col.startswith("final")]
            backbones.df.drop(eval_cols, axis=1, inplace=True)

        backbones.set_work_dir(os.path.join(args.working_dir, f"{eval_prefix}evaluation"))

        if args.eval_input_poses_per_bb:
            logging.info(f"Filtering evaluation input poses on per backbone level according to cycle_{last_ref_cycle}_refinement_composite_score...")
            backbones.filter_poses_by_rank(n=args.eval_input_poses_per_bb, score_col=f"cycle_{last_ref_cycle}_refinement_composite_score", remove_layers=1, prefix="evaluation_input_per_bb", plot=True)
        if args.eval_input_poses: 
            logging.info(f"Filtering evaluation input poses according to cycle_{last_ref_cycle}_refinement_composite_score...")
            backbones.filter_poses_by_rank(n=args.eval_input_poses, score_col=f"cycle_{last_ref_cycle}_refinement_composite_score", prefix="evaluation_input", plot=True)

        evaluation_input_poses_dir = os.path.join(backbones.work_dir, "evaluation_input_poses")
        os.makedirs(evaluation_input_poses_dir, exist_ok=True)
        backbones.save_poses(out_path=evaluation_input_poses_dir)
        backbones.save_poses(out_path=evaluation_input_poses_dir, poses_col="input_poses")
        backbones.save_scores(out_path=evaluation_input_poses_dir)

        # write pymol alignment script
        logging.info(f"Writing pymol alignment script for evaluation input poses at {evaluation_input_poses_dir}.")
        write_pymol_alignment_script(
            df = backbones.df,
            scoreterm = f"cycle_{last_ref_cycle}_refinement_composite_score",
            top_n = np.min([len(backbones.df.index), 25]),
            path_to_script = os.path.join(evaluation_input_poses_dir, "align_input_poses.pml"),
            ref_motif_col = "template_fixedres",
            ref_catres_col = "template_fixedres",
            target_catres_col = "fixed_residues",
            target_motif_col = "fixed_residues"
        )

        backbones.convert_pdb_to_fasta(prefix="final_fasta_conversion", update_poses=True)

        # run AF2
        backbones = colabfold.run(
            poses=backbones,
            prefix="final_AF2",
            return_top_n_poses=3,
            options="--msa-mode single_sequence"
        )

        # select top 3 predictions and filter for average score
        backbones.calculate_mean_score(name="final_AF2_plddt_mean", score_col="final_AF2_plddt", remove_layers=1)
        backbones.filter_poses_by_value(score_col="final_AF2_plddt_mean", value=args.eval_mean_plddt_cutoff, operator=">=", prefix="final_AF2_mean_plddt", plot=True)

        # calculate backbone rmsds
        backbones = catres_motif_bb_rmsd.run(poses=backbones, prefix=f"final_AF2_catres_bb")
        backbones = bb_rmsd.run(poses=backbones, prefix="final_AF2_backbone", ref_col=f"cycle_{last_ref_cycle}_bbopt_location")
        backbones = bb_rmsd.run(poses=backbones, prefix="final_AF2_ESM_bb", ref_col=f"cycle_{last_ref_cycle}_esm_location")
        backbones = tm_score_calculator.run(poses=backbones, prefix=f"final_AF2_tm", ref_col=f"cycle_{last_ref_cycle}_bbopt_location")
        backbones = tm_score_calculator.run(poses=backbones, prefix=f"final_AF2_ESM_tm", ref_col=f"cycle_{last_ref_cycle}_esm_location")
        backbones = calculate_mean_scores(poses=backbones, scores=["final_AF2_catres_bb_rmsd", "final_AF2_backbone_rmsd", "final_AF2_tm_TM_score_ref"], remove_layers=1)

        if args.attnpacker_repack:
            backbones = attnpacker.run(
                poses=backbones,
                prefix=f"final_packing"
            )

        backbones = catres_motif_heavy_rmsd.run(poses=backbones, prefix=f"final_AF2_catres_heavy")
        backbones = calculate_mean_scores(poses=backbones, scores=["final_AF2_catres_heavy_rmsd"], remove_layers=1 if not args.attnpacker_repack else 2)

        # filter for AF2 top model
        backbones.filter_poses_by_rank(n=1, score_col="final_AF2_plddt", ascending=False, remove_layers=1 if not args.attnpacker_repack else 2)

        # apply rest of the filters
        backbones.filter_poses_by_value(score_col="final_AF2_catres_bb_rmsd_mean", value=args.eval_mean_catres_bb_rmsd_cutoff, operator="<=", prefix=f"final_AF2_mean_catres_bb_rmsd", plot=True)
        backbones.filter_poses_by_value(score_col="final_AF2_plddt", value=args.eval_plddt_cutoff, operator=">=", prefix="final_AF2_plddt", plot=True)
        backbones.filter_poses_by_value(score_col="final_AF2_tm_TM_score_ref", value=0.9, operator=">=", prefix=f"final_AF2_TM_score", plot=True)
        backbones.filter_poses_by_value(score_col="final_AF2_ESM_bb_rmsd", value=2.0, operator="<=", prefix="final_AF2_ESM_bb_rmsd", plot=True) # check if AF2 and ESM predictions agree      
        backbones.filter_poses_by_value(score_col="final_AF2_catres_bb_rmsd", value=args.eval_catres_bb_rmsd_cutoff, operator="<=", prefix=f"final_AF2_catres_bb_rmsd", plot=True)

        # copy description column for merging with apo relaxed structures
        backbones.df['final_relax_input_description'] = backbones.df['poses_description']
        apo_backbones = copy.deepcopy(backbones)

        # add ligand chain 
        backbones = chain_adder.superimpose_add_chain(
            poses = backbones,
            prefix = f"final_ligand",
            ref_col = "updated_reference_frags_location",
            target_motif = "fixed_residues",
            copy_chain = "Z"
        )
        
        # calculate ligand clashes and ligand contacts
        backbones = ligand_clash.run(poses=backbones, prefix="final_AF2_ligand")
        backbones = ligand_contacts.run(poses=backbones, prefix="final_AF2_lig")

        # add covalent bonds info to poses pre-relax
        backbones = add_covalent_bonds_info(poses=backbones, prefix="final_fastrelax_cov_info", covalent_bonds_col="covalent_bonds")
        
        # relax predictions with ligand present
        backbones = rosetta.run(
            poses = backbones,
            prefix = "final_fastrelax",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 10,
            options = fr_options
        )

        # calculate RMSDs of relaxed poses
        backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"final_postrelax_catres_heavy")
        backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"final_postrelax_catres_bb")
        backbones = ligand_rmsd.run(poses = backbones, prefix = "final_postrelax_ligand")

        # average values for all relaxed poses
        backbones = calculate_mean_scores(poses=backbones, scores=["final_postrelax_catres_heavy_rmsd", "final_postrelax_catres_bb_rmsd", "final_postrelax_ligand_rmsd", "final_fastrelax_sap_score"], remove_layers=1)
        
        # filter to relaxed pose with best score
        backbones.filter_poses_by_rank(n=1, score_col="final_fastrelax_total_score", remove_layers=1)

        # relax apo poses
        apo_backbones = rosetta.run(
            poses = apo_backbones,
            prefix = "final_fastrelax_apo",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 10,
            options = fr_options
        )

        # calculate apo relaxed rmsds
        apo_backbones = catres_motif_heavy_rmsd.run(poses = apo_backbones, prefix = f"final_postrelax_apo_catres_heavy")
        apo_backbones = catres_motif_bb_rmsd.run(poses = apo_backbones, prefix = f"final_postrelax_apo_catres_bb")
        apo_backbones = calculate_mean_scores(poses=apo_backbones, scores=["final_postrelax_apo_catres_heavy_rmsd", "final_postrelax_apo_catres_bb_rmsd"], remove_layers=1)

        # filter to relaxed pose with best score, merge dataframes
        apo_backbones.filter_poses_by_rank(n=1, score_col="final_fastrelax_apo_total_score", remove_layers=1)
        preserve_cols = ['final_relax_input_description', 'final_fastrelax_apo_total_score', 'final_postrelax_apo_catres_heavy_rmsd', 'final_postrelax_apo_catres_heavy_rmsd_mean', 'final_postrelax_apo_catres_bb_rmsd', 'final_postrelax_apo_catres_bb_rmsd_mean']
        backbones.df = backbones.df.merge(apo_backbones.df[preserve_cols], on='final_relax_input_description')
        
        # calculate delta score between apo and holo poses
        backbones.df['final_delta_apo_holo'] = backbones.df['final_fastrelax_total_score'] - backbones.df['final_fastrelax_apo_total_score']

        # filter ligand rmsd
        backbones.filter_poses_by_value(score_col="final_postrelax_ligand_rmsd_mean", value=args.eval_mean_ligand_rmsd_cutoff, operator="<=", prefix="final_mean_ligand_rmsd", plot=True)        
        backbones.filter_poses_by_value(score_col="final_postrelax_ligand_rmsd", value=args.eval_ligand_rmsd_cutoff, operator="<=", prefix="final_ligand_rmsd", plot=True)        

        # calculate final composite score
        backbones.calculate_composite_score(
            name=f"final_composite_score",
            scoreterms=["final_AF2_plddt", "final_AF2_tm_TM_score_ref", "final_AF2_catres_bb_rmsd", "final_AF2_catres_heavy_rmsd", "final_delta_apo_holo", "final_postrelax_ligand_rmsd", "final_postrelax_catres_heavy_rmsd", "final_fastrelax_sap_score_mean", "final_postrelax_apo_catres_heavy_rmsd"],
            weights=[-2, -2, 8, 4, 2, 2, 2, 1, 2],
            plot=True
        )
    
        # plot mean results
        plots.violinplot_multiple_cols(
            dataframe=backbones.df,
            cols=["final_AF2_catres_heavy_rmsd_mean", "final_AF2_catres_bb_rmsd_mean", "final_postrelax_catres_heavy_rmsd_mean", "final_postrelax_catres_bb_rmsd_mean", "final_postrelax_ligand_rmsd_mean", "final_postrelax_apo_catres_heavy_rmsd_mean", "final_fastrelax_sap_score_mean"],
            y_labels=["Angstrom", "Angstrom", "Angstrom", "Angstrom", "Angstrom", "Angstrom", "SAP score"],
            titles=["Mean AF2\nSidechain RMSD", "Mean AF2 catres\nBB RMSD", "Mean Relaxed\nSidechain RMSD", "Mean Relaxed catres\nBB RMSD", "Mean Relaxed ligand\nRMSD", "Mean Apo Relaxed\nSidechain RMSD", "Spatial Aggregation Propensity"],
            out_path=os.path.join(backbones.plots_dir, "final_mean_rmsds.png"),
            show_fig=False
        )

        backbones.reindex_poses(prefix="final_reindex", remove_layers=2 if not args.attnpacker_repack else 3)

        # create plot for screening, refinement and evaluation
        trajectory_plots = instantiate_trajectory_plotting(plot_dir=backbones.plots_dir, df=backbones.df)
        for cycle in range(1, last_ref_cycle + 1):
            trajectory_plots = update_trajectory_plotting(trajectory_plots, backbones.df, cycle)
        trajectory_plots = add_final_data_to_trajectory_plots(backbones.df, trajectory_plots)

        eval_results_dir = os.path.join(args.working_dir, f"{eval_prefix}evaluation_results")
        create_final_results_dir(backbones, eval_results_dir)
        backbones.save_scores()
        backbones.save_scores(out_path=os.path.join(eval_results_dir, f"results_{eval_prefix}_evaluation"))


    ########################### VARIANT GENERATION ###########################
    if args.variants_input_json:

        if args.variants_prefix: variants_prefix = f"{args.variants_prefix}_"
        else: variants_prefix = eval_prefix if args.eval_prefix or args.ref_prefix else ""
        
        backbones.set_work_dir(os.path.join(args.working_dir, f"{variants_prefix}variants"))

        if args.variants_mutations_csv:
            mutations = pd.read_csv(args.variants_mutations_csv)
            backbones.df = backbones.df.merge(mutations, on="poses_description", how="right")
            backbones.df.reset_index(drop=True, inplace=True)
            mutations_dir = os.path.join(backbones.work_dir, "mutations")
            os.makedirs(mutations_dir, exist_ok=True)
            backbones.df["variants_pose_opts"] = backbones.df.apply(lambda row: omit_AAs(row['omit_AAs'], row['allow_AAs'], mutations_dir, row["poses_description"]), axis=1)

        if args.variants_input_poses_per_bb:
            backbones.filter_poses_by_rank(n=args.variants_input_poses_per_bb, score_col="final_composite_score", remove_layers=1)

        if args.variants_input_poses:
            backbones.filter_poses_by_rank(n=args.variants_input_poses, score_col="final_composite_score")

        #reset ligand position
        backbones = chain_remover.run(
            poses = backbones,
            prefix = f"variants_remove_ligand",
            chains = "Z"
        )

        logging.info("Adding ligand to ESMFold predictions...")
        backbones = chain_adder.superimpose_add_chain(
        poses = backbones,
        prefix = f"variants_reset_ligand",
        ref_col = "updated_reference_frags_location",
        target_motif = "fixed_residues",
        copy_chain = "Z"
        )

        # add covalent bonds info to poses pre-relax
        backbones = add_covalent_bonds_info(poses=backbones, prefix="variants_bbopt_cov_info", covalent_bonds_col="covalent_bonds")

        backbones.df[f'variants_bbopt_opts'] = [write_bbopt_opts(row=row, cycle=1, total_cycles=1, reference_location_col="updated_reference_frags_location", cat_res_col="fixed_residues", motif_res_col="motif_residues", ligand_chain="Z") for _, row in backbones.df.iterrows()]

        backbones = rosetta.run(
            poses = backbones,
            prefix = f"variants_bbopt",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 3,
            options = bb_opt_options,
            pose_options='variants_bbopt_opts'
        )

        # filter backbones down to starting backbones
        backbones.filter_poses_by_rank(n=1, score_col="variants_bbopt_total_score", remove_layers=1)

        # select shell around the ligand
        ligand_shell_selector = DistanceSelector(center="ligand_motif", distance=10, operator="<=", noncenter_atoms="CA")
        ligand_shell_selector.select(prefix="ligand_shell", poses=backbones)

        # create copy of backbones
        shell_backbones = copy.deepcopy(backbones)

        # optimize sequences
        backbones = ligand_mpnn.run(
            poses = backbones,
            prefix = f"variants_mpnn",
            nseq = 30,
            model_type = "ligand_mpnn",
            options = ligandmpnn_options,
            pose_options = "variants_pose_opts" if args.variants_mutations_csv else None,
            fixed_res_col = "fixed_residues"
        )

        # only diversify around ligand, keep rest fixed
        if args.variants_mutations_csv:
            shell_backbones.df["variants_pose_opts"] = [f"{mut_opts} --redesigned_residues {shell.to_string(delim=" ")}" for mut_opts, shell in zip(shell_backbones.df["variants_pose_opts"].to_list(), shell_backbones.df["ligand_shell"].to_list())]
        else:
            shell_backbones.df["variants_pose_opts"] = [f"--redesigned_residues {shell.to_string(delim=" ")}" for shell in shell_backbones.df["ligand_shell"].to_list()]

        shell_backbones = ligand_mpnn.run(
            poses = shell_backbones,
            prefix = "shell_diversification",
            nseq = 30,
            model_type = "ligand_mpnn",
            options = f"--ligand_mpnn_use_side_chain_context 1 --temperature 0.2 {args.ligandmpnn_options if args.ligandmpnn_options else ""}",
            pose_options = "variants_pose_opts",
            fixed_res_col = "fixed_residues"
        )

        backbones.df = pd.concat(backbones.df, shell_backbones.df)
        backbones.reindex_poses(prefix="post_mpnn", remove_layers=1, force_reindex=True)

        # predict structures using ESMFold
        backbones = esmfold.run(
            poses = backbones,
            prefix = f"variants_esm",
        )

        # calculate rmsds, TMscores and clashes
        backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"variants_esm_catres_heavy")
        backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"variants_esm_catres_bb")
        backbones = bb_rmsd.run(poses = backbones, ref_col=f"variants_bbopt_location", prefix = f"variants_esm_backbone")
        backbones = tm_score_calculator.run(poses = backbones, prefix = f"variants_esm_tm", ref_col = f"variants_bbopt_location")

        backbones.filter_poses_by_value(score_col=f"variants_esm_plddt", value=args.ref_plddt_cutoff_end, operator=">=", prefix=f"variants_esm_plddt", plot=True)
        backbones.filter_poses_by_value(score_col=f"variants_esm_tm_TM_score_ref", value=0.9, operator=">=", prefix=f"variants_esm_TM_score", plot=True)
        backbones.filter_poses_by_value(score_col=f"variants_esm_catres_bb_rmsd", value=args.ref_catres_bb_rmsd_cutoff_end, operator="<=", prefix=f"variants_esm_catres_bb", plot=True)


        # repack predictions with attnpacker, if set
        if args.attnpacker_repack:
            backbones = attnpacker.run(
                poses=backbones,
                prefix=f"variants_packing"
            )

        # copy description column for merging with holo relaxed structures later
        backbones.df[f'variants_rlx_description'] = backbones.df['poses_description']
        apo_backbones = copy.deepcopy(backbones)

        # relax apo poses
        apo_backbones = rosetta.run(
            poses = apo_backbones,
            prefix = f"variants_fastrelax_apo",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 3,
            options = fr_options
        )

        apo_backbones = catres_motif_heavy_rmsd.run(poses = apo_backbones, prefix = f"variants_postrelax_apo_catres_heavy")
        apo_backbones = catres_motif_bb_rmsd.run(poses = apo_backbones, prefix = f"variants_postrelax_apo_catres_bb")
        apo_backbones= calculate_mean_scores(poses=apo_backbones, scores=[f"variants_postrelax_apo_catres_heavy_rmsd", f"variants_postrelax_apo_catres_bb_rmsd"],  remove_layers=1)

        # filter for top relaxed apo pose and merge with original dataframe
        apo_backbones.filter_poses_by_rank(n=1, score_col="variants_fastrelax_apo_total_score", remove_layers=1)
        preserve_cols = [f'variants_rlx_description', f"variants_fastrelax_apo_total_score", f"variants_postrelax_apo_catres_heavy_rmsd", f"variants_postrelax_apo_catres_heavy_rmsd_mean", f"variants_postrelax_apo_catres_bb_rmsd", f"variants_postrelax_apo_catres_bb_rmsd_mean"]
        backbones.df = backbones.df.merge(apo_backbones.df[preserve_cols], on='variants_rlx_description')

        # add ligand to poses
        backbones = chain_adder.superimpose_add_chain(
            poses = backbones,
            prefix = f"variants_ligand",
            ref_col = "updated_reference_frags_location",
            target_motif = "fixed_residues",
            copy_chain = "Z"
        )

        # add covalent bonds info to poses pre-relax
        backbones = add_covalent_bonds_info(poses=backbones, prefix="variants_fastrelax_cov_info", covalent_bonds_col="covalent_bonds")

        # run rosetta_script to evaluate residuewise energy
        backbones = rosetta.run(
            poses = backbones,
            prefix = f"variants_fastrelax",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 3,
            options = fr_options
        )

        # calculate delta apo holo score
        backbones.df[f'variants_delta_apo_holo'] = backbones.df[f"variants_fastrelax_total_score"] - backbones.df[f"variants_fastrelax_apo_total_score"]

        # calculate RMSD on relaxed poses
        logging.info(f"Relax finished. Now calculating RMSD of catalytic residues for {len(backbones)} structures.")
        backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = "variants_postrelax_catres_heavy")
        backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = "variants_postrelax_catres_bb")
        backbones = ligand_rmsd.run(poses = backbones, prefix = "variants_postrelax_ligand")
        backbones = calculate_mean_scores(poses=backbones, scores=["variants_postrelax_catres_heavy_rmsd", "variants_postrelax_catres_bb_rmsd", "variants_postrelax_ligand_rmsd", "variants_fastrelax_sap_score"], remove_layers=1)

        # filter backbones down to relax input backbones
        backbones.filter_poses_by_rank(n=1, score_col="variants_fastrelax_total_score", remove_layers=1)

        # calculate multi-scoreterm score for the final backbone filter:
        backbones.calculate_composite_score(
            name="variants_composite_score",
            scoreterms=["variants_esm_plddt", "variants_esm_tm_TM_score_ref", "variants_esm_catres_bb_rmsd", "variants_esm_catres_heavy_rmsd", "variants_delta_apo_holo", "variants_postrelax_ligand_rmsd", "variants_postrelax_catres_heavy_rmsd", "variants_fastrelax_sap_score_mean", "variants_postrelax_apo_catres_heavy_rmsd"],
            weights=[-2, -2, 8, 4, 2, 2, 2, 1, 2],
            plot=True
        )
        
        # apply filters
        backbones.filter_poses_by_value(score_col=f"variants_postrelax_ligand_rmsd", value=args.ref_ligand_rmsd_end, operator="<=", prefix=f"variants_ligand_rmsd", plot=True)        

        # define number of index layers that were added during variants generation
        layers = 3
        if args.attnpacker_repack: layers += 1

        # manage screen output
        backbones.reindex_poses(prefix=f"variants_reindex", remove_layers=layers, force_reindex=True)

        # filter down to rfdiffusion backbones
        backbones.filter_poses_by_rank(
            n=args.variants_evaluation_input_poses_per_bb,
            score_col="variants_composite_score",
            prefix="variants_composite_score_per_bb",
            plot=True,
            remove_layers=2
        )

        backbones.filter_poses_by_rank(
            n=args.variants_evaluation_input_poses,
            score_col="variants_composite_score",
            prefix="variants_composite_score",
            plot=True,
            remove_layers=None
        )

        backbones.convert_pdb_to_fasta(prefix="variants_fasta_conversion", update_poses=True)

        # run AF2
        backbones = colabfold.run(
            poses=backbones,
            prefix="variants_AF2",
            return_top_n_poses=3,
            options="--msa-mode single_sequence"
        )

        # select top 3 predictions and filter for average score
        backbones.calculate_mean_score(name="variants_AF2_plddt_mean", score_col="variants_AF2_plddt", remove_layers=1)
        backbones.filter_poses_by_value(score_col="variants_AF2_plddt_mean", value=args.eval_mean_plddt_cutoff, operator=">=", prefix="variants_AF2_plddt_mean", plot=True)

        # calculate backbone rmsds
        backbones = catres_motif_bb_rmsd.run(poses=backbones, prefix=f"variants_AF2_catres_bb")
        backbones = bb_rmsd.run(poses=backbones, prefix="variants_AF2_backbone", ref_col=f"variants_bbopt_location")
        backbones = bb_rmsd.run(poses=backbones, prefix="variants_AF2_ESM_bb", ref_col=f"variants_esm_location")
        backbones = tm_score_calculator.run(poses=backbones, prefix=f"variants_AF2_tm", ref_col=f"variants_bbopt_location")
        backbones = tm_score_calculator.run(poses=backbones, prefix=f"variants_AF2_ESM_tm", ref_col=f"variants_esm_location")
        backbones = calculate_mean_scores(poses=backbones, scores=["variants_AF2_catres_bb_rmsd", "variants_AF2_backbone_rmsd", "variants_AF2_tm_TM_score_ref"], remove_layers=1)

        if args.attnpacker_repack:
            backbones = attnpacker.run(
                poses=backbones,
                prefix=f"variants_AF2_packing"
            )

        backbones = catres_motif_heavy_rmsd.run(poses=backbones, prefix=f"variants_AF2_catres_heavy")
        backbones = calculate_mean_scores(poses=backbones, scores=["variants_AF2_catres_heavy_rmsd"], remove_layers=1 if not args.attnpacker_repack else 2)

        # filter for AF2 top model
        backbones.filter_poses_by_rank(n=1, score_col="variants_AF2_plddt", ascending=False, remove_layers=1 if not args.attnpacker_repack else 2)

        # apply rest of the filters
        backbones.filter_poses_by_value(score_col="variants_AF2_catres_bb_rmsd_mean", value=args.eval_mean_catres_bb_rmsd_cutoff, operator="<=", prefix=f"variants_AF2_mean_catres_bb_rmsd", plot=True)
        backbones.filter_poses_by_value(score_col="variants_AF2_plddt", value=args.eval_plddt_cutoff, operator=">=", prefix="variants_AF2_plddt", plot=True)
        backbones.filter_poses_by_value(score_col="variants_AF2_tm_TM_score_ref", value=0.9, operator=">=", prefix=f"variants_AF2_TM_score", plot=True)
        backbones.filter_poses_by_value(score_col="variants_AF2_ESM_bb_rmsd", value=2.0, operator="<=", prefix="variants_AF2_ESM_bb_rmsd", plot=True) # check if AF2 and ESM predictions agree      
        backbones.filter_poses_by_value(score_col="variants_AF2_catres_bb_rmsd", value=args.eval_catres_bb_rmsd_cutoff, operator="<=", prefix=f"variants_AF2_catres_bb_rmsd", plot=True)

        # copy description column for merging with apo relaxed structures
        backbones.df['variants_AF2_relax_input_description'] = backbones.df['poses_description']
        apo_backbones = copy.deepcopy(backbones)

        # add ligand chain 
        backbones = chain_adder.superimpose_add_chain(
            poses = backbones,
            prefix = f"variants_AF2_ligand",
            ref_col = "updated_reference_frags_location",
            target_motif = "fixed_residues",
            copy_chain = "Z"
        )
        
        # calculate ligand clashes and ligand contacts
        backbones = ligand_clash.run(poses=backbones, prefix="variants_AF2_ligand")
        backbones = ligand_contacts.run(poses=backbones, prefix="variants_AF2_lig")

        # add covalent bonds info to poses pre-relax
        backbones = add_covalent_bonds_info(poses=backbones, prefix="variants_AF2_fastrelax_cov_info", covalent_bonds_col="covalent_bonds")

        # relax predictions with ligand present
        backbones = rosetta.run(
            poses = backbones,
            prefix = "variants_AF2_fastrelax",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 10,
            options = fr_options
        )

        # calculate RMSDs of relaxed poses
        backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"variants_AF2_postrelax_catres_heavy")
        backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"variants_AF2_postrelax_catres_bb")
        backbones = ligand_rmsd.run(poses = backbones, prefix = "variants_AF2_postrelax_ligand")

        # average values for all relaxed poses
        backbones = calculate_mean_scores(poses=backbones, scores=["variants_AF2_postrelax_catres_heavy_rmsd", "variants_AF2_postrelax_catres_bb_rmsd", "variants_AF2_postrelax_ligand_rmsd", "variants_AF2_fastrelax_sap_score"], remove_layers=1)
        
        # filter to relaxed pose with best score
        backbones.filter_poses_by_rank(n=1, score_col="variants_AF2_fastrelax_total_score", remove_layers=1)

        # relax apo poses
        apo_backbones = rosetta.run(
            poses = apo_backbones,
            prefix = "variants_AF2_fastrelax_apo",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 10,
            options = fr_options
        )

        # calculate apo relaxed rmsds
        apo_backbones = catres_motif_heavy_rmsd.run(poses = apo_backbones, prefix = f"variants_AF2_postrelax_apo_catres_heavy")
        apo_backbones = catres_motif_bb_rmsd.run(poses = apo_backbones, prefix = f"variants_AF2_postrelax_apo_catres_bb")
        apo_backbones = calculate_mean_scores(poses=apo_backbones, scores=["variants_AF2_postrelax_apo_catres_heavy_rmsd", "variants_AF2_postrelax_apo_catres_bb_rmsd"], remove_layers=1)

        # filter to relaxed pose with best score, merge dataframes
        apo_backbones.filter_poses_by_rank(n=1, score_col="variants_AF2_fastrelax_apo_total_score", remove_layers=1)
        preserve_cols = ['variants_AF2_relax_input_description', 'variants_AF2_fastrelax_apo_total_score', 'variants_AF2_postrelax_apo_catres_heavy_rmsd', 'variants_AF2_postrelax_apo_catres_heavy_rmsd_mean', 'variants_AF2_postrelax_apo_catres_bb_rmsd', 'variants_AF2_postrelax_apo_catres_bb_rmsd_mean']
        backbones.df = backbones.df.merge(apo_backbones.df[preserve_cols], on='variants_AF2_relax_input_description')
        
        # calculate delta score between apo and holo poses
        backbones.df['variants_AF2_delta_apo_holo'] = backbones.df['variants_AF2_fastrelax_total_score'] - backbones.df['variants_AF2_fastrelax_apo_total_score']

        # filter ligand rmsd
        backbones.filter_poses_by_value(score_col="variants_AF2_postrelax_ligand_rmsd_mean", value=args.eval_mean_ligand_rmsd_cutoff, operator="<=", prefix="variants_AF2_mean_ligand_rmsd", plot=True)        
        backbones.filter_poses_by_value(score_col="variants_AF2_postrelax_ligand_rmsd", value=args.eval_ligand_rmsd_cutoff, operator="<=", prefix="variants_AF2_ligand_rmsd", plot=True)        

        # calculate variants_AF2 composite score
        backbones.calculate_composite_score(
            name=f"variants_AF2_composite_score",
            scoreterms=["variants_AF2_plddt", "variants_AF2_tm_TM_score_ref", "variants_AF2_catres_bb_rmsd", "variants_AF2_catres_heavy_rmsd", "variants_AF2_delta_apo_holo", "variants_AF2_postrelax_ligand_rmsd", "variants_AF2_postrelax_catres_heavy_rmsd", "variants_AF2_fastrelax_sap_score_mean", "variants_AF2_postrelax_apo_catres_heavy_rmsd"],
            weights=[-2, -2, 8, 4, 2, 2, 2, 1, 2],
            plot=True
        )
    
        # plot mean results
        plots.violinplot_multiple_cols(
            dataframe=backbones.df,
            cols=["variants_AF2_catres_heavy_rmsd_mean", "variants_AF2_catres_bb_rmsd_mean", "variants_AF2_postrelax_catres_heavy_rmsd_mean", "variants_AF2_postrelax_catres_bb_rmsd_mean", "variants_AF2_postrelax_ligand_rmsd_mean", "variants_AF2_postrelax_apo_catres_heavy_rmsd_mean", "variants_AF2_fastrelax_sap_score_mean"],
            y_labels=["Angstrom", "Angstrom", "Angstrom", "Angstrom", "Angstrom", "Angstrom", "SAP score"],
            titles=["Mean AF2\nSidechain RMSD", "Mean AF2 catres\nBB RMSD", "Mean Relaxed\nSidechain RMSD", "Mean Relaxed catres\nBB RMSD", "Mean Relaxed ligand\nRMSD", "Mean Apo Relaxed\nSidechain RMSD", "Spatial Aggregation Propensity"],
            out_path=os.path.join(backbones.plots_dir, "variants_AF2_mean_rmsds.png"),
            show_fig=False
        )

        backbones.reindex_poses(prefix="variants_AF2_reindex", remove_layers=2 if not args.attnpacker_repack else 3)
        #backbones.filter_poses_by_rank(n=25, score_col='final_composite_score', prefix="final_composite_score", plot=True)

        # filter for unique diffusion backbones
        backbones.filter_poses_by_rank(n=5, score_col="variants_AF2_composite_score", remove_layers=2)

        create_variants_results_dir(backbones, os.path.join(args.working_dir, f"{variants_prefix}variants_results"))
        backbones.save_scores()



if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--riff_diff_dir", type=str, default=".", help="output_directory")
    argparser.add_argument("--working_dir", type=str, required=True, help="output_directory")

    # general optionals
    argparser.add_argument("--skip_refinement", action="store_true", help="Skip refinement and evaluation, only run screening.")
    argparser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation, only run screening and refinement.")
    argparser.add_argument("--params_files", type=str, default=None, help="Path to alternative params file. Can also be multiple paths separated by ';'.")
    argparser.add_argument("--attnpacker_repack", action="store_true", help="Run attnpacker on ESM and AF2 predictions")
    argparser.add_argument("--use_reduced_motif", action="store_true", help="Instead of using the full fragments during backbone optimization, just use residues directly adjacent to fixed_residues. Also affects motif_bb_rmsd etc.")

    # jobstarter
    argparser.add_argument("--cpu_only", action="store_true", help="Should only cpu's be used during the entire pipeline run?")
    argparser.add_argument("--max_gpus", type=int, default=10, help="How many GPUs do you want to use at once?")
    argparser.add_argument("--max_cpus", type=int, default=1000, help="How many cpus do you want to use at once?")

    # screening
    argparser.add_argument("--screen_input_json", type=str, default=None, help="Read in a poses json file containing input poses for screening (e.g. the successful_input_motifs.json from a previous screening run).")
    argparser.add_argument("--screen_decentralize_weights", type=str, default="20,40", help="Decentralize weights that should be tested during screening. Separated by ','.")
    argparser.add_argument("--screen_decentralize_distances", type=str, default="2,4", help="Decentralize distances that should be tested during screening. Separated by ','.")
    argparser.add_argument("--screen_input_poses", type=int, default=200, help="Number of input poses for screening. Poses will be selected according to <screen_input_selection>.")
    argparser.add_argument("--screen_input_selection", default="top", help="Can be either 'top' (default), 'random' or 'weighted'. Defines if motif library input poses are chosen based on score, at random or random weighted by score.")
    argparser.add_argument("--screen_num_rfdiffusions", type=int, default=5, help="Number of backbones to generate per input path during screening.")
    argparser.add_argument("--screen_skip_mpnn_rlx_mpnn", action="store_true", help="Skip LigandMPNN-RELAX-LigandMPNN steps and just run LigandMPNN once before prediction with ESMFold. Faster, but lower success rates (only recommended for initial screening purposes).")
    argparser.add_argument("--screen_num_mpnn_sequences", type=int, default=30, help="Number of LigandMPNN sequences per backbone that should be generated and predicted with ESMFold post-RFdiffusion.")
    argparser.add_argument("--screen_num_seq_thread_sequences", type=int, default=5, help="Number of LigandMPNN sequences that should be generated during the sequence threading phase (input for backbone optimization). Only used if <screen_mpnn_rlx_mpnn> is True.")
    argparser.add_argument("--screen_prefix", type=str, default=None, help="Prefix for screening runs for testing different settings. Will be reused for subsequent steps if not specified otherwise.")

    # refinement optionals
    argparser.add_argument("--ref_input_json", type=str, default=None, help="Read in a poses json file containing input poses for refinement. Screening will be skipped.")
    argparser.add_argument("--ref_prefix", type=str, default=None, help="Prefix for refinement runs for testing different settings.")
    argparser.add_argument("--ref_cycles", type=int, default=5, help="Number of Rosetta-MPNN-ESM refinement cycles.")
    argparser.add_argument("--ref_input_poses_per_bb", default=None, help="Filter the number of refinement input poses on an input-backbone level. This filter is applied before the ref_input_poses filter.")
    argparser.add_argument("--ref_input_poses", type=int, default=100, help="Maximum number of input poses for refinement cycles after initial RFDiffusion-MPNN-ESM-Rosetta run. Poses will be filtered by design_composite_score. Filter can be applied on a per-input-backbone level if using the flag --ref_input_per_backbone.")
    argparser.add_argument("--ref_num_mpnn_seqs", type=int, default=25, help="Number of sequences that should be created with LigandMPNN during refinement.")
    argparser.add_argument("--ref_catres_bb_rmsd_cutoff_end", type=float, default=0.7, help="End value for catres backbone rmsd filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_catres_bb_rmsd_cutoff_start", type=float, default=1.2, help="Start value for catres backbone rmsd filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_plddt_cutoff_end", type=float, default=85, help="End value for esm plddt filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_plddt_cutoff_start", type=float, default=75, help="Start value for esm plddt filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_ligand_rmsd_end", type=float, default=2.0, help="End value for esm plddt filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_ligand_rmsd_start", type=float, default=2.8, help="Start value for esm plddt filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_num_cycle_poses", type=int, default=3, help="Number of poses per unique diffusion backbone that should be passed on to the next refinement cycle.")
    argparser.add_argument("--ref_seq_thread_num_mpnn_seqs", type=float, default=3, help="Number of LigandMPNN output sequences during the initial, sequence-threading phase (pre-relax).")
    argparser.add_argument("--ref_start_cycle", type=int, default=1, help="Number from which to start cycles. Useful if adding additional refinement cycles after a run has completed.")

    # evaluation
    argparser.add_argument("--eval_prefix", type=str, default=None, help="Prefix for evaluation runs for testing different settings or refinement outputs.")
    argparser.add_argument("--eval_input_json", type=str, default=None, help="Read in a custom poses json containing input poses for evaluation.")
    argparser.add_argument("--eval_input_poses", type=int, default=100, help="Maximum number of input poses for evaluation with AF2 after refinement. Poses will be filtered by design_composite_score.")
    argparser.add_argument("--eval_input_poses_per_bb", type=int, default=10, help="Maximum number of input poses per unique diffusion backbone for evaluation with AF2 after refinement. Poses will be filtered by design_composite_score")
    argparser.add_argument("--eval_mean_plddt_cutoff", type=float, default=80, help="Cutoff for mean plddt over all AF2 models of each pose.")
    argparser.add_argument("--eval_mean_catres_bb_rmsd_cutoff", type=float, default=1.0, help="Cutoff for mean catres backbone rmsd over all AF2 models of each pose.")
    argparser.add_argument("--eval_mean_ligand_rmsd_cutoff", type=int, default=2.5, help="Cutoff for mean ligand rmsd over all relaxed models of the top AF2 model for each pose.")
    argparser.add_argument("--eval_plddt_cutoff", type=float, default=85, help="Cutoff for plddt for the AF2 top model for each pose.")
    argparser.add_argument("--eval_catres_bb_rmsd_cutoff", type=float, default=0.7, help="Cutoff for catres backbone rmsd for the AF2 top model for each pose.")
    argparser.add_argument("--eval_ligand_rmsd_cutoff", type=int, default=2, help="Cutoff for ligand rmsd for the top relaxed model of the top AF2 model for each pose.")
    argparser.add_argument("--eval_drop_previous_results", action="store_true", help="Drop all evaluation columns from poses dataframe (useful if running evaluation again, e.g. after refining first evaluation output)")

    # variant generation
    argparser.add_argument("--variants_prefix", type=str, default=None, help="Prefix for variant generation runs for testing different variants.")
    argparser.add_argument("--variants_input_json", type=str, default=None, help="Read in a custom json containing poses from evaluation output.")
    argparser.add_argument("--variants_mutations_csv", type=str, default=None, help="Read in a custom csv containing poses description and mutation columns.")
    argparser.add_argument("--variants_input_poses_per_bb", type=int, default=5, help="Number of poses per unique backbone that should go into the evaluation step of variant generation.")
    argparser.add_argument("--variants_input_poses", type=int, default=50, help="Number of poses per unique backbone that should go into the evaluation step of variant generation.")
    argparser.add_argument("--variants_num_poses_per_bb", type=int, default=5, help="Read in a custom csv containing poses description and mutation columns.")
    argparser.add_argument("--variants_evaluation_input_poses_per_bb", type=int, default=30, help="Number of poses per unique backbone that should go into the evaluation step of variant generation.")
    argparser.add_argument("--variants_evaluation_input_poses", type=int, default=200, help="Number of poses per unique backbone that should go into the evaluation step of variant generation.")

    # rfdiffusion optionals
    argparser.add_argument("--recenter", type=str, default=None, help="Point (xyz) in input pdb towards the diffusion should be recentered. example: --recenter=-13.123;34.84;2.3209")
    argparser.add_argument("--flanking", type=str, default="split", help="How flanking should be set. nterm or cterm also valid options.")
    argparser.add_argument("--flanker_length", type=int, default=30, help="Set Length of Flanking regions. For active_site model: 30 (recommended at least).")
    argparser.add_argument("--total_length", type=int, default=200, help="Total length of protein to diffuse. This includes flanker, linkers and input fragments.")

    # ligandmpnn optionals
    argparser.add_argument("--ligandmpnn_options", type=str, default=None, help="Options for ligandmpnn runs.")

    # filtering options
    argparser.add_argument("--rfdiffusion_max_clashes", type=int, default=20, help="Filter rfdiffusion output for ligand-backbone clashes before passing poses to LigandMPNN.")
    argparser.add_argument("--rfdiffusion_min_ligand_contacts", type=float, default=7, help="Filter rfdiffusion output for number of ligand contacts (Ca atoms within 8A divided by number of ligand atoms) before passing poses to LigandMPNN.")
    argparser.add_argument("--rfdiffusion_max_rog", type=float, default=19, help="Filter rfdiffusion output for radius of gyration before passing poses to LigandMPNN.")
    argparser.add_argument("--min_ligand_contacts", type=float, default=3, help="Minimum number of ligand contacts per ligand heavyatom for the design to be a success.")
    argparser.add_argument("--ligand_clash_factor", type=float, default=0.8, help="Factor for determining clashes. Set to 0 if ligand clashes should be ignored.")

    arguments = argparser.parse_args()

    
    main(arguments)

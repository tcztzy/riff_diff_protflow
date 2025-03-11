"""
MD Simulation Runner Script
===========================

This script automates the process of running Molecular Dynamics (MD) simulations on 
predicted protein structure (.pdb) files within a specified directory using ProtFlow. It leverages 
Gromacs for simulation execution and supports both individual .pdb files and directories 
containing multiple .pdb files. The script is designed to handle parameter validation, 
simulation setup, job management, and logging of the simulation process.

Dependencies:
    - protflow
    - Gromacs from protflow.tools.gromacs

Usage:
    python md_starter.py --input <path_to_pdb_or_directory> \
                                    --md_mdp <path_to_mdp_file> \
                                    --output_dir <output_directory> \
                                    [--n_sims <number_of_simulations>]

Example:
    python md_starter.py --input ./pdb_files \
                                    --md_mdp params/md.mdp \
                                    --output_dir ./md_outputs \
                                    --n_sims 5
"""
import os
import glob
import logging
import argparse
import pandas as pd

# dependencies
import protflow
from protflow.tools.gromacs import Gromacs, MDAnalysis

def setup_logging(output_dir):
    """
    Configures logging to output messages to both the console and a logfile.

    Args:
        output_dir (str): Path to the directory where the logfile will be stored.

    Returns:
        None
    """
    log_file = os.path.join(output_dir, 'md_simulations.log')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up. Log file: %s", log_file)

def main(args):
    ''
    # Initialize logging
    setup_logging(args.output_dir)

    # check if md-params file exists
    if not os.path.isfile(args.md_mdp):
        logging.error("MD parameter file not found: %s", args.md_mdp)
        raise ValueError(f"Parameter 'md_mdp' must be path to valid .mdp file! Current 'md_mdp': {args.md_mdp}")

    # prep inputs
    if os.path.isdir(args.input):
        pdb_fl = glob.glob(f"{args.input}/*.pdb")
        if not pdb_fl:
            logging.error("No .pdb files found in directory: %s", args.input)
            raise ValueError(f"No .pdb files found in the directory: {args.input}")
        logging.info("Found %d .pdb files in directory: %s", len(pdb_fl), args.input)

    elif os.path.isfile(args.input) and args.input.endswith(".pdb"):
        pdb_fl = [args.input]
        logging.info("Single .pdb file provided: %s", args.input)
    else:
        logging.error("Invalid input path: %s", args.input)
        raise ValueError(f"Parameter 'input' must be either a .pdb file or a directory that holds any number of .pdb files > 1. Current 'input': {args.input}")

    # setup poses
    sims = protflow.poses.Poses(
        poses = pdb_fl,
        work_dir = args.output_dir
    )

    # setup jobstarter
    sbatch_jst = protflow.jobstarters.SbatchArrayJobstarter(
        max_cores = 16,
        options = "--nodes=1 --ntasks-per-node=4 --gpus-per-node=1"
    )
    sbatch_jst._use_bash(True)
    logging.info("Jobstarter configured with max_cores=4 and specified options.")

    # setup gromacs runner
    gmx = Gromacs()
    gmx.md_params.set_params(md=args.md_mdp)
    logging.info("Gromacs runner initialized with MD parameters: %s", args.md_mdp)

    # start
    logging.info(f"Starting MD simulations on {len(sims)*args.n_sims} poses. This might take a while.")
    gmx.run(
        poses = sims,
        prefix = "md",
        jobstarter = sbatch_jst,
        n = args.n_sims
    )

    # wrap up
    logging.info("MD simulations finished and files are ready for analysis.")

    if args.ref_df and args.mdanalysis_script and args.reference_frags_dir:
        logging.info(f"--ref_df and --mdanalysis_script specified. Running {args.mdanalysis_script} on MD simulations based on information from {args.ref_df}")

        # merge ref_df into poses.df based on description?
        ref_df = pd.read_json(args.ref_df).reset_index(drop=True)

        # including RA95.5-8F in simulation (this needs to be fixed later)
        ra95_dict = {
            "poses_description": ["af2"],
            "fixed_residues": [{"A": [51, 83, 110, 180]}],
            "reference_filename": ["A0-B0-C0-D0_5an7.pdb"]
        }

        # add RA95.5-8F into ref_df
        ra95 = pd.DataFrame.from_dict(ra95_dict)
        ref_df = pd.concat([ref_df, ra95]).reset_index(drop=True)

        # prepare merging dataframes:
        sims.df["merge_col"] = sims.df["poses_description"].str.split("_").str[:-3].str.join("_")
        print(sims.df['merge_col'].values)
        print(ref_df["poses_description"].values)

        sims.df = sims.df.merge(ref_df, left_on="merge_col", right_on="poses_description")
        sims.df.rename(columns={"poses_description_x": "poses_description", "poses_description_y": "ref_poses_description"}, inplace=True)

        # now extract catalytic positions and reference frags location
        sims.df["reference_frags_location"] = args.reference_frags_dir + sims.df["reference_filename"]
        sims.df["catpos"] = sims.df["fixed_residues"].map(lambda val: "["+",".join([str(id) for id in val["A"]])+"]")

        # instantiate md analysis runner and its own jobstarter (will run on cpus for better performance)
        mda_jst = protflow.jobstarters.SbatchArrayJobstarter(max_cores=64)
        md_analysis = MDAnalysis(python=args.md_analysis_env, script_path=args.mdanalysis_script)

        # prep arguments for md_analysis
        mda_prefix = "md_analysis"
        md_analysis.set_pose_options({
            "gro_file": "md_t0_extract_poses", # location column for .gro file
            "trajectory_file": "md_fit_poses", # location column of .xtc file (refitted and waters removed)
            "reference_frag": "reference_frags_location", # location column of reference fragments (needs to be valid location)
            "catalytic_positions": "catpos", # column holding list of the catalytic positions as finished string for the commandline example: '[14,65,112,178]'
        })

        md_analysis.set_options(f"--output_dir {os.path.join(sims.work_dir, mda_prefix)}")

        # start md_analysis
        md_analysis.run(
            poses = sims,
            prefix = mda_prefix,
            jobstarter=mda_jst
        )

        # create plots


#trajectory_file:str, reference_frag: str, catalytic_positions: list[int], output_dir: str

if __name__ == "__main__":
    # MD Args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input", type=str, required=True, help="Path to directory or singular .pdb file that should be simulated with MD")
    argparser.add_argument("--md_mdp", type=str, required=True, help="Path to the md.mdp params file for the MD run.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Name of directory where outputs should be stored.")
    argparser.add_argument("--n_sims", type=int, default=1, help="Number of simulation replicates to run per input pose.")

    # analysis args
    argparser.add_argument("--ref_df", type=str, help="Path to DataFrame that holds reference fragments and catalytic residue IDs for which to run MDAnalysis.")
    argparser.add_argument("--mdanalysis_script", type=str, help="Path to md-analysis script to run on MD simulation outputs.")
    argparser.add_argument("--reference_frags_dir", type=str, help="Specify path to directory that holds reference frags.")
    argparser.add_argument("--md_analysis_env", type=str, default="/home/mabr3112/anaconda3/envs/mdanalysis/bin/python", help="Path to python env where you have mdanalysis dependencies installed.")
    arguments = argparser.parse_args()

    main(arguments)

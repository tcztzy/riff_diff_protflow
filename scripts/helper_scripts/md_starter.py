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
import logging
import glob
import argparse

# dependencies
import protflow
from protflow.tools.gromacs import Gromacs

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
        max_cores = 4,
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


if __name__ == "__main__":
    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input", type=str, required=True, help="Path to directory or singular .pdb file that should be simulated with MD")
    argparser.add_argument("--md_mdp", type=str, required=True, help="Path to the md.mdp params file for the MD run.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Name of directory where outputs should be stored.")
    argparser.add_argument("--n_sims", type=int, default=1, help="Number of simulation replicates to run per input pose.")
    arguments = argparser.parse_args()

    main(arguments)

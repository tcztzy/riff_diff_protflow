"""
Script to predict protein-ligand interactions for designed enzymes.
Predict protein ligand interactions with AF3.
"""
# imports
import os
import glob
import json
import logging
import argparse

# dependencies
import protflow
import numpy as np
from protflow.tools.alphafold3 import AlphaFold3

# Get matching between two RDKit molecules using Maximum Common Substructure
def main(args):
    '''does stuff'''
    # create output
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more details
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("log.txt"),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )
    logging.info(f"Running predict_protein_ligand.py on {args.input_dir} with ligand from {args.ligand_dir}")

    # parse jobstarter
    if args.jobstarter == "sbatch":
        jst = protflow.jobstarters.SbatchArrayJobstarter(max_cores=args.num_workers, gpus=1)
    elif args.jobstarter == "local":
        jst = protflow.jobstarters.LocalJobStarter(max_cores=args.num_workers)
    else:
        raise ValueError(f"Unsupported options for --jobstarter: {args.jobstarter}. Allowed options: {{sbatch, local}}")

    # setup af3 runner
    af3 = AlphaFold3(jobstarter=jst)

    # check if all ligand columns are present in poses_df
    ligands = glob.glob(os.path.join(args.ligand_dir, "*.json"))

    # if input poses are in .pdb format, convert to .fa files first.
    #if proteins.poses_list()[0].endswith(".pdb"):
    #    proteins.convert_pdb_to_fasta("pdbs_to_fasta", update_poses=True)

    # loop over ligands
    for i, ligand in enumerate(ligands):
        # first load in the ligand
        logging.info(f"Reading Ligand from {ligand}")
        with open(ligand, 'r', encoding="UTF-8") as f:
            ligand_dict = json.load(f)

        # load poses from DataFrame.
        batch = protflow.poses.Poses(poses=args.input_dir, glob_suffix="*.pdb", work_dir=os.path.join(args.output_dir, f"batch_{i+1}"))
        logging.info(f"Loaded {len(batch)} poses from input {args.input_dir}.")

        # run af3 on ligand
        logging.info(f"Predicting structure ligand from {ligand}")
        af3.run(
            poses=batch,
            prefix=ligand,
            nstruct=15,
            additional_entities=ligand_dict, # assumes file {ligand} points to correctly formatted ligand
            options="--flash_attention_implementation xla --cuda_compute_7x 1",
            single_sequence_mode=False,
            user_ccd=args.custom_ccd_dir,
            return_top_n_models=15
        )

    logging.info(f"Prediction of {len(args.ligand_cols)} complex structures completed")

    ### questionable assignment of ligand atoms.
    # setup MotifRMSD runner for ligands.


    # calculate Ligand RMSDs


if __name__ == "__main__":
    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required options
    argparser.add_argument("--input_dir", required=True, help="Directory that contains .fa files that should be predicted")
    argparser.add_argument("--ligand_dir", required=True, help="Columns in input_poses DataFrame that contain ligands to be predicted with poses. example: --ligand_cols='lig1_col,lig2_col,lig3_col'")
    argparser.add_argument("--custom_ccd_dir", type=str, default=None, help="Path to directory that contains custom ccd codes.")
    argparser.add_argument("--output_dir", required=True, help="Path to directory where predicted structures shall be stored")
    argparser.add_argument("--jobstarter", type=str, default="sbatch", help="{sbatch, local} Specify which jobstarter class to use for batch downloads.")
    argparser.add_argument("--num_workers", type=int, default=10, help="Number of processes to run in parallel")
    arguments = argparser.parse_args()

    main(arguments)

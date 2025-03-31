"""
Script to predict protein-ligand interactions for designed enzymes.
Predict protein ligand interactions with AF3.
"""
# imports
import os
import logging
import argparse


# dependencies
import protflow
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from Bio.PDB import PDBParser, Select
from Bio.PDB.Polypeptide import is_aa
from protflow.tools.alphafold3 import AlphaFold3

# Extract ligand coordinates from PDB file directly
def extract_ligand_coords(pdb_file, ligand_resname, chain_id, res_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)
    
    coords = []
    atom_names = []

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if residue.get_resname() == ligand_resname and residue.id[1] == res_id:
                        for atom in residue:
                            coords.append(atom.get_coord())
                            atom_names.append(atom.element)
    return atom_names, coords

## Create RDKit molecule from coordinates and atom types (without bonds)
#def create_rdkit_mol(atom_names, coords):
#    mol = Chem.RWMol()
#    conformer = Chem.Conformer(len(atom_names))
#
#    periodic_table = Chem.GetPeriodicTable()
#
#    for i, (element, coord) in enumerate(zip(atom_names, coords)):
#        atomic_num = periodic_table.GetAtomicNumber(element)
#        atom = Chem.Atom(atomic_num)
#        mol.AddAtom(atom)
#        conformer.SetAtomPosition(i, coord)
#
#    mol.AddConformer(conformer)
#
#    # Generate bonds based on distances
#    mol = Chem.rdmolops.AssignBondOrdersFromTemplate(Chem.AddHs(mol), mol)
#    mol = Chem.RemoveHs(mol)
#
#    return mol


# Get matching between two RDKit molecules using Maximum Common Substructure
def get_matching(mol1, mol2):
    res = rdFMCS.FindMCS([mol1, mol2])
    patt = Chem.rdmolfiles.MolFromSmarts(res.smartsString)

    match1 = mol1.GetSubstructMatch(patt)
    match2 = mol2.GetSubstructMatch(patt)

    mapping = dict(zip(match1, match2))

    return mapping

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
    logging.info(f"Running predict_protein_ligand.py on {args.input_dir} with ligand from {args.ligand_file}")

    # parse jobstarter
    if args.jobstarter == "sbatch":
        jst = protflow.jobstarters.SbatchArrayJobstarter(max_cores=args.num_workers, gpus=1)
    elif args.jobstarter == "local":
        jst = protflow.jobstarters.LocalJobStarter(max_cores=args.num_workers)
    else:
        raise ValueError(f"Unsupported options for --jobstarter: {args.jobstarter}. Allowed options: {{sbatch, local}}")

    # setup af3 runner
    af3 = AlphaFold3(jobstarter=jst)

    # load poses from DataFrame.
    proteins = protflow.poses.load_poses(poses_path=args.input_poses)
    logging.info(f"Loaded {len(proteins)} poses from input {args.poses_path}.")

    # check if all ligand columns are present in poses_df
    for ligand in args.ligand_cols:
        if ligand not in proteins.df.columns:
            raise KeyError(f"Ligand column {ligand} not found in poses DataFrame {args.input_poses}")

    # if input poses are in .pdb format, convert to .fa files first.
    if proteins.poses_list()[0].endswith(".pdb"):
        proteins.convert_pdb_to_fasta("pdbs_to_fasta", update_poses=True)

    # loop over ligands
    for ligand in args.ligand_cols:
        logging.info(f"Predicting structure using entity info of column {ligand}")
        af3.run(
            poses=proteins,
            prefix=ligand,
            nstruct=5,
            additional_entities=ligand,
            col_as_input=True,
            options="--flash_attention_implementation xla --cuda_compute_7x 1",
            single_sequence_mode=False
        )

    logging.info(f"Prediction of {len(args.ligand_cols)} complex structures completed")

    ### questionable assignment of ligand atoms.
    # setup MotifRMSD runner for ligands.


    # calculate Ligand RMSDs




if __name__ == "__main__":
    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required options
    argparser.add_argument("--input_poses", required=True, help="Directory that contains .fa files that should be predicted")
    argparser.add_argument("--ligand_cols", required=True, help="Columns in input_poses DataFrame that contain ligands to be predicted with poses. example: --ligand_cols='lig1_col,lig2_col,lig3_col'")
    argparser.add_argument("--ligand_rmsd_ref_cols", type=str, help="Columns that contain reference structures to calculate ligand RMSD for. example (where you calculate RMSD for lig1_col and lig3_col from the example of --ligand_cols): --ligand_rmsd_ref_cols='refcol_1,-,refcol_3'")
    argparser.add_argument("--output_dir", required=True, help="Path to directory where predicted structures shall be stored")
    argparser.add_argument("--jobstarter", type=str, default="sbatch", help="{sbatch, local} Specify which jobstarter class to use for batch downloads.")
    argparser.add_argument("--num_workers", type=int, default=10, help="Number of processes to run in parallel")

    arguments = argparser.parse_args()

    main(arguments)




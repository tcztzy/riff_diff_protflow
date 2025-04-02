"""
Script to create MD plots for RAD simulations.

Purpose of this script is to extract all data relevant 
to plotting from the raw MD Analysis DataFrames
so we can plot locally.
"""
# imports
import os
import logging
import argparse

# dependencies
import numpy as np
import pandas as pd

def compile_catres_functional_groups(row: pd.Series) -> list[str]:
    '''combines "reference_filename" column with catpos column to compile list of columns that selects for functional group atoms.'''
    # define which fragment specifies which atom names (catalytic residues)
    fragment_dict = {
        "A": "OH", # Tyr51
        "B": "NZ", # Lys83
        "C": "CG", # Asn110
        "D": "OH" # Tyr180
    }

    # derive order of catalytic residues from fragment ordering:
    fragment_order = [x[0] for x in row["reference_filename"].split("_")[0].split("-")] # will give list like this ["D", "C", "A", "B"]
    catpos = row["fixed_residues"]["A"] # will give list like this: [23, 70, 123, 175]

    # compile functional group columns, looks like this ["OH-23", "CG-70", "OH-123", "NZ-175"]
    catres_functional_groups = [f"{fragment_dict[frag]}-{str(atm_id)}" for frag, atm_id in zip(fragment_order, catpos)]
    return catres_functional_groups

def compile_internal_dist_cols(functional_atoms: list[str], cols: list[str]) -> list[str]:
    '''Columns for internal distances are pairs of functional group atoms. The pairings are sorted by residue index.'''
    # transform ["OH-23", ...] to {"23OH", ...}
    reparsed = {"".join(entry.split("-")[::-1]) for entry in functional_atoms}

    # transform columns from ["26C-26CA", ..] to [{"26C", "26CA"}, ...]
    columns_parsed = [set(x.split("-")) for x in cols]

    # extract functional group column index using set operation
    functional_group_col_idx_list = [i for i, col in enumerate(columns_parsed) if col <= reparsed] # checks if the set(col) is a subset of set(reparsed)

    # return columns from original cols list by index
    functional_group_cols = [cols[idx] for idx in functional_group_col_idx_list]

    return functional_group_cols

def main(args):
    '''Does stuff'''
    # setup output directory
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
    logging.info(f"Running extract_md_scores.py on {args.input_df}")

    if not os.path.isfile(args.input_df):
        raise FileNotFoundError(args.input_df)

    # input format check
    if not args.input_df.endswith(".json"):
        raise ValueError(f"--input_df must be in .json format. If you want a different format, edit the script (from pd.read_json() to pd.read_your_shitty_custom_format()) and remove this error. --input_df: {args.input_df}")

    # read in md_analysis scores DF (it references to all paths of MD runs)
    ref_df = pd.read_json(args.input_df) # this dataframe references the locations of all scorefiles that we want to analyze.
    
    # compile functional amino acids from reference_files col and fixedres:
    ref_df["functional_group_cols"] = ref_df.apply(compile_catres_functional_groups, axis=1)
    enzyme_dfs = [df for i, df in ref_df.groupby("input_poses")]
    enzyme_df_names = [name for name, df in ref_df.groupby("input_poses")]

    # write out enzyme df names
    with open(os.path.join(args.output_dir, "enzyme_names.csv"), 'w') as f:
        f.write(",".join(enzyme_df_names))

    # extract trajectory C-alpha RMSDs for (37 enzymes, 2001 time-points, 20 replicates)
    ca_rmsd_data = np.empty((len(enzyme_dfs), len(enzyme_dfs[0]), 2001))

    for i, enzyme_df in enumerate(enzyme_dfs):
        for j, replicate_df_path in enumerate(enzyme_df["md_analysis_rmsd_df_path"]):
            replicate_df = pd.read_csv(replicate_df_path)
            ca_rmsd_data[i, j, :] = replicate_df["C-alphas"].to_numpy() # [37, 20, 2001]
        print(f"extracting rmsd for enzyme {i+1}")

    # save
    np.savez(os.path.join(args.output_dir, "ca_rmsd.npz"), ca_rmsd_data)

    # extract reference distances for all four catalytic array amino acids
    sidechain_asmsd_data = np.empty((len(enzyme_dfs), 4, len(enzyme_dfs[0]), 2001)) # [37 enzymes, 4 catalytic amino acids, 20 replicates, 2001 timepoints]

    for enz_idx, enzyme_df in enumerate(enzyme_dfs):
        for repl_idx, (_, row) in enumerate(enzyme_df.iterrows()):
            # extract info
            replicate_df_path = row["md_analysis_reference_distances_df"]
            functional_group_atoms = row["functional_group_cols"]

            # load replicate data
            replicate_df = pd.read_csv(replicate_df_path)
            for aa_idx, amino_acid in enumerate(functional_group_atoms):
                sidechain_asmsd_data[enz_idx, aa_idx, repl_idx, :] = replicate_df[amino_acid].to_numpy() # [37, 4, 20, 2001]
        print(f"extracting sidechain refdist for enzyme {enz_idx+1}")

    # save ref dist data
    np.savez(os.path.join(args.output_dir, "as_reference_distances.npz"), sidechain_asmsd_data)

    # extract internal distance deviation for all four catalytic amino acid functional group atoms
    internal_distances_columns = np.empty((len(enzyme_dfs), 6), dtype=object)
    internal_distances_data = np.empty((len(enzyme_dfs), 6, 20, 2001)) # [37 enzymes, 6 functional group combinations, 20 replicates, 2001 timepoints]
    for enz_idx, enzyme_df in enumerate(enzyme_dfs):
        # collect columns for functional groups
        functional_atoms = enzyme_df.iloc[0]["functional_group_cols"]
    
        for repl_idx, (_, row) in enumerate(enzyme_df.iterrows()):
            # define data
            replicate_df_path = row["md_analysis_internal_df"]
            repl_ref_df_path = row["md_analysis_internal_ref_df"]
    
            # read dataframes
            internal_df = pd.read_csv(replicate_df_path)
            internal_ref_df = pd.read_csv(repl_ref_df_path)
    
            ### compile columns [70NZ-179OH, etc. ...]
            internal_cols = compile_internal_dist_cols(functional_atoms, list(internal_df.columns))
    
            # subtract measured internal distance from reference internal distances:
            for group_idx, col in enumerate(internal_cols):
                internal_df[col] = internal_df[col] - internal_ref_df.loc[0, col] # internal_ref_df is a DataFrame with only one row (reference distance is always the same)!
    
                # add data into numpy array
                internal_distances_data[enz_idx, group_idx, repl_idx, :] = internal_df[col].to_numpy()
    
                # add columns into index
                internal_distances_columns[enz_idx, group_idx] = col
        print(f"extracting sidechain refdist for enzyme {enz_idx+1}")
    # store internal distances data
    np.savez(os.path.join(args.output_dir, "internal_distances.npz"), internal_distances_data)
    np.savez(os.path.join(args.output_dir, "internal_distances_index.npz"), internal_distances_columns)

    # extract RMSF for the four catalytic amino acids
    rmsf_data = np.empty((len(enzyme_dfs), 4, 20)) # [37 enzymes, 4 catalytic amino acids, 20 replicates] (only one value per replicate)
    rmsf_columns = np.empty((len(enzyme_dfs), 4), dtype=object)
    for enz_idx, enzyme_df in enumerate(enzyme_dfs):
        for repl_idx, (_, row) in enumerate(enzyme_df.iterrows()):
            repl_df_path = row["md_analysis_rmsf_df"]
            functional_group_atoms = row["functional_group_cols"]
    
                # read data
            repl_rmsf_df = pd.read_csv(repl_df_path)
    
            # create hashable index for functional atoms
            repl_rmsf_df["resn_idx"] = repl_rmsf_df["atomname"].str.cat(repl_rmsf_df["resnum"].astype(str), sep="-")
            repl_rmsf_df = repl_rmsf_df.set_index("resn_idx")
    
            # collect functional atom RMSF values
            for fg_idx, functional_group in enumerate(functional_group_atoms):
                rmsf_data[enz_idx, fg_idx, repl_idx] = repl_rmsf_df.loc[functional_group, "RMSF"]
                rmsf_columns[enz_idx, fg_idx] = functional_group
        print(f"extracting rmsf refdist for enzyme {enz_idx+1}")

    # STORE
    np.savez(os.path.join(args.output_dir, "rmsf_data.npz"), rmsf_data)
    np.savez(os.path.join(args.output_dir, "rmsf_columns.npz"), rmsf_columns)


if __name__ == "__main__":
    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required options
    argparser.add_argument("--input_df", required=True, type=str, help="Path to DataFrame that was produced by md_starter.py")
    argparser.add_argument("--output_dir", required=True, type=str, help="Path to directory where extracted scores should be stored.")
    arguments = argparser.parse_args()

    main(arguments)

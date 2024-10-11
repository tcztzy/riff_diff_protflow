# riff_diff_protflow
ProtFlow Implementation of the RiffDiff pipeline to design enzymes from theozymes.

## INSTALLATION

The ProtFlow python library is required to run Riff-Diff. You can download it here: https://github.com/mabr3112/ProtFlow together with installation instructions. Required protein tools for Riff-Diff that need to be installed on your machine:
  - RFdiffusion (https://github.com/RosettaCommons/RFdiffusion)
  - ESMFold (https://github.com/facebookresearch/esm)
  - Rosetta (https://github.com/RosettaCommons/rosetta)
  - LigandMPNN (https://github.com/dauparas/LigandMPNN)
  - ColabFold (https://github.com/YoshitakaMo/localcolabfold)
  - AttnPacker (optional, https://github.com/MattMcPartlon/AttnPacker)
Make sure all of these are properly configured with ProtFlow (added to the .config file!)
The Riff-Diff pipeline was developed for SLURM-based job allocators.

All Riff-Diff scripts should be started within an active ProtFlow conda environment.

Clone this repository:

```
git clone https://github.com/mabr3112/riff_diff_protflow.git
```

Replace the potentials.py in your RFdiffusion installation folder with the potentials.py in the database directory to gain access to the custom pocket potential. Alternatively, you can just copy the relevant class (custom_recenter_ROG) into the RFdiffusion potentials.py script (do not forget to add the potentials to the dictionary of implemented potentials in the end of the script!)

```
cd riff_diff_protflow
cp database/potentials.py /path/to/RFdiffusion/rfdiffusion/potentials/potentials.py
```

If you want to experiment with the fragment picking pipeline, you need to download the fragment database from XYZ (optional).


## MANUAL

The riff-diff pipeline is divided into 3 major scripts:
  1) generation of fragments for a provided theozyme (identify_fragments.py)
  2) assembly of these fragments into a motif library (assemble_motif_library.py)
  3) generation of structures and refinement (structure_generation.py)
This manual will walk you through each of these steps to create proficient de novo enzymes or small molecule binders.

## FRAGMENT IDENTIFICATION

For each of the active site residues in the theozyme, fragments will be created by inverting the rotamers and attaching a backbone fragment. The fragments are mainly selected based on rotamer preference.
By default, the backbone fragment consists of a 7-residue idealized helical fragment, but this can be replaced by custom fragments if desired using the --fragment_pdb flag. The mandatory inputs for fragment identification (identify_fragments.py) are:
  --riff_diff_dir                 required to find paths to database etc
  --theozyme_pdb                  path to the theozyme input
  --theozyme_resnums              comma-separated list of all active site residues with chain information that should be present in the catalytic motif (e.g. 25A,38A,188B)
  --output_dir                    directory where all output will be saved
  --output_prefix                 to easily identify which fragments belong together, can be any string
  --ligands                       comma-separated list of all ligand residues with chain information that should be present in the catalytic motif (e.g. X188,Z1)
If you want to modify some options just for certain residues, you can run these separately using the same output directory and prefix.
Other important flags include:  
  --add_equivalent_func_groups    if one of the catalytic residues is ASP/ASN/ILE, also create fragments with GLU/GLN/VAL
  --channel_chain                 Riff-Diff uses a placeholder fragment to ensure binding pocket formation. If you want to provide a custom channel, provide the chain name
                                  of the respective chain in the theozyme pdb
  --rotamer_position              select position of the active site residue on the backbbone fragment (by default, all positions except for N- and C-terminus of the fragment are selected)
Instead of providing a predefined backbone fragment, Riff-Diff can also search the PDB for the most appropriate fragments based on rotamer and backbone occurrences using the flag --pick_frags_from_db. You need to download the fragment library from (insert_link_here) first and place it in the Riff-Diff database folder. This will result in fragments with higher rotamer probabilities, however these fragments typically yield worse output in the subsequent diffusion steps compared to the default idealized helix fragment. It is still experimental and not recommended!
After running the script, all selected fragments can be found in the output directory, together with some information about the selected rotamers and fragments in the respective subfolders.

## MOTIF LIBRARY ASSEMBLY

In the next step, fragments from all active site residues will be combined to create an active site motif (assemble_motif_library.py). Motifs with clashes will be removed automatically. The input is specified using the flag --input_dir, it should point to the output directory of the previous step (the folder containing .pdb and .json files for each active site residue). All .json files in this directory will be read in automatically. You also need to specifiy a working directory (this can be the same as the previous output directory) using --working_dir.
Other important options are:
  --channel_path                  replaces the default channel with the channel from this pdb
  --channel_chain                 in which chain the channel in the pdb at channel_path is found
  --preserve_channel_coordinates  preserves the coordinates of the custom channel instead of automatically placing it
  --no_channel_placeholder        do not add a channel
The main output of this script are multiple pdb files containing the active site motifs in the folder working_dir/motif_library_assembly/motif_library. This script will also create a .params for use in Rosetta in the ligand directory and a .json file in the working directory called selected_paths.json. This is the main input for the next step.

## PROTEIN GENERATION

Protein generation proceeds in several stages, all bundled within the structure_generation.py script. Specifying a working directory (--working_dir) is mandatory for all.

# Stage 1: Screening

During the screening stage, individual fragments within the .pdbs of the motif library will be connected by RFdiffusion. Sequences are generated using a combination of ProteinMPNN and Rosetta Relax, which are then predicted via ESMFold. You need to specify an input .json file (--screen_input_json), this is usually the output of the previous step (selected_paths.json).
Different RFdiffusion settings will be sampled according to the flags --screen_decentralize_weights (influences the strength of the custom ROG potential, higher = stronger) and --screen_decentralize_distances (places the center of the denoising trajectory along an axis from the center of the motif in the direction of the channel). Increasing the decentralize weights leads to more globular, compact proteins, but increasing it too far results in implausible backbones. If your screening output structures have very exposed active sites, it is recommended to increase the decentralize distances, if the substrate is too buried, it is generally recommended to decrease them. You can also set the center of the denoising trajectory manually, by providing coordinates to the --recenter flag.
Other important flags are:
  --screen_input_poses             determines how many motifs of the motif library are selected as input for rfdiffusion.
  --screen_input_selection         determines how structures are selected, it is generally recommended to use either "top" or "weighted".
  --screen_num_rfdiffusions        number of backbones that will be generated with RFdiffusion for each selected motif.
It is generally recommended to have a look at the screening output before continuing to the next stage using the flag --skip_refinement. This way the script will not continue automatically. You can find the output of the first stage in the directory XYZ.

# Stage 2: Refinement

In this stage, the screening output is iteratively refined by cycling sequence optimization-backbone optimization-sequence-optimization-structure prediction steps (LigandMPNN-Rosetta Relax-LigandMPNN-ESMFold). If you did not provide the --skip_refinement flag in the previous stage, it will start automatically. Otherwise, you can provide the output of the previous step (XYZ.json) via the flag --ref_input_json.
  --ref_cycles                      number of iterations for sequence optimization-backbone optimization-sequence-optimization-structure prediction.
  --ref_input_poses                 recommended to look at the screening output to determine how many structures are actually worth refining and select the number of input poses based on that.
The various filter steps can be adjusted via --ref_catres_bb_rmsd_cutoff_start/--ref_catres_bb_rmsd_cutoff_end, --ref_plddt_cutoff_start/--ref_plddt_cutoff_end. Filter values are ramped with each cycle, start and end values correspond to the first and last cycle.
You can again pause the script using --skip_evaluation. Due to prediction with AF2, evaluation is quite time-consuming.

# Stage 3: Evaluation

The refined structures are predicted using AF2 and evaluated by various metrics. If --skip_evaluation was set in the previous stage, you can provide the refinement output with --eval_input_json.
  --eval_input_poses                define based on number of promising refinement structures.
The output is is XYZ.

Stage 4: Diversification
This is an optional stage. It is used to diversify sequences for successful backbones and to fine-tune active sites. You can manually provide a list of mutations you want to introduce for each input structure (for instance, to open channels that are blocked by a sidechain).

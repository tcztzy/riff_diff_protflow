# riff_diff_protflow
ProtFlow Implementation of the RiffDiff pipeline to design enzymes from theozymes.


####################### MANUAL #######################

The riff-diff pipeline is divided into 3 major scripts:
  1) generation of fragments for a provided theozyme (identify_fragments.py)
  2) assembly of these fragments into a motif library (assemble_motif_library.py)
  3) generation of structures and refinement (structure_generation.py)
This manual will walk you through each of these steps to create proficient de novo enzymes or small molecule binders.

################### FRAGMENT IDENTIFICATION ##########
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
  --rotamer_position              select position of the active site residue on the backbbone fragment (by default, central position is selected)
Instead of providing a predefined backbone fragment, Riff-Diff can also search the PDB for the most appropriate fragments based on rotamer and backbone occurrences using the flag --pick_frags_from_db. You need to download the fragment library from (insert_link_here) first and place it in the Riff-Diff database folder. This will result in fragments with higher rotamer probabilities, however these fragments typically yield worse output in the subsequent diffusion steps compared to the default idealized helix fragment. It is still experimental and not recommended!
After running the script, all selected fragments can be found in the output directory, together with some information about the selected rotamers and fragments in the respective subfolders.

################### MOTIF LIBRARY ASSEMBLY ##########
In the next step, fragments from all active site residues will be combined to create an active site motif (assemble_motif_library.py). Motifs with clashes will be removed automatically. The input can be provided by either the paths to the .json output files in the output directory of the previous step (--json_files) or by providing the path including output_prefix (--output_prefix, in the format /path/to/output/output_prefix_). You also need to specifiy a working directory (this can be the same as the previous output directory) using --working_dir.
Other important options are:
  --channel_path                  replaces the default channel with the channel from this pdb
  --channel_chain                 in which chain the channel in the pdb at channel_path is found
  --preserve_channel_coordinates  preserves the coordinates of the custom channel instead of automatically placing it
  --no_channel_placeholder        do not add a channel
The main output of this script are multiple pdb files containing the active site motifs in the folder pdb_in. This script will also create a .params for use in Rosetta in the ligand directory.

##################

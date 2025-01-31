#!/usr/bin/bash
# Example command to generate an artificial motif library.
# In this case, we directly use the .pdb file of RA95.5-8F as input.
# This can be run locally. The script should complete in less than a minute.
# Make sure you run this with your python environment that has protflow installed.

python identify_fragments.py --jobstarter Local --riff_diff_dir ./ --theozyme_pdb examples/inputs/5an7.pdb --theozyme_resnums 1083A,1051A,1110A,1180A --working_dir examples/outputs/rad_fragments/ --ligands 5001A --add_equivalent_func_groups

#!/usr/bin/bash
# Example command to generate an artificial motif library.
# In this case, we directly use the .pdb file of RA95.5-8F as input.
# Make sure you run this with your python environment that has protflow installed.

python create_fragment_library.py --riff_diff_dir . --input_json sarah/in.json

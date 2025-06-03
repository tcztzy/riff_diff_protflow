#!/usr/bin/bash
# Example command to generate de novo enzymes from a motif library.
# Make sure you run this with your python environment that has protflow installed.

python ../structure_generation.py --riff_diff_dir ../ --working_dir outputs --screen_input_json outputs/selected_paths.json --attnpacker_repack

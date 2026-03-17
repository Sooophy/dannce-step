#!/bin/bash
#
#SBATCH --job-name=stroke_exp
#SBATCH --mem=20000                     # Job memory request
#SBATCH -t 6-23:59               # Time limit days-hrs:min:sec
#SBATCH --output=stroke_exp.out
set -e
source ~/.bashrc
source activate neuroposelib
python stroke_exp.py
#!/usr/bin/env bash
trap '{ rm "$input_file"; }' EXIT
set -e
set -o pipefail

export OMP_NUM_THREADS=16
a=2.46                        # Lattice constant in Å
t=2.7                         # Hopping parameter in eV
n=32                          # System size
kT=0.0256                     # Temperature in eV
mu=1.0                        # Chemical potential μ in eV
damping=0.1                   # Damping η in eV
freq=$(seq -s, 0.0 0.1 20.0)  # Comma-separated list of frequencies ω in eV
input_file="temp_square_sheet_${n}x${n}.h5"

# Generate input using TiPSi
. $(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh
conda activate tipsi_devel
python3 build.py -n "$n" -t "$t" -a "$a" "$input_file"

# Compute χ using Plasmons.jl
julia -e 'import Plasmons; Plasmons.julia_main()' -- \
	--kT "$kT" --mu "$mu" --damping "$damping" \
	--hamiltonian H \
	--coulomb V \
	--frequency "$freq" \
	--cuda 0 \
	"$input_file" "result_square_sheet_${n}x${n}.h5"
# TMPDIR=$HOME/tmp singularity run --fakeroot --bind=$PWD ../../Plasmons.sif \
# 	--kT "$kT" --mu "$mu" --damping "$damping" \
# 	--hamiltonian H \
# 	--frequency "$freq" \
# 	"$input_file" "square_sheet_${n}x${n}.h5"

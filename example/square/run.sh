#!/bin/sh
trap '{ rm "$input_file"; }' EXIT

export OMP_NUM_THREADS=4
a=2.46                        # Lattice constant in Å
t=2.7                         # Hopping parameter in eV
n=5                           # System size
kT=0.0256                     # Temperature in eV
mu=3.0                        # Chemical potential μ in eV
damping=0.1                   # Damping η in eV
freq=$(seq -s, 0.0 0.05 12.0) # Comma-separated list of frequencies ω in eV
input_file="temp_square_sheet_${n}x${n}.h5"

# Generate input using TiPSi
python3 build.py -n "$n" -t "$t" -a "$a" "$input_file"

# Compute χ using Plasmons.jl
julia -e 'import Plasmons; Plasmons.julia_main()' -- \
	--kT "$kT" --mu "$mu" --damping "$damping" \
	--hamiltonian H \
	--frequency "$freq" \
	"$input_file" "square_sheet_${n}x${n}_new.h5"
# TMPDIR=$HOME/tmp singularity run --fakeroot --bind=$PWD ../../Plasmons.sif \
# 	--kT "$kT" --mu "$mu" --damping "$damping" \
# 	--hamiltonian H \
# 	--frequency "$freq" \
# 	"$input_file" "square_sheet_${n}x${n}.h5"

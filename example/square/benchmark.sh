#!/bin/sh
set -e
set -o pipefail

. $(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh
conda activate tipsi_devel

export OMP_NUM_THREADS=16
a=2.46                        # Lattice constant in Å
t=2.7                         # Hopping parameter in eV
kT=0.0256                     # Temperature in eV
mu=3.0                        # Chemical potential μ in eV
damping=0.1                   # Damping η in eV
freq='0.1,0.2'                # Comma-separated list of frequencies ω in eV


run_once() {
	declare -r n="$1"
	declare -r use_cuda="$2"
	declare -r input_file="input_square_${n}x${n}.h5"
	if [ $use_cuda -eq 1 ]; then
        declare -r output_file="output_square_${n}x${n}_cuda.h5"
		declare -r extra_args=("--cuda" "0")
	else
        declare -r output_file="output_square_${n}x${n}_cpu.h5"
		declare -r extra_args=()
	fi
    # Compute χ using Plasmons.jl
    julia -O3 --project=../.. -e 'import Plasmons; Plasmons.julia_main()' -- \
        --kT "$kT" --mu "$mu" --damping "$damping" \
        --hamiltonian H \
        --frequency "$freq" \
		"${extra_args[@]}" \
        "$input_file" "$output_file"
}

# system_sizes=(5 10 15 20 25 30 35 40)
system_sizes=(45 50 55)
for n in "${system_sizes[@]}"; do
	input_file="input_square_${n}x${n}.h5"
	if [ ! -f "$input_file" ]; then
        # Generate input using TiPSi
        python3 build.py -n "$n" -t "$t" -a "$a" "$input_file"
	fi
	run_once $n 0
	run_once $n 1
done

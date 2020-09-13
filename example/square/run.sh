#!/bin/sh

export OMP_NUM_THREADS=4
exec julia -e 'import Plasmons; Plasmons.julia_main()' -- \
	--kT 0.025 --mu 0.4 --damping 0.006 \
	--hamiltonian H --coulomb V \
	--frequency "$(seq -s, 0.0 0.05 15.0)" \
	square_sheet_10x10.h5 result_10x10.h5

# Plasmons.jl

Within *real-space* RPA (Random Phase Approximation) the calculation of
polarizability matrix ``\chi(\omega)`` amounts to evaluation of the following
equation
```math
\begin{aligned}
    \chi_{a,b}(\omega)
        &= \langle a |\hat\chi(\omega)| b \rangle
        = 2\cdot \sum_{i,j} \langle i |\hat G(\omega)| j \rangle
           \langle j | a \rangle \langle a | i \rangle
           \langle i | b \rangle \langle b | j \rangle\; , \\
    G_{i,j}(\omega)
        &= \langle i | \hat G(\omega) | j \rangle
        = \frac{f_i - f_j}{E_i - E_j - (\hbar\omega + i\eta)} \;.
\end{aligned}
```

Here ``E_i`` is the ``i``'th energy eigenvalue, ``\langle a | i \rangle = \psi_i
(a)`` -- the ``i``'th eigenstate, ``f_i = f(E_i)`` -- occupation of the ``i``'th
state given by the Fermi-Dirac distribution, ``\eta`` is the Landau damping, and
``\omega`` is the frequency.

Then, if you have access to *bare* (also called *unscreened*) Coulomb interaction
``V``, dielectric function ``\varepsilon(\omega)`` can be computed as follows:

```math
    \varepsilon(\omega) = 1 - \chi(\omega) \cdot V \;.
```

And with ``\varepsilon(\omega)`` you essentially have full optical description
of your system and can do all kind of interesting physics.


!!! note

    This package allows you to compute ``\chi(\omega)`` and ``\varepsilon(\omega)``
    as efficiently as possible. Think of it as
    [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) for
    real-space RPA: writing matrix-matrix multiplication function is trivial,
    but good luck making it competitive with OpenBLAS or Intel MKL! Same thing
    here: calculation of ``\chi(\omega)`` can be done in 10 lines of code, but
    boy will it be slow! 😝


## Installing

[`Plasmons.jl`](https://github.com/twesterhout/Plasmons.jl) can be used either
as a Julia package or as a [Singularity](https://sylabs.io/) container.

#### Julia package

Install using Pkg:

```julia
import Pkg; Pkg.add(url="https://github.com/twesterhout/Plasmons.jl")
```


#### Singularity container

This option is provided for people who do not care about Julia and just want to
do physics.

Download `Plasmons.sif` file from the
[Releases](https://github.com/twesterhout/Plasmons.jl/releases/latest) page and
you are good to go.


## Example: square lattice

This is a very simple example of using `Plasmons.jl` to calculate dielectric
function of a square lattice sheet with periodic boundary conditions.

All code for this example is located in
[`examples/square`](https://github.com/twesterhout/Plasmons.jl/tree/master/example/square)
folder.

### Generating input HDF5 file

First, we need to generate the Hamiltonian for our system. For this we will use
[TiPSi](http://www.edovanveen.com/tipsi/) Python package. There is a Conda
environment file
[`conda-tipsi.yml`](https://github.com/twesterhout/Plasmons.jl/tree/master/example/square/conda-tipsi.yml). So if you do not yet have TiPSi installed, run

```sh
conda env create --file conda-tipsi.yml # Creates tipsi_devel environment
conda activate tipsi_devel # Now you have access to TiPSi
```

We can not run `build.py`:

```sh
python3 build.py -n 10 input_square_10x10.h5
```

This will build the sample and write it to `input_square_10x10.h5` file. See the
source of
[`build.py`](https://github.com/twesterhout/Plasmons.jl/tree/master/example/square/build.py)
for more info.

**TODO:** Finish this


## Reference

```@docs
    Plasmons.polarizability
    Plasmons.dielectric
    Plasmons.dispersion
```

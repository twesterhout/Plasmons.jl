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
    real-space RPA: writing matrix-matrix multiplication function is trivial and
    everybody can do it, but good luck making it competitive with OpenBLAS or Intel
    MKL! Same thing here: calculation of ``\chi(\omega)`` can be done in 10 lines of
    code, but boy will it be slow! 😝


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


## Using the executable

After downloading the container (`Plasmons.sif` file) from the
[Releases](https://github.com/twesterhout/Plasmons.jl/releases) page one can
simply run it. It follows the UNIX philosophy and tries to do one thing and do
it well.

The one thing is calculating polarizability ``\chi`` (or dielectric function
``\varepsilon``). It can be described by the following two functions:

```math
\begin{aligned}
    \left(H, \omega, \mu, T\right) &\mapsto \chi \\
    \left(\chi, V\right) &\mapsto \varepsilon
\end{aligned}
```

This means that if you provide a Hamiltonian ``H`` and frequency ``\omega`` (and
some information about the environment, namely chemical potential ``\mu`` and
temperature ``T``), `Plasmons.sif` will compute ``\chi(\omega)`` for your
system. If you additionally provide unscreened Coulomb interaction ``V``,
`Plasmons.sif` will compute ``varepsilon(\omega)`` as well.



## Reference

```@docs
    Plasmons.polarizability
    Plasmons.dielectric
    Plasmons.dispersion
```

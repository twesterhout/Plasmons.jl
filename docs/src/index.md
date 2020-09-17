# Plasmons.jl

## Installing

Plasmons.jl can be used either as a Julia package or as a standalone executable.
The latter option is provided for people who do not care about Julia and just
want to do physics. If you are one of them just download `Plasmons.sif`
executable file from the
[Releases](https://github.com/twesterhout/Plasmons.jl/releases) page and you are
good to go.

Otherwise, install Plasmons.jl using Pkg:
```julia
import Pkg; Pkg.add(url="https://github.com/twesterhout/Plasmons.jl")
```


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



## Algorithm

Within real-space RPA (Random Phase Approximation) the calculation of
polarizability matrix ``\chi(\omega)`` amounts to evaluation of the following equation
```math
\begin{aligned}
    \chi_{a,b}
        &= \langle a |\hat\chi| b \rangle
        = 2\cdot \sum_{i,j} \langle i |\hat G| j \rangle
           \langle j | a \rangle \langle a | i \rangle
           \langle i | b \rangle \langle b | j \rangle\; , \\
    G_{i,j}
        &= \langle i | \hat G | j \rangle
        = \frac{f_i - f_j}{E_i - E_j - (\hbar\omega + i\eta)} \;.
\end{aligned}
```

Here ``E_i``s are energy eigenvalues, ``\langle a | i \rangle = \psi_i (a)`` are
energy eigenstates, ``f_i = f(E_i)`` is the occupation of the ``i``'th state
given by the Fermi-Dirac distribution, ``\eta`` is Landau damping, and
``\omega`` is the frequency.

First of all, for a given temperature ``T`` and chemical potential ``\mu``, we
compute occupation numbers ``f(E_i)``.

```@docs
    Plasmons.fermidirac
```

Next, we compute the matrix ``G``. This is done by either [`Plasmons._g`](@ref)
or [`Plasmons._g_blocks`](@ref) functions.

```@docs
    Plasmons._g
    Plasmons._g_blocks
```

Let us now turn to the computation of polarizability matrix ``\chi``. The naive
approach is to write 4 nested loops. However, this is tremendously slow! A
slightly better approach which I used for my Bachelor thesis is to rewrite
the computation of each ``\chi_{a, b}`` as a combination of `GEMV` and `DOT`
operations:

```math
\begin{aligned}
    \chi_{a, b}
        &= \sum_{i, j} \left(\langle a | i \rangle \langle i | b \rangle\right)
            G_{i, j}
            \left(\langle j | a \rangle \langle b | j \rangle\right)
         = A(a,b)^\dagger G A(a,b) \;, \\
        &\text{where } A(a,b)_j = \langle j | a \rangle \langle b | j \rangle \;.
\end{aligned}
```

We use the following data structure to hold ``A`` as well as the temporary ``G
A``.

```@docs
    Plasmons._Workspace
```

**TODO**: Finish this.

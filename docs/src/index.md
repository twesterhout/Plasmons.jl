# Plasmons.jl


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

Let us now turn to the computation of polarizability matrix elements ``\chi_{a, b}``.

```@index
```

```@autodocs
Modules = [Plasmons]
```

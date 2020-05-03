module Plasmons

export dielectric
export polarizability_batched, polarizability_thesis, polarizability_simple
export coulomb_simple
export read_hamiltonian, read_coordinates

using LinearAlgebra

"""
    fermidirac(E; mu, kT)

Return Fermi-Dirac distribution at energy `E`, chemical potential `mu`, and temperature
`kT`. Note that `kT` is assumed to be temperature which has been multiplied by Boltzmann
constant, i.e. physical dimension of `kT` is the same as `E` (e.g. electron-volts).
"""
@inline fermidirac(E; mu, kT) = 1 / (1 + exp((E - mu) / kT))


@doc raw"""
    _g(ħω, E; mu, kT) -> G

**This is an internal function!**

Compute matrix G
```math
    G_{ij} = \frac{f(E_i) - f(E_j)}{E_i - E_j - \hbar\omega}
```
where ``f`` is Fermi-Dirac distribution at chemical potential `mu` and temperature `kT`. `E`
is a vector of eigenenergies. `ħω` is a complex frequency including Landau damping (i.e.
``\hbar(\omega + i\eta)``). All arguments are assumed to be in energy units.
"""
function _g(ħω::Complex{R}, E::AbstractVector{R}; mu::R, kT::R) where {R <: Real}
    # Compared to a simple `map` the following saves one allocation
    f = similar(E)
    map!(x -> fermidirac(x, mu = mu, kT = kT), f, E)
    n = length(E)
    G = similar(E, complex(R), n, n)
    @inbounds for j in 1:n
        @. G[:, j] = (f - f[j]) / (E - E[j] - ħω)
    end
    return G
end


@doc raw"""
    polarizability_thesis(ħω, E, ψ; mu, kT) -> χ

Calculates the polarizability matrix ``\chi`` by using methods from the Bachelor thesis.
"""
polarizability_thesis(ħω, E, ψ; mu, kT) =
    _polarizability_thesis(_g(ħω, E; mu = mu, kT = kT), ψ)

"""
    _Workspace{<: AbstractArray}

**This is an internal data structure!**

A workspace which is used by `_polarizability_*` functions to avoid allocating many
temporary arrays.
"""
struct _Workspace{T <: AbstractArray}
    A::T
    temp::T
end

# An optimisation for the case when ψ is real. This reduces the amount of computations by a
# factor 2.
function _polarizability_thesis(
    G::AbstractMatrix{Complex{T}},
    ψ::AbstractMatrix{T},
) where {T <: Real}
    χ = similar(G)
    ws = _Workspace(similar(G, size(G, 2)), similar(G, size(G, 2)))
    @inbounds for b in 1:size(G, 2)
        χ[b, b] = _thesis_mat_el!(ws, b, b, G, ψ)
        for a in (b + 1):size(G, 1)
            χ[a, b] = _thesis_mat_el!(ws, a, b, G, ψ)
            χ[b, a] = χ[a, b]
        end
    end
    return χ
end

# General fallback for the case when both G and ψ are complex. No implementation for real G
# is provided since it's singular without Landau damping η.
function _polarizability_thesis(
    G::AbstractMatrix{Complex{T}},
    ψ::AbstractMatrix{Complex{T}},
) where {T <: Real}
    χ = similar(G)
    ws = _Workspace(similar(G, size(G, 2)), similar(G, size(G, 2)))
    @inbounds for b in 1:size(G, 2)
        for a in 1:size(G, 1)
            χ[a, b] = _thesis_mat_el!(ws, a, b, G, ψ)
        end
    end
    return χ
end

@doc raw"""
    _thesis_mat_el!(ws, a::Int, b::Int, G, ψ) -> χ[a, b]

**This is an internal function!**

Compute entry `χ[a, b]` of the polarizability matrix using the method described in the
Bachelor thesis. It uses a combination of GEMV & CDOT.
"""
function _thesis_mat_el!(ws::_Workspace{<:AbstractVector}, a::Int, b::Int, G, ψ)
    for i in 1:size(ψ, 2)
        @inbounds ws.A[i] = ψ[a, i] * conj(ψ[b, i])
    end
    mul!(ws.temp, transpose(G), ws.A)
    return 2 * dot(ws.A, ws.temp)
end


polarizability_batched(ħω, E, ψ; mu, kT) =
    _polarizability_batched(_g_blocks(ħω, E; mu = mu, kT = kT)..., ψ)

function _polarizability_batched(Gᵣ, Gᵢ, ψ::AbstractMatrix{T}) where {T <: Real}
    χ = similar(ψ, complex(T))
    fill!(χ, zero(eltype(χ)))
    ws = _Workspace(similar(ψ), similar(ψ))
    for b in 1:size(ψ, 2)
        a = b:size(ψ, 1)
        # BANG! view of χ is allocated on the heap...
        _batched_mat_el!(view(χ, a, b), ws, a, b, Gᵣ, Gᵢ, ψ)
    end
    for b in 2:size(ψ, 2)
        for a in 1:(b - 1)
            @inbounds χ[a, b] = χ[b, a]
        end
    end
    return χ
end

# TODO: Add dispatch for CuArrays because _compute_A_loops! will fail
_compute_A!(A, as, b, ψ) = _compute_A_loops!(A, as, b, ψ)

function _compute_A_loops!(
    out::AbstractMatrix{T},
    as::UnitRange{Int},
    b::Int,
    ψ::AbstractMatrix{T},
) where {T}
    # Effectively we want this:
    #     A = view(ψ, as, :) .* transpose(view(ψ, b, :))
    # but then without memory allocations
    offset = as.start - 1
    @inbounds for j in 1:size(ψ, 2)
        scale = conj(ψ[b, j])
        @simd for a in as
            out[a - offset, j] = scale * ψ[a, j]
        end
    end
end

# This was an attempt to improve _compute_A_generic by using SIMD intrinsics. It amounted to
# a whole bowl of nothing :) Julia optimises the loops well enough by itself. Performance of
# _compute_A_generic! is within a factor 2 of copyto!.

# Base.@propagate_inbounds function _compute_column!(
#     out::Ptr{T},
#     blocks::Int,
#     remainder::Int,
#     scale::T,
#     psi::Ptr{T},
# ) where {T}
#     N = div(64, sizeof(T))
#     c = Vec{N, T}(scale)
#     i = 0
#     while i < 64 * blocks
#         vstore(c * vload(Vec{N, T}, psi + i), out + i)
#         i += 64
#     end
#     mask = Vec{N, T}(remainder) > Vec{N, T}((0, 1, 2, 3, 4, 5, 6, 7))
#     vstore(c * vload(Vec{N, T}, psi + i, mask), out + i, mask)
# end
#
# function _compute_A_simd!(
#     A::AbstractMatrix{T},
#     a::Int,
#     b::Int,
#     ψ::AbstractMatrix{T},
# ) where {T}
#     blocks, remainder = divrem(size(A, 1), div(64, sizeof(T)))
#     δA = 1
#     δψ = a
#     @inbounds for j in 1:size(A, 2)
#         scale = conj(ψ[b, j])
#         _compute_column!(pointer(A, δA), blocks, remainder, scale, pointer(ψ, δψ))
#         δA += stride(A, 2)
#         δψ += stride(ψ, 2)
#     end
# end


_dot_many!(out, A, B, scale) = _dot_many_loops!(out, A, B, complex(scale))

function _dot_many_loops!(
    out::AbstractVector{Complex{T}},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    scale::Complex{T},
) where {T}
    @inbounds for j in 1:size(A, 2)
        @simd for i in 1:length(out)
            out[i] += scale * A[i, j] * B[i, j]
        end
    end
end

@noinline function _batched_mat_el!(
    out::AbstractVector,
    ws::_Workspace{<:AbstractMatrix},
    as::UnitRange{Int},
    b::Int,
    Gᵣ,
    Gᵢ,
    ψ,
)
    A = view(ws.A, 1:length(as), :) # BANG! Allocated on the heap...
    temp = view(ws.temp, 1:length(as), :) # BANG! Allocated on the heap...
    _compute_A!(A, as, b, ψ)
    mul!(temp, A, Gᵣ)
    _dot_many!(out, A, temp, 2.0)
    mul!(temp, A, Gᵢ)
    _dot_many!(out, A, temp, 2.0im)
end

struct _ThreeBlockMatrix{
    T,
    T₁ <: AbstractMatrix{T},
    T₂ <: AbstractMatrix{T},
    T₃ <: AbstractMatrix{T},
}
    block₁::T₁
    block₂::T₂
    block₃::T₃
end

function _ThreeBlockMatrix(G::AbstractMatrix, n₁::Int, n₂::Int)
    @assert size(G, 1) == size(G, 2)
    @assert 0 < n₁ && n₁ < size(G, 1)
    @assert 0 < n₂ && n₂ < size(G, 1)
    return _ThreeBlockMatrix(
        view(G, (n₁ + 1):size(G, 1), 1:(size(G, 2) - n₂)),
        view(G, 1:n₁, (n₁ + 1):(size(G, 2) - n₂)),
        view(G, 1:(size(G, 2) - n₂), (size(G, 1) - n₂ + 1):size(G, 1)),
    )
end

function makeblocks(G::AbstractMatrix)
    if size(G, 1) != size(G, 2)
        throw(DimensionMismatch("Expected a square matrix, but G has size $(size(G))"))
    end
    n₁, n₂ = _analyze(G)
    if n₁ + n₂ >= size(G, 1)
        throw(Exception("Overlapping zero regions are not yet supported: n₁=$n₁, n₂=$n₂"))
    else
        return _ThreeBlockMatrix(G, n₁, n₂)
    end
end

function LinearAlgebra.mul!(
    C::AbstractMatrix,
    A::AbstractMatrix,
    B::_ThreeBlockMatrix,
    α,
    β,
)
    n₁ = size(B.block₂, 1)
    n₂ = size(B.block₃, 2)
    #! format: off
    mul!(view(C, :, 1:size(B.block₁, 2)), view(A, :, (n₁ + 1):size(A, 2)), B.block₁, α, β)
    mul!(view(C, :, (n₁ + 1):size(B.block₁, 2)), view(A, :, 1:n₁), B.block₂, α, one(eltype(C)))
    mul!(view(C, :, (size(B.block₁, 2) + 1):size(C, 2)), view(A, :, 1:size(B.block₁, 2)), B.block₃, α, β)
    #! format: on
    return C
end

function _analyze(G::AbstractMatrix{<:Real})::Tuple{Int, Int}
    @assert size(G, 1) == size(G, 2)
    cutoff = mapreduce(abs, max, G) * eps(eltype(G))
    return _analyze_top_left(G, cutoff), _analyze_bottom_right(G, cutoff)
end

function _analyze_top_left(G::AbstractMatrix{T}, cutoff::T)::Int where {T <: Real}
    # Find an upper bound on n
    n = size(G, 1)
    @inbounds for i in 1:size(G, 1)
        if abs(G[i, 1]) >= cutoff
            n = i - 1
            break
        end
    end
    # Fine-tune it
    j = 2
    @inbounds while j <= n
        i = 1
        while i <= n
            if abs(G[i, j]) >= cutoff
                n = max(i, j) - 1
                break
            end
            i += 1
        end
        j += 1
    end
    return n
end

function _analyze_bottom_right(G::AbstractMatrix{T}, cutoff::T)::Int where {T <: Real}
    # Find a lower bound on n
    n = 1
    @inbounds for i in size(G, 1):-1:1
        if abs(G[i, size(G, 2)]) >= cutoff
            n = i + 1
            break
        end
    end
    # Fine-tune it
    j = size(G, 2) - 1
    @inbounds while j >= n
        i = size(G, 1)
        while i >= n
            if abs(G[i, j]) >= cutoff
                n = min(i, j) + 1
                break
            end
            i -= 1
        end
        j -= 1
    end
    return size(G, 1) + 1 - n
end


@doc raw"""
    _g_blocks(ħω, E; mu, kT) -> (Gᵣ, Gᵢ)

**This is an internal function!**

Calculate matrix `G(ħω)` given a vector of eigenenergies `E`, chemical potential `mu`, and
temperature `kT`. See `_g` for the definition of `G(ħω)`.

Compared to `_g` this function applies to tricks:
  * `G` is split into real and complex parts `Gᵣ` and `Gᵢ`.
  * We exploit the "block-sparse" structure of `G`.
"""
function _g_blocks(ħω::Complex{R}, E::AbstractVector{R}; mu::R, kT::R) where {R <: Real}
    # Compared to a simple `map` the following saves one allocation
    f = similar(E)
    map!(x -> fermidirac(x, mu = mu, kT = kT), f, E)
    n = length(E)
    Gᵣ = similar(E, n, n)
    Gᵢ = similar(E, n, n)
    @inbounds for j in 1:n
        for i in 1:n
            value = (f[i] - f[j]) / (E[i] - E[j] - ħω)
            Gᵣ[i, j] = real(value)
            Gᵢ[i, j] = imag(value)
        end
    end
    return makeblocks(Gᵣ), makeblocks(Gᵢ)
end


@doc raw"""
    polarizability_simple(ħω, E, ψ; mu, kT) -> χ

Calculates the polarizability matrix ``\chi`` by direct evaluation of eq. (3). This approach
is single-threaded and doesn't utilize the CPU well. It is provided here for testing
purposes only.
"""
function polarizability_simple(
    ħω,
    E::AbstractVector{R},
    ψ::AbstractMatrix{C};
    mu::R,
    kT::R,
) where {R, C}
    if size(ψ, 1) != size(ψ, 2) || size(E, 1) != size(ψ, 1)
        throw(DimensionMismatch(
            "dimensions of E and ψ do not match: $(size(E)) & $(size(ψ)); " *
            "expected ψ to be a square matrix of the same dimension as E",
        ))
    end
    G = _g(ħω, E; mu = mu, kT = kT)
    χ = similar(G)
    @inbounds for a in 1:size(χ, 2)
        for b in 1:size(χ, 1)
            acc = zero(eltype(χ))
            for i in 1:length(E)
                for j in 1:length(E)
                    acc += G[i, j] * conj(ψ[a, j]) * ψ[a, i] * conj(ψ[b, i]) * ψ[b, j]
                end
            end
            χ[a, b] = 2 * acc
        end
    end
    return χ
end


@doc raw"""
    dielectric(χ, V) -> ε

Compute dielectric function ``\varepsilon`` given polarizability matrix ``\chi`` and Coulomb
interaction potential ``V``. Dielectric function is simply ``\varepsilon = 1 - \chi V``.
"""
function dielectric(χ::AbstractMatrix{C}, V::AbstractMatrix{R}) where {C, R}
    if size(χ, 1) != size(χ, 2) || size(χ) != size(V)
        throw(DimensionMismatch(
            "dimensions of χ and V do not match: $(size(χ)) != $(size(V)); " *
            "expected two square matrices of the same size",
        ))
    end
    ε = similar(χ)
    fill!(ε, zero(C))
    @inbounds ε[diagind(ε)] .= one(C)
    mul!(ε, V, χ, -one(C), one(C))
    return ε
end


@doc raw"""
    coulomb_simple(x, y, z; v0) -> V

Given site coordinates construct the simplest approximation of Coulomb interaction potential:
```math
    V_{ab} = \left\{ \begin{aligned}
        &\frac{e^2}{4\pi \varepsilon_0} \frac{1}{|\mathbf{r}_a - \mathbf{r}_b|},
            \text{ when } a \neq b \\
        &V_0, \text{ otherwise}
    \end{aligned} \right.
```
Coordinates are assumed to be in meters, `v0` -- in electron-volts, and the returned matrix
is also in electron-volts.

Note, that electron charge ``e > 0`` in the above formula.
"""
function coulomb_simple(
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T};
    v0::T,
) where {T <: Real}
    e = 1.602176634E-19 # [C], from https://en.wikipedia.org/wiki/Elementary_charge
    ε₀ = 8.8541878128E-12 # [F/m], from https://en.wikipedia.org/wiki/Vacuum_permittivity
    # NOTE: `scale` contains the first power of `e`. THIS IS NOT A MISTAKE!
    # [e / ε₀] = C * m / F= C * m / (C / V) = V * m, and dividing by distance we get volts.
    # And since we work in electron-volts the second multiplication with e is implicit.
    scale = T(e / (4 * π * ε₀))
    n = length(x)
    if length(y) != n || length(z) != n
        throw(DimensionMismatch(
            "coordinates have diffent lengths: length(x)=$(length(x)), " *
            "length(y)=$(length(y)), length(z)=$(length(z))",
        ))
    end
    distance = let
        x = x
        y = y
        z = z
        (i, j) -> hypot(x[i] - x[j], y[i] - y[j], z[i] - z[j])
    end

    v = similar(x, n, n)
    @inbounds v[diagind(v)] .= v0
    @inbounds for b in 1:(n - 1)
        for a in (b + 1):n
            v[a, b] = scale / distance(a, b)
            v[b, a] = v[a, b]
        end
    end
    return v
end


# Interoperability with TiPSi
include("tipsi.jl")

end # module

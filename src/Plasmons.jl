module Plasmons

export dielectric
export dispersion
export polarizability_batched, polarizability_thesis, polarizability_simple
export coulomb_simple
export read_hamiltonian, read_coordinates

using LinearAlgebra
using ArgParse
using HDF5
import Printf: @sprintf

"""
    fermidirac(E; mu, kT) -> f

Return Fermi-Dirac distribution ``f`` at energy `E`, chemical potential `mu`, and temperature
`kT`. Note that `kT` is assumed to be temperature multiplied by the Boltzmann constant, i.e.
physical dimension of `kT` is the same as `E` (e.g. electron-volts).
"""
@inline fermidirac(E; mu, kT) = 1 / (1 + exp((E - mu) / kT))


@doc raw"""
    _g(ħω, E; mu, kT) -> G

!!! warning
    This is an internal function!

Compute matrix G
```math
    G_{ij} = \frac{f(E_i) - f(E_j)}{E_i - E_j - \hbar\omega}
```
where ``f`` is [Fermi-Dirac
distribution](https://en.wikipedia.org/wiki/Fermi%E2%80%93Dirac_statistics) at chemical
potential `mu` (``\mu``) and temperature `kT` (``k_B T``). `E` is a vector of eigenenergies. `ħω` is a complex
frequency including Landau damping (i.e.  ``\hbar\omega + i\eta``). All arguments are
assumed to be in energy units.

Sometimes one can further exploit the structure of ``G``. For ``E \ll \mu`` or ``E \gg \mu``
Fermi-Dirac distribution is just a constant and ``G`` goes to 0 for all ``\omega``.
[`Plasmons._g_blocks`](@ref) uses this fact to construct a block-sparse version of
``G``. The reason why such a block-sparse version is useful will become apparent later.
"""
function _g(ħω::Complex{ℝ}, E::AbstractVector{ℝ}; mu::ℝ, kT::ℝ) where {ℝ <: Real}
    # Compared to a simple `map` the following saves one allocation
    f = similar(E)
    map!(x -> fermidirac(x, mu = mu, kT = kT), f, E)
    n = length(E)
    G = similar(E, complex(ℝ), n, n)
    @inbounds for j in 1:n
        @. G[:, j] = (f - f[j]) / (E - E[j] - ħω)
    end
    return G
end

@doc raw"""
    _ThreeBlockMatrix

!!! warning
    This is an internal data structure

A dense matrix with top-left and bottom-right blocks assumed to be zero:
![three-block-structure](three-blocks.jpg)
`_ThreeBlockMatrix` stores blocks 1, 2, and 3 as dense matrices. There is also a special
overload of `*` operator for faster matrix-matrix multiplication.
"""
struct _ThreeBlockMatrix{M <: AbstractMatrix{<:Real}}
    block₁::M
    block₂::M
    block₃::M
end

@doc raw"""
    _ThreeBlockMatrix(G, n₁, n₂)

!!! warning
    This is an internal function

``n₁`` and ``n₂`` are top-left and bottom-right zero subblock sizes respectively.
"""
_ThreeBlockMatrix(G::AbstractMatrix, n₁::Int, n₂::Int) = _ThreeBlockMatrix(
    G[(n₁ + 1):size(G, 1), 1:(size(G, 2) - n₂)],
    G[1:n₁, (n₁ + 1):(size(G, 2) - n₂)],
    G[1:(size(G, 1) - n₂), (size(G, 2) - n₂ + 1):size(G, 2)],
)

@doc raw"""
    _ThreeBlockMatrix(G)

!!! warning
    This is an internal function

Analyze matrix ``G`` and construct three-block version of it for faster matrix-matrix
multiplication.
"""
function _ThreeBlockMatrix(G::AbstractMatrix)
    if size(G, 1) != size(G, 2)
        throw(DimensionMismatch("Expected a square matrix, but G has size $(size(G))"))
    end
    # All elements smaller than ε·‖G‖ are assumed to be negligible and are set to zero.
    cutoff = mapreduce(abs, max, G) * eps(eltype(G))
    n₁ = _analyze_top_left(G, cutoff)
    n₂ = _analyze_bottom_right(G, cutoff)
    if n₁ + n₂ >= size(G, 1)
        throw(Exception("Overlapping zero regions are not yet supported: n₁=$n₁, n₂=$n₂"))
    else
        return _ThreeBlockMatrix(G, n₁, n₂)
    end
end

@doc raw"""
    _analyze_top_left(G::AbstractMatrix{ℝ}, cutoff::ℝ) -> Int

!!! warning
    This is an internal function!

We assume that `G` is a square matrix with the following block structure:
![top-left-block](top-left-block.jpg)
I.e. the top-left corner of size `n₁ x n₁` consists of zeros. This function determines `n₁`
by treating all matrix elements smaller than `cutoff` as zeros.
"""
function _analyze_top_left(G::AbstractMatrix{ℝ}, cutoff::ℝ)::Int where {ℝ <: Real}
    # Find an upper bound on n. We follow the first column of G and find the first element
    # which exceeds cutoff.
    n = size(G, 1)
    @inbounds for i in 1:size(G, 1)
        if abs(G[i, 1]) >= cutoff
            n = i - 1
            break
        end
    end
    # Fine-tune it by considering other columns.
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

@doc raw"""
    _analyze_bottom_right(G::AbstractMatrix{ℝ}, cutoff::ℝ) -> Int

!!! warning
    This is an internal function!

Very similar to [`_analyze_top_left`](@ref) except that now the bottom right corner of G is
assumed to contain zeros. This function determines the size of this block.
"""
function _analyze_bottom_right(G::AbstractMatrix{ℝ}, cutoff::ℝ)::Int where {ℝ <: Real}
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

"""
    mul!(C, A, B::_ThreeBlockMatrix, α, β) -> C

Overload of matrix-matrix multiplication for _ThreeBlockMatrix. We can spare some flops
because of the zero blocks.

We replace one bigg GEMM by three smaller:

  1) ![first-gemm](gemm-1.jpg)
  2) ![second-gemm](gemm-2.jpg)
  3) ![third-gemm](gemm-3.jpg)
"""
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


@doc raw"""
    _g_blocks(ħω, E; mu, kT) -> (Gᵣ, Gᵢ)

!!! warning
    This is an internal function!

Calculate matrix ``G`` given a vector of eigenenergies `E`, chemical potential `mu`, and
temperature `kT`. See [`_g`](@ref) for the definition of ``G``.

Compared to [`_g`](@ref) this function applies to tricks:
  * `G` is split into real and imaginary parts `Gᵣ` and `Gᵢ`.
  * We exploit the "block-sparse" structure of `G` (see
    [`Plasmons._ThreeBlockMatrix`](@ref)).
"""
function _g_blocks(ħω::Complex{ℝ}, E::AbstractVector{ℝ}; mu::ℝ, kT::ℝ) where {ℝ <: Real}
    G = _g(ħω, E; mu = mu, kT = kT)
    return _ThreeBlockMatrix(real(G)), _ThreeBlockMatrix(imag(G))
end


@doc raw"""
    polarizability_thesis(ħω, E, ψ; mu, kT) -> χ

Calculate the polarizability matrix ``\chi`` by using methods from the Bachelor thesis.
"""
polarizability_thesis(ħω, E, ψ; mu, kT) =
    _polarizability_thesis(_g(ħω, E; mu = mu, kT = kT), ψ)

"""
    _Workspace{<: AbstractArray}

**This is an internal data structure!**

A workspace which is used by [`polarizability_thesis`](@ref) and
[`polarizability_batched`](@ref) functions to avoid allocating many temporary arrays.
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
    out::AbstractMatrix{ℂ},
    as::UnitRange{Int},
    b::Int,
    ψ::AbstractMatrix{ℂ},
) where {ℂ <: Union{Real, Complex}}
    # Effectively we want this:
    #     A = view(ψ, as, :) .* transpose(view(ψ, b, :))
    # but without memory allocations
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



@doc raw"""
    polarizability_simple(ħω, E, ψ; mu, kT) -> χ

Calculates the polarizability matrix ``\chi`` by direct evaluation of eq. (3). This approach
is single-threaded and doesn't utilize the CPU well. It is provided here for testing
purposes only.
"""
function polarizability_simple(
    ħω::Complex{ℝ},
    E::AbstractVector{ℝ},
    ψ::AbstractMatrix{<:Union{ℝ, Complex{ℝ}}};
    mu::ℝ,
    kT::ℝ,
) where {ℝ <: Real}
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
function dielectric(χ::AbstractMatrix{Complex{ℝ}}, V::AbstractMatrix{ℝ}) where {ℝ <: Real}
    ℂ = complex(ℝ)
    if size(χ, 1) != size(χ, 2) || size(χ) != size(V)
        throw(DimensionMismatch(
            "dimensions of χ and V do not match: $(size(χ)) != $(size(V)); " *
            "expected two square matrices of the same size",
        ))
    end
    ε = similar(χ)
    fill!(ε, zero(ℂ))
    @inbounds ε[diagind(ε)] .= one(ℂ)
    mul!(ε, V, χ, -one(ℂ), one(ℂ))
    return ε
end


@doc raw"""
    coulomb_simple(x, y, z; V₀) -> V

Given site coordinates (as vectors of real numbers) construct the simplest approximation of
Coulomb interaction potential:
```math
    V_{ab} = \left\{ \begin{aligned}
        &\frac{e^2}{4\pi \varepsilon_0} \frac{1}{|\mathbf{r}_a - \mathbf{r}_b|},
            \text{ when } a \neq b \\
        &V_0, \text{ otherwise}
    \end{aligned} \right.
```
Coordinates are assumed to be in meters, `V₀` -- in electron-volts, and the returned matrix
is also in electron-volts.

Note, that electron charge ``e > 0`` in the above formula.
"""
function coulomb_simple(
    x::AbstractVector{ℝ},
    y::AbstractVector{ℝ},
    z::AbstractVector{ℝ};
    V₀::ℝ,
) where {ℝ <: Real}
    e = 1.602176634E-19 # [C], from https://en.wikipedia.org/wiki/Elementary_charge
    ε₀ = 8.8541878128E-12 # [F/m], from https://en.wikipedia.org/wiki/Vacuum_permittivity
    # NOTE: `scale` contains the first power of `e`. THIS IS NOT A MISTAKE!
    # [e / ε₀] = C * m / F= C * m / (C / V) = V * m, and dividing by distance we get volts.
    # And since we work in electron-volts the second multiplication with e is implicit.
    scale = ℝ(e / (4 * π * ε₀))
    n = length(x)
    if length(y) != n || length(z) != n
        throw(DimensionMismatch(
            "coordinates have diffent lengths: length(x)=$(length(x)), " *
            "length(y)=$(length(y)), length(z)=$(length(z))",
        ))
    end
    distance = let x = x, y = y, z = z
        (i, j) -> hypot(x[i] - x[j], y[i] - y[j], z[i] - z[j])
    end

    v = similar(x, n, n)
    @inbounds v[diagind(v)] .= V₀
    @inbounds for b in 1:(n - 1)
        for a in (b + 1):n
            v[a, b] = scale / distance(a, b)
            v[b, a] = v[a, b]
        end
    end
    return v
end

_momentum_eigenvector(
    k::NTuple{3, ℝ},
    x::AbstractVector{ℝ},
    y::AbstractVector{ℝ},
    z::AbstractVector{ℝ},
) where {ℝ <: Real} = map(let kˣ = k[1], kʸ = k[2], kᶻ = k[3]
    (xᵢ, yᵢ, zᵢ) -> exp(1im * (kˣ * xᵢ + kʸ * yᵢ + kᶻ * zᵢ))
end, x, y, z)

function dispersion(
    k::NTuple{3, ℝ},
    ε::AbstractMatrix{Complex{ℝ}},
    x::AbstractVector{ℝ},
    y::AbstractVector{ℝ},
    z::AbstractVector{ℝ},
) where {ℝ <: Real}
    if length(x) != length(y)
        throw(DimensionMismatch("'x' and 'y' have different lengths: $(length(x)) != $(length(y))"))
    end
    if size(ε, 1) != length(x)
        throw(DimensionMismatch("'ε' and 'x' have incompatible dimensions: $(size(ε)) and $(length(x))"))
    end
    q = _momentum_eigenvector(k, x, y, z)
    dot(q, ε, q)
end

# Interoperability with TiPSi
include("tipsi.jl")

function main(
    E::Vector{ℝ},
    ψ::Matrix{ℂ};
    kT::Real,
    μ::Real,
    η::Real,
    ωs::Vector{<:Real},
    out::Union{HDF5File, HDF5Group},
    V::Union{Matrix{ℝ}, Nothing} = nothing,
) where {ℝ <: Real, ℂ <: Union{ℝ, Complex{ℝ}}}
    if kT <= 0
        throw(ArgumentError("invalid 'kT': $kT; expected a positive real number"))
    end
    if η <= 0
        throw(ArgumentError("invalid 'η': $η; expected a positive real number"))
    end
    if isnothing(E) || isnothing(ψ)
        if isnothing(H)
            throw(ArgumentError(
                "if Hamiltonian 'H' is not specified, both eigenvalues 'E'" *
                " and eigenvectors 'ψ' must be provided",
            ))
        end
        @info "Eigenvalues or eigenvectors not provided: diagonalizing the Hamiltonian"
        factorization = eigen(Hermitian(H))
        E = factorization.values
        ψ = factorization.vectors
        @info "Diagonalization completed"
    end
    group_χ = g_create(out, "χ")
    group_ε::Union{HDF5Group, Nothing} = isnothing(V) ? nothing : g_create(out, "ε")
    @info "Polarizability matrices χ(ω) will be saved to group 'χ'"
    if !isnothing(V)
        @info "Dielectric functions ε(ω) will be saved to group 'ε'"
    else
        @warn "Coulomb interaction matrix not provided: dielectric function ε(ω) will not be computed"
    end

    for (i, ω) in enumerate(map(x -> x + 1im * η, ωs))
        @info "Calculating χ(ω = $ω) ..."
        name = @sprintf "%04i" i
        χ = polarizability_batched(
            convert(complex(ℝ), ω),
            E,
            ψ;
            mu = convert(ℝ, μ),
            kT = convert(ℝ, kT),
        )
        group_χ[name] = χ
        attrs(group_χ[name])["ħω"] = ω
        flush(group_χ)
        if !isnothing(V)
            group_ε[name] = dielectric(χ, V)
            attrs(group_ε[name])["ħω"] = ω
            flush(group_ε)
        end
    end
end
function main(H::Matrix{ℂ}; kwargs...) where {ℂ}
    @info "Eigenvalues or eigenvectors not provided: diagonalizing the Hamiltonian"
    factorization = eigen(Hermitian(H))
    E = factorization.values
    ψ = factorization.vectors
    @info "Diagonalization completed"
    main(E, ψ; kwargs...)
end



ArgParse.parse_item(::Type{Vector{Float64}}, x::AbstractString) =
    map(s -> parse(Float64, s), split(x, ","))

function _parse_commandline()
    s = ArgParseSettings()
    #! format: off
    @add_arg_table! s begin
        "--kT"
            help = "Temperature kʙ·T in eletron-volts"
            arg_type = Float64
            required = true
        "--mu"
            help = "Chemical potential μ in eletron-volts"
            arg_type = Float64
            required = true
        "--damping"
            help = "Landau damping η in eletron-volts"
            arg_type = Float64
            required = true
        "--frequency"
            help = "Comma-separated list of ħω values in eletron-volts for which to compute χ and ε"
            arg_type = Vector{Float64}
            required = true
        "--hamiltonian"
            help = "Path to Hamiltonian matrix in the input HDF5 file"
            arg_type = String
        "--coulomb"
            help = "Path to unscreened Coulomb interaction matrix in the input HDF5 file"
            arg_type = String
        "--eigenvalues"
            help = "Path to eigenvalues vector in the input HDF5 file"
            arg_type = String
        "--eigenvectors"
            help = "Path to eigenvectors matrix in the input HDF5 file"
            arg_type = String
        "input_file"
            help = "Input HDF5 file"
            required = true
        "output_file"
            help = "Output HDF5 file"
            required = true
    end
    #! format: on
    return parse_args(s, as_symbols = true)
end

function tryread(io::HDF5File, path::Union{<:AbstractString, Nothing} = nothing)
    isnothing(path) && return nothing
    if !has(io, path)
        @warn "Path '$path' not found in input HDF5 file"
        return nothing
    end
    read(io, path)
end

function entry_main()
    args = _parse_commandline()
    H, E, ψ, V = h5open(args[:input_file], "r") do io
        tryread(io, get(args, :hamiltonian, nothing)),
        tryread(io, get(args, :eigenvalues, nothing)),
        tryread(io, get(args, :eigenvectors, nothing)),
        tryread(io, get(args, :coulomb, nothing))
    end
    if (isnothing(E) || isnothing(ψ)) && isnothing(H)
        throw(ArgumentError(
            "if Hamiltonian 'H' is not specified, both eigenvalues 'E'" *
            " and eigenvectors 'ψ' must be provided",
        ))
    end
    h5open(args[:output_file], "w") do io
        kwargs = (
            kT = args[:kT],
            μ = args[:mu],
            η = args[:damping],
            ωs = args[:frequency],
            out = io,
            V = V,
        )
        if (isnothing(E) || isnothing(ψ))
            main(H; kwargs...)
        else
            main(E, ψ; kwargs...)
        end
    end
end

end # module

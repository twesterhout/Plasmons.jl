module Plasmons

export dielectric, polarizability, dispersion
export julia_main

using CUDA
using Adapt
using LinearAlgebra
using ArgParse
using HDF5

include("utilities.jl")

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
potential `mu` (``\mu``) and temperature `kT` (``k_B T``). `E` is a vector of eigenenergies.
`ħω` is a complex frequency including Landau damping (i.e.  ``\hbar\omega + i\eta``). All
arguments are assumed to be in energy units.

Sometimes one can further exploit the structure of ``G``. For ``E \ll \mu`` or ``E \gg \mu``
Fermi-Dirac distribution is just a constant and ``G`` goes to 0 for all ``\omega``.
[`Plasmons._g_blocks`](@ref) uses this fact to construct a block-sparse version of
``G``. The reason why such a block-sparse version is useful will become apparent later.
"""
function _build_g(ħω::Complex{ℝ}, E::AbstractVector{ℝ}; mu::ℝ, kT::ℝ) where {ℝ <: Real}
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
function _postprocess_g(
    ::Type{T},
    G::AbstractMatrix,
    blocks::Bool,
) where {T <: Union{Real, Complex}}
    split_if(x) = T <: Real ? (real(x), imag(x)) : x
    compress_if(x) = blocks ? ThreeBlockMatrix(x) : x
    return _map_tuple(compress_if, split_if(G))
end
function _map_tuple(f::Function, x::Tuple)
    # @info "_map_tuple($f, $x::Tuple) = "
    f.(x)
end
_map_tuple(f::Function, x::Any) = f(x)

g_matrix(
    ::Type{T},
    ħω::Complex{ℝ},
    E::Vector{ℝ};
    mu::ℝ,
    kT::ℝ,
    blocks::Bool,
) where {T, ℝ <: Real} = _postprocess_g(T, _build_g(ħω, E; mu = mu, kT = kT), blocks)
function g_matrix(
    ::Type{T},
    ħω::Complex{ℝ},
    E::CuVector{ℝ};
    mu::ℝ,
    kT::ℝ,
    blocks::Bool,
) where {T, ℝ <: Real}
    G = _postprocess_g(T, _build_g(ħω, Vector(E); mu = mu, kT = kT), blocks)
    T <: Real ? (adapt(CuArray, G[1]), adapt(CuArray, G[2])) : adapt(CuArray, G)
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
# function _g_blocks(ħω::Complex{ℝ}, E::AbstractVector{ℝ}; mu::ℝ, kT::ℝ) where {ℝ <: Real}
#     G = _g(ħω, E; mu = mu, kT = kT)
#     # TODO(twesterhout): Are these extra copies important?
#     ThreeBlockMatrix(real(G)), ThreeBlockMatrix(imag(G))
# end
# function _g_blocks(ħω::Complex{ℝ}, E::CuVector{ℝ}; mu::ℝ, kT::ℝ) where {ℝ <: Real}
#     (Gᵣ, Gᵢ) = _g_blocks(ħω, Vector(E); mu = mu, kT = kT)
#     adapt(CuArray, Gᵣ), adapt(CuArray, Gᵢ)
# end

@doc raw"""
    polarizability(ħω, E, ψ; mu, kT, method = :batched) -> χ

Compute polarizability matrix ``\chi`` using method `method` (either `:simple`, `:thesis`,
or `:batched`).
"""
polarizability(ħω, E, ψ; mu, kT, method::Symbol = :batched) = Dict{Symbol, Function}(
    :simple => polarizability_simple,
    :thesis => polarizability_thesis,
    :batched => polarizability_batched,
)[method](
    ħω,
    E,
    ψ;
    mu = mu,
    kT = kT,
)

@doc raw"""
    _Workspace{<: AbstractArray}

!!! warning
    This is an internal data structure!

A workspace which is used by [`polarizability`](@ref) function to avoid allocating many
temporary arrays. Stores two attributes:

  * A vector `A` which is defined by ``A_j = \langle j | a \rangle \langle b | j \rangle``.
  * A vector `temp` which is the product ``G A``.
"""
struct Workspace{M <: AbstractMatrix}
    A::M
    temp::M
end

function polarizability_batched(
    ħω::Complex,
    E::AbstractVector{ℝ},
    ψ::AbstractMatrix{ℂ};
    mu::Real,
    kT::Real,
    blocks::Bool,
) where {ℝ <: Real, ℂ <: Union{ℝ, Complex{ℝ}}}
    G = g_matrix(ℂ, ħω, E; mu = mu, kT = kT, blocks = blocks)
    ℂ <: Real ? polarizability_batched(G[1], G[2], ψ) : polarizability_batched(G, ψ)
end

function polarizability_batched(
    Gᵣ::Union{AbstractMatrix{ℝ}, ThreeBlockMatrix{ℝ}},
    Gᵢ::Union{AbstractMatrix{ℝ}, ThreeBlockMatrix{ℝ}},
    ψ::AbstractMatrix{ℝ},
) where {ℝ <: Real}
    N = size(ψ, 1)
    χ = similar(ψ, complex(ℝ))
    fill!(χ, zero(eltype(χ)))
    ws = Workspace(similar(ψ), similar(ψ))
    for b in 1:N
        a = b:N
        _batched_mat_el!(view(χ, a, b), ws, a, b, Gᵣ, Gᵢ, ψ)
    end
    χ .= 2 .* (χ .+ transpose(χ))
    χ[diagind(χ)] ./= 2
    χ
end
function polarizability_batched(
    G::Union{AbstractMatrix{ℂ}, ThreeBlockMatrix{ℂ}},
    ψ::AbstractMatrix{ℂ},
) where {ℂ <: Complex}
    N = size(ψ, 1)
    χ = similar(ψ)
    fill!(χ, zero(eltype(χ)))
    ws = Workspace(similar(ψ), similar(ψ))
    for b in 1:N
        _batched_mat_el!(view(χ, :, b), ws, b, G, ψ)
    end
    χ .*= 2
    χ
end

function _batched_mat_el!(
    out::AbstractVector{Complex{ℝ}},
    ws,
    as::UnitRange{Int},
    b::Int,
    Gᵣ::Union{AbstractMatrix{ℝ}, ThreeBlockMatrix{ℝ}},
    Gᵢ::Union{AbstractMatrix{ℝ}, ThreeBlockMatrix{ℝ}},
    ψ::AbstractMatrix{ℝ},
) where {ℝ <: Real}
    batch_size = length(as)
    A = view(ws.A, 1:batch_size, :)
    temp = view(ws.temp, 1:batch_size, :)
    A .= view(ψ, as, :) .* transpose(view(ψ, b, :))
    # @info "" typeof(temp) typeof(A) typeof(Gᵣ)
    _mul_with_strides!(temp, A, Gᵣ, one(ℝ), zero(ℝ))
    outᵣ = view(reinterpret(ℝ, out), 1:2:(2 * batch_size))
    dot_batched!(outᵣ, A, temp)
    _mul_with_strides!(temp, A, Gᵢ, one(ℝ), zero(ℝ))
    outᵢ = view(reinterpret(ℝ, out), 2:2:(2 * batch_size))
    dot_batched!(outᵢ, A, temp)
    out
end
function _batched_mat_el!(
    out::AbstractVector{ℂ},
    ws,
    b::Int,
    G::Union{AbstractMatrix{ℂ}, ThreeBlockMatrix{ℂ}},
    ψ::AbstractMatrix{ℂ},
) where {ℂ <: Complex}
    A = ws.A
    temp = ws.temp
    A .= ψ .* conj.(transpose(view(ψ, b, :)))
    # @info "" A G
    _mul_with_strides!(temp, A, G, one(ℂ), zero(ℂ))
    dot_batched!(out, A, temp)
    out
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
    G = _build_g(ħω, E; mu = mu, kT = kT)
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

function _momentum_eigenvectors(q::NTuple{3, <:Real}, x, y, z; n::Int)
    ks = 0:(π / (n - 1)):π
    @assert length(ks) == n
    out = similar(x, complex(eltype(x)), length(x), length(ks))
    for i in 1:size(out, 2)
        fn = let kˣ = q[1] * ks[i], kʸ = q[2] * ks[i], kᶻ = q[3] * ks[i]
            (xᵢ, yᵢ, zᵢ) -> exp(1im * (kˣ * xᵢ + kʸ * yᵢ + kᶻ * zᵢ))
        end
        map!(fn, view(out, :, i), x, y, z)
    end
    out
end
function _dispersion_function(
    q::NTuple{3, ℝ},
    x::AbstractVector{ℝ},
    y::AbstractVector{ℝ},
    z::AbstractVector{ℝ};
    n::Int,
) where {ℝ <: Real}
    if length(x) != length(y) || length(x) != length(z)
        throw(DimensionMismatch("'x', 'y', and 'z' have different lengths: $(length(x)) vs. $(length(y)) vs. $(length(z))"))
    end
    return let ks = _momentum_eigenvectors(q, x, y, z; n = n),
        temp = similar(ks, length(x), size(ks, 2))

        (out, ε) -> begin
            mul!(temp, ε, ks)
            for i in 1:length(out)
                out[i] = dot(view(ks, :, i), view(temp, :, i))
            end
        end
    end
end
function dispersion(εs, q, x, y, z; n::Int = 100)
    fn! = _dispersion_function(q, x, y, z; n = n)
    out = similar(x, complex(eltype(x)), length(εs), n)
    for (i, ε) in enumerate(εs)
        fn!(view(out, i, :), ε)
    end
    out
end

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
        name = string(i, pad = 4)
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

function julia_main()::Cint
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
    return 0
end

end # module

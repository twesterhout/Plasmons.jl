module Plasmons

export dielectric, polarizability, dispersion
export main, julia_main

using CUDA
using Adapt
using LinearAlgebra
using ArgParse
using HDF5
using Dates
using Logging
using LoggingExtras


include("utilities.jl")

"""
    fermidirac(E; mu, kT) -> f

Return Fermi-Dirac distribution ``f`` at energy `E`, chemical potential `mu`, and temperature
`kT`. Note that `kT` is assumed to be temperature multiplied by the Boltzmann constant, i.e.
physical dimension of `kT` is the same as `E` (e.g. electron-volts).
"""
@inline fermidirac(E; mu, kT) = 1 / (1 + exp((E - mu) / kT))

function _build_g(ħω::Complex{ℝ}, E::AbstractVector{ℝ}; mu::ℝ, kT::ℝ) where {ℝ <: Real}
    f = map(x -> fermidirac(x, mu = mu, kT = kT), E)
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
"""
    _map_tuple(f::Function, x)

Apply `f` to every element of `x`, but only if `x` is a `Tuple`. If not, just apply `f` to
the whole `x`.
"""
_map_tuple(f::Function, x::Tuple) = f.(x)
_map_tuple(f::Function, x::Any) = f(x)

@doc raw"""
    g_matrix(::Type{ℝ}, ħω::ℂ, E; mu::ℝ, kT::ℝ, blocks::Bool) -> (Gᵣ, Gᵢ)
    g_matrix(::Type{ℂ}, ħω::ℂ, E; mu::ℝ, kT::ℝ, blocks::Bool) -> G

Compute matrix ``G``
```math
    G_{ij} = \frac{f(E_i) - f(E_j)}{E_i - E_j - \hbar\omega}
```
where ``f`` is [Fermi-Dirac
distribution](https://en.wikipedia.org/wiki/Fermi%E2%80%93Dirac_statistics) at chemical
potential `mu` (``\mu``) and temperature `kT` (``k_B T``). `E` is a vector of eigenenergies.
`ħω` is a complex frequency including Landau damping (i.e.  ``\hbar\omega + i\eta``). All
arguments are assumed to be in energy units.

Matrix ``G`` is always complex, but the first argument specifies which type should be used
to represent it. For example, if you call `g_matrix(Float32, ...)` then a tuple of two
values will be returned -- representing real and imaginary parts of ``G``. If, however, you
call `g_matrix(ComplexF64, ...)` then ``G`` will not be split into real and imaginary parts
and a single object is returned.

Sometimes one can further exploit the structure of ``G``. For ``E \ll \mu`` or ``E \gg \mu``
Fermi-Dirac distribution is just a constant and ``G`` goes to 0 for all ``\omega``. We can
use this fact to construct a block-sparse version of ``G`` to speed up matrix-matrix
products containing it. To return ``G`` in block-sparse form, set `blocks = true`.

This function works with both CPU and GPU arrays. If vector `E` is stored on the GPU, then
the returned matrix (or all submatrices for block-sparse version) will also reside on GPU.
"""
g_matrix(
    ::Type{T},
    ħω::Complex{ℝ},
    E::Vector{ℝ};
    mu::ℝ,
    kT::ℝ,
    blocks::Bool,
) where {T, ℝ <: Real} = _postprocess_g(T, _build_g(ħω, E; mu = mu, kT = kT), blocks)
g_matrix(
    ::Type{T},
    ħω::Complex{ℝ},
    E::CuVector{ℝ};
    mu::ℝ,
    kT::ℝ,
    blocks::Bool,
) where {T, ℝ <: Real} = _map_tuple(
    x -> adapt(CuArray, x),
    _postprocess_g(T, _build_g(ħω, Vector(E); mu = mu, kT = kT), blocks),
)

@doc raw"""
    polarizability(ħω::ℂ, E, ψ; mu::ℝ, kT::ℝ, method = :batched, blocks = true) -> χ

Compute polarizability matrix ``\chi`` using method `method` (either `:simple` or
`:batched`). `ħω` specifies complex frequency including Landau damping (i.e. ``\hbar\omega +
i\eta``). `mu` and `kT` specify chemical potential and temperature respectively. `E` and `ψ`
are eigenvalues and eigenvectors of the system Hamiltonian. They can be either CPU or GPU
arrays.
"""
function polarizability(
    ħω::Complex,
    E::AbstractVector{<:Real},
    ψ::AbstractMatrix{<:Union{Real, Complex}};
    mu::Real,
    kT::Real,
    method::Symbol = :batched,
    blocks::Bool = true,
)
    if size(ψ, 1) != size(ψ, 2)
        throw(DimensionMismatch("'ψ' has wrong shape: $(size(ψ)); expected a square matrix"))
    end
    if size(E, 1) != size(ψ, 1)
        throw(DimensionMismatch(
            "dimensions of 'E' and 'ψ' do not match: $(size(E)) & $(size(ψ)); " *
            "expected 'ψ' to be of the same dimension as 'E'",
        ))
    end
    if method == :simple
        polarizability_simple(ħω, E, ψ; mu = mu, kT = kT)
    elseif method == :batched
        polarizability_batched(ħω, E, ψ; mu = mu, kT = kT, blocks = blocks)
    else
        throw(ArgumentError("invalid 'method': $method; expected either :simple or :batched"))
    end
end


struct Workspace{M <: AbstractMatrix}
    A::M
    temp::M
end

@doc raw"""
    polarizability_batched(ħω::ℂ, E, ψ; mu::ℝ, kT::ℝ, blocks::Bool) -> χ

Compute polarizability matrix ``\chi`` using "batched" method.
"""
function polarizability_batched(
    ħω::Complex,
    E::AbstractVector{ℝ},
    ψ::AbstractMatrix{ℂ};
    mu::ℝ,
    kT::ℝ,
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
    ws::Workspace,
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
    ws::Workspace,
    b::Int,
    G::Union{AbstractMatrix{ℂ}, ThreeBlockMatrix{ℂ}},
    ψ::AbstractMatrix{ℂ},
) where {ℂ <: Complex}
    A = ws.A
    temp = ws.temp
    A .= ψ .* conj.(transpose(view(ψ, b, :)))
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


function _momentum_eigenvectors(
    q::NTuple{3, ℝ},
    x::AbstractVector{ℝ},
    y::AbstractVector{ℝ},
    z::AbstractVector{ℝ};
    n::Int,
) where {ℝ <: Real}
    if n < 1
        throw(ArgumentError("invalid number of points 'n': $n; expected a positive number"))
    end
    ks = 0:(π / (n - 1)):π
    @assert length(ks) == n
    out = similar(x, complex(eltype(x)), length(x), length(ks))
    @inbounds for i in 1:size(out, 2)
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

        (out, A) -> begin
            mul!(temp, A, ks)
            @inbounds for i in 1:length(out)
                out[i] = dot(view(ks, :, i), view(temp, :, i))
            end
        end
    end
end
"""
    dispersion(A(ω, r, r'), q, x, y, z; n = 100) -> A(ω, q)

Calculate diagonal elements of the Fourier transform of matrix A.
"""
function dispersion(As, q, x, y, z; n::Int = 100)
    fn! = _dispersion_function(q, x, y, z; n = n)
    out = similar(x, complex(eltype(x)), length(εs), n)
    for (i, A) in enumerate(As)
        fn!(view(out, i, :), A)
    end
    out
end


function main(
    E::AbstractVector{ℝ},
    ψ::AbstractMatrix{ℂ};
    kT::Real,
    μ::Real,
    η::Real,
    ωs::Vector{<:Real},
    out::Union{HDF5.File, HDF5.Group},
    V::Union{AbstractMatrix{ℝ}, Nothing} = nothing,
) where {ℝ <: Real, ℂ <: Union{ℝ, Complex{ℝ}}}
    if kT <= 0
        throw(ArgumentError("invalid 'kT': $kT; expected a positive real number"))
    end
    if η <= 0
        throw(ArgumentError("invalid 'η': $η; expected a positive real number"))
    end
    HDF5.attributes(out)["kT"] = kT
    HDF5.attributes(out)["μ"] = μ
    HDF5.attributes(out)["η"] = η
    group_χ = create_group(out, "χ")
    group_ε::Union{HDF5.Group, Nothing} = isnothing(V) ? nothing : create_group(out, "ε")
    @info "Polarizability matrices χ(ω) will be saved to group 'χ'"
    if !isnothing(V)
        @info "Dielectric functions ε(ω) will be saved to group 'ε'"
    else
        @warn "Coulomb interaction matrix not provided: dielectric function ε(ω) will not be computed"
    end

    for (i, ω) in enumerate(map(x -> x + 1im * η, ωs))
        @info "Calculating χ(ω = $ω) ..."
        name = string(i, pad = 4)
        t₀ = time_ns()
        χ = polarizability(
            convert(complex(ℝ), ω),
            E,
            ψ;
            mu = convert(ℝ, μ),
            kT = convert(ℝ, kT),
        )
        if typeof(χ) <: CuArray
            synchronize()
        end
        t₁ = time_ns()
        group_χ[name] = Array(χ)
        HDF5.attributes(group_χ[name])["ħω"] = ω
        HDF5.attributes(group_χ[name])["time"] = (t₁ - t₀) / 1e9
        flush(group_χ)
        if !isnothing(V)
            t₀ = time_ns()
            ε = dielectric(χ, V)
            if typeof(ε) <: CuArray
                synchronize()
            end
            t₁ = time_ns()
            group_ε[name] = Array(ε)
            HDF5.attributes(group_ε[name])["ħω"] = ω
            HDF5.attributes(group_ε[name])["time"] = (t₁ - t₀) / 1e9
            flush(group_ε)
        end
    end
end
function main!(H::AbstractMatrix{ℂ}; kwargs...) where {ℂ}
    @warn "Eigenvalues or eigenvectors not provided: diagonalizing the Hamiltonian..."
    E, ψ = _eigen!(Hermitian(H))
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
        "--cuda"
            help = "Index of CUDA device to use"
            arg_type = Int
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

function tryread(io::HDF5.File, path::Union{<:AbstractString, Nothing} = nothing)
    isnothing(path) && return nothing
    if !haskey(io, path)
        @warn "Path '$path' not found in input HDF5 file"
        return nothing
    end
    read(io, path)
end

const date_format = "yyyy-mm-dd HH:MM:SS"

timestamp_logger(logger) =
    TransformerLogger(logger) do log
        merge(log, (; message = "$(Dates.format(now(), date_format))  $(log.message)"))
    end

function julia_main()::Cint
    CUDA.allowscalar(false)
    args = _parse_commandline()
    H, E, ψ, V = h5open(args[:input_file], "r") do io
        tryread(io, get(args, :hamiltonian, nothing)),
        tryread(io, get(args, :eigenvalues, nothing)),
        tryread(io, get(args, :eigenvectors, nothing)),
        tryread(io, get(args, :coulomb, nothing))
    end
    if (isnothing(E) || isnothing(ψ)) && isnothing(H)
        @error "if Hamiltonian 'H' is not specified, both eigenvalues 'E'" *
               " and eigenvectors 'ψ' must be provided"
        return 1
    end
    cuda_device = args[:cuda]
    if !isnothing(cuda_device)
        @info "Setting CUDA device to $cuda_device..."
        device!(cuda_device)
        if !isnothing(H)
            H = adapt(CuArray, H)
        end
        if !isnothing(E)
            E = adapt(CuArray, E)
        end
        if !isnothing(ψ)
            ψ = adapt(CuArray, ψ)
        end
        if !isnothing(V)
            V = adapt(CuArray, V)
        end
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
        with_logger(ConsoleLogger(stdout, Logging.Info) |> timestamp_logger) do
            if (isnothing(E) || isnothing(ψ))
                main!(H; kwargs...)
            else
                main(E, ψ; kwargs...)
            end
        end
    end
    return 0
end

end # module

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
_g(ħω::Complex{ℝ}, E::CuVector{ℝ}; mu::ℝ, kT::ℝ) where {ℝ <: Real} =
    CuArray(_g(ħω, Vector(E); mu = mu, kT = kT))

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
    # TODO(twesterhout): Are these extra copies important?
    ThreeBlockMatrix(real(G)), ThreeBlockMatrix(imag(G))
end
function _g_blocks(ħω::Complex{ℝ}, E::CuVector{ℝ}; mu::ℝ, kT::ℝ) where {ℝ <: Real}
    (Gᵣ, Gᵢ) = _g_blocks(ħω, Vector(E); mu = mu, kT = kT)
    adapt(CuArray, Gᵣ), adapt(CuArray, Gᵢ)
end

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
struct _Workspace{T₁ <: AbstractArray, T₂ <: AbstractArray}
    A::T₁
    temp::T₂
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

function _compute_A!(
    out::AbstractMatrix{ℂ},
    as::UnitRange{Int},
    b::Int,
    ψ::AbstractMatrix{ℂ},
) where {ℂ <: Union{Real, Complex}}
    out .= view(ψ, as, :) .* transpose(view(ψ, b, :))
    # Previously, we had an explicit loop on CPU, but the above expression compiles the
    # pretty much same code and works on GPU.
    # Old loop (kept here for reference):
    # offset = as.start - 1
    # @inbounds for j in 1:size(ψ, 2)
    #     scale = conj(ψ[b, j])
    #     @simd for a in as
    #         out[a - offset, j] = scale * ψ[a, j]
    #     end
    # end
end

_dot_many!(out, A, B, scale) = _dot_many_loops!(out, A, B, complex(scale))

_dot_many_loops(A, B) = _dot_many_loops!(similar(A, size(A, 1)), A, B)
function _dot_many_loops!(
    out::AbstractVector,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    scale::Complex{T} = complex(one(T)),
) where {T}
    @inbounds for j in 1:size(A, 2)
        @simd for i in 1:length(out)
            out[i] += scale * A[i, j] * B[i, j]
        end
    end
    out
end

dot_batched_simple(A, B) = sum(A .* B, dims = 2)
function dot_batched_simple!(
    out::AbstractVector,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T}
    copy!(out, dot_batched_simple(A, B))
end

function dot_kernel!(
    n,
    out::CuDeviceVector{T},
    _A::CuDeviceVector{T},
    _B::CuDeviceVector{T},
) where {T}
    A = CUDA.Const(_A)
    B = CUDA.Const(_B)
    block_acc = @cuDynamicSharedMem(T, blockDim().x)
    # Initialize block_acc for each block within the "first grid"
    if blockIdx().x <= gridDim().x
        block_acc[threadIdx().x] = zero(T)
    end

    # Perform thread-local accumulation in each block
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_acc = zero(T)
    while i <= n
        thread_acc += A[i] * B[i]
        i += blockDim().x * gridDim().x
    end
    block_acc[threadIdx().x] += thread_acc
    sync_threads()

    # Reduction within each block
    stride = blockDim().x
    while stride > 1
        stride >>= 1
        if threadIdx().x <= stride
            block_acc[threadIdx().x] += block_acc[threadIdx().x + stride]
        end
        sync_threads()
    end
    if threadIdx().x == 1
        out[blockIdx().x] = block_acc[1]
    end
    nothing
end
function dot_cuda(A::CuVector{T}, B::CuVector{T}) where {T}
    n = length(A)
    num_threads(threads) = prevpow(2, threads)
    num_blocks(threads) = div(n + threads - 1, threads)
    amount_shmem(threads) = threads * sizeof(T)
    kernel = @cuda launch = false dot_kernel!(n, A, A, B)
    config = launch_configuration(kernel.fun, shmem = amount_shmem)
    threads = num_threads(config.threads)
    blocks = num_blocks(threads)
    shmem = amount_shmem(threads)
    out = similar(A, blocks)
    kernel(n, out, A, B; threads = threads, blocks = blocks, shmem = shmem)
    sum(out, dims = 1)
end

function dot_kernel_batched!(
    batch_size,
    n,
    out::CuDeviceMatrix{T},
    _A::CuDeviceMatrix{T},
    _B::CuDeviceMatrix{T},
) where {T}
    A = CUDA.Const(_A)
    B = CUDA.Const(_B)
    block_acc = @cuDynamicSharedMem(T, blockDim().x, blockDim().y)
    # Initialize block_acc for each block within the "first grid"
    if blockIdx().x <= gridDim().x && blockIdx().y <= gridDim().y
        block_acc[threadIdx().x, threadIdx().y] = zero(T)
    end

    # Perform thread-local accumulation in each block
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= batch_size
        thread_acc = zero(T)
        while j <= n
            thread_acc += A[i, j] * B[i, j]
            j += blockDim().y * gridDim().y
        end
        block_acc[threadIdx().x, threadIdx().y] += thread_acc
    end
    sync_threads()

    # Reduction within each block
    stride = blockDim().y
    while stride > 1
        stride >>= 1
        if threadIdx().y <= stride
            block_acc[threadIdx().x, threadIdx().y] +=
                block_acc[threadIdx().x, threadIdx().y + stride]
        end
        sync_threads()
    end
    if i <= batch_size && threadIdx().y == 1
        out[i, blockIdx().y] = block_acc[threadIdx().x, 1]
    end
    nothing
end
function dot_batched_cuda(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T}
    function num_threads(threads)
        threads_x = 2 # 64
        threads_y = 1 # prevpow(2, div(threads, threads_x))
        threads_x, threads_y
    end
    num_blocks(threads) = cld.((size(A, 1), size(A, 2)), threads)
    amount_shmem(threads) = prod(threads) * sizeof(T)
    kernel = @cuda launch = false dot_kernel_batched!(size(A, 1), size(A, 2), A, A, B)
    config = launch_configuration(kernel.fun, shmem = amount_shmem)
    threads = num_threads(config.threads)
    blocks = num_blocks(threads)
    @info "" blocks
    shmem = amount_shmem(threads)
    out = similar(A, size(A, 1), blocks[2])
    kernel(size(A, 1), size(A, 2), out, A, B; threads = threads, blocks = blocks, shmem = shmem)
    sum(out, dims = 1)
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

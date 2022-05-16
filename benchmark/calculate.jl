using Adapt
using CUDA
using HDF5
using LinearAlgebra
using Plasmons

load(n) = h5open(io -> read(io, "/H"), "input_square_$(n)x$(n).h5", "r")
prepare(n) = eigen(load(n))
ℝ = Float32 # Float64

function run(E, ψ; usecuda=false, method=:simple, blocks=true)
    # @assert method == :batched || method == :thesis
    if usecuda
        E = adapt(CuArray{ℝ}, E)
        ψ = adapt(CuArray{ℝ}, ψ)
    end
    η = 6e-3
    μ = 0.4
    kT = 8.617333262145E-5 * 300.0
    ωs = [0.43, 0.44]
    χs = []
    ts = Float64[]
    for (i, ω) in enumerate(ωs)
        t₀ = time_ns()
        χ = Array(polarizability(
            convert(complex(ℝ), ω),
            E,
            ψ;
            mu = convert(ℝ, μ),
            kT = convert(ℝ, kT),
            method = method,
            blocks = blocks,
        ))
        t₁ = time_ns()
        push!(χs, χ)
        push!(ts, (t₁ - t₀) / 1e9)
    end
    return ts
end

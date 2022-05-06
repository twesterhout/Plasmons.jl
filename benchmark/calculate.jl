
using LinearAlgebra
using HDF5
using Plasmons

function load(n)
    h5open("square_sheet_$(n)x$n.hdf5") do f
        H = read_hamiltonian(f)
        p = read_coordinates(f)
        return H, p
    end
end

function prepare(n = 20)
    H, _ = load(n)
    E, ψ = eigen(H)
    return E, ψ
end

function run(E, ψ; method=:batched)
    @assert method == :batched || method == :thesis
    η = 6e-3
    μ = 0.4
    kT = 8.617333262145E-5 * 300.0
    ωs = [0.43, 0.44, 0.45, 0.46]
    χ = Array{Any, 1}(undef, length(ωs))
    for (i, ω) in enumerate(ωs)
        if method == :batched
            χ[i] = polarizability_batched(ω + η * 1im, E, ψ; mu = μ, kT = kT)
        else
            χ[i] = polarizability_thesis(ω + η * 1im, E, ψ; mu = μ, kT = kT)
        end
    end
    return χ
end


using LinearAlgebra
using HDF5
using Plasmons

function load()
    h5open("sample.hdf5") do f
        H = read_hamiltonian(f)
        p = read_coordinates(f)
        return H, p
    end
end

function main()
    H, p = load()
    V = coulomb_simple(p...; v0=15.78)
    E, ψ = eigen(H)

    for ħω ∈ [0.45, 0.46, 0.47, 0.48, 0.49, 0.5]
        G = g(ħω + 6e-3im, E; mu=0.4, kT=8.617333262145E-5 * 300)
        χ = polarizability(G=G, ψ=ψ)
        ε = dielectric(χ, V)
        loss = eigvals(ε)
        loss = maximum(-imag(1 ./ loss))
        println(loss)
    end
end

main()

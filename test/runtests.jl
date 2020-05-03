using Plasmons
using LinearAlgebra
using HDF5
using Test

@testset "Plasmons.jl" begin

    @testset "fermidirac" begin
        @test Plasmons.fermidirac(0.52193; mu=0.4, kT=0.1) ≈ 0.228059661030488549677
        @test Plasmons.fermidirac(2.0; mu=-0.8, kT=0.13) ≈ 4.425527104542660880979e-10
        @test Plasmons.fermidirac(-2.0; mu=-0.8, kT=0.13) ≈ 0.9999020317590553138168788
        @test Plasmons.fermidirac(100.0; mu=0.82, kT=0.1) ≈ 0
        @test Plasmons.fermidirac(-100.0; mu=0.87, kT=0.1) ≈ 1
    end

    @testset "_g" begin
        for T in (Float32, Float64)
            E = rand(T, 10)
            μ = T(0.45)
            kT = T(0.01)
            ħω = complex(T)(0.5 + 1e-2im)
            G = Plasmons._g(ħω, E; mu = μ, kT = kT)
            @test eltype(G) == complex(T)
            @test size(G) == (length(E), length(E))
        end
    end

    @testset "read_coordinates & read_hamiltonian" begin
        H, p = h5open("square_8x8.hdf5") do f
            H = read_hamiltonian(f)
            p = read_coordinates(f)
            H, p
        end
        @test size(H) == (64, 64)
        @test ishermitian(H)
        @test size(p[1]) == (64,)
        @test size(p[2]) == (64,)
        @test size(p[3]) == (64,)
    end

    @testset "coulomb_simple" begin
        x = [1.0e-9; 2.0e-9; 3.0e-9]
        y = [4.0e-9; 5.0e-9; 6.0e-9]
        z = [7.0e-9; 8.0e-9; 9.0e-9]
        V = coulomb_simple(x, y, z; v0 = 4.5)
        @test size(V) == (3, 3)
        @test issymmetric(V)
        @test V[diagind(V)] == [4.5; 4.5; 4.5]
    end

    @testset "polarizability_thesis" begin
        kT = 8.617333262145E-5 * 300.0
        μ = 0.4

        H₁ = h5open("square_8x8.hdf5") do f
            read_hamiltonian(f)
        end
        E, ψ = eigen(H₁)
        for ω in [0.4; 0.43; 0.45]
            ħω = ω + 1e-3im
            χ₁ = polarizability_simple(ħω, E, ψ; mu = μ, kT = kT)
            χ₂ = polarizability_thesis(ħω, E, ψ; mu = μ, kT = kT)
            @test χ₁ ≈ χ₂
        end

        # Add random hermitian noise to H
        δH = rand(ComplexF64, size(H₁)...) .- (0.5 + 0.5im)
        H₂ = H₁ + δH + δH'
        E, ψ = eigen(H₂)
        for ω in [0.4; 0.43; 0.45]
            ħω = ω + 1e-3im
            χ₁ = polarizability_simple(ħω, E, ψ; mu = μ, kT = kT)
            χ₂ = polarizability_thesis(ħω, E, ψ; mu = μ, kT = kT)
            @test χ₁ ≈ χ₂
        end
    end

    @testset "_analyze_top_left" begin
        G = rand(50, 50)
        fill!(view(G, 1:5, 1:7), zero(eltype(G)))
        G[6, 2] = 0.1
        @test Plasmons._analyze_top_left(G, 1e-2) == 5

        G = rand(50, 50)
        fill!(view(G, 1:5, 1:5), zero(eltype(G)))
        G[5, 6] = -0.1
        @test Plasmons._analyze_top_left(G, 1e-2) == 5

        G = rand(50, 50)
        fill!(view(G, :, 1:5), zero(eltype(G)))
        G[4, 6] = 0.1
        @test Plasmons._analyze_top_left(G, 1e-2) == 5

        G = rand(50, 50)
        fill!(view(G, :, 1:5), zero(eltype(G)))
        G[1, 1] = 0.1
        @test Plasmons._analyze_top_left(G, 1e-2) == 0
    end

    @testset "_analyze_bottom_right" begin
        G = rand(50, 50)
        fill!(view(G, 25:50, 27:50), zero(eltype(G)))
        G[26, 48] = 0.1
        @test Plasmons._analyze_bottom_right(G, 1e-2) == 24

        G = rand(50, 50)
        fill!(view(G, 27:50, 27:50), zero(eltype(G)))
        G[48, 27] = 0.1
        @test Plasmons._analyze_bottom_right(G, 1e-2) == 23
    end

    @testset "makeblocks & mul!" begin
        G = rand(50, 50) .- 0.5
        fill!(view(G, 1:3, 1:11), zero(eltype(G)))
        fill!(view(G, 35:50, 37:50), zero(eltype(G)))
        B = Plasmons.makeblocks(G)

        A = rand(38, 50)
        C = rand(38, 50)
        D = deepcopy(C)
        @test mul!(C, A, B, 0.183, -0.482) ≈ 0.183 .* (A * G) .+ (-0.482) .* D
    end

    @testset "polarizability_batched" begin
        kT = 8.617333262145E-5 * 300.0
        μ = 0.4

        H₁ = h5open("square_8x8.hdf5") do f
            read_hamiltonian(f)
        end
        E, ψ = eigen(H₁)
        for ω in [0.4; 0.43; 0.45]
            ħω = ω + 1e-3im
            χ₁ = polarizability_thesis(ħω, E, ψ; mu = μ, kT = kT)
            χ₂ = polarizability_batched(ħω, E, ψ; mu = μ, kT = kT)
            @test χ₁ ≈ χ₂
        end
    end
end

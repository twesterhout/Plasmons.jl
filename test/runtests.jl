using Plasmons
using LinearAlgebra
using HDF5
using Test

@testset "Plasmons.jl" begin

    @testset "fermidirac" begin
        @test Plasmons.fermidirac(0.52193; mu = 0.4, kT = 0.1) ≈ 0.228059661030488549677
        @test Plasmons.fermidirac(2.0; mu = -0.8, kT = 0.13) ≈ 4.425527104542660880979e-10
        @test Plasmons.fermidirac(-2.0; mu = -0.8, kT = 0.13) ≈ 0.9999020317590553138168788
        @test Plasmons.fermidirac(100.0; mu = 0.82, kT = 0.1) ≈ 0
        @test Plasmons.fermidirac(-100.0; mu = 0.87, kT = 0.1) ≈ 1
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

        E = rand(Float64, 10)
        μ = 0.45
        kT = 0.01
        ħω = 1e-2im
        G = Plasmons._g(ħω, E; mu = μ, kT = kT)
        @test ishermitian(G)
    end

    @testset "coulomb_simple" begin
        x = [1.0e-9; 2.0e-9; 3.0e-9]
        y = [4.0e-9; 5.0e-9; 6.0e-9]
        z = [7.0e-9; 8.0e-9; 9.0e-9]
        V = Plasmons.coulomb_simple(x, y, z; V₀ = 4.5)
        @test size(V) == (3, 3)
        @test issymmetric(V)
        @test V[diagind(V)] == [4.5; 4.5; 4.5]
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

    @testset "_ThreeBlockMatrix" begin
        G = rand(50, 60) .- 0.5
        fill!(view(G, 1:3, 1:11), zero(eltype(G)))
        fill!(view(G, 35:50, 37:60), zero(eltype(G)))
        @test_throws DimensionMismatch Plasmons._ThreeBlockMatrix(G)

        G = rand(50, 50) .- 0.5
        fill!(view(G, 1:26, 1:25), zero(eltype(G)))
        fill!(view(G, 26:50, 26:50), zero(eltype(G)))
        @test_throws ErrorException Plasmons._ThreeBlockMatrix(G)
    end

    @testset "mul!" begin
        G = rand(50, 50) .- 0.5
        fill!(view(G, 1:3, 1:11), zero(eltype(G)))
        fill!(view(G, 35:50, 37:50), zero(eltype(G)))
        B = Plasmons._ThreeBlockMatrix(G)

        A = rand(38, 50)
        C = rand(38, 50)
        D = deepcopy(C)
        @test mul!(C, A, B, 0.183, -0.482) ≈ 0.183 .* (A * G) .+ (-0.482) .* D
    end

    @testset "polarizability & dielectric" begin
        kT = 8.617333262145E-5 * 300.0
        μ = 0.4

        H = h5open(io -> read(io, "H"), "input.h5")
        V = h5open(io -> read(io, "V"), "input.h5")
        @test issymmetric(V)
        E, ψ = eigen(H)
        for ω in [0.0; 0.4; 0.43; 0.45]
            ħω = ω + 1e-3im
            χ₁ = polarizability(ħω, E, ψ; mu = μ, kT = kT, method = :simple)
            χ₂ = polarizability(ħω, E, ψ; mu = μ, kT = kT, method = :thesis)
            χ₃ = polarizability(ħω, E, ψ; mu = μ, kT = kT, method = :batched)
            @test χ₁ ≈ χ₂
            @test χ₁ ≈ χ₃
            @test χ₁ ≈ transpose(χ₁)
            @test issymmetric(χ₂)
            @test issymmetric(χ₃)
            ε₃ = dielectric(χ₃, V)
            @test size(ε₃) == size(H)
        end
    end
end

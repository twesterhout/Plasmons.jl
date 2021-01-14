using Plasmons
using LinearAlgebra
using HDF5
using Test
using CUDA
using Adapt

CUDA.allowscalar(false)

@testset "Plasmons.jl" begin
    if false
        @testset "fermidirac" begin
            @test Plasmons.fermidirac(0.52193; mu = 0.4, kT = 0.1) ≈ 0.228059661030488549677
            @test Plasmons.fermidirac(2.0; mu = -0.8, kT = 0.13) ≈
                  4.425527104542660880979e-10
            @test Plasmons.fermidirac(-2.0; mu = -0.8, kT = 0.13) ≈
                  0.9999020317590553138168788
            @test Plasmons.fermidirac(100.0; mu = 0.82, kT = 0.1) ≈ 0
            @test Plasmons.fermidirac(-100.0; mu = 0.87, kT = 0.1) ≈ 1
        end

        @testset "g_matrix" begin
            for T in (Float32, Float64)
                E = rand(T, 10)
                μ = T(0.45)
                kT = T(0.01)
                ħω = complex(T)(0.5 + 1e-2im)
                G = Plasmons.g_matrix(complex(T), ħω, E; mu = μ, kT = kT, blocks = false)
                @test eltype(G) == complex(T)
                @test size(G) == (length(E), length(E))
                @test Diagonal(G) ≈ UniformScaling(0.0)
                if CUDA.functional()
                    G₂ = Plasmons.g_matrix(
                        complex(T),
                        ħω,
                        CuVector(E);
                        mu = μ,
                        kT = kT,
                        blocks = false,
                    )
                    @test eltype(G₂) == complex(T)
                    @test size(G₂) == (length(E), length(E))
                    @test typeof(G₂) <: CuArray
                    @test G ≈ Array(G₂)
                end
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

        @testset "ThreeBlockMatrix" begin
            G = rand(50, 60) .- 0.5
            fill!(view(G, 1:3, 1:11), zero(eltype(G)))
            fill!(view(G, 35:50, 37:60), zero(eltype(G)))
            @test_throws DimensionMismatch Plasmons.ThreeBlockMatrix(G)

            for T in (Float32, Float64)
                H = rand(T, 50, 50)
                H = Hermitian(H + transpose(H))
                E = eigvals(H)
                μ = T(0.45)
                for kT in T[0.001, 0.01, 0.1, 1.0, 10.0]
                    for _ω in -1.0:0.5:1.0
                        ħω = complex(T)(_ω + 1e-3im)
                        G₁ = Plasmons._build_g(ħω, E; mu = μ, kT = kT)
                        # full matrix
                        G₂ = Plasmons.ThreeBlockMatrix(G₁)
                        @test G₁ ≈ convert(Array, G₂)
                        # real and imaginary parts
                        G₂ = Plasmons.ThreeBlockMatrix(real(G₁))
                        @test real(G₁) ≈ convert(Array, G₂)
                        G₂ = Plasmons.ThreeBlockMatrix(imag(G₁))
                        @test imag(G₁) ≈ convert(Array, G₂)
                    end
                end
            end
        end

        @testset "mul! for ThreeBlockMatrix" begin
            G = rand(50, 50) .- 0.5
            fill!(view(G, 1:3, 1:11), zero(eltype(G)))
            fill!(view(G, 35:50, 37:50), zero(eltype(G)))
            B = Plasmons.ThreeBlockMatrix(G)
            A = rand(38, 50)
            C = rand(38, 50)
            D = deepcopy(C)
            @test mul!(C, A, B, 0.183, -0.482) ≈ 0.183 .* (A * G) .+ (-0.482) .* D

            for T in (Float32, Float64)
                for n in [1, 2, 3, 10, 50]
                    for use_cuda in (false, true)
                        if use_cuda && !CUDA.functional()
                            continue
                        end
                        G = rand(T, n, n) .- 0.5
                        n₁ = rand(0:div(n, 2))
                        n₂ = rand(0:div(n, 2))
                        fill!(view(G, 1:n₁, 1:n₁), zero(eltype(G)))
                        fill!(view(G, (n - n₂ + 1):n, (n - n₂ + 1):n), zero(eltype(G)))
                        B = Plasmons.ThreeBlockMatrix(G)
                        use_cuda && (G = CuArray(G))
                        use_cuda && (B = adapt(CuArray, B))

                        A = rand(T, 7, n)
                        use_cuda && (A = CuArray(A))
                        C = rand(T, 7, n)
                        use_cuda && (C = CuArray(C))
                        D = deepcopy(C)
                        mul!(C, A, B, 0.183, -0.482)
                        mul!(D, A, G, 0.183, -0.482)
                        @test C ≈ D
                    end
                end
            end
        end

        @testset "g_matrix with blocks" begin
            for T in (Float32, Float64)
                for use_cuda in (false, true)
                    if use_cuda && !CUDA.functional()
                        continue
                    end
                    E = rand(T, 10)
                    use_cuda && (E = CuArray(E))
                    μ = T(0.45)
                    kT = T(0.01)
                    ħω = complex(T)(0.5 + 1e-2im)
                    (Gᵣ, Gᵢ) = Plasmons.g_matrix(T, ħω, E; mu = μ, kT = kT, blocks = true)
                    @test eltype(Gᵣ) == T
                    @test size(Gᵣ) == (length(E), length(E))
                    @test eltype(Gᵢ) == T
                    @test size(Gᵢ) == (length(E), length(E))
                end
            end
        end
    end

    if false
        @testset "dot_batched" begin
            reference(x, y) = sum(x .* y, dims = 2)
            for T in (Float32, Float64)
                for n in [1, 2, 3, 30, 50, 5000]
                    for m in [1, 2, 3, 30, 50, 5000]
                        A = rand(n, m) .- 0.5
                        B = rand(n, m) .- 0.5
                        expected = reference(A, B)
                        @test Plasmons.dot_batched(A, B) ≈ expected
                        if CUDA.functional()
                            A = CuArray(A)
                            B = CuArray(B)
                            @test Array(Plasmons.dot_batched(A, B)) ≈ expected
                        end
                    end
                end
            end
        end
    end

    @testset "polarizability_batched (ℝ)" begin
        for T in (Float32, Float64)
            H = rand(T, 50, 50)
            H = Hermitian(H + transpose(H))
            E, ψ = eigen(H)
            μ = T(0.45)
            for kT in T[0.01, 0.1, 1.0, 10.0]
                for _ω in -1.0:0.5:1.0
                    ħω = complex(T)(_ω + 1e-3im)
                    χ₁ = Plasmons.polarizability_batched(
                        ħω,
                        E,
                        ψ;
                        mu = μ,
                        kT = kT,
                        blocks = false,
                    )
                    χ₂ = Plasmons.polarizability_batched(
                        ħω,
                        E,
                        ψ;
                        mu = μ,
                        kT = kT,
                        blocks = true,
                    )
                    χ₃ = Plasmons.polarizability_simple(ħω, E, ψ; mu = μ, kT = kT)
                    @test χ₁ ≈ χ₃
                    @test χ₂ ≈ χ₃
                    if CUDA.functional()
                        E₀ = CuArray(E)
                        ψ₀ = CuArray(ψ)
                        χ₁ = Array(Plasmons.polarizability_batched(
                            ħω,
                            E₀,
                            ψ₀;
                            mu = μ,
                            kT = kT,
                            blocks = false,
                        ))
                        χ₂ = Array(Plasmons.polarizability_batched(
                            ħω,
                            E₀,
                            ψ₀;
                            mu = μ,
                            kT = kT,
                            blocks = true,
                        ))
                        @test χ₁ ≈ χ₃
                        @test χ₂ ≈ χ₃
                    end
                end
            end
        end
    end

    @testset "polarizability_batched (ℂ)" begin
        for T in (Float32, Float64)
            H = (rand(T, 50, 50) .- T(0.5)) + (rand(T, 50, 50) .- T(0.5)) * 1im
            H = Hermitian(H)
            E, ψ = eigen(H)
            μ = T(0.45)
            for kT in T[0.01, 0.1, 1.0, 10.0]
                for _ω in -1.0:0.5:1.0
                    ħω = complex(T)(_ω + 1e-3im)
                    χ₁ = Plasmons.polarizability_batched(
                        ħω,
                        E,
                        ψ;
                        mu = μ,
                        kT = kT,
                        blocks = false,
                    )
                    χ₂ = Plasmons.polarizability_batched(
                        ħω,
                        E,
                        ψ;
                        mu = μ,
                        kT = kT,
                        blocks = true,
                    )
                    χ₃ = Plasmons.polarizability_simple(ħω, E, ψ; mu = μ, kT = kT)
                    @test χ₁ ≈ χ₃
                    @test χ₂ ≈ χ₃
                    if CUDA.functional()
                        E₀ = CuArray(E)
                        ψ₀ = CuArray(ψ)
                        χ₁ = Array(Plasmons.polarizability_batched(
                            ħω,
                            E₀,
                            ψ₀;
                            mu = μ,
                            kT = kT,
                            blocks = false,
                        ))
                        χ₂ = Array(Plasmons.polarizability_batched(
                            ħω,
                            E₀,
                            ψ₀;
                            mu = μ,
                            kT = kT,
                            blocks = true,
                        ))
                        @test χ₁ ≈ χ₃
                        @test χ₂ ≈ χ₃
                    end
                end
            end
        end
    end

    if false
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
end

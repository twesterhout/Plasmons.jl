
for (fname, elty) in (
    (Symbol("cublasDgemm_v2"), :Float64),
    (Symbol("cublasSgemm_v2"), :Float32),
    (Symbol("cublasZgemm_v2"), :ComplexF64),
    (Symbol("cublasCgemm_v2"), :ComplexF32),
)
    @eval begin
        function _gemm!(
            transA::Char,
            transB::Char,
            alpha::Number,
            A::StridedCuMatrix{$elty},
            B::StridedCuMatrix{$elty},
            beta::Number,
            C::StridedCuMatrix{$elty},
        )
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C, 1) || n != size(C, 2) || k != size(B, transB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            if stride(A, 1) != 1 || stride(B, 1) != 1 || stride(C, 1) != 1
                throw(DimensionMismatch("cuBLAS requires matrices to be in column-major order"))
            end
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            ldc = max(1, stride(C, 2))
            CUDA.CUBLAS.$fname(
                CUDA.CUBLAS.handle(),
                transA,
                transB,
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
            C
        end

    end
end
for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function _mul_with_strides!(
            C::StridedCuMatrix{$elty},
            A::StridedCuMatrix{$elty},
            B::StridedCuMatrix{$elty},
            a::$elty,
            b::$elty,
        )
            if C isa CuArray
                mul!(C, A, B, a, b)
            else
                _gemm!('N', 'N', a, A, B, b, C)
            end
        end
        _mul_with_strides!(C::StridedMatrix{$elty}, A, B, a::$elty, b::$elty) =
            mul!(C, A, B, a, b)
    end
end

@doc raw"""
    ThreeBlockMatrix

!!! warning
    This is an internal data structure

A dense matrix with top-left and bottom-right blocks assumed to be zero:
![three-block-structure](three-blocks.jpg)

`ThreeBlockMatrix` stores blocks 1, 2, and 3 as dense matrices. There is also a special
overload of [`LinearAlgebra.mul!`](@ref) function for faster matrix-matrix multiplication.
"""
struct ThreeBlockMatrix{T, M <: AbstractMatrix{T}}
    block₁::M
    block₂::M
    block₃::M
end

Base.ndims(::ThreeBlockMatrix) = 2

Base.eltype(::Type{ThreeBlockMatrix{T, M}}) where {T, M} = T
# Base.eltype(::Type{ThreeBlockMatrix{T}}) where {T} = T
Base.eltype(::Type{ThreeBlockMatrix}) = Any

Base.size(x::ThreeBlockMatrix) = (size(x, 1), size(x, 2))
function Base.size(x::ThreeBlockMatrix, dim::Integer)
    dim > 0 || throw(DimensionMismatch("dimension out of range: $dim"))
    dim <= 2 ? size(x.block₁, 1) + size(x.block₂, 1) : 1
end

Base.real(x::ThreeBlockMatrix) =
    ThreeBlockMatrix(real(x.block₁), real(x.block₂), real(x.block₃))
Base.imag(x::ThreeBlockMatrix) =
    ThreeBlockMatrix(imag(x.block₁), imag(x.block₂), imag(x.block₃))

Base.show(io::IO, x::ThreeBlockMatrix) = print(io, "ThreeBlockMatrix")

function Base.convert(::Type{M₁}, x::ThreeBlockMatrix{T, M₂}) where {M₁, T, M₂ <: M₁}
    N = size(x.block₁, 1) + size(x.block₂, 1)
    n₁ = size(x.block₂, 1)
    n₂ = size(x.block₃, 2)
    out = convert(M₁, similar(x.block₁, N, N))
    fill!(view(out, 1:n₁, 1:n₁), zero(eltype(out)))
    fill!(view(out, (N - n₂ + 1):N, (N - n₂ + 1):N), zero(eltype(out)))
    copy!(view(out, (n₁ + 1):N, 1:(N - n₂)), x.block₁)
    copy!(view(out, 1:n₁, (n₁ + 1):(N - n₂)), x.block₂)
    copy!(view(out, 1:(N - n₂), (N - n₂ + 1):N), x.block₃)
    return out
end

Adapt.adapt_structure(to, x::ThreeBlockMatrix) =
    ThreeBlockMatrix(adapt(to, x.block₁), adapt(to, x.block₂), adapt(to, x.block₃))

@doc raw"""
    sparsity(x::ThreeBlockMatrix)

Get the fraction of zeros which are optimized away during GEMM.
"""
function sparsity(x::ThreeBlockMatrix)
    N = size(x.block₁, 1) + size(x.block₂, 1)
    n₁ = size(x.block₂, 1)
    n₂ = size(x.block₃, 2)
    (n₁^2 + n₂^2) / N^2
end

@doc raw"""
    ThreeBlockMatrix(G, n₁, n₂)

!!! warning
    This is an internal function

Given a square dense matrix `G` and sizes `n₁` and `n₂` of top-left and bottom-right blocks
respectively, construct a `ThreeBlockMatrix`. All three blocks are copied (this ensure
contiguous storage for faster matrix-matrix products).
"""
function ThreeBlockMatrix(G::AbstractMatrix, n₁::Integer, n₂::Integer)
    if size(G, 1) != size(G, 2)
        throw(DimensionMismatch("'G' must be a square matrix, but got a matrix of shape $(size(G))"))
    end
    n₁ >= 0 || throw(ArgumentError("Invalid 'n₁': $n₁"))
    n₂ >= 0 || throw(ArgumentError("Invalid 'n₂': $n₂"))
    N = size(G, 1)
    if N == 0
        throw(DimensionMismatch("Empty matrices are not (yet) supported: 'G' has shape $(size(G))"))
    end
    if (n₁ + n₂) >= N
        throw(ErrorException("Overlapping zero regions are not (yet) supported: $n₁ + $n₂ >= $N"))
    end
    ThreeBlockMatrix(
        G[(n₁ + 1):N, 1:(N - n₂)],
        G[1:n₁, (n₁ + 1):(N - n₂)],
        G[1:(N - n₂), (N - n₂ + 1):N],
    )
end

@doc raw"""
    ThreeBlockMatrix(G)

!!! warning
    This is an internal function

Analyze matrix `G` and construct three-block version of it for faster matrix-matrix
multiplication. This function computes `n₁` and `n₂` and constructs `ThreeBlockMatrix`
using [`ThreeBlockMatrix(G, n₁, n₂)`](@ref).
"""
function ThreeBlockMatrix(G::AbstractMatrix)
    if size(G, 1) != size(G, 2)
        throw(DimensionMismatch("Expected a square matrix, but G has size $(size(G))"))
    end
    N = size(G, 1)
    N > 1 || return ThreeBlockMatrix(G, 0, 0)
    # All elements smaller than ε·‖G‖ are assumed to be negligible and are set to zero.
    cutoff = mapreduce(abs, max, G) * eps(real(eltype(G)))
    n₁ = _analyze_top_left(G, cutoff)
    n₁ < N - 1 || (n₁ = N - 2)
    n₂ = _analyze_bottom_right(G, cutoff)
    n₂ < N - 1 || (n₂ = N - 2)
    if n₁ + n₂ >= N
        @warn "Temperature is so low that G contains overlapping zero regions. " *
              "Reducing n₁ or n₂. This will hurt performance..."
        # Overlapping zero regions are not yet supported... We reduce one of the blocks to
        # ensure that n₁ + n₂ == size(G, 1) - 1
        if n₁ >= n₂
            n₂ = N - n₁ - 1
        else
            n₁ = N - n₂ - 1
        end
    end
    ThreeBlockMatrix(G, n₁, n₂)
end
# function ThreeBlockMatrix(G::CuArray)
#     @warning "There are no CUDA kernels for _analyze_top_left and _analyze_bottom_right. " *
#              "Matrix 'G' will be temporarily copied to CPU..."
# end

@doc raw"""
    _analyze_top_left(G::AbstractMatrix{ℂ}, cutoff::ℝ) -> Int

!!! warning
    This is an internal function!

We assume that `G` is a square matrix with the following block structure:
![top-left-block](top-left-block.jpg)
I.e. the top-left corner of size `n₁ x n₁` consists of zeros. This function determines `n₁`
by treating all matrix elements smaller than `cutoff` as zeros.
"""
function _analyze_top_left(
    G::AbstractMatrix{ℂ},
    cutoff::ℝ,
)::Int where {ℝ <: Real, ℂ <: Union{ℝ, Complex{ℝ}}}
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
    _analyze_bottom_right(G::AbstractMatrix{ℂ}, cutoff::ℝ) -> Int

!!! warning
    This is an internal function!

Very similar to [`_analyze_top_left`](@ref) except that now the bottom right corner of G is
assumed to contain zeros. This function determines the size of this block.
"""
function _analyze_bottom_right(
    G::AbstractMatrix{ℂ},
    cutoff::ℝ,
)::Int where {ℝ <: Real, ℂ <: Union{ℝ, Complex{ℝ}}}
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

Overload of matrix-matrix multiplication for ThreeBlockMatrix. We can spare some flops
because of the zero blocks.

We replace one bigg GEMM by three smaller:

  1) ![first-gemm](gemm-1.jpg)
  2) ![second-gemm](gemm-2.jpg)
  3) ![third-gemm](gemm-3.jpg)
"""
function LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::ThreeBlockMatrix, α, β)
    N = size(B, 1)
    if size(A, 2) != N
        throw(DimensionMismatch("Dimensions of 'A' and 'B' are incompatible: $(size(A)) vs $(size(B))"))
    end
    if size(C) != size(A)
        throw(DimensionMismatch("'C' has wrong dimension: $(size(C)), expected $(size(A))"))
    end
    n₁ = size(B.block₂, 1)
    n₂ = size(B.block₃, 2)
    _mul_with_strides!(
        view(C, :, 1:size(B.block₁, 2)),
        view(A, :, (n₁ + 1):N),
        B.block₁,
        α,
        β,
    )
    if n₁ > 0
        # NOTE: If n₁ is zero, then A B α + C β amount to C β, and since β is 1 we can drop
        # it altogerher.
        #! format: off
        _mul_with_strides!(view(C, :, (n₁ + 1):size(B.block₁, 2)), view(A, :, 1:n₁), B.block₂, α, one(eltype(C)))
        #! format: on
    end
    if n₂ > 0
        #! format: off
        _mul_with_strides!(view(C, :, (size(B.block₁, 2) + 1):size(C, 2)), view(A, :, 1:size(B.block₁, 2)), B.block₃, α, β)
        #! format: on
    end
    return C
end

dot_batched(A, B) = dot_batched!(similar(A, size(A, 1)), A, B)
function dot_batched!(
    out::StridedVector{T},
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where {T}
    if size(A) != size(B)
        throw(DimensionMismatch("Dimensions of 'A' and 'B' do not match: $(size(A)) != $(size(B))"))
    end
    if length(out) != size(A, 1)
        throw(DimensionMismatch("'out' has wrong length: $(length(out)); expected $(size(A, 1))"))
    end
    (n, m) = size(A)
    fill!(out, zero(T))
    @inbounds for j in 1:m
        @simd for i in 1:n
            out[i] += conj(A[i, j]) * B[i, j]
        end
    end
    out
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
    block_acc = @cuDynamicSharedMem(T, (blockDim().x, blockDim().y))
    # Initialize block_acc for each block within the "first grid"
    if blockIdx().x <= gridDim().x && blockIdx().y <= gridDim().y
        @inbounds block_acc[threadIdx().x, threadIdx().y] = zero(T)
    end

    # Perform thread-local accumulation in each block
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= batch_size
        thread_acc = zero(T)
        @inbounds while j <= n
            thread_acc += conj(A[i, j]) * B[i, j]
            j += blockDim().y * gridDim().y
        end
        @inbounds block_acc[threadIdx().x, threadIdx().y] += thread_acc
    end
    sync_threads()

    # Reduction within each block
    stride = blockDim().y
    while stride > 1
        stride >>= 1
        if threadIdx().y <= stride
            @inbounds block_acc[threadIdx().x, threadIdx().y] +=
                block_acc[threadIdx().x, threadIdx().y + stride]
        end
        sync_threads()
    end
    if i <= batch_size && threadIdx().y == 1
        @inbounds out[i, blockIdx().y] = block_acc[threadIdx().x, 1]
    end
    nothing
end
function dot_batched!(
    out::StridedCuVector{T},
    A::StridedCuMatrix{T},
    B::StridedCuMatrix{T},
) where {T}
    if size(A) != size(B)
        throw(DimensionMismatch("Dimensions of 'A' and 'B' do not match: $(size(A)) != $(size(B))"))
    end
    if length(out) != size(A, 1)
        throw(DimensionMismatch("'out' has wrong length: $(length(out)); expected $(size(A, 1))"))
    end
    (n, m) = size(A)
    function num_threads(threads)
        threads_y = 32
        threads_x = div(threads, threads_y)
        threads_x, threads_y
    end
    num_blocks(threads) = cld.(size(A), threads)
    amount_shmem(threads) = prod(threads) * sizeof(T)
    @assert stride(A, 1) == 1 && stride(B, 1) == 1
    _A = unsafe_wrap(CuArray{T, 2}, Base.unsafe_convert(CuPtr{T}, A), (stride(A, 2), size(A, 2)))
    _B = unsafe_wrap(CuArray{T, 2}, Base.unsafe_convert(CuPtr{T}, B), (stride(B, 2), size(B, 2)))
    kernel = @cuda launch = false dot_kernel_batched!(n, m, _A, _A, _B)
    config = launch_configuration(kernel.fun, shmem = amount_shmem)
    threads = num_threads(config.threads)
    blocks = num_blocks(threads)
    shmem = amount_shmem(threads)
    temp = similar(A, n, blocks[2])
    GC.@preserve A B kernel(
        n,
        m,
        temp,
        _A,
        _B;
        threads = threads,
        blocks = blocks,
        shmem = shmem,
    )
    sum!(out, temp)
end

function _eigen!(A::Hermitian{T, CuArray{T, 2, B}}) where {T, B}
    eigenvalues, eigenvectors = T <: Complex ? CUSOLVER.heevd!('V', A.uplo, A.data) :
        CUSOLVER.syevd!('V', A.uplo, A.data)
    return eigenvalues, eigenvectors
end
function _eigen!(A::Hermitian{T, Array{T, 2}}) where {T}
    factorization = eigen!(A)
    return factorization.values, factorization.vectors
end

using LinearAlgebra

@doc raw"""
    ThreeBlockMatrix

!!! warning
    This is an internal data structure

A dense matrix with top-left and bottom-right blocks assumed to be zero:
![three-block-structure](three-blocks.jpg)

`ThreeBlockMatrix` stores blocks 1, 2, and 3 as dense matrices. There is also a special
overload of [`LinearAlgebra.mul!`](@ref) function for faster matrix-matrix multiplication.
"""
struct ThreeBlockMatrix{M <: AbstractMatrix}
    block₁::M
    block₂::M
    block₃::M
end

Base.ndims(::ThreeBlockMatrix) = 2

Base.eltype(::Type{ThreeBlockMatrix{T}}) where {T} = eltype(T)
Base.eltype(::Type{ThreeBlockMatrix}) = Any

Base.size(x::ThreeBlockMatrix) = (size(x, 1), size(x, 2))
function Base.size(x::ThreeBlockMatrix, dim::Integer)
    dim > 0 || throw(DimensionMismatch("dimension out of range: $dim"))
    dim <= 2 ? size(x.block₁, 1) + size(x.block₂, 1) : 1
end

# Base.show(io::IO, x::ThreeBlockMatrix) = print(io, "ThreeBlockMatrix")

function Base.convert(::Type{M₁}, x::ThreeBlockMatrix{M₂}) where {M₁, M₂ <: M₁}
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
    mul!(view(C, :, 1:size(B.block₁, 2)), view(A, :, (n₁ + 1):N), B.block₁, α, β)
    if n₁ > 0
        # NOTE: If n₁ is zero, then A B α + C β amount to C β, and since β is 1 we can drop
        # it altogerher.
        #! format: off
        mul!(view(C, :, (n₁ + 1):size(B.block₁, 2)), view(A, :, 1:n₁), B.block₂, α, one(eltype(C)))
        #! format: on
    end
    if n₂ > 0
        #! format: off
        mul!(view(C, :, (size(B.block₁, 2) + 1):size(C, 2)), view(A, :, 1:size(B.block₁, 2)), B.block₃, α, β)
        #! format: on
    end
    return C
end

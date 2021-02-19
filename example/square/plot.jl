using Plots
using LaTeXStrings
using LinearAlgebra
using HDF5
using DelimitedFiles
# using Plasmons
gr()

theme(
    :solarized_light;
    fg = RGB(88 / 255, 110 / 255, 117 / 255),
    fgtext = RGB(7 / 255, 54 / 255, 66 / 255),
    fgguide = RGB(7 / 255, 54 / 255, 66 / 255),
    fglegend = RGB(7 / 255, 54 / 255, 66 / 255),
)

function extract_time(file::HDF5.File, i::Int; group = "/χ")
    i = string(i, pad = 4)
    read(attributes(file["$group/$i"]), "time")
end
extract_time(file::AbstractString, i::Int; kwargs...) = h5open(io -> extract_time(io, i; kwargs...), file, "r")

function extract_time(sizes::AbstractVector{Int}; suffix)
    map(n -> extract_time("output_square_$(n)x$(n)_$suffix.h5", 2), sizes)
end

function save_timings(sizes)
    sizes = collect(sizes)
    cpu = extract_time(sizes; suffix="cpu")
    cuda = extract_time(sizes; suffix="cuda")
    writedlm("benchmark_result_square.dat", [sizes cpu cuda])
end

function plot_times()
    sizes = collect(5:5:40)
    cpu = extract_time(sizes; suffix="cpu")
    cuda = extract_time(sizes; suffix="cuda")
    g = plot(sizes, cpu, label = "Xeon E5-2450 v2")
    plot!(g, sizes, cuda, label = "Tesla K40")
    savefig(g, "")
end

# function plot_sample(x::AbstractVector, y::AbstractVector)
#     scatter(
#         x,
#         y,
#         label = "",
#         size = (400, 400),
#         aspect_ratio = :equal,
#         grid = false,
#         showaxis = false,
#         ticks = nothing,
#     )
# end
# plot_sample(file::HDF5File) = plot_sample(read(file, "x"), read(file, "y"))
# plot_sample(file::AbstractString) =
#     h5open(io -> plot_sample(read(io, "x"), read(io, "y")), file, "r")

# function plot_dos(E::AbstractVector)
#     histogram(E, bins = 30, xlabel = "E", ylabel = "ρ", normalize = true, label = nothing)
# end
# plot_dos(H::AbstractMatrix) = plot_dos(eigvals(Hermitian(H)))
# plot_dos(file::HDF5File) = plot_sample(read(file, "H"))

# function make_dispersion(ε, ks, x, y, z)
#     map(k -> Plasmons.dispersion((k, 0.0, 0.0), ε, x, y, z), ks)
# end
# function make_dispersion(file::HDF5File, x, y, z)
#     a = minimum(filter(_x -> _x > 0, x))
#     ks = (-π / a):(π / a / 100):(π / a)
#     ωs = map(d -> real(read(attrs(d), "ħω")), file["ε"])
#     out = Array{ComplexF64}(undef, length(ωs), length(ks))
#     for (i, ε) in enumerate(map(read, file["ε"]))
#         out[i, :] = make_dispersion(ε, ks, x, y, z)
#     end
#     out = -1 ./ out
#     heatmap(ks, ωs, imag(out))
# end
# function make_dispersion(file::HDF5File, x, y, z)
#     # ωs = map(d -> real(read(attrs(d), "ħω")), file["ε"])
#     out = dispersion(map(read, file["ε"]), (1.0, 0.0, 0.0), x, y, z; n = 1000)
#     # @info out
#     out = imag(-1 ./ out)
#     heatmap(out)
# end

using Plots
using LaTeXStrings
using LinearAlgebra
using HDF5
using DelimitedFiles
using LsqFit
using Printf

pgfplotsx()
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
extract_time(file::AbstractString, i::Int; kwargs...) =
    h5open(io -> extract_time(io, i; kwargs...), file, "r")

function extract_time(sizes::AbstractVector{Int}; suffix)
    map(n -> extract_time("output_square_$(n)x$(n)_$suffix.h5", 2), sizes)
end

function save_timings(sizes)
    sizes = collect(sizes)
    cpu = extract_time(sizes; suffix = "cpu")
    cuda = extract_time(sizes; suffix = "cuda")
    writedlm("benchmark_result_square.dat", [sizes cpu cuda])
end

function determine_scaling(x, y)
    @. model(x, p) = p[1] + p[2] * x
    p0 = [-10.0, 4.0]
    fit = curve_fit(model, log.(x), log.(y), p0)

    f = n -> exp(fit.param[1]) * n .^ fit.param[2]
    return fit.param[2], f
end

function plot_times()
    table = readdlm("benchmark_result_square.dat")
    sizes, cpu, cuda = table[:, 1], table[:, 2], table[:, 3]
    sizes = sizes .^ 2 # benchmark.sh stores width of the sample rather
    cpu_exp, cpu_line = determine_scaling(sizes[5:end], cpu[5:end])
    cuda_exp, cuda_line = determine_scaling(sizes[5:end], cuda[5:end])
    continuous_size = collect(sizes[begin]:0.1:sizes[end])

    g = plot(xlabel = "System size", ylabel = "Time per frequency, seconds", legend = :topleft)
    plot!(
        g,
        continuous_size,
        cpu_line(continuous_size),
        linewidth = 2,
        linecolor = :black,
        linealpha = 0.6,
        label = nothing,
    )
    plot!(
        g,
        continuous_size,
        cuda_line(continuous_size),
        linewidth = 2,
        linecolor = :black,
        linealpha = 0.6,
        label = nothing,
    )
    scatter!(g, sizes, cpu, label = @sprintf("CPU (Xeon E5-2450 v2), ν = %.2f", cpu_exp))
    scatter!(g, sizes, cuda, label = @sprintf("GPU (Tesla K40), ν = %.2f", cuda_exp))
    savefig(g, "benchmark_result_square.png")
    g
end

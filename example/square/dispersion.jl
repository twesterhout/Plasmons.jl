using HDF5
using LaTeXStrings
using Plasmons
using Plots

pgfplotsx()
theme(
    :solarized_light;
    fg = RGB(88 / 255, 110 / 255, 117 / 255),
    fgtext = RGB(7 / 255, 54 / 255, 66 / 255),
    fgguide = RGB(7 / 255, 54 / 255, 66 / 255),
    fglegend = RGB(7 / 255, 54 / 255, 66 / 255),
)

function data_for_figure_1()
    x, y, z = h5open("input_square_sheet_32x32.h5", "r") do io
        read(io["x"]), read(io["y"]), read(io["z"])
    end
    χ, ε, ωs, qs = h5open("result_square_sheet_32x32.h5", "r") do io
        ωs = map(d->real(read(attributes(d)["ħω"])), io["/ε"])
        n = 32 + 1
        qs = collect(0:(π / (n - 1)):π)
        direction = (1.0, 0.0, 0.0)
        # Each row corresponds to an ħω, each column -- to a q
        χ = dispersion((read(d) for d in io["/χ"]), map(q-> q .* direction, qs), x, y, z)
        ε = dispersion((read(d) for d in io["/ε"]), map(q-> q .* direction, qs), x, y, z)
        χ, ε, ωs, qs
    end
    h5open("dispersion_square_sheet_32x32.h5", "w") do io
        io["ε"] = ε
        io["χ"] = χ
        io["ħω"] = ωs
        io["q"] = qs
    end
    nothing
end

function figure_1()
    matrix, ωs, qs = h5open("dispersion_square_sheet_32x32.h5", "r") do io
        read(io["data"]), read(io["ħω"]), read(io["q"])
    end
    # Compute loss function
    loss = @. -imag(1 / matrix)
    loss = transpose(loss)
    g = heatmap(qs, ωs, loss, ylims = (0, 15), xlabel = "q, 1/a", ylabel = "ħω, eV",
                title = "-Im[1 / ε(q, ω)]")
    savefig(g, "dispersion_square_sheet_32x32.png")
    nothing
end

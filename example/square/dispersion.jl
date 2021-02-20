using HDF5
using LaTeXStrings
using Plasmons
using Plots

pgfplotsx()

function data_for_figure_1(n::Int = 32)
    x, y, z = h5open("input_square_sheet_$(n)x$(n).h5", "r") do io
        read(io["x"]), read(io["y"]), read(io["z"])
    end
    χ, ε, ωs, qs = h5open("result_square_sheet_$(n)x$(n).h5", "r") do io
        ωs = map(d->real(read(attributes(d)["ħω"])), io["/ε"])
        number_qs = n + 1
        qs = collect(0:(π / (number_qs - 1)):π)
        direction = (1.0, 0.0, 0.0)
        # Each row corresponds to an ħω, each column -- to a q
        χ = dispersion((read(d) for d in io["/χ"]), map(q-> q .* direction, qs), x, y, z)
        ε = dispersion((read(d) for d in io["/ε"]), map(q-> q .* direction, qs), x, y, z)
        χ, ε, ωs, qs
    end
    h5open("dispersion_square_sheet_$(n)x$(n).h5", "w") do io
        io["ε"] = ε
        io["χ"] = χ
        io["ħω"] = ωs
        io["q"] = qs
    end
    nothing
end

function figure_1()
    theme(
        :solarized_light;
        # fg = RGB(88 / 255, 110 / 255, 117 / 255),
        fg = RGB(7 / 255, 54 / 255, 66 / 255),
        fgtext = RGB(7 / 255, 54 / 255, 66 / 255),
        fgguide = RGB(7 / 255, 54 / 255, 66 / 255),
        fglegend = RGB(7 / 255, 54 / 255, 66 / 255),
    )
    χ, ε, ωs, qs = h5open("dispersion_square_sheet_32x32.h5", "r") do io
        read(io["χ"]), read(io["ε"]), read(io["ħω"]), read(io["q"])
    end
    # Compute loss function
    g₁ = heatmap(
        qs,
        ωs,
        transpose(@. -imag(1 / ε)),
        ylims = (0, 15),
        xlabel = raw"$q$, $1/a$",
        ylabel = raw"$\hbar\omega$, eV",
        title = L"$-\mathrm{Im}\left[1 / \varepsilon(q, \omega) \right]$",
    )
    g₂ = heatmap(
        qs,
        ωs,
        transpose(@. -imag(χ)),
        ylims = (0, 15),
        xlabel = nothing,
        ylabel = nothing,
        title = L"$-\mathrm{Im}\left[\chi(q, \omega)\right]$",
    )
    savefig(plot(g₁, g₂, size = (700, 350)), "dispersion_square_sheet_32x32.png")
    nothing
end

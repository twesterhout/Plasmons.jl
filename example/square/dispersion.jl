using HDF5
using Plasmons

load_coordinates(file::HDF5.File) = read(file["x"]), read(file["y"]), read(file["z"])
load_coordinates(file::AbstractString) = h5open(load_coordinates, file, "r")

matrix_iterator(file::HDF5.File, group::AbstractString) = (read(d) for d in file[group])
function matrix_iterator(file::AbstractString, group::AbstractString)
    io = h5open(file, "r")
    matrix_iterator(io, group)
end

function figure_1()
    x, y, z = load_coordinates("input_square_sheet_32x32.h5")
    εs = matrix_iterator("result_square_sheet_32x32.h5", "/ε")
    q = (1.0, 0.0, 0.0)

    m = dispersion(εs, q, x, y, z; n = 100)


    h5open("dispersion_square_sheet.h5", "w") do io
        io["m"] = m
    end
end

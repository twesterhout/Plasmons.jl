import argparse
import math
import h5py
import numpy as np
import scipy.sparse
import tipsi


def square(width: int, a: float, t: float, periodic: bool) -> tipsi.Sample:
    vectors = [[a, 0, 0], [0, a, 0]]
    orbital_coords = [[0, 0, 0]]
    lattice = tipsi.builder.Lattice(vectors, orbital_coords)

    site_set = tipsi.builder.SiteSet()
    for i in range(width):
        for j in range(width):
            site_set.add_site((i, j, 0))

    hop_dict = tipsi.builder.HopDict()
    hop_dict.set((1, 0, 0), t)
    hop_dict.set((0, 1, 0), t)

    if periodic:

        def pbc_func(unit_cell_coords, orbital):
            n0, n1, n2 = unit_cell_coords
            return (n0 % width, n1 % width, n2), orbital

    else:
        pbc_func = tipsi.builder.bc_default

    sample = tipsi.Sample(lattice, site_set, pbc_func)
    sample.add_hop_dict(hop_dict)
    return sample


def get_hamiltonian(sample: tipsi.Sample) -> np.ndarray:
    n = len(sample.site_x)  # number of sites
    if np.all(sample.hop.imag == 0):
        data = sample.hop.real
    else:
        data = sample.hop
    sparse = scipy.sparse.csr_matrix((data, sample.indices, sample.indptr), shape=(n, n))
    return sparse.todense()


def get_coulomb(sample: tipsi.Sample, v0: float) -> np.ndarray:
    e = 1.602176634e-19  # [C], from https://en.wikipedia.org/wiki/Elementary_charge
    ε0 = 8.8541878128e-12  # [F/m], from https://en.wikipedia.org/wiki/Vacuum_permittivity
    # NOTE: `scale` contains the first power of `e`. THIS IS NOT A MISTAKE!
    # [e / ε₀] = C * m / F= C * m / (C / V) = V * m, and dividing by distance we get volts.
    # And since we work in electron-volts the second multiplication with e is implicit.
    scale = e / (4 * math.pi * ε0)
    x, y, z = sample.site_x, sample.site_y, sample.site_z
    n = len(x)
    distance = lambda i, j: math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)

    v = np.empty((n, n), dtype=np.float64)
    for b in range(n):
        v[b, b] = v0
        for a in range(b + 1, n):
            v[a, b] = scale / distance(a, b)
            v[b, a] = v[a, b]
    return v


def generate_input_for_julia(filename: str, n: int, a: float, t: float, v0: float):
    """Generate HDF5 which can be fed to `Plasmons.jl`.

    :param filename: Path to output HDF5 file.
    :param n: Width/height of the sample. We always create square samples.
    :param a: Lattice constant in Å (Angstrom).
    :param t: Hopping parameter in eV.
    """
    assert n > 1
    assert a > 0
    # Convert a to meters
    a *= 1e-10
    sample = square(n, a=a, t=t, periodic=False)
    with h5py.File(filename, "w") as f:
        f.create_dataset("H", data=get_hamiltonian(sample))
        f.create_dataset("V", data=get_coulomb(sample, v0))
        # Convert x, y, and z from meters to lattice constants
        x = sample.site_x / a
        y = sample.site_y / a
        z = sample.site_z / a
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("z", data=z)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tight-binding Hamiltonian for square lattice."
    )
    parser.add_argument("-n", type=int, help="Width/height of the sample.")
    parser.add_argument("-a", type=float, default=2.46, help="Lattice constant in Å.")
    parser.add_argument("-t", type=float, default=2.7, help="Hopping parameter in eV.")
    parser.add_argument("-v0", type=float, default=15.78, help="Coulomb self interaction in eV.")
    parser.add_argument("filename", type=str, help="Path to output HDF5 file.")
    args = parser.parse_args()
    generate_input_for_julia(args.filename, args.n, args.a, args.t, args.v0)


if __name__ == "__main__":
    main()

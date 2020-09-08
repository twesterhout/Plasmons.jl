import math
import h5py
import numpy as np
import scipy.sparse
import tipsi


def square(width: int, a: float, t: float, periodic: bool) -> tipsi.builder.Sample:
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


def main():
    n = 10
    # Graphene parameters, but square lattice :P
    sample = square(n, a=0.246e-9, t=2.8, periodic=False)
    with h5py.File("square_sheet_{}x{}.h5".format(n, n), "w") as f:
        f.create_dataset("H", data=get_hamiltonian(sample))
        f.create_dataset("V", data=get_coulomb(sample, v0=15.78))
        f.create_dataset("x", data=sample.site_x)
        f.create_dataset("y", data=sample.site_y)
        f.create_dataset("z", data=sample.site_z)


if __name__ == "__main__":
    main()

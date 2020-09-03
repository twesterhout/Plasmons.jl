import tipsi
import numpy as np
import scipy.sparse
import h5py


def square(width: int, a: float, t: float) -> tipsi.builder.Sample:
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

    sample = tipsi.Sample(lattice, site_set)
    sample.add_hop_dict(hop_dict)
    return sample


def get_hamiltonian(sample: tipsi.Sample) -> np.ndarray:
    n = len(sample.site_x) # number of sites
    sparse = scipy.sparse.csr_matrix((sample.hop, sample.indices, sample.indptr), shape=(n, n))
    return sparse.todense()


def get_coulomb(sample: tipsi.Sample) -> np.ndarray:
    pass


def main():
    n = 10
    sample = square(n, 1.0, 1.0)
    with h5py.File("square_sheet_{}x{}.h5".format(n, n), "w") as f:
        f.create_dataset("H", data=get_hamiltonian(sample))


if __name__ == "__main__":
    main()

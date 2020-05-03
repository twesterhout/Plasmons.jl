import tipsi


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


def main():
    n = 35
    sample = square(n, 1.0, 1.0)
    sample.save("square_sheet_{}x{}.hdf5".format(n, n))


if __name__ == "__main__":
    main()

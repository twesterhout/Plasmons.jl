import tipsi


def square_lattice(a: float) -> tipsi.builder.Lattice:
    vectors = [[a, 0, 0], [0, a, 0]]
    orbital_coords = [[0, 0, 0]]
    return tipsi.builder.Lattice(vectors, orbital_coords)


def carpet_sites(start_width, iteration):
    width = start_width * 3 ** iteration
    height = width
    site_set = tipsi.builder.SiteSet()
    for i in range(height):
        for j in range(width):
            site_set.add_site((i, j, 0))

    def in_hole(tag, scale):
        x, y, _, _ = tag
        x = x % scale
        y = y % scale
        return (
            x >= scale // 3
            and x < 2 * scale // 3
            and y >= scale // 3
            and y < 2 * scale // 3
        )

    delete = []
    scale = width
    for i in range(iteration):
        for tag in site_set.sites:
            if in_hole(tag, scale):
                delete.append(tag)
        scale /= 3

    for tag in delete:
        x, y, z, orbital = tag
        site_set.delete_site((x, y, z), orbital)

    return site_set


def carpet_hoppings(t):
    hop_dict = tipsi.builder.HopDict()
    hop_dict.set((1, 0, 0), t)
    hop_dict.set((0, 1, 0), t)
    return hop_dict


def carpet(start_width, iteration, lattice_constant, hopping_value):
    lattice = square_lattice(lattice_constant)
    sites = carpet_sites(start_width, iteration)
    sample = tipsi.Sample(lattice, sites, bc_func=lambda p, o: (p, o))
    sample.add_hop_dict(carpet_hoppings(hopping_value))
    return sample


def main():
    sample = carpet(1, 3, 0.246e-9, 2.8)
    # sample.plot()
    sample.save()


if __name__ == "__main__":
    main()

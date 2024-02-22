from collections.abc import Sequence

from util.hyperbox.hyperbox import Hyperbox


def check_dimensions(hyperbox: Hyperbox, expected_dimensions: int):
    h_dim = hyperbox.n_dimensions()
    if h_dim != expected_dimensions:
        raise ValueError(
            "Hyperbox does not have the expected number of dimensions.\n" +
            "Dimensions of hyperbox: " + str(h_dim) + "\n" +
            "Expected dimensions: " + str(expected_dimensions) + "\n")


def check_dimensions_all(hyperboxes: Sequence[Hyperbox], expected_dimensions: int):
    for h in hyperboxes:
        check_dimensions(hyperbox=h, expected_dimensions=expected_dimensions)

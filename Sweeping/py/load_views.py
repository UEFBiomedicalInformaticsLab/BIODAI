from collections.abc import Iterable, Sequence

import os
from typing import Optional

from input_data.fallback_locations import FallbackLocations, EMPTY_FALLBACK_LOCATIONS
from util.table.load_table import load_table, load_table_by_path
from util.table.table import Table
from util.table.table_utils import n_col, n_row
from util.printer.printer import Printer
from views.views import Views, JustViews


def view_exists(directory: str, view_type: str) -> bool:
    to_load_path_csv = os.path.join(directory, view_type + ".csv")
    to_load_path_hdf5 = os.path.join(directory, view_type + ".hdf5")
    return os.path.isfile(to_load_path_csv) or os.path.isfile(to_load_path_hdf5)


def load_view_with_fallbacks(directory: str, view_type: str, printer: Printer,
                             fallback_locations: Sequence[str] = ()) -> Optional[Table]:
    printer.print("Loading view " + view_type)
    v_res = None
    if view_exists(directory, view_type):
        printer.print("Loading view from default directory " + str(directory))
        v_res = load_table(directory=directory, table_name=view_type)
    else:
        printer.print("View not found in default directory " + str(directory))
        i = 0
        while v_res is None and i < len(fallback_locations):
            f = fallback_locations[i]
            if os.path.isfile(f):
                printer.print("Loading view from fallback location " + str(f))
                v_res = load_table_by_path(path=f)
            else:
                printer.print("View not found in fallback location " + str(f))
            i += 1
    if v_res is None:
        printer.print("View does not exist.")
        return None
    else:
        nrows = n_row(v_res)
        ncols = n_col(v_res)
        printer.print("View loaded with " + str(nrows) + " rows and " + str(ncols) + " columns.")
        if nrows > 0 and ncols > 0:
            return v_res
        else:
            printer.print("Discarding view for lack of data.")
            return None


def load_all_views(directory: str, view_types: Iterable[str], printer: Printer,
                   fallback_locations: FallbackLocations = EMPTY_FALLBACK_LOCATIONS) -> Views:
    """Does not include checks for consistency between the views. Keeps row names."""
    res = {}
    for v in view_types:
        v_res= load_view_with_fallbacks(directory=directory, view_type=v, printer=printer,
                                        fallback_locations=fallback_locations.locations_for_view(view_name=v))
        if v_res is not None:
            res[v] = v_res
    return JustViews(views_dict=res)

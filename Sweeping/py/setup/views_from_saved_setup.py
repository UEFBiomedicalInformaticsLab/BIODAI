import os
from collections.abc import Sequence

from input_data.input_creator.input_creator import INPUT_DIR_NAME
from load_views import view_exists
from setup.setup_reader import read_setup
from setup.setup_utils import SAVED_CONFIG_FILE_NAME


def up_dir_tree(path: str, levels: int) -> str:
    for _ in range(levels):
        path, _ = os.path.split(path)
    return path


def optimizer_dir_to_input_dir(optimizer_dir: str) -> str:
    base_dir = up_dir_tree(path=optimizer_dir, levels=4)
    for _ in range(4):
        candidate = os.path.join(base_dir, INPUT_DIR_NAME)
        if os.path.isdir(candidate):
            return candidate
        else:
            base_dir = up_dir_tree(path=base_dir, levels=1)
    raise ValueError("No input directory found.")


def views_from_saved_setup(optimizer_dir: str, check_existence_in_input_dir: bool = False) -> Sequence[str]:
    """Returns view names sorted alphabetically.
    If check_existence_in_input_dir is True, will exclude views not present in the input directory.
    Also views read from an alternative fallback directory would be excluded."""
    file_path = os.path.join(optimizer_dir, SAVED_CONFIG_FILE_NAME)
    if os.path.isfile(file_path):
        setup = read_setup(file=file_path)
        setup_views = setup.views_to_use().all_views_seq()
        input_dir = optimizer_dir_to_input_dir(optimizer_dir=optimizer_dir)
        if check_existence_in_input_dir:
            actual_views = []
            for v in setup_views:
                if view_exists(directory=input_dir, view_type=v):
                    actual_views.append(v)
            return actual_views
        else:
            return setup_views
    else:
        raise ValueError("Setup file not found: " + file_path)

import os

from location_manager.location_consts import HOFS_STR


def create_optimizer_save_path(save_path: str, optimizer_nick: str) -> str:
    """Ends with '/'."""
    return save_path + optimizer_nick + "/"


def hofs_path_from_optimizer_path(optimizer_path: str) -> str:
    return os.path.join(optimizer_path, HOFS_STR) + "/"


def main_path_for_saves(base_path: str, optimizer_nick: str) -> str:
    optimizer_path = create_optimizer_save_path(save_path=base_path, optimizer_nick=optimizer_nick)
    return hofs_path_from_optimizer_path(optimizer_path=optimizer_path)


def path_for_saves(base_path: str, optimizer_nick: str, hof_nick: str) -> str:
    return main_path_for_saves(base_path=base_path, optimizer_nick=optimizer_nick) + hof_nick + "/"

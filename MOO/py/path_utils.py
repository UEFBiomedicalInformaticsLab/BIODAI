def create_optimizer_save_path(save_path: str, optimizer_nick: str) -> str:
    """Ends with '/'."""
    return save_path + optimizer_nick + "/"

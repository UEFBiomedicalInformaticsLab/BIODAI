def row_by_index(index: int, ncol: int) -> int:
    return index // ncol


def col_by_index(index: int, ncol: int) -> int:
    return index % ncol


def row_col_by_index(index: int, ncol: int) -> (int, int):
    """First row and then col"""
    return row_by_index(index=index, ncol=ncol), col_by_index(index=index, ncol=ncol)

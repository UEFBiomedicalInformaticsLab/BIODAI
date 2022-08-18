def check_none(x):
    if x is None:
        raise ValueError("Unexpected None")
    return x

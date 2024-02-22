from util.distribution.distribution import Distribution


class UniformDistribution(Distribution):
    __len: int
    __item_val: float

    def __init__(self, size: int):
        self.__len = size
        if size > 0:
            self.__item_val = 1.0 / size
        else:
            self.__item_val = 0.0

    def __getitem__(self, i: int) -> float:
        s_len = self.__len
        if -s_len <= i < s_len:
            return self.__item_val
        else:
            raise IndexError()

    def __len__(self) -> int:
        return self.__len

    def is_uniform(self) -> bool:
        return True

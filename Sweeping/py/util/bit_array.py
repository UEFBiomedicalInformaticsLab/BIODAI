import sys

import numpy as np
from numpy import ndarray


# Shape is lost upon packing.
class BitArray:

    def __init__(self, array):
        if isinstance(array, list):
            self.__original_size = len(array)
            self.__packed = np.packbits(array, axis=None)
        elif isinstance(array, ndarray):
            self.__original_size = array.size
            self.__packed = np.packbits(array, axis=None)
        else:
            raise Exception('Input type not supported: ' + str(type(array)))
        if False:
            print("size before")
            print(sys.getsizeof(array))
            print("size after")
            print(sys.getsizeof(self))
        if False:
            if not np.array_equal(array, self.unpack()):
                raise Exception("Equality test failed.")

    def unpack(self):
        return np.unpackbits(self.__packed, axis=None)[:self.__original_size].astype(bool)

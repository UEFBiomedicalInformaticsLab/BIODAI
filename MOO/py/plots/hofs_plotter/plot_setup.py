from typing import Optional


class PlotSetup:
    __x_min: Optional[float]
    __x_max: Optional[float]
    __y_min: Optional[float]
    __y_max: Optional[float]
    __alpha: Optional[float]
    __labels_map: dict[str, str]

    def __init__(self,
                 x_min: Optional[float] = None, x_max: Optional[float] = None,
                 y_min: Optional[float] = None, y_max: Optional[float] = None,
                 alpha: Optional[float] = None, labels_map: Optional[dict[str, str]] = None):
        self.__x_min = x_min
        self.__x_max = x_max
        self.__y_min = y_min
        self.__y_max = y_max
        self.__alpha = alpha
        if labels_map is None:
            self.__labels_map = {}
        else:
            self.__labels_map = labels_map

    def x_min(self) -> Optional[float]:
        return self.__x_min

    def x_max(self) -> Optional[float]:
        return self.__x_max

    def y_min(self) -> Optional[float]:
        return self.__y_min

    def y_max(self) -> Optional[float]:
        return self.__y_max

    def alpha(self) -> Optional[float]:
        return self.__alpha

    def labels_map(self) -> dict[str, str]:
        return self.__labels_map

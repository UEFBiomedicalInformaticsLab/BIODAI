from typing import Optional

from consts import FONT_SIZE
from plots.default_labels_map import LabelsTransformer, DUMMY_LABELS_TRANSFORMER, DEFAULT_LABELS_TRANSFORMER


class PlotSetup:
    __x_min: Optional[float]
    __x_max: Optional[float]
    __y_min: Optional[float]
    __y_max: Optional[float]
    __alpha: Optional[float]
    __labels_map: LabelsTransformer
    __font_size: int

    def __init__(self,
                 x_min: Optional[float] = None, x_max: Optional[float] = None,
                 y_min: Optional[float] = None, y_max: Optional[float] = None,
                 alpha: Optional[float] = None, labels_map: LabelsTransformer = DUMMY_LABELS_TRANSFORMER,
                 font_size: int = FONT_SIZE):
        self.__x_min = x_min
        self.__x_max = x_max
        self.__y_min = y_min
        self.__y_max = y_max
        self.__alpha = alpha
        self.__labels_map = labels_map
        self.__font_size = font_size

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

    def labels_map(self) -> LabelsTransformer:
        return self.__labels_map

    def font_size(self) -> int:
        return self.__font_size

    def label_transform(self, label: str) -> str:
        return self.__labels_map.apply(label=label)


class PlotSetupWithDefaultLabels(PlotSetup):

    def __init__(self,
                 x_min: Optional[float] = None, x_max: Optional[float] = None,
                 y_min: Optional[float] = None, y_max: Optional[float] = None,
                 alpha: Optional[float] = None,
                 font_size: int = FONT_SIZE):
        PlotSetup.__init__(
            self=self, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, alpha=alpha, font_size=font_size,
            labels_map=DEFAULT_LABELS_TRANSFORMER)

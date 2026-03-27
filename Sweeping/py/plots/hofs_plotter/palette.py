from abc import ABC, abstractmethod

import seaborn

from plots.plot_utils import default_color_list


class Palette(ABC):

    def colors(self, n_colors: int):
        res = self.inner_colors(n_colors=n_colors)
        if len(res) < n_colors:
            res = seaborn.color_palette(n_colors=n_colors)
        return res

    @abstractmethod
    def inner_colors(self, n_colors: int):
        raise NotImplementedError()


class DefaultPalette(Palette):

    def inner_colors(self, n_colors: int):
        return default_color_list(n_colors=n_colors, invert=True)


class SeabornPalette(Palette):
    __palette_name: str
    __invert: bool

    def __init__(self, palette_name: str, invert: bool = False):
        self.__palette_name = palette_name
        self.__invert = invert

    def inner_colors(self, n_colors: int):
        colors = seaborn.color_palette(palette=self.__palette_name, n_colors=n_colors)
        if self.__invert:
            colors = colors[::-1]
        return colors


class BaselinePalette(SeabornPalette):

    def __init__(self):
        SeabornPalette.__init__(self=self, palette_name="mako", invert=True)


DEFAULT_PALETTE = DefaultPalette()
BASELINE_PALETTE = BaselinePalette()
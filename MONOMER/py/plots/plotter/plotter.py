from abc import ABC, abstractmethod
from matplotlib.axes import Axes


class Plotter(ABC):

    @abstractmethod
    def plot(self, ax: Axes, color=None):
        """Plots to ax."""
        raise NotImplementedError()

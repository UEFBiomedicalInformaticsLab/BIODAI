from abc import ABC, abstractmethod

from cross_validation.multi_objective.cross_evaluator.hof_saver import hofs_path_from_optimizer_path
from plots.explored_features_plot import explored_features_plots_for_directory
from plots.feature_diversity_plots import feature_diversity_plots_for_directory
from plots.generations_plot import generations_plots_for_directory
from plots.hof_stability_plots import hof_stability_plots_for_main_directory
from plots.monotonic_front import single_front_plot_every_hof
from plots.performance_by_class import performance_by_class_plots_every_hof
from util.printer.printer import OutPrinter, Printer


class Postprocessing(ABC):

    def run_postprocessing(self, optimizer_dir: str):
        self.inner_run_postprocessing(optimizer_dir=optimizer_dir,
                                      main_hofs_dir=hofs_path_from_optimizer_path(optimizer_path=optimizer_dir))

    @abstractmethod
    def inner_run_postprocessing(self, optimizer_dir: str, main_hofs_dir: str):
        raise NotImplementedError()

    def description(self) -> str:
        return "Postprocessing with missing description."


class GenerationsPlotsPostprocessing(Postprocessing):

    def inner_run_postprocessing(self, optimizer_dir: str, main_hofs_dir: str):
        generations_plots_for_directory(optimizer_dir=optimizer_dir)

    def description(self) -> str:
        return "Creating generation plots if possible"


class ExploredFeaturesPlotsPostprocessing(Postprocessing):

    def inner_run_postprocessing(self, optimizer_dir: str, main_hofs_dir: str):
        explored_features_plots_for_directory(optimizer_dir=optimizer_dir)

    def description(self) -> str:
        return "Creating plots of explored features if possible"


class FeatureDiversityPlotsPostprocessing(Postprocessing):

    def inner_run_postprocessing(self, optimizer_dir: str, main_hofs_dir: str):
        feature_diversity_plots_for_directory(direct=optimizer_dir)

    def description(self) -> str:
        return "Creating plots of stability if possible"


class HofStabilityPlotsPostprocessing(Postprocessing):

    def inner_run_postprocessing(self, optimizer_dir: str, main_hofs_dir: str):
        hof_stability_plots_for_main_directory(main_hofs_dir=main_hofs_dir)

    def description(self) -> str:
        return "Creating plots of hall of fame stability if possible"


class SingleFrontPlotEveryHofPostprocessing(Postprocessing):

    def inner_run_postprocessing(self, optimizer_dir: str, main_hofs_dir: str):
        single_front_plot_every_hof(main_hofs_dir=main_hofs_dir)

    def description(self) -> str:
        return "Creating plots of monotonic test fronts if possible"


class PerformanceByClassPlotsPostprocessing(Postprocessing):

    def inner_run_postprocessing(self, optimizer_dir: str, main_hofs_dir: str):
        performance_by_class_plots_every_hof(main_hofs_dir=main_hofs_dir)

    def description(self) -> str:
        return "Creating plots of performance by class if possible"


POSTPROCESSING_ARCHIVE = (
    GenerationsPlotsPostprocessing(),
    ExploredFeaturesPlotsPostprocessing(),
    FeatureDiversityPlotsPostprocessing(),
    HofStabilityPlotsPostprocessing(),
    SingleFrontPlotEveryHofPostprocessing(),
    PerformanceByClassPlotsPostprocessing()
)


def run_postprocessing_archive(optimizer_dir: str, printer: Printer = OutPrinter()):
    for p in POSTPROCESSING_ARCHIVE:
        printer.title_print(p.description())
        p.run_postprocessing(optimizer_dir=optimizer_dir)

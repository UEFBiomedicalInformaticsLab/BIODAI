from abc import ABC, abstractmethod
from collections.abc import Sequence

from matplotlib import pyplot as plt

from location_manager.path_utils import hofs_path_from_optimizer_path
from plots.explored_features_plot import explored_features_plots_for_directory
from plots.feature_diversity_plots import feature_diversity_plots_for_directory
from plots.generations_plot import generations_plots_for_directory
from plots.hof_stability_plots import hof_stability_plots_for_main_directory
from plots.monotonic_front import single_front_plot_every_hof
from plots.performance_by_class import performance_by_class_plots_every_hof
from plots.runnable.summary_statistics_plotter_from_registries import DEFAULT_REGISTRY_PROPERTIES
from plots.view_contributions import view_contributions_every_hof
from util.printer.printer import OutPrinter, Printer
from validation_registry.fill_missing_properties import fill_missing_properties_every_hof
from validation_registry.registry_property import RegistryProperty


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


class ViewContributionsPlotsPostprocessing(Postprocessing):

    def inner_run_postprocessing(self, optimizer_dir: str, main_hofs_dir: str):
        view_contributions_every_hof(main_hofs_dir=main_hofs_dir)

    def description(self) -> str:
        return "Creating plots of view contributions if possible"


class FillMissingPropertiesPostprocessing(Postprocessing):
    __include_folds: bool

    def __init__(self, include_folds: bool = True,
                 properties: Sequence[RegistryProperty] = DEFAULT_REGISTRY_PROPERTIES):
        self.__include_folds = include_folds
        self.__properties = properties

    def inner_run_postprocessing(self, optimizer_dir: str, main_hofs_dir: str):
        fill_missing_properties_every_hof(
            main_hofs_dir=main_hofs_dir, include_folds=self.__include_folds, properties=self.__properties)

    def description(self) -> str:
        return "Computing remaining properties"


POSTPROCESSING_ARCHIVE_CV = (
    GenerationsPlotsPostprocessing(),
    ExploredFeaturesPlotsPostprocessing(),
    FeatureDiversityPlotsPostprocessing(),
    HofStabilityPlotsPostprocessing(),
    SingleFrontPlotEveryHofPostprocessing(),
    PerformanceByClassPlotsPostprocessing(),
    ViewContributionsPlotsPostprocessing(),
    FillMissingPropertiesPostprocessing()
)


def run_postprocessing_archive(optimizer_dir: str, archive: Sequence[Postprocessing], printer: Printer = OutPrinter()):
    for p in archive:
        printer.title_print(p.description())
        try:
            p.run_postprocessing(optimizer_dir=optimizer_dir)
        except BaseException as e:
            printer.print("Postprocessing " + p.description() + " failed with the following exception.\n" +
                          str(e) + "\n" +
                          "The program will try to continue.")
            plt.close("all")  # In case some fig is not closed properly.


def run_postprocessing_archive_cv_and_final(optimizer_dir: str, printer: Printer = OutPrinter()):
    run_postprocessing_archive(optimizer_dir=optimizer_dir, archive=POSTPROCESSING_ARCHIVE_CV, printer=printer)

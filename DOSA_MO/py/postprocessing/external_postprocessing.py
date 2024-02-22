from plots.performance_by_class import performance_by_class_external_plots_every_hof
from postprocessing.postprocessing import Postprocessing, run_postprocessing_archive, \
    FillMissingPropertiesPostprocessing
from util.printer.printer import Printer, OutPrinter
from validation_registry.registry_property_archive import TEST_HV_PROPERTY, CROSS_HV_PROPERTY, INNER_CV_HV_PROPERTY, \
    MEAN_JACCARD_PROPERTY, PERFORMANCE_GAP_PROPERTY, PERFORMANCE_ERROR_PROPERTY, PARETO_DELTA_PROPERTY

EXTERNAL_REGISTRY_PROPERTIES = (
    TEST_HV_PROPERTY, CROSS_HV_PROPERTY, INNER_CV_HV_PROPERTY, MEAN_JACCARD_PROPERTY,
    PERFORMANCE_GAP_PROPERTY, PERFORMANCE_ERROR_PROPERTY, PARETO_DELTA_PROPERTY)


class PerformanceByClassPlotsExternalPostprocessing(Postprocessing):

    def inner_run_postprocessing(self, optimizer_dir: str, main_hofs_dir: str):
        performance_by_class_external_plots_every_hof(main_hofs_dir=main_hofs_dir)

    def description(self) -> str:
        return "Creating plots of external performance by class if possible"


POSTPROCESSING_ARCHIVE_EXTERNAL = (
    PerformanceByClassPlotsExternalPostprocessing(),
    FillMissingPropertiesPostprocessing(include_folds=False, properties=EXTERNAL_REGISTRY_PROPERTIES)
)


def run_postprocessing_archive_external(optimizer_dir: str, printer: Printer = OutPrinter()):
    run_postprocessing_archive(optimizer_dir=optimizer_dir, archive=POSTPROCESSING_ARCHIVE_EXTERNAL, printer=printer)

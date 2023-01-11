from plots.performance_by_class import performance_by_class_external_plots_every_hof
from postprocessing.postprocessing import Postprocessing


class PerformanceByClassPlotsExternalPostprocessing(Postprocessing):

    def inner_run_postprocessing(self, optimizer_dir: str, main_hofs_dir: str):
        performance_by_class_external_plots_every_hof(main_hofs_dir=main_hofs_dir)

    def description(self) -> str:
        return "Creating plots of external performance by class if possible"

from abc import ABC, abstractmethod
from typing import Sequence

from cross_validation.multi_objective.optimizer.optimizer_names import nick_from_optimizer_and_fi_and_fs, \
    name_from_optimizer_and_fi_and_fs
from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from feature_importance.multi_outcome_feature_importance import MultiOutcomeFeatureImportance
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from input_data.input_data import InputData
from univariate_feature_selection.feature_selector_multi_target import FeatureSelectorMO, DummySelectorMO
from util.feature_space_lifter import FeatureSpaceLifterMV
from util.printer.printer import Printer, UnbufferedOutPrinter, NULL_PRINTER
from util.randoms import random_state_context_blended
from util.sparse_bool_list_by_set import SparseBoolListBySet
from util.str_utils import name_value
from view_adjuster.adjust_input_data import adjust_input_data
from views.views import Views, EMPTY_VIEWS
import csv


DEFAULT_MAX_FEATURES_TO_WRITE = 50000


def active_features_by_view(
        input_data: InputData, feature_selector: FeatureSelectorMO, printer: Printer, n_proc: int = 1,
        very_verbose: bool = False) -> dict[str,Sequence[bool]]:
    """Uses a separate random context and restores the current one at the end.
    Returns a Boolean mask for each view."""
    if input_data.needs_adjustment():
        raise ValueError("Active features must be computed after adjusting the input data.")
    printer.title_print("Computing local active features in a separate random context.")
    res = {}
    if input_data.has_covariates():
        covariates_table = input_data.covariates_table()
    else:
        covariates_table = None
    with random_state_context_blended(additional_seed=23465, printer=printer):
        for v_name in input_data.view_names_seq():
            if input_data.is_covariate_view(view_name=v_name):
                view_n_cols = input_data.view(view_name=v_name).n_col()
                printer.print("All features of covariate view " + str(v_name) +
                              " (" + str(view_n_cols) + ") are set as active.")
                res[v_name]=DummySelectorMO().selection_mask(
                    x=input_data.view(view_name=v_name), outcomes=input_data.outcomes(), n_proc=n_proc,
                    printer=printer)
            else:
                printer.print("Computing active features for view " + str(v_name))
                res[v_name] = feature_selector.selection_mask(
                    x=input_data.view(view_name=v_name), outcomes=input_data.outcomes(), n_proc=n_proc,
                    printer=printer, covariates=covariates_table)
    if very_verbose:
        printer.print_variable("number features available", [len(v) for v in res])
        printer.print_variable("number of local active features", [sum(v) for v in res])
    return res


class ActiveFeaturesObserver(ABC):

    @abstractmethod
    def signal_active_features(
            self, active_views: Views, adjusting_views: Views = EMPTY_VIEWS, printer: Printer = NULL_PRINTER):
        raise NotImplementedError()


class ActiveFeaturesCsvWriter(ActiveFeaturesObserver):
    __csv_file_path: str
    __max_features: int

    def __init__(self, csv_file_path: str, max_features: int = DEFAULT_MAX_FEATURES_TO_WRITE):
        self.__csv_file_path = csv_file_path
        self.__max_features = max_features

    def signal_active_features(
            self, active_views: Views, adjusting_views: Views = EMPTY_VIEWS, printer: Printer = NULL_PRINTER):
        n_features = self.__max_features
        if active_views.n_features() <= n_features:
            printer.print("Saving the active features in " + str(self.__csv_file_path))
            if adjusting_views.n_features() > 0:
                printer.print("There are also adjusting views. They are not saved.")
            with open(self.__csv_file_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(["view", "feature"])

                # Write each view and its features
                for view_name in active_views.keys():
                    view = active_views.view(key=view_name)
                    col_names = view.colnames()
                    for col_name in col_names:
                        writer.writerow([view_name, col_name])
        else:
            printer.print("Not saving the active features because there are too many.")
            printer.print_variable("Active features", n_features)
            printer.print_variable("Max saved features", self.__max_features)


class ActiveFeaturesObservable(ABC):

    @abstractmethod
    def add_active_feature_observer(self, observer: ActiveFeaturesObserver):
        raise NotImplementedError()



class ConcreteActiveFeaturesObservable(ActiveFeaturesObservable):
    __active_feature_observers: list[ActiveFeaturesObserver]

    def __init__(self):
        self.__active_feature_observers = []

    def add_active_feature_observer(self, observer: ActiveFeaturesObserver):
        # noinspection PyUnreachableCode
        if isinstance(observer, ActiveFeaturesObserver):
            self.__active_feature_observers.append(observer)
        else:
            raise TypeError("Observer must be an instance of ActiveFeaturesObserver")

    def signal_active_features(
            self, active_views: Views, adjusting_views: Views = EMPTY_VIEWS, printer: Printer = NULL_PRINTER):
        for o in self.__active_feature_observers:
            o.signal_active_features(active_views=active_views, adjusting_views=adjusting_views, printer=printer)



class PrefilteredMOOptimizerIncludingFI(MultiObjectiveOptimizer, ConcreteActiveFeaturesObservable):
    """A pipeline that computes feature importance and then uses it while optimizing."""
    __feature_selector: FeatureSelectorMO
    __feature_importance: MultiOutcomeFeatureImportance
    __optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance
    __optimizer_type: MOOptimizerType

    def __init__(self,
                 feature_importance: MultiOutcomeFeatureImportance,
                 optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance,
                 feature_selector: FeatureSelectorMO):
        ConcreteActiveFeaturesObservable.__init__(self=self)
        self.__feature_selector = feature_selector
        self.__feature_importance = feature_importance
        self.__optimizer = optimizer
        self.__optimizer_type = ConcreteMOOptimizerType(
            uses_inner_models=True,
            nick=nick_from_optimizer_and_fi_and_fs(optimizer=optimizer, fi=feature_importance, fs=feature_selector),
            name=name_from_optimizer_and_fi_and_fs(optimizer=optimizer, fi=feature_importance, fs=feature_selector))

    def optimize(self, input_data: InputData, printer: Printer, n_proc=1, workers_printer=UnbufferedOutPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver(),
                 very_verbose: bool = False
                 ) -> list[MultiObjectiveOptimizerResult]:
        """Make sure we do not include outcomes (potentially affecting feature selection) that are not in objectives."""

        printer.title_print("Adjusting input data for computing active features and feature importance")
        adjusted_input_data = adjust_input_data(input_data=input_data, printer=printer)
        active_by_view = active_features_by_view(
            input_data=adjusted_input_data, feature_selector=self.__feature_selector, printer=printer, n_proc=n_proc,
            very_verbose=very_verbose)
        # Now we add the views that were not in the adjusted input data so that the lifter will know of their number
        # of features.
        for view_name in input_data.view_names_seq():
            if not view_name in active_by_view:
                active_by_view[view_name] = SparseBoolListBySet(min_size=input_data.view(view_name=view_name).n_col())
        lifter = FeatureSpaceLifterMV.create_from_active_features_and_adjusted_view_def(
            active_by_view=active_by_view, adjusted_view_def=input_data.adjusted_view_def())
        lifted_input_data = adjusted_input_data.uplift(lifter=lifter)
        printer.print("Computing feature importance")
        computed_feature_importance = self.__feature_importance.compute(
            input_data=lifted_input_data, n_proc=n_proc, printer=printer)
        for k,v in computed_feature_importance.items():
            printer.print(k + " feature importance plot:\n" + v.ascii_plot() + "\n")
        selected_input_data = input_data.select_active_features(active_by_view=active_by_view)
        self.signal_active_features(
            active_views=selected_input_data.predictive_views(),
            adjusting_views=selected_input_data.adjuster_views(), printer=printer)
        printer.print("Running optimizer")
        lifted_res = self.__optimizer.optimize_with_feature_importance(
            input_data=selected_input_data, printer=printer,
            feature_importance=computed_feature_importance,
            n_proc=n_proc,
            workers_printer=workers_printer, logbook_saver=logbook_saver, feature_counts_saver=feature_counts_saver)
        return [r.downlift(lifter) for r in lifted_res]

    def optimizer_type(self) -> MOOptimizerType:
        return self.__optimizer_type

    def __str__(self) -> str:
        res = "Prefiltered optimizer with feature importance\n"
        res += name_value("Nick", self.nick()) + "\n"
        res += "Feature selector:\n"
        res += str(self.__feature_selector)
        res += "Inner optimizer:\n"
        res += str(self.__optimizer)
        res += "Multi-view feature importance strategy:\n"
        res += str(self.__feature_importance) + "\n"
        return res

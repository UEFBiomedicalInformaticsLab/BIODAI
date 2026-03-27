from collections.abc import Sequence

import numpy

from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.mv_predictor import MVPredictor
from objective.objective_computer import ObjectiveComputer


def feature_importance_by_permutation_mv(
        objective_computer: ObjectiveComputer, predictor: MVPredictor,
        test_data: ModelReadyInputData, seed: int = 764254) -> dict[str,Sequence[float]]:
    performance_full = objective_computer.compute_from_predictor_and_test_mv(
        predictor=predictor, test_data=test_data).fitness()
    generator = numpy.random.default_rng(seed=seed)
    shuffling_idx = numpy.arange(test_data.n_samples())
    generator.shuffle(shuffling_idx)
    importances = {}
    for k, v in test_data.views_dict().items():
        n_features = v.n_col()
        importances[k] = [0.0] * n_features
        view_importances = importances[k]
        for i in range(n_features):
            temp_col = v.np_col(selected_col=i)[shuffling_idx]
            # Must use list to ignore previous indices that are still in Series.
            v_temp = v.replace_column(new_column_pos=i, new_column=temp_col)
            performance_i = objective_computer.compute_from_predictor_and_test_mv(
                predictor=predictor, test_data=test_data.set_view(view_name=k, table=v_temp)).fitness()
            view_importances[i] = max(performance_full - performance_i, 0.0)
    return importances

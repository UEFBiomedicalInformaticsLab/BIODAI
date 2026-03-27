import numpy
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score

from prediction_stats import stat_keys as keys


class StatCreator:

    # Stats are in the form dictionary key -> stat_value
    # Notice that resulting prediction_stats are arrays with an element for each class.
    def create_stats(self, test_predicted_y, test_true_y, train_predicted_y=None, train_true_y=None,
                     confusion_m=None):
        raise NotImplementedError()


class StatsFromConfusion(StatCreator):

    # You can pass a confusion matrix so the method will not need to compute it by itself.
    def create_stats(self, predicted_y, true_y, train_predicted_y=None, train_true_y=None,
                     confusion_m=None):

        if confusion_m is None:
            confusion_m = confusion_matrix(y_true=true_y, y_pred=predicted_y)

        len_y = len(true_y)
        diag = np.diag(confusion_m)

        # FP, FN, TP and TN are normalized so they sum to 1.
        fp = (confusion_m.sum(axis=0) - diag) / len_y
        fn = (confusion_m.sum(axis=1) - diag) / len_y
        tp = diag / len_y
        tn = 1.0 - (fp + fn + tp)

        with np.errstate(divide='ignore', invalid='ignore'):
            #  Ignoring division by zero and by nan that can legitimately happen
            #  if there are no positive or no negative values in fold

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            specificity = tn / (tn + fp)

            accuracy = sum(tp)

            res = {}
            res[keys.ACCURACY] = accuracy
            res[keys.BALANCED_ACCURACY] = np.mean(recall)
            res[keys.COHEN_K] = cohen_kappa_score(y1=predicted_y, y2=true_y)
            res[keys.ZERO_ONE_LOSS] = 1. - accuracy
            res[keys.FALSE_POSITIVES_PREVALENCE] = fp  # Prevalence of false alarm, type I error or underestimation.
            res[keys.FALSE_NEGATIVES_PREVALENCE] = fn  # Prevalence of miss, type II error or overestimation.
            res[keys.TRUE_POSITIVES_PREVALENCE] = tp
            res[keys.TRUE_NEGATIVES_PREVALENCE] = tn
            res[keys.SENSITIVITY] = recall  # Sensitivity, hit rate, recall, or true positive rate.
            res[keys.SPECIFICITY] = specificity  # Specificity, selectivity or true negative rate.
            res[keys.PRECISION] = precision  # Precision or positive predictive value.
            res[keys.NEGATIVE_PREDICTIVE_VALUE] = tn / (tn + fn)
            res[keys.FALL_OUT] = fp / (fp + tn)  # Fall-out or false positive rate.
            res[keys.FALSE_NEGATIVE_RATE] = fn / (tp + fn)  # False negative rate or miss rate
            res[keys.FALSE_DISCOVERY_RATE] = fp / (tp + fp)
            res[keys.FALSE_OMISSION_RATE] = fn / (fn + tn)
            res[keys.CLASS_ACCURACY] = tp + tn
            res[keys.CLASS_BALANCED_ACCURACY] = (recall + specificity) / 2
            res[keys.F_SCORE] = (2.0 * precision * recall) / (precision + recall)  # F-score or F1 score
            res[keys.G_SCORE] = numpy.sqrt(precision * recall)
            res[keys.PREVALENCE_THRESHOLD] =\
                (numpy.sqrt(recall * (-specificity+1)) + specificity - 1) / (recall + specificity - 1)
            res[keys.THREAT_SCORE] = tp / (tp + fn + fp)  # Threat score or critical success index
            res[keys.CONFUSION_MATRIX] = confusion_m

        return res

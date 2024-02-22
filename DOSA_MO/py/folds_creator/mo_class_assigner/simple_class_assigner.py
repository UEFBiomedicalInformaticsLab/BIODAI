from typing import Sequence

from folds_creator.mo_class_assigner.mo_class_assigner import MOClassAssigner
from input_data.input_data import InputData
from util.printer.printer import Printer, NullPrinter


class SimpleClassAssigner(MOClassAssigner):

    def assign_classes(self, data: InputData, printer: Printer = NullPrinter()) -> Sequence[int]:
        return data.stratify_outcome_data()

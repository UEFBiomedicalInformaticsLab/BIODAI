import os.path

from matplotlib import pyplot as plt

from plots.archives.test_battery_cv import TestBatteryCV
from postprocessing.postprocessing import run_postprocessing_archive_cv


def postprocessing_battery(test_battery: TestBatteryCV):
    for optimizer_dir in test_battery.optimizer_directories():
        if os.path.isdir(optimizer_dir):
            print("Postprocessing for " + optimizer_dir)
            run_postprocessing_archive_cv(optimizer_dir=optimizer_dir)
            plt.close("all")  # In case some fig is not closed properly.

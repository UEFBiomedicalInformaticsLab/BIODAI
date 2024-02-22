from util.cross_hypervolume.cross_hypervolume import cross_hypervolume, hypervolume
from util.hyperbox.hyperbox import ConcreteHyperbox0B

train_hyperboxes = []
test_hyperboxes = []

res = hypervolume(hyperboxes=[])
if res != 0.0:
    raise Exception("Test failed")

empty_box = ConcreteHyperbox0B.create_by_b_vals([])

res = hypervolume(hyperboxes=[empty_box])
if res != 0.0:
    raise Exception("Test failed")

res = cross_hypervolume(train_hyperboxes=[], test_hyperboxes=[])
if res != 0.0:
    raise Exception("Test failed")

res = cross_hypervolume(train_hyperboxes=[empty_box], test_hyperboxes=[empty_box])
if res != 0.0:
    raise Exception("Test failed")

one_box = ConcreteHyperbox0B.create_by_b_vals([1.0])

res = hypervolume(hyperboxes=[one_box])
if res != 1.0:
    raise Exception("Test failed")

res = cross_hypervolume(train_hyperboxes=[one_box], test_hyperboxes=[one_box])
if res != 1.0:
    raise Exception("Test failed")

half_box = ConcreteHyperbox0B.create_by_b_vals([0.5])

res = hypervolume(hyperboxes=[half_box])
if res != 0.5:
    raise Exception("Test failed")

res = cross_hypervolume(train_hyperboxes=[one_box], test_hyperboxes=[half_box])
if res != 0.5:
    raise Exception("Test failed")

res = cross_hypervolume(train_hyperboxes=[half_box], test_hyperboxes=[one_box])
if res != 1.0:
    raise Exception("Test failed")


res = hypervolume(hyperboxes=[half_box, one_box])
if res != 1.0:
    raise Exception("Test failed")

res = cross_hypervolume(train_hyperboxes=[half_box, one_box], test_hyperboxes=[half_box, half_box])
if res != 0.5:
    raise Exception("Test failed")

res = cross_hypervolume(train_hyperboxes=[half_box, one_box], test_hyperboxes=[one_box, half_box])
if res != 0.5:
    raise Exception("Test failed")

res = cross_hypervolume(train_hyperboxes=[half_box, one_box], test_hyperboxes=[half_box, one_box])
if res != 1.0:
    raise Exception("Test failed")

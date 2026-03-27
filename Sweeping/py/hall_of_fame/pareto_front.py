from hall_of_fame.fronts import Fronts


class ParetoFront(Fronts):
    """Just a sequence of fronts with only the first front."""

    def __init__(self):
        Fronts.__init__(self=self, number_of_fronts=1)

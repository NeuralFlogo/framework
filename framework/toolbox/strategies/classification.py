from abc import ABC

from framework.toolbox.strategy import Strategy


class ClassificationStrategy(Strategy, ABC):
    pass

    def type_measurement(self):
        return "accuracy"

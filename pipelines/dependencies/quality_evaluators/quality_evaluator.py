from abc import abstractmethod


class QualityEvaluator:
    @abstractmethod
    def evaluate(self, truth, processed):
        ...
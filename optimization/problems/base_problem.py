from abc import ABC, abstractmethod


class AnnealableProblem(ABC):

    @abstractmethod
    def randomize(self):
        pass

    @abstractmethod
    def propose_move(self):
        pass

    @abstractmethod
    def delta_cost(self, move):
        pass

    @abstractmethod
    def accept_move(self, move):
        pass

    @abstractmethod
    def reject_move(self):
        pass

    @abstractmethod
    def energy(self):
        pass
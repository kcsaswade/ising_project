import numpy as np


class SimulatedAnnealer:

    def __init__(
        self,
        problem,
        T_start=5.0,
        T_end=0.1,
        steps=10000,
    ):

        self.problem = problem

        self.T_start = T_start
        self.T_end = T_end
        self.steps = steps

        self.current_step = 0
        self.temperature = T_start


    def step(self):

        # stop condition
        if self.current_step >= self.steps:
            return False

        # progress [0,1]
        alpha = self.current_step / self.steps

        # geometric cooling
        self.temperature = (
            self.T_start
            * (self.T_end / self.T_start)**alpha
        )

        move = self.problem.propose_move()

        dE = self.problem.delta_cost(move)

        if dE <= 0:
            accept = True
        else:
            accept = (
                np.random.random()
                < np.exp(-dE / self.temperature)
            )

        if accept:
            self.problem.accept_move(move)

        self.current_step += 1

        return True
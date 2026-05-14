import numpy as np


class SimulatedAnnealer:

    def __init__(
        self,
        problem,
        T_start=5.0,
        T_end=0.1,
        steps=10000,
        schedule="EXP",
    ):

        self.problem = problem

        self.T_start = T_start
        self.T_end = T_end
        self.steps = steps
        self.schedule = schedule

        self.current_step = 0
        self.temperature = T_start


    def compute_temperature(self):

        alpha = self.current_step / self.steps

        if self.schedule == "EXP":

            return (
                self.T_start
                * (self.T_end / self.T_start)**alpha
            )

        elif self.schedule == "LINEAR":

            return (
                self.T_start
                - (self.T_start-self.T_end)
                * alpha
            )

        elif self.schedule == "LOG":

            return (
                self.T_start
                / np.log(self.current_step + 2)
            )

        elif self.schedule == "FAST":

            return (
                self.T_start
                / (1 + self.current_step)
            )

        return self.T_start


    def step(self):

        if self.current_step >= self.steps:
            return False

        self.temperature = self.compute_temperature()

        move = self.problem.propose_move()

        dE = self.problem.delta_cost(move)

        if dE <= 0:
            accept = True
        else:
            accept = (
                np.random.random()
                < np.exp(-dE/self.temperature)
            )

        if accept:
            self.problem.accept_move(move)

        self.current_step += 1

        return True
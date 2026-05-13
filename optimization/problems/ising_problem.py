import numpy as np

from optimization.problems.base_problem import AnnealableProblem


class IsingProblem(AnnealableProblem):

    def __init__(self, sim):
        self.sim = sim

    def randomize(self):
        self.sim.reset("random")

    def propose_move(self):

        N = self.sim.size

        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        return (i, j)

    def delta_cost(self, move):

        i, j = move

        spins = self.sim.spins

        s = spins[i, j]

        up = spins[(i - 1) % self.sim.size, j]
        down = spins[(i + 1) % self.sim.size, j]
        left = spins[i, (j - 1) % self.sim.size]
        right = spins[i, (j + 1) % self.sim.size]

        neighbor_sum = up + down + left + right

        return 2.0 * s * (
            self.sim.J * neighbor_sum + self.sim.h
        )

    def accept_move(self, move):

        i, j = move

        self.sim.spins[i, j] *= -1

    def reject_move(self):
        pass

    def energy(self):
        return self.sim.energy_per_spin()
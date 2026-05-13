from ising.metropolis import MetropolisIsing
from optimization.problems.ising_problem import IsingProblem
from optimization.annealer import SimulatedAnnealer


sim = MetropolisIsing(size=64)

problem = IsingProblem(sim)

annealer = SimulatedAnnealer(
    problem,
    T_start=4.0,
    T_end=0.1,
    steps=10000,
)

for step in range(10000):

    annealer.step()

    if step % 1000 == 0:
        print(step, problem.energy())
import numpy as np

from optimization.problems.base_problem import AnnealableProblem

def distance(a, b):

    dx = a[0] - b[0]
    dy = a[1] - b[1]

    return np.sqrt(dx * dx + dy * dy)

class TSPProblem(AnnealableProblem):

    def __init__(self, cities, route):

        self.cities = cities
        self.route = route

        self.best_route = route.copy()
        self.best_cost = self.total_distance(route)

        self.nn_route = self.nearest_neighbor_route()
        self.nn_cost = self.total_distance(self.nn_route)

    def total_distance(self, route=None):

        if route is None:
            route = self.route

        total = 0.0

        for i in range(len(route)):

            a = self.cities[route[i]]
            b = self.cities[route[(i + 1) % len(route)]]

            dx = a[0] - b[0]
            dy = a[1] - b[1]

            total += (dx * dx + dy * dy) ** 0.5

        return total

    def route_length(self):

        total = 0.0

        n = len(self.route)

        for i in range(n):

            a = self.cities[self.route[i]]
            b = self.cities[self.route[(i + 1) % n]]

            total += distance(a, b)

        return total
    
    def propose_move(self):

        n = len(self.route)

        i = np.random.randint(0, n)
        j = np.random.randint(0, n)

        if i > j:
            i, j = j, i

        if abs(i - j) < 2:
            return None

        return (i, j)
    
    def delta_cost(self, move):

        if move is None:
            return 0.0

        i, j = move

        n = len(self.route)

        a = self.cities[self.route[(i - 1) % n]]
        b = self.cities[self.route[i]]

        c = self.cities[self.route[j]]
        d = self.cities[self.route[(j + 1) % n]]

        old_cost = (
            distance(a, b)
            + distance(c, d)
        )

        new_cost = (
            distance(a, c)
            + distance(b, d)
        )

        return new_cost - old_cost
    
    def accept_move(self, move):

        if move is None:
            return

        i, j = move

        self.route[i:j+1] = reversed(
            self.route[i:j+1]
        )

    def reject_move(self):
        pass

    def energy(self):
        return self.total_distance()
    
    def randomize(self):
        np.random.shuffle(self.route)   

    def nearest_neighbor_route(self):

        N = len(self.cities)

        if N == 0:
            return []

        unvisited = set(range(N))

        current = 0

        route = [current]
        unvisited.remove(current)

        while unvisited:

            current_city = self.cities[current]

            next_city = min(
                unvisited,
                key=lambda j: (
                    (current_city[0] - self.cities[j][0]) ** 2
                    + (current_city[1] - self.cities[j][1]) ** 2
                )
            )

            route.append(next_city)

            unvisited.remove(next_city)

            current = next_city

        return route
    
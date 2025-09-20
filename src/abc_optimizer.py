import numpy as np

class ABCOptimizer:
    def __init__(self, func, bounds, colony_size=20, max_iter=100):
        self.func = func
        self.bounds = np.array(bounds)
        self.colony_size = colony_size
        self.max_iter = max_iter

    def optimize(self):
        D = len(self.bounds)
        solutions = np.random.uniform(self.bounds[:,0], self.bounds[:,1],
                                      (self.colony_size, D))
        fitness = np.array([self.func(sol) for sol in solutions])

        best_sol, best_fit = solutions[fitness.argmin()], fitness.min()

        for _ in range(self.max_iter):
            for i in range(self.colony_size):
                k = np.random.randint(self.colony_size)
                while k == i:
                    k = np.random.randint(self.colony_size)
                phi = np.random.uniform(-1, 1, D)
                cand = solutions[i] + phi * (solutions[i] - solutions[k])
                cand = np.clip(cand, self.bounds[:,0], self.bounds[:,1])
                f_cand = self.func(cand)

                if f_cand < fitness[i]:
                    solutions[i], fitness[i] = cand, f_cand

            if fitness.min() < best_fit:
                best_sol, best_fit = solutions[fitness.argmin()], fitness.min()

        return best_sol, best_fit

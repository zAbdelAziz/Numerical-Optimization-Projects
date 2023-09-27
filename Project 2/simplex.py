from problems import *
from datetime import datetime

class Simplex:
    def __init__(self, p, feasible=True):
        """
        Initialize the Simplex algorithm with problem data.

        Args:
        - p: SimplexProblem object containing the problem data (A, b, c)
        - feasible: Boolean indicating whether a feasible starting point is known
                    Default is True.
        """
        self.A, self.b, self.c = p.A, p.b, p.c
        self.x0 = p.x_f if feasible else p.x_nf
        self.m, self.n = self.A.shape

    def run(self):
        """
        Initialize the Simplex algorithm with problem data.

        Args:
        - p: SimplexProblem object containing the problem data (A, b, c)
        - feasible: Boolean indicating whether a feasible starting point is known
                    Default is True.
        """
        iteration_1, b_bar, b_bars1 = self.phase_1()
        # if np.any(b_bar > self.n):
        #     print('Infeasible Error')
        iteration, solution, b_bars2 = self.phase_2(b_bar)
        solution = solution[:10] if solution.shape[0] > 10 else solution
        return (iteration_1, iteration), (b_bars1, b_bars2), solution

    def phase_1(self):
        """
        Perform the Phase 1 of the Simplex algorithm to find a feasible starting point.

        Returns:
        - b_bar: Basis indices after Phase 1
        """
        if self.x0 is not None:
            z = self.x0
        else:
            z = np.ones(self.m)
        Z = np.diag([1 if i >= 0 else -1 for i in self.b])
        # print(z)
        A = np.hstack((self.A, Z))
        b = self.b
        c = np.hstack((np.zeros(self.n), z))

        _init = Simplex(SimplexProblem(A, b, c))
        b_bar = np.arange(0, self.m) + self.n
        i, b_bar, b_bars = _init.optimize(b_bar)
        return i, b_bar, b_bars

    def phase_2(self, b_bar):
        """
        Perform the Phase 2 of the Simplex algorithm using a feasible starting point.

        Args:
        - b_bar: Basis indices from Phase 1

        Returns:
        - iteration: Number of iterations required for convergence
        - solution: Optimal solution vector
        """
        if self.x0 is not None:
            z = self.x0
        else:
            z = np.ones(self.m)
        A = np.hstack((np.vstack((self.A, np.zeros((self.m, self.n)))),
                        np.vstack((np.eye(self.m), np.eye(self.m)))))
        b = np.hstack((self.b, np.zeros(self.m)))
        c = np.hstack((self.c, np.zeros_like(z)))

        simplex = Simplex(SimplexProblem(A, b, c))
        iteration, b_bar, b_bars = simplex.optimize(b_bar)
        solution = simplex.get_vertex(b_bar)
        return iteration, solution[:self.n], b_bars

    def optimize(self, b_bar):
        """
        Run the optimization loop until convergence.

        Args:
        - b_bar: Initial basis indices

        Returns:
        - i: Number of iterations required for convergence
        - b_bar: Basis indices after convergence
        """
        done = False
        i = 0
        b_bars = []
        while not done:
            b_bar, done = self.step(b_bar)
            b_bars.append(b_bar.tolist())
            i +=1
        return i, b_bar, b_bars

    def step(self, b_bar):
        """
        Run the optimization loop until convergence.

        Args:
        - b_bar: Initial basis indices

        Returns:
        - k: Number of iterations required for convergence
        - b_bar: Basis indices after convergence
        """
        # Get the number of variables in the problem
        n = self.A.shape[1]
        # Sort the basis indices in ascending order
        b_inds = np.sort(b_bar)
        # Get the non-basis indices
        n_inds = np.sort(np.setdiff1d(np.arange(0, n), b_bar))
        # Select the columns of A corresponding to the basis indices
        AB = self.A[:, b_inds]
        # Select the columns of A corresponding to the non-basis indices
        AV = self.A[:, n_inds]

        # Select the objective coefficients corresponding to the basis indices
        cB = self.c[b_inds]
        # Solve the system of equations for the dual variables
        l = np.linalg.lstsq(AB.T, cB, rcond=None)[0]
        # Select the objective coefficients corresponding to the non-basis indices
        cV = self.c[n_inds]
        # Calculate the reduced costs of the non-basis variables
        mV = cV - np.matmul(AV.T, l)
        # Initialize variables for entering index and step size
        q, p, xq, delta = -1, 0, np.inf, np.inf
        # Find the entering index with the most negative reduced cost and determine the step size
        for i in range(0, mV.shape[0]):
            if mV[i] < 0:
                # Compute the edge transition
                pi, xi = self.trans_edge(b_bar, i)
                if np.any(mV[i] * xi < delta):
                    q, p, xq, delta = i, pi, xi, mV[i] * xi
         # If no entering index found, the algorithm has converged
        if q == -1:
            return b_bar, True
        # If step size is infinite, the problem is unbounded
        if np.isinf(xq):
            print('Unbounded Error')
        # Find the index of the leaving variable in the basis
        j = np.argwhere(b_inds[p] == b_bar)[0]
        # Update the basis indices by replacing the leaving variable with the entering variable
        b_bar[j] = n_inds[q]
        return b_bar, False

    def trans_edge(self, B, q):
        """
        Find the entering index and the step size for the edge transition.

        Args:
        - B: Current basis indices
        - q: Index of the leaving variable

        Returns:
        - p: Index of the entering variable
        - xq: Step size for the edge transition
        """
        n = self.A.shape[1]
        b_inds = np.sort(B)
        # Get the non-basis indices
        n_inds = np.sort(np.setdiff1d(np.arange(0, n), B))
        AB = self.A[:, b_inds]
        # Compute the direction ratios
        d = np.linalg.lstsq(AB, self.A[:, n_inds[q]], rcond=None)[0]
        # Compute the values of the basic variables
        xB = np.linalg.lstsq(AB, self.b, rcond=None)[0]
        # Initialize variables for entering index and step size
        p, xq = 0, np.inf
        # Find the entering index and determine the step size for the edge transition
        for i in range(0, d.shape[0]):
            if d[i] > 0:
                v = xB[i] / d[i]
                if v < xq:
                    p, xq = i, v
        return p, xq

    def get_vertex(self, B):
        """
        Get the vertex of the polyhedron corresponding to the basis indices.

        Args:
        - B: Basis indices

        Returns:
        - x: Vertex of the polyhedron
        """
        b_inds = np.sort(B)
        AB = self.A[:, b_inds]
        xB = np.linalg.lstsq(AB, self.b, rcond=None)[0]
        x = np.zeros(self.c.shape[0])
        x[b_inds] = xB
        return x


print('#'*11)
print('# Simplex #')
print('#'*11)
for i, p in enumerate(simplex_problems):
    print(f'Problem {i+1}:')
    print('-'*10)
    print('A:', p.A.tolist())
    print('b:', p.b.tolist())
    print('c:', p.c.tolist())
    print('-'*20)

    print(f'\t- (Feasible) Starting Point: {p.x_f.tolist()}')
    print('\t ', '-'*26)
    st = datetime.now()
    simp = Simplex(p, feasible=True)
    iterations, basis, result = simp.run()
    print('\t\tOptimum x: ', result.tolist())
    print('\t\tPhase 1 iterations: ', iterations[0])
    print('\t\t\tPhase 1 basis: ', basis[0])
    print('\t\tPhase 2 iterations: ', iterations[1])
    print('\t\t\tPhase 2 basis: ', basis[1])
    print('\t\tTotal Time: ', datetime.now() - st)
    print('\n')

    print(f'\t- (In-Feasible) Starting Point: {p.x_nf.tolist()}')
    print('\t ', '-'*29)
    st = datetime.now()
    simp = Simplex(p, feasible=False)
    iterations, basis, result = simp.run()
    print('\t\tOptimum x: ', result.tolist())
    print('\t\tPhase 1 iterations: ', iterations[0])
    print('\t\t\tPhase 1 basis: ', basis[0])
    print('\t\tPhase 2 iterations: ', iterations[1])
    print('\t\t\tPhase 2 basis: ', basis[1])
    print('\t\tTotal Time: ', datetime.now() - st)

    print('='*20)
    print('\n\n')

"""Module that provides utilities to obtain data."""
import numpy as np


class Vortex2dDataset:
    """Generator of synthetic observations from vortex 2D velocity field."""
    def __init__(self, n):
        self.n = n

        self.lb_x = -1.0
        self.ub_x = +1.0
        self.lb_y = -1.0
        self.ub_y = +1.0

    def load_data(self):
        X_star, u_star = self._generate_data()

        # Choose randomly indexes without replacement (all indices are unique).
        idx = np.random.choice(X_star.shape[0], self.n, replace=False)

        X_observed = X_star[idx]
        u_observed = u_star[idx]

        return X_observed, u_observed


    def _generate_data(self):
        """Generate feature and label matrices."""
        x = np.linspace(self.lb_x, self.ub_x, num=201)
        y = np.linspace(self.lb_y, self.ub_y, num=201)

        assert len(x)*len(y) > 5*self.n

        X, Y = np.meshgrid(x, y)

        X_col = X.flatten()[:, None]
        Y_col = Y.flatten()[:, None]
        X_star = np.hstack((X_col, Y_col))

        U = +np.sin(X) * np.cos(Y)
        V = -np.cos(X) * np.sin(Y)

        U_col = U.flatten()[:, None]
        V_col = V.flatten()[:, None]
        y_star = np.hstack((U_col, V_col))

        return X_star, y_star

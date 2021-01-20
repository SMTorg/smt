"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <steven.berguin@gtri.gatech.edu>

This package is distributed under New BSD license.
"""

import numpy as np

EPS = np.finfo(float).eps  # small number to avoid division by zero


# ------------------------------------ S U P P O R T   F U N C T I O N S -----------------------------------------------


def finite_difference(parameters, fun=None, dx=1e-6):
    """
    Compute gradient using central difference

    :param parameters: point at which to evaluate gradient
    :param fun: function handle to use for finite difference
    :param dx: finite difference step
    :return: dy: the derivative of fun with respect to x
    """
    grads = dict()
    for key in parameters.keys():
        x = np.copy(parameters[key])
        n, p = x.shape
        dy = np.zeros((n, p))
        for i in range(0, n):
            for j in range(0, p):
                # Forward step
                parameters[key][i, j] = x[i, j] + dx
                y_fwd = fun(parameters)
                parameters[key] = np.copy(x)

                # Backward step
                parameters[key][i, j] = x[i, j] - dx
                y_bwd = fun(parameters)
                parameters[key] = np.copy(x)

                # Central difference
                dy[i, j] = np.divide(y_fwd - y_bwd, 2 * dx)

        grads[key] = dy

    return grads


# ------------------------------------ O P T I M I Z E R   C L A S S ---------------------------------------------------


class Optimizer(object):
    @property
    def optimum(self):
        return self._optimum_design

    @property
    def current_design(self):
        return self._current_design

    def search_direction(self):
        return self._search_direction

    @property
    def cost_history(self):
        return self._cost_history

    @property
    def design_history(self):
        return self._design_history

    @property
    def cost(self):
        return self._current_cost

    def __init__(self, **kwargs):

        self.learning_rate = 0.1
        self.beta_1 = 0.9
        self.beta_2 = 0.99
        self.user_cost_function = None
        self.user_grad_function = None
        self._current_design = None
        self._previous_design = None
        self._search_direction = None
        self._cost_history = []
        self._design_history = []
        self._optimum_design = None
        self._current_cost = None
        self._current_iteration = 0
        self.initial_guess = None

        for name, value in kwargs.items():
            setattr(self, name, value)

    @classmethod
    def initialize(
        cls,
        initial_guess,
        cost_function,
        grad_function=None,
        learning_rate=0.05,
        beta1=0.9,
        beta2=0.99,
    ):
        attributes = {
            "user_cost_function": cost_function,
            "user_grad_function": grad_function,
            "learning_rate": learning_rate,
            "beta_1": beta1,
            "beta_2": beta2,
            "initial_guess": initial_guess,
            "_current_design": initial_guess.copy(),
        }
        return cls(**attributes)

    def _cost_function(self, x):
        return self.user_cost_function(x)

    def _grad_function(self, x):
        if self.user_grad_function is not None:
            return self.user_grad_function(x)
        else:
            return finite_difference(x, fun=self.user_cost_function)

    def _update_current_design(self, learning_rate=0.05):
        """
        Implement one step of gradient descent
        """
        pass

    def grad_check(self, parameters, tol=1e-6):  # pragma: no cover
        """
        Check analytical gradient against to finite difference

        :param parameters: point at which to evaluate gradient
        :param tol: acceptable error between finite difference and analytical
        """
        grads = self._grad_function(parameters)
        grads_FD = finite_difference(parameters, fun=self.user_cost_function)
        for key in parameters.keys():
            numerator = np.linalg.norm(grads[key] - grads_FD[key])
            denominator = np.linalg.norm(grads[key]) + np.linalg.norm(grads_FD[key])
            difference = numerator / (denominator + EPS)
            if difference <= tol or numerator <= tol:
                print("The gradient of {} is correct".format(key))
            else:
                print("The gradient of {} is wrong".format(key))
            print("Finite dif: grad[{}] = {}".format(key, str(grads_FD[key].squeeze())))
            print("Analytical: grad[{}] = {}".format(key, str(grads[key].squeeze())))

    def backtracking_line_search(self, tau=0.5):
        """
        Perform backtracking line search

        :param x0: initial inputs understood by the function 'update' and 'evaluate'
        :param alpha: learning rate (maximum step size allowed)
        :param update: function that updates X given alpha, i.e. X = update(alpha)
        :param evaluate: function that updates cost given X, i.e. cost = evaluate(X)
        :param tau: hyper-parameter between 0 and 1 used to reduce alpha during backtracking line search
        :return: x: update inputs understood by the function 'update' and 'evaluate'
        """
        tau = max(0.0, min(1.0, tau))  # make sure 0 < tau < 1
        converged = False
        self._previous_design = self._current_design.copy()
        while not converged:
            self._update_current_design(learning_rate=self.learning_rate * tau)
            if self._cost_function(self._current_design) < self._cost_function(
                self._previous_design
            ):
                converged = True
            elif self.learning_rate * tau < 1e-6:
                converged = True
            else:
                tau *= tau

    def optimize(self, max_iter=100, is_print=True):
        """
        Optimization logic (main driver)

        :param max_iter: maximum number of iterations
        :param is_print: True = print cost at every iteration, False = silent
        :return: optimum
        """
        # Stopping criteria (Vanderplaats, ch. 3, p. 121)
        converged = False
        N1 = 0
        N1_max = 100  # num consecutive passes over which abs convergence criterion must be satisfied before stopping
        N2 = 0
        N2_max = 100  # num of consecutive passes over which rel convergence criterion must be satisfied before stopping
        epsilon_absolute = 1e-7  # absolute error criterion
        epsilon_relative = 1e-7  # relative error criterion

        self._current_cost = self._cost_function(self._current_design).squeeze()
        self._cost_history.append(self._current_cost)
        self._design_history.append(self._current_design.copy())

        # Iterative update
        for i in range(0, max_iter):
            self._current_iteration = i
            self._search_direction = self._grad_function(self._current_design)
            self.backtracking_line_search()

            self._current_cost = self._cost_function(self._current_design).squeeze()
            self._cost_history.append(self._current_cost)
            self._design_history.append(self._current_design.copy())

            if is_print:
                print(
                    "iteration = {:d}, cost = {:6.3f}".format(
                        i, float(self._current_cost)
                    )
                )

            # Absolute convergence criterion
            if i > 1:
                dF1 = abs(self._cost_history[-1] - self._cost_history[-2])
                if dF1 < epsilon_absolute * self._cost_history[0]:
                    N1 += 1
                else:
                    N1 = 0
                if N1 > N1_max:
                    converged = True
                    if is_print:
                        print("Absolute stopping criterion satisfied")

                # Relative convergence criterion
                dF2 = abs(self._cost_history[-1] - self._cost_history[-2]) / max(
                    abs(self._cost_history[-1]), 1e-6
                )
                if dF2 < epsilon_relative:
                    N2 += 1
                else:
                    N2 = 0
                if N2 > N2_max:
                    converged = True
                    if is_print:
                        print("Relative stopping criterion satisfied")

                # Maximum iteration convergence criterion
                if i == max_iter:
                    if is_print:
                        print("Maximum optimizer iterations reached")

                if converged:
                    break

        self._optimum_design = self._current_design.copy()

        return self.optimum


class GD(Optimizer):
    def _update_current_design(self, learning_rate=0.05):
        """Gradient descent update"""
        for key in self._previous_design.keys():
            self._current_design[key] = (
                self._previous_design[key] - learning_rate * self._search_direction[key]
            )


class Adam(Optimizer):
    def __init__(self, **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.v = None
        self.s = None

    def _update_current_design(self, learning_rate=0.05, beta_1=0.9, beta_2=0.99):
        """Adam update"""
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        t = self._current_iteration + 1

        if self.v is None:
            self.v = {
                key: np.zeros(value.shape)
                for key, value in self._current_design.items()
            }
        if self.s is None:
            self.s = {
                key: np.zeros(value.shape)
                for key, value in self._current_design.items()
            }

        for key in self._current_design.keys():
            self.v[key] = (
                self.beta_1 * self.v[key] + (1.0 - beta_1) * self._search_direction[key]
            )
            self.s[key] = self.beta_2 * self.s[key] + (1.0 - beta_2) * np.square(
                self._search_direction[key]
            )
            v_corrected = self.v[key] / (1.0 - self.beta_1 ** t)
            s_corrected = self.s[key] / (1.0 - self.beta_2 ** t)
            self._current_design[key] = self._previous_design[
                key
            ] - learning_rate * v_corrected / (np.sqrt(s_corrected) + EPS)


def run_example(use_adam=True):  # pragma: no cover
    """visual example using 2D rosenbrock function"""
    import matplotlib.pyplot as plt

    # Test function
    def rosenbrock(parameters):
        x1 = parameters["x1"]
        x2 = parameters["x2"]

        y = (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2
        y = y.reshape(1, 1)

        dydx = dict()
        dydx["x1"] = -2 * (1 - x1) - 400 * x1 * (x2 - x1 ** 2)
        dydx["x2"] = 200 * (x2 - x1 ** 2)

        return y, dydx

    # Initial guess
    initial_guess = dict()
    initial_guess["x1"] = np.array([1.25]).reshape((1, 1))
    initial_guess["x2"] = np.array([-1.75]).reshape((1, 1))

    # Function handles to be pass
    f = lambda x: rosenbrock(parameters=x)[0]
    dfdx = lambda x: rosenbrock(parameters=x)[1]

    # Learning rate
    alpha = 0.5

    # Optimize
    if use_adam:
        optimizer = Adam.initialize(
            initial_guess=initial_guess,
            cost_function=f,
            grad_function=dfdx,
            learning_rate=alpha,
        )
    else:
        optimizer = GD.initialize(
            initial_guess=initial_guess,
            cost_function=f,
            grad_function=dfdx,
            learning_rate=alpha,
        )
    optimizer.grad_check(initial_guess)
    optimizer.optimize(max_iter=1000)

    # For plotting initial and final answer
    x0 = np.array([initial_guess["x1"].squeeze(), initial_guess["x2"].squeeze()])

    xf = np.array(
        [optimizer.optimum["x1"].squeeze(), optimizer.optimum["x2"].squeeze()]
    )

    # For plotting contours
    lb = -2.0
    ub = 2.0
    m = 100
    x1 = np.linspace(lb, ub, m)
    x2 = np.linspace(lb, ub, m)
    X1, X2 = np.meshgrid(x1, x2)
    Y = np.zeros(X1.shape)
    for i in range(0, m):
        for j in range(0, m):
            Y[i, j] = f({"x1": np.array([X1[i, j]]), "x2": np.array([X2[i, j]])})

    # Plot
    x1_his = np.array([design["x1"] for design in optimizer.design_history]).squeeze()
    x2_his = np.array([design["x2"] for design in optimizer.design_history]).squeeze()
    plt.plot(x1_his, x2_his)
    plt.plot(x0[0], x0[1], "+", ms=15)
    plt.plot(xf[0], xf[1], "o")
    plt.plot(np.array([1.0]), np.array([1.0]), "x")
    plt.legend(["history", "initial guess", "predicted optimum", "true optimum"])
    plt.contour(X1, X2, Y, 50, cmap="RdGy")
    if use_adam:
        plt.title("adam")
    else:
        plt.title("gradient descent")
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    run_example()

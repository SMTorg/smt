"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        
This package is distributed under New BSD license.
"""

import numpy as np
import scipy.sparse

VALID_LINE_SEARCHES = ("backtracking", "bracketed", "quadratic", "cubic", "null")


def get_line_search_class(line_search):
    if line_search == "backtracking":
        return BacktrackingLineSearch
    elif line_search == "bracketed":
        return BracketedLineSearch
    elif line_search == "quadratic":
        return QuadraticLineSearch
    elif line_search == "cubic":
        return CubicLineSearch
    elif line_search == "null":
        return NullLineSearch


class LineSearch(object):
    """
    Base line search class.
    """

    def __init__(self, x, dx, func, grad, u1=1.0e-4, u2=0.9):
        """
        Initialize all attributes for the given problem.

        Arguments
        ---------
        x : ndarray[:]
            Vector representing the current location in the n-D space.
        dx : ndarray[:]
            Search direction.
        func : function
            scalar function of x.
        grad : function
            vector function that yields the gradient of func.
        u1 : float
            Parameter in the sufficient decrease criterion to ensure non-zero decrease.
        u2 : float
            Parameter in the curvature criterion to ensure gradient norm decreases.
        """
        self.x = x
        self.dx = dx
        self.func = func
        self.grad = grad

        self.u1 = u1
        self.u2 = u2

        self.phi_0 = self._phi(0.0)
        self.dphi_0 = self._dphi(0.0)

    def _phi(self, a):
        """
        Function in terms of alpha (a).

        phi(a) = func(x + a dx)
        """
        return self.func(self.x + a * self.dx)

    def _dphi(self, a):
        """
        Derivative of phi w.r.t. alpha (a).
        """
        return np.dot(self.grad(self.x + a * self.dx), self.dx)

    def _func_decreased(self, a):
        """
        Check sufficient decrease criterion.
        """
        return self._phi(a) <= self.phi_0 + self.u1 * a * self.dphi_0

    def _grad_decreased(self, a):
        """
        Check curvature criterion.
        """
        return np.abs(self._dphi(a)) <= np.abs(self.u2 * self.dphi_0)


class NullLineSearch(object):
    """
    Base line search class.
    """

    def __init__(self, x, dx, func, grad, u1=1.0e-4, u2=0.9):
        """
        Initialize all attributes for the given problem.

        Arguments
        ---------
        x : ndarray[:]
            Vector representing the current location in the n-D space.
        dx : ndarray[:]
            Search direction.
        func : function
            scalar function of x.
        grad : function
            vector function that yields the gradient of func.
        u1 : float
            Parameter in the sufficient decrease criterion to ensure non-zero decrease.
        u2 : float
            Parameter in the curvature criterion to ensure gradient norm decreases.
        """
        self.x = x
        self.dx = dx

    def __call__(self, initial_a=1):
        return self.x + initial_a * self.dx


class BacktrackingLineSearch(LineSearch):
    """
    Simple backtracking line search enforcing only sufficient decrease.
    """

    def __call__(self, initial_a=1.0, rho=0.5):
        a = initial_a
        while not self._func_decreased(a):
            a *= rho
        return self.x + a * self.dx


class BracketedLineSearch(LineSearch):
    """
    Base class for line search algorithms enforcing the Strong Wolfe conditions.
    """

    def __call__(self, initial_a=1):
        a1 = 0
        a2 = initial_a
        p1 = self._phi(a1)
        p2 = self._phi(a2)
        dp1 = self._dphi(a1)
        dp2 = self._dphi(a2)

        for ind in range(20):
            if not self._func_decreased(a2) or p2 > p1:
                # We've successfully bracketed if
                # 1. The function value is greater than at a=0
                # 2. The function value has increased from the previous iteration
                return self._zoom(a1, p1, dp1, a2, p2, dp2)

            if self._grad_decreased(a2):
                # At this point, the func decrease condition is satisfied,
                # so if the grad decrease also is satisfied, we're done.
                return self.x + a2 * self.dx
            elif dp2 >= 0:
                # If only the func decrease is satisfied, but the phi' is positive
                # we've successfully bracketed.
                return self._zoom(a2, p2, dp2, a1, p1, dp1)
            else:
                # Otherwise, we're lower than initial f and previous f,
                # and the slope is still negative and steeper than initial.
                # We can get more aggressive and increase the step.
                a1 = a2
                p1 = p2
                dp1 = dp2
                a2 = a2 * 1.5
                p2 = self._phi(a2)
                dp2 = self._dphi(a2)

    def _zoom(self, a1, p1, dp1, a2, p2, dp2):
        """
        Find a solution in the interval, [a1, a2], assuming that phi(a1) < phi(a2).
        """
        while True:
            a, p, dp = self._compute_minimum(a1, p1, dp1, a2, p2, dp2)

            if not self._func_decreased(a) or p > p1:
                # If still lower than initial f or still higher than low
                # then make this the new high.
                a2 = a
                p2 = p
                dp2 = dp
            else:
                if self._grad_decreased(a):
                    # Both conditions satisfied, so we're done.
                    return self.x + a * self.dx
                elif dp * (a2 - a1) >= 0:
                    # We have a new low and the slope has the right sign.
                    a2 = a1
                    p2 = p1
                    dp2 = dp1
                a1 = a
                p1 = p
                dp1 = dp

    def _compute_minimum(self, a1, p1, dp1, a2, p2, dp2):
        """
        Estimate the minimum as the midpoint.
        """
        a = 0.5 * a1 + 0.5 * a2
        p = self._phi(a)
        dp = self._dphi(a)
        return a, p, dp


class QuadraticLineSearch(BracketedLineSearch):
    """
    Use quadratic interpolation in the zoom method.
    """

    def _compute_minimum(self, a1, p1, dp1, a2, p2, dp2):
        quadratic_mtx = np.zeros((3, 3))
        quadratic_mtx[0, :] = [1.0, a1, a1 ** 2]
        quadratic_mtx[1, :] = [1.0, a2, a2 ** 2]
        quadratic_mtx[2, :] = [0.0, 1.0, 2 * a1]
        c0, c1, c2 = np.linalg.solve(quadratic_mtx, [p1, p2, dp1])

        d0 = c1
        d1 = 2 * c2

        a = -d0 / d1
        p = self._phi(a)
        dp = self._dphi(a)
        return a, p, dp


class CubicLineSearch(BracketedLineSearch):
    """
    Use cubic interpolation in the zoom method.
    """

    def _compute_minimum(self, a1, p1, dp1, a2, p2, dp2):
        cubic_mtx = np.zeros((4, 4))
        cubic_mtx[0, :] = [1.0, a1, a1 ** 2, a1 ** 3]
        cubic_mtx[1, :] = [1.0, a2, a2 ** 2, a2 ** 3]
        cubic_mtx[2, :] = [0.0, 1.0, 2 * a1, 3 * a1 ** 2]
        cubic_mtx[3, :] = [0.0, 1.0, 2 * a2, 3 * a2 ** 2]
        c0, c1, c2, c3 = np.linalg.solve(cubic_mtx, [p1, p2, dp1, dp2])

        d0 = c1
        d1 = 2 * c2
        d2 = 3 * c3
        r1, r2 = np.roots([d2, d1, d0])

        a = None
        p = max(p1, p2)
        if (a1 <= r1 <= a2 or a2 <= r1 <= a1) and np.isreal(r1):
            px = self._phi(r1)
            if px < p:
                a = r1
                p = px
                dp = self._dphi(r1)
        if (a1 <= r2 <= a2 or a2 <= r2 <= a1) and np.isreal(r2):
            px = self._phi(r2)
            if px < p:
                a = r2
                p = px
                dp = self._dphi(r2)

        return a, p, dp

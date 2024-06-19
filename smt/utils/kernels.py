import numpy as np
from abc import ABCMeta  # , abstractmethod


class Kernel(metaclass=ABCMeta):
    def __init__(self, theta):
        self.theta = np.array(theta)


class PowExp(Kernel):
    def __call__(self, d, grad_ind=None, hess_ind=None, derivative_params=None):
        r = np.zeros((d.shape[0], 1))
        n_components = d.shape[1]

        # Construct/split the correlation matrix
        i, nb_limit = 0, int(1e4)
        while i * nb_limit <= d.shape[0]:
            r[i * nb_limit : (i + 1) * nb_limit, 0] = np.exp(
                -np.sum(
                    self.theta.reshape(1, n_components)
                    * d[i * nb_limit : (i + 1) * nb_limit, :],
                    axis=1,
                )
            )
            i += 1

        i = 0
        if grad_ind is not None:
            while i * nb_limit <= d.shape[0]:
                r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                    -d[i * nb_limit : (i + 1) * nb_limit, grad_ind]
                    * r[i * nb_limit : (i + 1) * nb_limit, 0]
                )
                i += 1

        i = 0
        if hess_ind is not None:
            while i * nb_limit <= d.shape[0]:
                r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                    -d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                    * r[i * nb_limit : (i + 1) * nb_limit, 0]
                )
                i += 1

        if derivative_params is not None:
            dd = derivative_params["dd"]
            r = r.T
            dr = -np.einsum("i,ij->ij", r[0], dd)
            return r.T, dr

        return r


class Matern52(Kernel):
    def __call__(self, d, grad_ind=None, hess_ind=None, derivative_params=None):
        r = np.zeros((d.shape[0], 1))
        n_components = d.shape[1]

        # Construct/split the correlation matrix
        i, nb_limit = 0, int(1e4)
        theta = self.theta
        while i * nb_limit <= d.shape[0]:
            ll = (
                theta.reshape(1, n_components) * d[i * nb_limit : (i + 1) * nb_limit, :]
            )
            r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                1.0 + np.sqrt(5.0) * ll + 5.0 / 3.0 * ll**2.0
            ).prod(axis=1) * np.exp(-np.sqrt(5.0) * (ll.sum(axis=1)))
            i += 1
        i = 0

        M52 = r.copy()

        if grad_ind is not None:
            theta_r = theta.reshape(1, n_components)
            while i * nb_limit <= d.shape[0]:
                fact_1 = (
                    np.sqrt(5) * d[i * nb_limit : (i + 1) * nb_limit, grad_ind]
                    + 10.0
                    / 3.0
                    * theta_r[0, grad_ind]
                    * d[i * nb_limit : (i + 1) * nb_limit, grad_ind] ** 2.0
                )
                fact_2 = (
                    1.0
                    + np.sqrt(5)
                    * theta_r[0, grad_ind]
                    * d[i * nb_limit : (i + 1) * nb_limit, grad_ind]
                    + 5.0
                    / 3.0
                    * (theta_r[0, grad_ind] ** 2)
                    * (d[i * nb_limit : (i + 1) * nb_limit, grad_ind] ** 2)
                )
                fact_3 = np.sqrt(5) * d[i * nb_limit : (i + 1) * nb_limit, grad_ind]

                r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                    fact_1 / fact_2 - fact_3
                ) * r[i * nb_limit : (i + 1) * nb_limit, 0]
                i += 1
        i = 0

        if hess_ind is not None:
            while i * nb_limit <= d.shape[0]:
                fact_1 = (
                    np.sqrt(5) * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                    + 10.0
                    / 3.0
                    * theta_r[0, hess_ind]
                    * d[i * nb_limit : (i + 1) * nb_limit, hess_ind] ** 2.0
                )
                fact_2 = (
                    1.0
                    + np.sqrt(5)
                    * theta_r[0, hess_ind]
                    * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                    + 5.0
                    / 3.0
                    * (theta_r[0, hess_ind] ** 2)
                    * (d[i * nb_limit : (i + 1) * nb_limit, hess_ind] ** 2)
                )
                fact_3 = np.sqrt(5) * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]

                r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                    fact_1 / fact_2 - fact_3
                ) * r[i * nb_limit : (i + 1) * nb_limit, 0]

                if hess_ind == grad_ind:
                    fact_4 = (
                        10.0
                        / 3.0
                        * d[i * nb_limit : (i + 1) * nb_limit, hess_ind] ** 2.0
                        * fact_2
                    )
                    r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                        (fact_4 - fact_1**2) / (fact_2) ** 2
                    ) * M52[i * nb_limit : (i + 1) * nb_limit, 0] + r[
                        i * nb_limit : (i + 1) * nb_limit, 0
                    ]

                i += 1
        if derivative_params is not None:
            dx = derivative_params["dx"]

            abs_ = abs(dx)
            sqr = np.square(dx)
            abs_0 = np.dot(abs_, theta)

            dr = np.zeros(dx.shape)

            A = np.zeros((dx.shape[0], 1))
            for i in range(len(abs_0)):
                A[i][0] = np.exp(-np.sqrt(5) * abs_0[i])

            der = np.ones(dx.shape)
            for i in range(len(der)):
                for j in range(n_components):
                    if dx[i][j] < 0:
                        der[i][j] = -1

            dB = np.zeros((dx.shape[0], n_components))
            for j in range(dx.shape[0]):
                for k in range(n_components):
                    coef = 1
                    for ll in range(n_components):
                        if ll != k:
                            coef = coef * (
                                1
                                + np.sqrt(5) * abs_[j][ll] * theta[ll]
                                + (5.0 / 3) * sqr[j][ll] * theta[ll] ** 2
                            )
                    dB[j][k] = (
                        np.sqrt(5) * theta[k] * der[j][k]
                        + 2 * (5.0 / 3) * der[j][k] * abs_[j][k] * theta[k] ** 2
                    ) * coef

            for j in range(dx.shape[0]):
                for k in range(n_components):
                    dr[j][k] = (
                        -np.sqrt(5) * theta[k] * der[j][k] * r[j] + A[j][0] * dB[j][k]
                    ).item()

            return r, dr

        return r


class Matern32(Kernel):
    def __call__(self, d, grad_ind=None, hess_ind=None, derivative_params=None):
        r = np.zeros((d.shape[0], 1))
        n_components = d.shape[1]
        theta = self.theta
        # Construct/split the correlation matrix
        i, nb_limit = 0, int(1e4)

        theta_r = theta.reshape(1, n_components)

        while i * nb_limit <= d.shape[0]:
            ll = theta_r * d[i * nb_limit : (i + 1) * nb_limit, :]
            r[i * nb_limit : (i + 1) * nb_limit, 0] = (1.0 + np.sqrt(3.0) * ll).prod(
                axis=1
            ) * np.exp(-np.sqrt(3.0) * (ll.sum(axis=1)))
            i += 1
        i = 0

        M32 = r.copy()

        if grad_ind is not None:
            while i * nb_limit <= d.shape[0]:
                fact_1 = (
                    1.0
                    / (
                        1.0
                        + np.sqrt(3.0)
                        * theta_r[0, grad_ind]
                        * d[i * nb_limit : (i + 1) * nb_limit, grad_ind]
                    )
                    - 1.0
                )
                r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                    fact_1
                    * r[i * nb_limit : (i + 1) * nb_limit, 0]
                    * np.sqrt(3.0)
                    * d[i * nb_limit : (i + 1) * nb_limit, grad_ind]
                )

                i += 1
            i = 0

        if hess_ind is not None:
            while i * nb_limit <= d.shape[0]:
                fact_2 = (
                    1.0
                    / (
                        1.0
                        + np.sqrt(3.0)
                        * theta_r[0, hess_ind]
                        * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                    )
                    - 1.0
                )
                r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                    r[i * nb_limit : (i + 1) * nb_limit, 0]
                    * fact_2
                    * np.sqrt(3.0)
                    * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                )
                if grad_ind == hess_ind:
                    fact_3 = (
                        3.0 * d[i * nb_limit : (i + 1) * nb_limit, hess_ind] ** 2.0
                    ) / (
                        1.0
                        + np.sqrt(3.0)
                        * theta_r[0, hess_ind]
                        * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                    ) ** 2.0
                    r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                        r[i * nb_limit : (i + 1) * nb_limit, 0]
                        - fact_3 * M32[i * nb_limit : (i + 1) * nb_limit, 0]
                    )
                i += 1
        if derivative_params is not None:
            dx = derivative_params["dx"]

            abs_ = abs(dx)
            abs_0 = np.dot(abs_, theta)
            dr = np.zeros(dx.shape)

            A = np.zeros((dx.shape[0], 1))
            for i in range(len(abs_0)):
                A[i][0] = np.exp(-np.sqrt(3) * abs_0[i])

            der = np.ones(dx.shape)
            for i in range(len(der)):
                for j in range(n_components):
                    if dx[i][j] < 0:
                        der[i][j] = -1

            dB = np.zeros((dx.shape[0], n_components))
            for j in range(dx.shape[0]):
                for k in range(n_components):
                    coef = 1
                    for ll in range(n_components):
                        if ll != k:
                            coef = coef * (1 + np.sqrt(3) * abs_[j][ll] * theta[ll])
                    dB[j][k] = np.sqrt(3) * theta[k] * der[j][k] * coef

            for j in range(dx.shape[0]):
                for k in range(n_components):
                    dr[j][k] = (
                        -np.sqrt(3) * theta[k] * der[j][k] * r[j] + A[j][0] * dB[j][k]
                    ).item()
            return r, dr

        return r


class Operator(Kernel):
    def __init__(self, corr1, corr2):
        self.theta = np.array([corr1.theta, corr2.theta])
        self.corr1 = corr1
        self.corr2 = corr2


class Sum(Operator):
    def __call__(self, d, grad_ind=None, hess_ind=None, derivative_params=None):
        return self.corr1(d, grad_ind, hess_ind, derivative_params) + self.corr2(
            d, grad_ind, hess_ind, derivative_params
        )


class Product(Operator):
    def __call__(self, d, grad_ind=None, hess_ind=None, derivative_params=None):
        return self.corr1(d, grad_ind, hess_ind, derivative_params) * self.corr2(
            d, grad_ind, hess_ind, derivative_params
        )


if __name__ == "__main__":
    d = np.abs(np.array([[-1, -1.0], [1.0, 1.0], [-1, 1.0], [1.0, -1.0]]))
    theta1 = np.array([1.0, 2.0])
    theta2 = np.array([0.9, 0.5])
    k1 = PowExp(theta1)
    k2 = PowExp(theta2)
    k3 = Sum(k1, k2)
    k4 = Product(k1, k2)
    k5 = Product(k4, k3)
    print(k1(d))
    print(k2(d))
    print(k3(d))
    print(k4(d))
    print(k5(d))

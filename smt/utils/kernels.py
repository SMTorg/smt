import numpy as np
from abc import ABCMeta  # , abstractmethod


class Kernel(metaclass=ABCMeta):
    def __init__(self, theta):
        self.theta = np.array(theta)


class PowExp(Kernel):
    def __call__(self, d, grad_ind=None, hess_ind=None, derivative_params=None):
        theta = self.theta
        r = np.zeros((d.shape[0], 1))
        n_components = d.shape[1]

        # Construct/split the correlation matrix
        i, nb_limit = 0, int(1e4)
        while i * nb_limit <= d.shape[0]:
            r[i * nb_limit : (i + 1) * nb_limit, 0] = np.exp(
                -np.sum(
                    theta.reshape(1, n_components)
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


class SquarSinExp(Kernel):
    def __call__(self, d, grad_ind=None, hess_ind=None, derivative_params=None):
        theta = self.theta
        r = np.zeros((d.shape[0], 1))
        # Construct/split the correlation matrix
        i, nb_limit = 0, int(1e4)
        while i * nb_limit <= d.shape[0]:
            theta_array = theta.reshape(1, len(theta))
            r[i * nb_limit : (i + 1) * nb_limit, 0] = np.exp(
                -np.sum(
                    np.atleast_2d(theta_array[0][0 : int(len(theta) / 2)])
                    * np.sin(
                        np.atleast_2d(
                            theta_array[0][int(len(theta) / 2) : int(len(theta))]
                        )
                        * d[i * nb_limit : (i + 1) * nb_limit, :]
                    )
                    ** 2,
                    axis=1,
                )
            )
            i += 1
        kernel = r.copy()

        i = 0
        if grad_ind is not None:
            cut = int(len(theta) / 2)
            if (
                hess_ind is not None and grad_ind >= cut and hess_ind < cut
            ):  # trick to use the symetry of the hessian when the hessian is asked
                grad_ind, hess_ind = hess_ind, grad_ind

            if grad_ind < cut:
                grad_ind2 = cut + grad_ind
                while i * nb_limit <= d.shape[0]:
                    r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                        -(
                            np.sin(
                                theta_array[0][grad_ind2]
                                * d[i * nb_limit : (i + 1) * nb_limit, grad_ind]
                            )
                            ** 2
                        )
                        * r[i * nb_limit : (i + 1) * nb_limit, 0]
                    )
                    i += 1
            else:
                hess_ind2 = grad_ind - cut
                while i * nb_limit <= d.shape[0]:
                    r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                        -theta_array[0][hess_ind2]
                        * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                        * np.sin(
                            2
                            * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                            * theta_array[0][grad_ind]
                        )
                        * r[i * nb_limit : (i + 1) * nb_limit, 0]
                    )
                    i += 1

        i = 0
        if hess_ind is not None:
            cut = int(len(theta) / 2)
            if grad_ind < cut and hess_ind < cut:
                hess_ind2 = cut + hess_ind
                while i * nb_limit <= d.shape[0]:
                    r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                        -(
                            np.sin(
                                theta_array[0][hess_ind2]
                                * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                            )
                            ** 2
                        )
                        * r[i * nb_limit : (i + 1) * nb_limit, 0]
                    )
                    i += 1
            elif grad_ind >= cut and hess_ind >= cut:
                hess_ind2 = hess_ind - cut
                if grad_ind == hess_ind:
                    while i * nb_limit <= d.shape[0]:
                        r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                            -2
                            * theta_array[0][hess_ind2]
                            * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2] ** 2
                            * np.cos(
                                2
                                * theta_array[0][grad_ind]
                                * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                            )
                            * kernel[i * nb_limit : (i + 1) * nb_limit, 0]
                            - theta_array[0][hess_ind2]
                            * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                            * np.sin(
                                2
                                * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                                * theta_array[0][hess_ind]
                            )
                            * r[i * nb_limit : (i + 1) * nb_limit, 0]
                        )
                        i += 1
                else:
                    while i * nb_limit <= d.shape[0]:
                        r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                            -theta_array[0][hess_ind2]
                            * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                            * np.sin(
                                2
                                * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                                * theta_array[0][hess_ind]
                            )
                            * r[i * nb_limit : (i + 1) * nb_limit, 0]
                        )
                        i += 1
            elif grad_ind < cut and hess_ind >= cut:
                hess_ind2 = hess_ind - cut
                while i * nb_limit <= d.shape[0]:
                    r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                        -theta_array[0][hess_ind2]
                        * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                        * np.sin(
                            2
                            * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                            * theta_array[0][hess_ind]
                        )
                        * r[i * nb_limit : (i + 1) * nb_limit, 0]
                    )
                    if hess_ind2 == grad_ind:
                        r[i * nb_limit : (i + 1) * nb_limit, 0] += (
                            -d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                            * np.sin(
                                2
                                * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                                * theta_array[0][hess_ind]
                            )
                            * kernel[i * nb_limit : (i + 1) * nb_limit, 0]
                        )

                    i += 1
            i = 0

        if derivative_params is not None:
            cut = int(len(theta) / 2)
            dx = derivative_params["dx"]
            dr = np.empty(dx.shape)
            for j in range(dx.shape[0]):
                for k in range(dx.shape[1]):
                    dr[j, k] = (
                        -theta_array[0][k]
                        * theta_array[0][k + cut]
                        * np.sin(2 * theta_array[0][k + cut] * dx[j][k])
                        * kernel[j][0]
                    )
            return r, dr
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


class ActExp(Kernel):
    def __call__(
        self, d, grad_ind=None, hess_ind=None, d_x=None, derivative_params=None
    ):
        r = np.zeros((d.shape[0], 1))
        n_components = d.shape[1]

        if len(self.theta) % n_components != 0:
            raise Exception("Length of theta must be a multiple of n_components")

        n_small_components = len(self.theta) // n_components

        A = np.reshape(self.theta, (n_small_components, n_components)).T

        d_A = d.dot(A)

        # Necessary when working in embeddings space
        if d_x is not None:
            d = d_x
            n_components = d.shape[1]

        r[:, 0] = np.exp(-(1 / 2) * np.sum(d_A**2.0, axis=1))

        if grad_ind is not None:
            d_grad_ind = grad_ind % n_components
            d_A_grad_ind = grad_ind // n_components

            if hess_ind is None:
                r[:, 0] = -d[:, d_grad_ind] * d_A[:, d_A_grad_ind] * r[:, 0]

            elif hess_ind is not None:
                d_hess_ind = hess_ind % n_components
                d_A_hess_ind = hess_ind // n_components
                fact = -d_A[:, d_A_grad_ind] * d_A[:, d_A_hess_ind]
                if d_A_hess_ind == d_A_grad_ind:
                    fact = 1 + fact
                r[:, 0] = -d[:, d_grad_ind] * d[:, d_hess_ind] * fact * r[:, 0]

        if derivative_params is not None:
            raise ValueError("Jacobians are not available for this correlation kernel")

        return r

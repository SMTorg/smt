Proper Orthogonal Decomposition + Interpolation (PODI)
======================================================

PODI is an application used to predict vectorial outputs.
It combines Proper Orthogonal Decomposition (POD) and Kriging based surrogate models to perform the estimations.

Context
-------

We seek for an approximation of a vector :math:`u(\mathbf{x}) \in \mathbb{R}^p`, with :math:`p>>1` and :math:`\mathbf{x}= \in \mathcal{X} \subset \mathbb{R}^d` an input vector. 
PODI being a supervised learning approach we assume that a Design of Experiments of size :math:`N` is available, i.e. :math:`u(\mathbf{x_k})` for  :math:`k \in [\![1,N]\!]`.    
In the model order reduction, a vector :math:`u(\mathbf{x_k})` is called a **snapshot**.
The PODI application aims at building an approximation :math:`\hat{u}(\mathbf{x})` of :math:`u(\mathbf{x})` for any :math:`\mathbf{x}\in\mathcal{X}`. 


To construct this approximation, the :math:`N` snapshots are first gathered in a database called the **snapshot matrix**:

.. math ::
	S=
	\begin{bmatrix}
		u( \mathbf{x}_1)_1 & \dots & u( \mathbf{x}_N)_1 \\
		\vdots & \ddots & \vdots \\
		u( \mathbf{x}_1)_p & \dots & u( \mathbf{x}_N)_p \\
	\end{bmatrix}
	\in \mathbb{R}^{p \times N}

Each column of the matrix corresponds to a snapshot output :math:`u(\mathbf{x_k})`.

Proper Orthogonal Decomposition (POD)
-------------------------------------
Global POD
----------
The Proper Orthogonal Decomposition of :math:`u` reads, 

.. math ::
	\begin{equation}\label{e:pod}
	u({\mathbf x})\approx \hat{u}({\mathbf{x}}) =u_0 + \sum_{i=1}^{M} \alpha_i(\mathbf x)\phi_i
	\end{equation}

* :math:`u` is decomposed as a sum of :math:`M` modes and :math:`u_0` corresponds to the mean value of :math:`u`.

* each mode :math:`i` is defined by a scalar coefficient :math:`\alpha_i`, called generalized coordinate, and  a vector :math:`\phi_{i}` of dimension :math:`p`.

* the :math:`\phi_i` vectors are orthogonal and form the **POD basis** :math:`\Phi`. Note that they are independent of :math:`x`, hence the name "global" POD basis. 



In practice, the mean value :math:`u_0` of :math:`u` is not available and will be estimated by the mean value of the :math:`N` snapshots. It can be shown that the basis :math:`\Phi` that leads to the best approximation in the mean square error sense is the singular vector of the matrix :math:`S-u_0`.
The generalized coordinates :math:`\alpha_i(\mathbf{x}), i = 1,\cdots,M` will be interpolated by GP.  

Local POD
---------

Local POD starts by assuming that the input vector :math:`\mathbf{x}` can be splitted into two subsets of variables i.e. :math:`\mathbf{x}=\left\lbrace \mathbf{x_1},\mathbf{x_2}\right\rbrace`. The spliting is not automatic and left to the user.
Then, the local POD approximation reads

.. math ::
	\begin{equation}\label{e:local_pod}
	u({\mathbf x})\approx u_0 + \sum_{i=1}^{M} \alpha_i(\mathbf{x_1},\mathbf{x_2})\phi_i(\mathbf{x_1})
	\end{equation}

where :math:`\phi_i` is the local POD basis at input point :math:`\mathbf{x_1}`. In practice this POD basis must be approximated by interpolation on the Grassmann manifold of a database of local POD bases. Hence this approach further assumes that a DoE of local POD bases is provided. More information on this interpolation can be found in ref [1]. The generalized coordinates :math:`\alpha_i(\mathbf{x_1},\mathbf{x_2})` are interpolated by GPs as for the global POD case. 

[1] Porrello, C., Dubreuil, S., and Farhat, C. Bayesian framework with projection-based model order reduction for efficient global optimization. In AIAA AVIATION FORUM AND ASCEND 2024 (2024)


Singular Values Decomposition (SVD)
-------------------------------------
To perform the POD, the SVD of the snapshot matrix :math:`S` is used:

.. math ::
	\begin{equation}\label{e:svd}
	S=U\Sigma{V}^{T}
	\end{equation}

The :math:`(p \times p)` :math:`U` and :math:`(N \times N)` :math:`{V}^{T}` matrices are orthogonal and contain the **singular vectors**.
These vectors are the directions of maximum variance in the data and are ranked by decreasing order of importance.
Each vector corresponds to a mode of :math:`u`. The total number of available modes is limited by the number of snapshots:

.. math ::
	\begin{equation}\label{e:M<=N}
	M \le N
	\end{equation}

The importance of each mode is represented by the diagonal values of the :math:`(p \times N)` :math:`\Sigma` matrix. They are known as the *singular values* :math:`\sigma_i` and are positive numbers ranked by decreasing value.
It is then needed to filter the modes to keep those that represent most of the data structure.
To do this, we use the **explained variance**. It represents the data variance that we keep when filtering the modes.

If :math:`m` modes are kept, their explained variance :math:`EV_m` is:

.. math ::
	\begin{equation}\label{e:ev_m}
	EV_m=\frac{\sum_{i=1}^{m} \sigma_i^2}{\sum_{i=1}^{N} \sigma_i^2}
	\end{equation}

The number of kept modes is defined by a tolerance :math:`\eta \in ]0,1]` that represents the minimum variance we desire to explain during the SVD:

.. math ::
	\begin{equation}\label{e:M_def}
	M = \min\{m \in [\![1,N]\!]: EV_m \ge \eta\}
	\end{equation}

Then, the first :math:`M` singular vectors of the :math:`U` matrix correspond to the :math:`\phi_i` vectors in the POD.
The :math:`\alpha_i` coefficients of the :math:`A` matrix can be deduced:

.. math ::
	\begin{equation}\label{e:A}
	A={\Phi}^{T}(S-U_0)
	\end{equation}

Use of Surrogate models
---------------------------------

To compute :math:`u` at a new value :math:`\mathbf{x}_*`, the values of :math:`\alpha_i(\mathbf{x}_*)` at each mode :math:`i` are needed.

To estimate them, **Kriging based surrogate models** are used:


.. math ::
	\mathbf{x}=(\mathbf{x}_1,\dots,\mathbf{x}_k,\dots,\mathbf{x}_N)
	\longrightarrow
	\begin{cases}
		\alpha_1(\mathbf{x}) \longrightarrow \text{model 1} \\
		\vdots \\
		\alpha_i(\mathbf{x}) \longrightarrow \text{model i} \\
		\vdots \\
		\alpha_M(\mathbf{x}) \longrightarrow \text{model M} \\
	\end{cases}

For each kept mode :math:`i`, we use a surrogate model that is trained with the inputs :math:`\mathbf{x}_k` and outputs :math:`\alpha_i(\mathbf{x}_k)`.

These models are able to compute an estimation denoted :math:`\hat\alpha_i(\mathbf{x}_*)`. It is normally distributed:

.. math ::
	\hat\alpha_i(\mathbf{x}_*) \hookrightarrow \mathcal{N}(\mu_i(\mathbf{x}_*),\sigma_i^{2}(\mathbf{x}_*))

The mean, variance and derivative of :math:`u(\mathbf{x}_*)` can be deduced:

.. math ::
	\begin{cases}
		\mathbb{E}[u(\mathbf{x}_*)]=u_0+\sum_{i=1}^{M} \mu_i(\mathbf{x}_*)\phi_i \\
		\mathbb{V}[u(\mathbf{x}_*)]=\sum_{i=1}^{M} \sigma_i^{2}(\mathbf{x}_*)\phi_i^{2} \\
		u'(\mathbf{x}_*)=\sum_{i=1}^{M} \hat\alpha_i'(\mathbf{x}_*)\phi_i
	\end{cases}

NB: The variance equation takes in consideration that:

- the models are pairwise independent, so are the coefficients :math:`\hat\alpha_i(\mathbf{x}_*)`.

Usage
-----
Example 1: global POD case for 1D function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Example 2: local POD case for 2D function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


PODI class API
--------------

.. autoclass:: smt.applications.podi.PODI

	.. automethod:: smt.applications.podi.PODI.interp_subspaces

  	.. automethod:: smt.applications.podi.PODI.compute_pod

	.. automethod:: smt.application.podi.PODI.compute_pod_errors

    .. automethod:: smt.applications.podi.PODI.get_singular_vectors

	.. automethod:: smt.applications.podi.PODI.get_basis

	.. automethod:: smt.applications.podi.PODI.get_singular_values

	.. automethod:: mst.applications.podi.PODI.get_ev_list

	.. automethod:: smt.applications.podi.PODI.get_ev_ratio

	.. automethod:: smt.applications.podi.PODI.get_n_modes

	.. automethod:: smt.applications.podi.PODI.set_interp_options

	.. automethod:: smt.applications.podi.PODI.set_training_values

	.. automethod:: smt.applications.podi.PODI.train

	.. automethod:: smt.applications.podi.PODI.get_interp_coeff

	.. automethod:: smt.applications.podi.PODI.predict_values

	.. automethod:: smt.applications.podi.PODI.predict_variances

	.. automethod:: smt.applications.podi.PODI.predict_derivatives

	.. automethod:: smt.applications.podi.PODI.predict_variance_derivativess
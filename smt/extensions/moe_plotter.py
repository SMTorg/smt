import six
import numpy as np

from matplotlib import colors
import matplotlib.pyplot as plt        

class MOEPlotter(object):

    def __init__(self, moe, xlimits):
        self.moe = moe
        self.xlimits = xlimits

    ################################################################################
    def plot_cluster(self, x_, y_):
        """
        Plot distribsian cluster
        Parameters:
        -----------
        xlimits: array_like
            array[ndim, 2]
        x_: array_like
        Input training samples
        y_: array_like
        Output training samples
        Optionnals:
        -----------
        heaviside: float
        Heaviside factor. Default to False
        """

        GMM=self.moe.cluster
        xlim = self.xlimits

        if GMM.n_components > 1:

            colors_ = list(six.iteritems(colors.cnames))

            dim = xlim.shape[0]
            weight = GMM.weights_
            mean = GMM.means_
            cov = GMM.covars_
            prob_ = self.moe._proba_cluster(x_)
            sort = np.apply_along_axis(np.argmax, 1, prob_)

            if dim == 1:
                fig = plt.figure()
                x = np.linspace(xlim[0, 0], xlim[0, 1])
                prob = self.moe._proba_cluster(x)
                for i in range(len(weight)):
                    plt.plot(x, prob[:, i], ls='--')
                plt.xlabel('Input Values')
                plt.ylabel('Membership probabilities')
                plt.title('Cluster Map')

                fig = plt.figure()
                for i in range(len(sort)):
                    color_ind = int(((len(colors_) - 1) / sort.max()) * sort[i])
                    color = colors_[color_ind][0]
                    plt.plot(x_[i], y_[i], c=color, marker='o')
                plt.xlabel('Input Values')
                plt.ylabel('Output Values')
                plt.title('Samples with clusters')

            if dim == 2:
                x0 = np.linspace(xlim[0, 0], xlim[0, 1], 20)
                x1 = np.linspace(xlim[1, 0], xlim[1, 1], 20)
                xv, yv = np.meshgrid(x0, x1)
                x = np.array(zip(xv.reshape((-1,)), yv.reshape((-1,))))
                prob = self.moe._proba_cluster(x)

                fig = plt.figure()
                ax1 = fig.add_subplot(111, projection='3d')
                for i in range(len(weight)):
                    color = colors_[int(((len(colors_) - 1) / len(weight)) * i)][0]
                    ax1.plot_trisurf(x[:, 0], x[:, 1], prob[:, i], alpha=0.4, linewidth=0,
                                     color=color)
                plt.title('Cluster Map 3D')

                fig1 = plt.figure()
                for i in range(len(weight)):
                    color = colors_[int(((len(colors_) - 1) / len(weight)) * i)][0]
                    plt.tricontour(x[:, 0], x[:, 1], prob[:, i], 1, colors=color, linewidths=3)
                plt.title('Cluster Map 2D')

                fig = plt.figure()
                ax2 = fig.add_subplot(111, projection='3d')
                for i in range(len(sort)):
                    color = colors_[int(((len(colors_) - 1) / sort.max()) * sort[i])][0]
                    ax2.scatter(x_[i][0], x_[i][1], y_[i], c=color)
                plt.title('Samples with clusters')
            plt.show()


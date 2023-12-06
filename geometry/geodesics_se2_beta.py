import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.special_euclidean import SpecialEuclidean

SE2_GROUP = SpecialEuclidean(n=2, point_type='matrix')
N_STEPS = 15


def main():
    """Plot geodesics on SE(2) with different structures."""
    theta = gs.pi / 4
    initial_tangent_vec = gs.array([
        [0., - theta, 1],
        [theta, 0., 1],
        [0., 0., 0.]])
    t = gs.linspace(0, 1., N_STEPS + 1)
    tangent_vec = gs.einsum('t,ij->tij', t, initial_tangent_vec)

    fig = plt.figure(figsize=(10, 10))
    maxs_x = []
    mins_y = []
    maxs = []
    for i, beta in enumerate([1., 2., 3., 5.]):
        ax = plt.subplot(2, 2, i + 1)
        metric_mat = from_vector_to_diagonal_matrix(gs.array([1, beta, 1.]))
        metric = InvariantMetric(SE2_GROUP, metric_mat, point_type='matrix')
        points = metric.exp(tangent_vec, base_point=SE2_GROUP.identity)
        ax = visualization.plot(
            points, ax=ax, space='SE2_GROUP', color='black',
            label=r'$\beta={}$'.format(beta))
        mins_y.append(min(points[:, 1, 2]))
        maxs.append(max(points[:, 1, 2]))
        maxs_x.append(max(points[:, 0, 2]))
        plt.legend(loc='best')

    for ax in fig.axes:
        x_lim_inf, _ = plt.xlim()
        x_lims = [x_lim_inf, 1.1 * max(maxs_x)]
        y_lims = [min(mins_y) - .1, max(maxs) + .1]
        ax.set_ylim(y_lims)
        ax.set_xlim(x_lims)
        ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    main()

"""Requires backend with automatic differentiation."""

import matplotlib.pyplot as plt
from numpy import logspace

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.special_euclidean import SpecialEuclidean

N = 3
SPACE = SpecialEuclidean(N)

gs.random.seed(0)


def main():
    point = SPACE.random_point()

    # numbers of rungs between 10 and 100
    n_numbers = 10
    n_range = [int(k) for k in logspace(1, 2, n_numbers)]
    tan_a = SPACE.lie_algebra.matrix_representation(gs.random.rand(SPACE.dim))
    tan_b = SPACE.lie_algebra.matrix_representation(gs.random.rand(SPACE.dim))
    left_metric = SPACE.left_canonical_metric
    tan_a -= left_metric.inner_product(
        tan_a, tan_b) / left_metric.squared_norm(tan_b) * tan_b
    tan_b = SPACE.compose(point, tan_b)
    tan_a = SPACE.compose(point, tan_a)

    # define anisotropic metric by setting beta!=1
    fig = plt.figure()
    for beta, col in zip([1.5, 2.5],  ['b', 'g', 'r']):
        metric_mat = gs.eye(SPACE.dim)
        metric_mat[N, N] = beta
        metric = InvariantMetric(SPACE, metric_mat_at_identity=metric_mat)

        # exact parallel transport
        if beta == 1:
            exact_transport = left_metric.parallel_transport(tan_a, tan_b, point)
        else:
            exact_transport, end_point_ = metric.parallel_transport(
                tan_a, tan_b, point, n_steps=1100, step='rk4', return_endpoint=True)

        abs_error_pole = []
        abs_error_rk2 = []
        abs_error_rk4 = []
        alpha = 1
        for n_rungs in n_range:
            ladder = metric.ladder_parallel_transport(
                tan_a, tan_b, point, n_rungs=n_rungs, step='rk4', n_steps=1, tol=1e-14,
                alpha=alpha)
            transported = ladder['transported_tangent_vec']
            end_point = ladder['end_point']
            abs_error_pole.append(metric.norm(transported - exact_transport, end_point))

            ladder, end_point = metric.parallel_transport(
                tan_a, tan_b, point, n_steps=n_rungs, step='rk2', return_endpoint=True)
            transported = ladder
            abs_error_rk2.append(metric.norm(transported - exact_transport, end_point))

            ladder, end_point = metric.parallel_transport(
                tan_a, tan_b, point, n_steps=n_rungs, step='rk4', return_endpoint=True)
            transported = ladder
            abs_error_rk4.append(metric.norm(transported - exact_transport, end_point))

        plt.loglog(
            n_range, abs_error_rk2,
            marker='x', linewidth=1, c=col, linestyle='dashed',
            fillstyle='none', label=r'Integration, rk2, $\beta={}$'.format(beta))
        plt.loglog(
            n_range, abs_error_rk4,
            marker='+', linewidth=1, c=col, linestyle='dashed',
            fillstyle='none', label=r'Integration, rk4, $\beta={}$'.format(beta))
        plt.loglog(
            n_range, abs_error_pole,
            marker='o', linewidth=1, c=col, linestyle='dashed',
            fillstyle='none', label=r'Pole ladder, $\beta={}$'.format(beta))
        plt.title('Absolute error with respect to the number of steps', fontsize=14)
        plt.xlabel(r'$n$')
        plt.legend(loc='best')


if __name__ == '__main__':
    main()

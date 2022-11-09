import matplotlib.pyplot as plt
from numpy import logspace

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.special_euclidean import SpecialEuclidean

n = 3
space = SpecialEuclidean(n)

n_samples = 1

gs.random.seed(0)
point = space.random_point(n_samples)

# numbers of rungs between 10 and 100
n_numbers = 10
n_range = [int(k) for k in logspace(1, 3., n_numbers)]
tan_a = space.lie_algebra.matrix_representation(gs.random.rand(n_samples, space.dim))
tan_b = space.lie_algebra.matrix_representation(gs.random.rand(n_samples, space.dim))
left_metric = space.left_canonical_metric
tan_a -= left_metric.inner_product(
    tan_a, tan_b) / left_metric.squared_norm(tan_b) * tan_b
tan_b = space.compose(point, tan_b)
tan_a = space.compose(point, tan_a)

# define anisotropic metric by setting beta!=1
fig = plt.figure()
for beta, col in zip([1., 3., 5.],  ['b', 'g', 'r']):
    metric_mat = gs.eye(space.dim)
    metric_mat[n, n] = beta
    metric = InvariantMetric(space, metric_mat_at_identity=metric_mat)

    # exact parallel transport
    if beta == 1:
        exact_transport = left_metric.parallel_transport(tan_a, tan_b, point)
        end_point_ = left_metric.exp(tan_b, point)
        exact_transport_tan_b = left_metric.parallel_transport(tan_b, tan_b, point)
    else:
        exact_transport, end_point_ = metric.parallel_transport(
            tan_a, tan_b, point, n_steps=1100, step='rk4', return_endpoint=True)
        exact_transport_tan_b = metric.log(
            metric.exp(2 * tan_b, point, n_steps=30), end_point_, tol=1e-14, n_steps=30)

    abs_error_pole = []
    abs_error_rk2 = []
    abs_error_rk4 = []
    alpha = 1
    for n_rungs in n_range:
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
    plt.title('absolute error with respect to the number of steps')
    plt.xlabel(r'1 / $n$')
    plt.legend(loc='best')

plt.savefig(f'../figures/se3-pt-integration.eps', bbox_inches='tight', pad_inches=0)

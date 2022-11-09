import matplotlib.pyplot as plt
from numpy import logspace, split
from time import time

import geomstats.backend as gs
from geomstats.geometry.pre_shape import PreShapeSpace, KendallShapeMetric, Matrices

m = 3
k = 6
space = PreShapeSpace(k, m)
metric = KendallShapeMetric(k, m)

n_samples = 1

gs.random.seed(0)
point = space.random_uniform(n_samples)

# numbers of rungs between 10 and 100
n_numbers = 15
n_range = [int(k) for k in logspace(1, 3, n_numbers)]

# velocity of the geodesic to transport along
tan_b = Matrices(k, m).random_point(n_samples)
tan_b = space.to_tangent(tan_b, point)
tan_b = space.horizontal_projection(tan_b, point)

# use a vector orthonormal to tan_b
tan_a = Matrices(k, m).random_point(n_samples)
tan_a = space.to_tangent(tan_a, point)
tan_a = space.horizontal_projection(tan_a, point)

# orthonormalize and move to base_point
tan_a -= gs.einsum(
    '...,...ij->...ij',
    metric.inner_product(tan_a, tan_b, point) / metric.squared_norm(tan_b, point), tan_b)
tan_b = gs.einsum('...ij,...->...ij', tan_b, 1. / metric.norm(tan_b, point))
tan_a = gs.einsum('...ij,...->...ij', tan_a, 1. / metric.norm(tan_a, point))

# most accurate parallel transport
exact_transport = metric.parallel_transport(tan_a, tan_b, point, n_steps=1100, step='rk4')
end_point = metric.exp(tan_b, point)
exact_transport_tan_b = metric.log(metric.exp(2 * tan_b, point), end_point)

integrated_euler = []
integrated_rk4 = []
time_integrated_euler = []
time_integrated_rk4 = []
integrated_rk2 = []
time_integrated_rk2 = []
pl = []
time_pl = []
sl = []
time_sl = []
pl_2 = []
time_pl_2 = []
sl_2 = []
time_sl_2 = []
for n_rungs in n_range:
    s = time()
    integrated = metric.parallel_transport(tan_a, tan_b, point, n_steps=n_rungs, step='euler')
    time_integrated_euler.append(time() - s)
    integrated_euler.append(metric.norm(integrated - exact_transport, end_point))

    s = time()
    ladder = metric.ladder_parallel_transport(tan_a, tan_b, point, n_rungs=n_rungs)
    time_pl.append(time() - s)
    transported = ladder['transported_tangent_vec']
    end_point = ladder['end_point']
    pl.append(metric.norm(transported - exact_transport, end_point))

    s = time()
    ladder = metric.ladder_parallel_transport(tan_a, tan_b, point, n_rungs=n_rungs, alpha=2)
    time_pl_2.append(time() - s)
    transported = ladder['transported_tangent_vec']
    end_point = ladder['end_point']
    pl_2.append(metric.norm(transported - exact_transport, end_point))

    s = time()
    ladder = metric.ladder_parallel_transport(tan_a, tan_b, point, n_rungs=n_rungs, scheme='schild', alpha=1)
    time_sl.append(time() - s)
    transported = ladder['transported_tangent_vec']
    end_point = ladder['end_point']
    sl.append(metric.norm(transported - exact_transport, end_point))

    s = time()
    ladder = metric.ladder_parallel_transport(tan_a, tan_b, point, n_rungs=n_rungs, scheme='schild', alpha=2)
    time_sl_2.append(time() - s)
    transported = ladder['transported_tangent_vec']
    end_point = ladder['end_point']
    sl_2.append(metric.norm(transported - exact_transport, end_point))

    s = time()
    integrated = metric.parallel_transport(tan_a, tan_b, point, n_steps=n_rungs, step='rk2')
    time_integrated_rk2.append(time() - s)
    integrated_rk2.append(metric.norm(integrated - exact_transport, end_point))

    s = time()
    integrated = metric.parallel_transport(tan_a, tan_b, point, n_steps=n_rungs, step='rk4')
    time_integrated_rk4.append(time() - s)
    integrated_rk4.append(metric.norm(integrated - exact_transport, end_point))

plt.rc('font', family='serif')
plt.rc('text', usetex=True)

# Plot time
plt.figure()
plt.loglog(
    time_integrated_rk4, integrated_rk4,
    marker='o', linewidth=1, c='b', linestyle='dashed',
    fillstyle='none', label=r'Integration, RK4')
plt.loglog(
    time_integrated_rk2, integrated_rk2,
    marker='o', linewidth=1, c='g', linestyle='dashed',
    fillstyle='none', label=r'Integration, RK2')

plt.loglog(
    time_integrated_euler, integrated_euler,
    marker='o', linewidth=1, c='r', linestyle='dashed',
    fillstyle='none', label=r'Integration, Euler')
plt.title(f'Absolute error wrt computation time, k={k}, m={m}')
plt.xlabel('time in seconds')
plt.legend(loc='best')

# Plot error
plt.figure()
plt.loglog(
    n_range, integrated_rk4, marker='o', linewidth=1, c='b',
    linestyle='dashed', fillstyle='none', label=r'Integration, RK4')
plt.loglog(
    n_range, integrated_euler, marker='o', linewidth=1, c='r',
    linestyle='dashed', fillstyle='none',
    label=r'Integration, Euler')
plt.loglog(
    n_range, integrated_rk2, marker='o', linewidth=1, c='g',
    linestyle='dashed', fillstyle='none', label=r'Integration, RK2')
plt.loglog(
    n_range, pl, marker='o', linewidth=1, c='gold',
    linestyle='dashed', fillstyle='none', label=r'PL, $\alpha=1$')
plt.loglog(
    n_range, sl, marker='o', linewidth=1, c='gray',
    linestyle='dashed', fillstyle='none', label=r"SL, $\alpha=1$")
plt.loglog(
    n_range, pl_2, marker='x', linewidth=1, c='gold',
    linestyle='dashed', fillstyle='none', label=r'PL, $\alpha=2$')
plt.loglog(
    n_range, sl_2, marker='x', linewidth=1, c='gray',
    linestyle='dashed', fillstyle='none', label=r"SL, $\alpha=2$")
plt.title(f'Absolute error wrt number of time steps, k={k}, m={m}')
plt.xlabel('n')
plt.legend(loc='best')
plt.savefig(f'/home/nguigui/PycharmProjects/phd-diss/figures/kendall-pl-integration_{k}_{m}.eps', bbox_inches='tight', pad_inches=0)

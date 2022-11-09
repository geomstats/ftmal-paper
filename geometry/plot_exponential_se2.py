import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean

SE2_GROUP = SpecialEuclidean(n=2, point_type='matrix')
N_STEPS = 30
end_time = 2.7

theta = gs.pi / 3
initial_tangent_vecs = gs.array([
    [[0., - theta, 2], [theta, 0., 2], [0., 0., 0.]],
    [[0., - theta, 1.2], [theta, 0., 1.2], [0., 0., 0.]],
    [[0., - theta, 1.6], [theta, 0., 1.6], [0., 0., 0.]]])
t = gs.linspace(-end_time, end_time, N_STEPS + 1)

fig = plt.figure(figsize=(6, 6))
for tv, col in zip(initial_tangent_vecs, ['black', 'y', 'g']):
    tangent_vec = gs.einsum('t,ij->tij', t, tv)
    group_geo_points = SE2_GROUP.exp(tangent_vec)
    ax = visualization.plot(
        group_geo_points, space='SE2_GROUP', color=col)
ax = visualization.plot(
    gs.eye(3)[None, :, :], space='SE2_GROUP', color='slategray')
ax.set_aspect('equal')
ax.axis("off")
plt.savefig('../figures/exponential_se2.eps')
plt.show()

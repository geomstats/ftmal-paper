import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as viz
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean

n = 2
space = Hypersphere(n)
gs.random.seed(1)

n_samples = 15
precision = 5
last_meeting_point = gs.array([0., 1., 0.])
samples = space.random_riemannian_normal(
    mean=last_meeting_point,
    precision=precision, n_samples=n_samples)

estimator = FrechetMean(space.metric)
estimator.fit(samples)
new_meeting_point = estimator.estimate_

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
sphere_visu = viz.Sphere(n_meridians=30)
ax = sphere_visu.set_ax(ax=ax)
ax.grid(False)
sphere_visu.plot_heatmap(ax, lambda x: 1.)
sphere_visu.draw_points(ax, samples, marker='o', c='black', s=8)
sphere_visu.draw(ax, linewidth=1)
plt.axis('off')
plt.show()

logs = space.metric.log(samples, new_meeting_point)
geo = space.metric.geodesic(initial_point=new_meeting_point, initial_tangent_vec=logs)
t = gs.linspace(0, 1, 50)
points = geo(t)
sphere_visu.draw_points(ax, points.reshape(-1, 3), s=1, c='g')
sphere_visu.draw_points(ax, [new_meeting_point], marker='x', c='red', s=56)

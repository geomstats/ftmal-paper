import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as viz
from geomstats.geometry.hypersphere import Hypersphere

n = 2
space = Hypersphere(n)

n_samples = 5000

naive_uniform = gs.random.rand(n_samples, 2) * gs.pi * gs.array([1., 2.])[None, :]
mapped = space.spherical_to_extrinsic(naive_uniform)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
sphere_visu = viz.Sphere(n_meridians=30)
ax = sphere_visu.set_ax(ax=ax)
ax.grid(False)
sphere_visu.draw_points(ax, mapped, marker='o', c='b', s=2)
sphere_visu.draw(ax, linewidth=1)
plt.axis('off')
plt.savefig('../figures/sphere-uniform-naive.png', bbox_inches='tight', pad_inches=0)

uniform = space.random_uniform(n_samples)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
sphere_visu = viz.Sphere(n_meridians=30)
ax = sphere_visu.set_ax(ax=ax)
ax.grid(False)
sphere_visu.draw_points(ax, uniform, marker='o', c='b', s=2)
sphere_visu.draw(ax, linewidth=1)
plt.axis('off')
plt.savefig('../figures/sphere-uniform.png', bbox_inches='tight', pad_inches=0)

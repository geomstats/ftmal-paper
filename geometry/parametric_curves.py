import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.05
bound = 1.
t = np.linspace(-bound, bound, 100)
fig = plt.figure(frameon=False, figsize=(6, 8))
ax = fig.add_axes([0., 0., 1., 1.])
ax.axis("off")
ax.annotate("", xy=(0, bound + epsilon), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
ax.annotate("", xy=(-epsilon, 0.), xytext=(0, 0), arrowprops=dict(arrowstyle="-"))
ax.annotate("", xy=(bound + epsilon, 0.), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
ax.annotate("", xy=(0., -bound + epsilon), xytext=(0, 0), arrowprops=dict(arrowstyle="-"))
x, y = (t ** 2, t ** 3)
ax.plot(x, y, c='b')
plt.savefig('figures/cusp.eps')
fig.show()

bound = 1.3
t = np.linspace(-bound, bound, 100)
fig = plt.figure(figsize=(6, 8))
ax = fig.add_axes([0., 0., 1., 1.])
ax.axis("off")
x = 1 - t ** 2
y = t * x
ax.annotate("", xy=(0, max(y) + epsilon), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
ax.annotate("", xy=(max(x) + epsilon, 0.), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
ax.annotate("", xy=(min(x) - epsilon, 0.), xytext=(0, 0), arrowprops=dict(arrowstyle="-"))
ax.annotate("", xy=(0., min(y) - epsilon), xytext=(0, 0), arrowprops=dict(arrowstyle="-"))
ax.plot(x, y, c='b')
plt.savefig('figures/node.eps')
fig.show()

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Retour sur code d'affichage des densité
# Issu de rcp208_tpModelesMelange

# %%
# Code orginal
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# générer l'échantillon
N = 10 #= 100
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]

# préparer les données où on calculera la densité
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

true_density = (0.3*norm(0,1).pdf(X_plot[:,0]) + 0.7*norm(5,1).pdf(X_plot[:,0]))

# estimation par mélange gaussien, avec le « bon » nombre de composantes
gmm = GaussianMixture(n_components=2,n_init=3).fit(X)
print(gmm.converged_)
print(gmm.lower_bound_)

# calcul de la densité pour les données de X_plot
density = np.exp(gmm.score_samples(X_plot))

# affichage : vraie densité et estimation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(X_plot[:,0], true_density, fc='b', alpha=0.2, label='Vraie densité')
ax.plot(X_plot[:,0], density, '-', label="Estimation")
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
ax.legend(loc='upper left')
plt.show()

# %%
# Etude pas à pas
N = 10
X_raw = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))
X = X_raw[:, np.newaxis]
print("X : \n", X)

# %%
np.set_printoptions(precision=2)
X_plot_raw = np.linspace(-5, 10, 10)
print("X_plot_raw : \n", X_plot_raw)
X_plot = X_plot_raw[:, np.newaxis]
small_gmm = GaussianMixture(n_components=2,n_init=3).fit(X)
density = np.exp(small_gmm.score_samples(X_plot))
print("density : \n", density)


# %%
# Finit ...

# %%
## 2D Data (Density Surface)

# %%
small_md = np.array([ 
    [-1.0, -1.0],
    [-1.1, -1.0],
    [-1.1, -1.1],
    [-1.0, -1.1],
    [0.0, -1.25],
    [0.5, 0.5],
    [0.5, 1.2]
])
# estimation par mélange gaussien
gmm = GaussianMixture(n_components=1,n_init=3).fit(small_md)


# %%
# %matplotlib widget

# %%
# préparer les données où on calculera la densité
grid_size = 100

## Here we have to match the x&y range of value
Gx = np.arange(-1.5, 1.5, 3/grid_size) # 3/grid_size
Gy = np.arange(-1.5, 1.5, 3/grid_size)
Gx, Gy = np.meshgrid(Gx, Gy)


# calcul de la densité pour les données de la grille
density = np.exp(gmm.score_samples(np.hstack(
    (
        (Gx.reshape(grid_size*grid_size))[:,np.newaxis],
        (Gy.reshape(grid_size*grid_size)[:,np.newaxis]
        )
    )
)))

# affichage : données et estimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(Gx, Gy, density.reshape(grid_size,grid_size), rstride=1,
                    cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
ax.scatter(small_md[:,0], small_md[:,1], -0.025)
plt.show()

# %%

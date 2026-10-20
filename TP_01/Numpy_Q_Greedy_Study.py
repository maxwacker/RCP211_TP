# ---
# jupyter:
#   jupytext:
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

# %%
import numpy as np

# %%
a2D = np.array([[.4, .5, .1], [.3, .3, .4], [0, .55, .45]])

# %%
# Max sur axe1
# |     [.4, .5, .1]         -> .5
# |     [.3, .3, .4]         -> .4
# |     [0, .55, .45]        -> .55
# v     ------------> axe1
#axe0   

# %%
a2D.max(axis=1, keepdims=True)

# %%
atol=1e-10
best = np.isclose(a2D, a2D.max(axis=1, keepdims=True), atol=atol, rtol=0)
best

# %%
policy = best / best.sum(axis=1, keepdims=True)
policy # On obtient bien la matrice de transition déterministe de la meilleure action pour chaque ligne (s)

# %%

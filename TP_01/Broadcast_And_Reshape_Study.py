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
v = np.arange(4)
v 

# %%
v.shape

# %%
vr = v.reshape(1,-1)

# %%
vr.shape

# %%
vr[0]

# %%
vr[1]

# %%
np.sum(vr, axis=0)

# %%
np.sum(vr, axis=1)

# %%
# Exercice from
# https://www.youtube.com/watch?v=tKcLaGdvabM

# Calorite provided by Carbs / Protein / Fat, from different foods
apple = np.array([56.0, 1.2, 1.8]).reshape((3,1))
beef = np.array([0.0, 104.0, 135.0]).reshape((3,1))
eggs = np.array([4.4, 52.0, 99.0]).reshape((3,1))
potatoes = np.array([68.0, 8.0, 0.9]).reshape((3,1))
# .reshape((3,1) make array be a tensor of 3 lines 1 column (so column matrix) 

# The data is given by one sheet :
data = np.concatenate((apple, beef, eggs, potatoes), axis=1)
data.shape # (3, 4)
data

# %%
# Ex : Calculate percentage of calories comming from carbs/protein/fats for each food type
# WITHOUT LOOP

# 1 line of python code to compute the colums sum
# 1 other line to compute each % 

# %%
# Trying by my self
# Adding Calorie sum line
data_ext = data[:, :, np.newaxis]
data_ext.shape # (3, 4, 1)
data_ext[0]

# %%
# colums Sum:
csums = data.sum(axis=0) # axis=0 means sum accros line axis (arrow of progressing line number axe, so vertical) 
csums.shape # (4,)

# %%
percent = data / csums, #.reshape(1,4)

# %%
percent

# %%
#https://perso.univ-lyon1.fr/laurent.martin-witkowski/teaching/MGC2005L/Broadcasting.html

# %%
v1 = np.array([1,3,5,7])
v2 = np.array([2,4,6])
print(f"Shape de v1 : {v1.shape} ")
print(f"Shape de v2 : {v2.shape} ")


# %%
print(v1)
print(v2)

# %%
# reshapings 
rv1 = v1[:,np.newaxis]
print(rv1)
print(rv1.shape)

rv2 = v2[np.newaxis, :]
print(rv2)
print(rv2.shape)

# %%
rv1 + rv2

# %%

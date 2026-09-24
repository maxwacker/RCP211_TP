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
walls = None

# %%
walls = set() if walls is None else set(walls)

# %%
walls

# %%
n_rows, n_cols = (2, 2)

# %%
coords = [ (i,j)  
           for i in range(n_rows)
           for j in range(n_cols)
           if (i,j) not in walls
         ]

# %%
terminal_coords = set([(0,0), (1,1)])

# %%
terminal_coords <= set(coords)

# %%
set([(0,0), (2,1)]) <= set(coords)

# %%
n_states = len(coords)

# %%
ACTION_NAMES = ("gauche", "haut", "droite", "bas")

# MOVES contient les vecteurs de déplacement (di, dj) associés aux quatre actions.
# Depuis (i, j), la case candidate est (i + di, j + dj).
# Par exemple, pour "gauche" : (di, dj) = (0, -1), donc la case candidate est (i, j - 1).
MOVES = ((0, -1), (-1, 0), (0, 1), (1, 0))
N_ACTIONS = len(ACTION_NAMES)

# %%
P = np.zeros((n_states, N_ACTIONS, n_states))

# %%
R = np.zeros_like(P)

# %%
coord_to_state = {coord: s for s, coord in enumerate(coords)}

# %%
coord_to_state

# %%
terminal_states = np.array(
        sorted(coord_to_state[coord] for coord in terminal_coords), dtype=int
    )
terminal_set = set(terminal_states)

# %%
terminal_set

# %%
for s, (i, j) in enumerate(coords): # index, valeur in enumarate(someArray)
    for a, (di, dj) in enumerate(MOVES):
        if s in terminal_set: # Cette condition peut être sortie de cette boucle
            s_next = s
            reward = 0.0
        else:
            candidate = (i + di, j + dj)
            next_coord = candidate if candidate in coord_to_state else (i, j)
            s_next = coord_to_state[next_coord]
            reward = 0.0 if s_next in terminal_set else -1.0

        P[s, a, s_next] = 1.0 # Déterministe: Pour s donné, a choisie,
        R[s, a, s_next] = reward

# %% [markdown]
# ## Visualiser les quatre matrices de transition
#
# Le tableau `P`, de forme `(4, 4, 4)`, contient une matrice de transition par action. Pour une action fixée $a$,
#
# $$
# P^a=
# \bigl(P(s'\mid s,a)\bigr)_{s,s'}
# \in\mathbb R^{4\times4}.
# $$
#
# En Python, cette matrice s’obtient avec `P[:, a, :]`.

# %%
P[:, 0, :] #0: Action Gauche

# %%
P[:, 1, :] #1: Action Haut

# %%

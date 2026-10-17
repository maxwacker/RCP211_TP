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
# Partie 4 
def deterministic_policy(actions, n_actions=N_ACTIONS):
    # Convertit un tableau d'actions en politique au format (état, action).
    policy = np.zeros((len(actions), n_actions))
    policy[np.arange(len(actions)), actions] = 1.0
    return policy


# %%
actions = np.array([0,2,0,3])
n_actions = N_ACTIONS
policy = np.zeros((len(actions), n_actions))
policy[np.arange(len(actions)), actions] = 1.0
policy # This is the transition Matrix from actions array (s,a) 

# %%
np.arange(len(actions))

# %% [markdown]
# # Pourquoi le résultat de 'Value iteration' stochastique (plusieurs actions posible dans certains états) alors que 'Policy Iteration' donne un résultat déterrministe.
#
# En fait la description de 'Partie 4' choisit explicitment d'établir une politique déterministe.
# C'est un choix d'implémentation.
# L'algo général décrit dans le CM RL1_3.3.5 n'impose pas une telle contrainte : Dans le cas général, on peut utiliser l'algo 'Policy Iteration' pour trouver une politique optimale non déterministe.

# %%
# une fonction Etat-Action Q représenter par un tenseur Q[s,a] (Chaque ligne représentat la valeur de chaque depuis s, en colonne)
Q = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0], [0.3, 0.2, 0.5]])
Q              

# %%
V_new = Q.max(axis=1)
V_new # -> array([1. , 0.5, 0.5]) - La valeur max est donc bien unique (même en cas de ex-equo linge 2). Le non déterminisme ne vient pas de là

# %%
# Mais lorsqu'on cherche les actions qui permettent d'attendre ce maximum
best_vi = np.isclose(Q, Q.max(axis=1, keepdims=True), atol=1e-8, rtol=0)
best_vi # Il est bien possible d'obtenir plusieurs actions maximisantes (ex-eaquo) sur certaines ligne (ici : ligne 2)

# %% [markdown]
# # Partie 6 - Labyrinthe
# ## Etude de définition des murs
#

# %%
maze_array = np.array([
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
], dtype=int)

# %%
walls = set(map(tuple, np.argwhere(maze_array == 1)))

# %%
# Etude de argwhere
# Def np.argwhere : Find the indices of array elements that are non-zero, grouped by element.
# Ex :
import numpy as np
x = np.arange(6).reshape(2,3) 
# 2 lines x 3 cols :
# array([[0, 1, 2],
#      [3, 4, 5]])
np.argwhere(x>1) # Indices (ndim de x) des cellules t.q cell-value > 1

# %%
list(map(tuple, np.argwhere(x>1))) # Applique le consrtuceur tuple sur chaque [a, b] 

# %%
# Etude de build_gridworld

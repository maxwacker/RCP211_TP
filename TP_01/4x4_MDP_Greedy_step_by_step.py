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

# %% [markdown]
# # Application du code TP1 sur ex MCP grid 4x4
# Execution pas à pas de RCP211_RL1_TP1_Programmation_Dynamique.ipynb sur un exemple grid 4x4 comme illustré au slide 62/65 de RCP211_RL1_Presentation.pdf

# %%
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
plt.rcParams["figure.figsize"] = (8, 6)

ACTION_NAMES = ("gauche", "haut", "droite", "bas")

# MOVES contient les vecteurs de déplacement (di, dj) associés aux quatre actions.
# Depuis (i, j), la case candidate est (i + di, j + dj).
# Par exemple, pour "gauche" : (di, dj) = (0, -1), donc la case candidate est (i, j - 1).
MOVES = ((0, -1), (-1, 0), (0, 1), (1, 0))
N_ACTIONS = len(ACTION_NAMES)


# %%
def build_gridworld(shape, terminal_coords, walls=None):
    # Construit P(s'|s,a) et r(s,a,s') pour une grille déterministe.
    n_rows, n_cols = shape
    walls = set() if walls is None else set(walls) #Perso : init walls as empty set if not set as func parma
    terminal_coords = set(terminal_coords)

    coords = [
        (i, j)
        for i in range(n_rows)
        for j in range(n_cols)
        if (i, j) not in walls
    ]
    coord_to_state = {coord: s for s, coord in enumerate(coords)}

    if not terminal_coords <= set(coords):
        raise ValueError("Chaque terminal doit être une case accessible.")

    n_states = len(coords)
    P = np.zeros((n_states, N_ACTIONS, n_states))
    R = np.zeros_like(P)
    terminal_states = np.array(
        sorted(coord_to_state[coord] for coord in terminal_coords), dtype=int
    )
    terminal_set = set(terminal_states)

    for s, (i, j) in enumerate(coords):
        for a, (di, dj) in enumerate(MOVES):
            if s in terminal_set:
                s_next = s
                reward = 0.0
            else:
                candidate = (i + di, j + dj)
                next_coord = candidate if candidate in coord_to_state else (i, j)
                s_next = coord_to_state[next_coord]
                reward = 0.0 if s_next in terminal_set else -1.0

            P[s, a, s_next] = 1.0
            R[s, a, s_next] = reward

    return {
        "shape": shape,
        "walls": walls,
        "coords": coords,
        "coord_to_state": coord_to_state,
        "terminal_states": terminal_states,
        "P": P,
        "R": R,
    }


def check_mdp(env):
    # Vérifications élémentaires : dimensions et sommes des probabilités.
    P, R = env["P"], env["R"]
    assert P.shape == R.shape
    assert np.all(P >= 0)
    assert np.allclose(P.sum(axis=2), 1.0)
    return True



# %%
grid_4x4 = build_gridworld(
    shape=(4, 4),
    terminal_coords={(0, 0), (3, 3)}
)

P, R = grid_4x4["P"], grid_4x4["R"]
print("MDP valide :", check_mdp(grid_4x4))
print("Nombre d'états :", len(grid_4x4["coords"]))
print("Forme de P :", P.shape)
print("Forme de R :", R.shape)
print("États terminaux :", grid_4x4["terminal_states"])


# %%
def state_of(env, coord):
    return env["coord_to_state"][coord]


def transition_summary(env, coord, action_name):
    s = state_of(env, coord)
    a = ACTION_NAMES.index(action_name)
    # P[s, a, :] contient une unique valeur égale à 1.
    # Sa position est donc l'indice s_next de l'état suivant.
    s_next = int(np.argmax(env["P"][s, a]))
    next_coord = env["coords"][s_next]
    reward = env["R"][s, a, s_next]
    return {"s": s, "action": action_name, "s_next": s_next,
            "case_suivante": next_coord, "recompense": float(reward)}

print("Depuis (2, 2 ), action haut :", transition_summary(grid_4x4, (2, 2), "haut"))
print("Depuis (0, 3), action haut :", transition_summary(grid_4x4, (0, 3), "haut"))
print("Depuis (0, 1), action gauche :", transition_summary(grid_4x4, (0, 1), "gauche"))


# %%
#P[:, a, :] Matrice de transition de action a 
P[:, ACTION_NAMES.index("droite"), :]

# %%
proba_s_next_s01_droite =  P[state_of(grid_4x4, (0,1)), ACTION_NAMES.index("droite"), :]
print(proba_s_next_s01_droite)
s_next_s01_droite = int(np.argmax(proba_s_next_s01_droite))
print(s_next_s01_droite)


# %%
def draw_values(V, env, title="Fonction de valeur", ax=None):
    # Affiche un vecteur de valeurs sur la géométrie de la grille.
    if ax is None:
        _, ax = plt.subplots()

    image = np.full(env["shape"], np.nan)
    for s, coord in enumerate(env["coords"]):
        image[coord] = V[s]

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("black")
    shown = ax.imshow(image, cmap=cmap)
    terminal_set = set(env["terminal_states"])

    for s, (i, j) in enumerate(env["coords"]):
        label = "T" if s in terminal_set else f"{V[s]:.1f}"
        ax.text(j, i, label, ha="center", va="center", fontsize=7,
                color="white" if V[s] < np.nanmean(image) else "black")

    ax.set_title(title)
    ax.set_xticks(range(env["shape"][1]))
    ax.set_yticks(range(env["shape"][0]))
    ax.set_xticks(np.arange(-0.5, env["shape"][1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env["shape"][0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    return shown


def draw_policy(policy, env, title="Politique", ax=None):
    # Place chaque action de probabilité non nulle dans sa direction géométrique.
    if ax is None:
        _, ax = plt.subplots()

    background = np.zeros(env["shape"])
    for wall in env["walls"]:
        background[wall] = 1
    ax.imshow(background, cmap="Greys", vmin=0, vmax=1)
    terminal_set = set(env["terminal_states"])

    for s, (i, j) in enumerate(env["coords"]):
        if s in terminal_set:
            ax.text(j, i, "T", ha="center", va="center", fontsize=11)
        else:
            selected = np.flatnonzero(policy[s] > 1e-12)
            for action in selected:
                di, dj = MOVES[action]
                ax.arrow(
                    j, i, 0.28 * dj, 0.28 * di,
                    width=0.012,
                    head_width=0.11,
                    head_length=0.09,
                    length_includes_head=True,
                    color="black",
                )

    ax.set_title(title)
    ax.set_xticks(range(env["shape"][1]))
    ax.set_yticks(range(env["shape"][0]))
    ax.set_xticks(np.arange(-0.5, env["shape"][1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env["shape"][0], 1), minor=True)
    ax.grid(which="minor", color="tab:gray", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)



# %%
uniform_policy = np.full((len(grid_4x4["coords"]), N_ACTIONS), 1 / N_ACTIONS)
draw_policy(uniform_policy, grid_4x4, "Politique uniforme")
plt.show()

# %%

# %%
P.shape # = (16, 4, 16) ie (s, a, s') : 4 matrice P_up, P_left, P_down, P-right of 16 lines of 16 probability
P[:, 0, :] # left
P[0, 0, :] # probas des états atteints partant de : s0, action left

# %%
R.shape # (16, 4, 16) : Récompense

# %%
uniform_policy.shape # = ((4*4), 4) : 16 states, 4 actions possibles per state
uniform_policy[0] # array([0.25, 0.25, 0.25, 0.25]) uniform distributionof proba for each possible actions

# %%
gamma = 0.9

# %%
V0.shape # = (16,) 4x4 (states) value for state 

# %%
V0[None, None, :].shape  # (1, 1, nombre_etats)
V0[None, None, :] # array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
V_ = gamma * V0[None, None, :] # seqOf SeqOf V0 ?

V0[None, None, :].shape


# %%
def action_values(V, P,R, gamma):
    # Calcule q(s,a) pour tous les états et toutes les actions.
    return np.sum(P * (R + gamma * V[None, None, :]), axis=2)


# %%
#policy_evaluation_0
V0 = np.zeros(P.shape[0])
Vinit = action_values(V0, P,R, gamma)
Vinit.shape

# %%
draw_values(V0, grid_4x4)

# %%
sweep = 1


# %%
Q = action_values(V0, P, R, gamma)
Q

# %%

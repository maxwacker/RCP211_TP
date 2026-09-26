# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TP 1 — Programmation dynamique dans un MDP fini
#
# **Durée indicative : 2 h**  
# **Environnement :** Python, NumPy et Matplotlib
#
# ## Objectifs
#
# Ce TP met en pratique les méthodes de programmation dynamique étudiées dans les notes **RL1 — Model-based RL**. Nous allons :
#
# 1. construire explicitement un GridWorld sous la forme d'un MDP fini ;
# 2. évaluer une politique fixée ;
# 3. améliorer cette politique par un choix glouton ;
# 4. implémenter *Policy Iteration* puis *Value Iteration* ;
# 5. réutiliser **sans les modifier** ces algorithmes dans un labyrinthe.
#
# Le parcours principal est entièrement codé : on peut donc lire, exécuter et interpréter le notebook du début à la fin sans être bloqué par une question de programmation.
#
# > **À propos des questions facultatives et des questions d'approfondissement.**  
# > Elles sont intéressantes et méritent le détour, mais elles ne sont pas nécessaires pour comprendre la suite. Si le temps manque, poursuivez le parcours principal et revenez-y lors d'une deuxième lecture.
#
# Les questions ordinaires sont courtes : essayez d'y répondre mentalement ou en quelques phrases avant d'exécuter la cellule suivante.
#
# ### Références dans RL1
#
# - §2.7 : exemple du GridWorld et équations de Bellman ;
# - §3.2 : évaluation d'une politique ;
# - §3.3 : *Policy Iteration* ;
# - §3.4 : *Value Iteration* ;
# - §3.5 : lecture commune par la *Generalized Policy Iteration* (GPI).

# %% [markdown]
# ## 0 — Imports et conventions
#
# Nous n'utilisons que NumPy et Matplotlib. Les quatre actions sont toujours rangées dans l'ordre
#
# $$
# \mathcal A=\{\text{gauche},\text{haut},\text{droite},\text{bas}\}.
# $$
#
# Cet ordre sera le même dans les tableaux NumPy, les fonctions et les figures.

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


# %% [markdown]
# ### Coordonnées de la grille et indices de matrice
#
# Les coordonnées $(i,j)$ suivent la convention utilisée pour les matrices et les tableaux NumPy :
#
# - $i$ est l'indice de **ligne** : il détermine la position verticale ;
# - $j$ est l'indice de **colonne** : il détermine la position horizontale.
#
# La seconde coordonnée varie donc lors d'un déplacement horizontal :
#
# $$
# \text{gauche} : (i,j)\longrightarrow(i,j-1),
# \qquad
# \text{droite} : (i,j)\longrightarrow(i,j+1).
# $$
#
# La première coordonnée varie lors d'un déplacement vertical :
#
# $$
# \text{haut} : (i,j)\longrightarrow(i-1,j),
# \qquad
# \text{bas} : (i,j)\longrightarrow(i+1,j).
# $$
#
# Ces déplacements donnent la case candidate ; si elle sort de la grille ou correspond à un mur, l'agent reste sur place. Cette convention diffère de la notation cartésienne habituelle $(x,y)$, dans laquelle la première coordonnée représente la position horizontale. Ici, on peut faire la correspondance
#
# $$
# x=j,\qquad y=i,
# $$
#
# dans les coordonnées de l'affichage, avec une particularité supplémentaire : dans l'affichage d'une matrice, l'indice de ligne $i$ augmente du haut vers le bas.

# %% [markdown]
# # Partie 1 — Le GridWorld et son MDP
#
# Nous considérons une grille $10\times10$ avec deux états terminaux : $(0,0)$ et $(9,9)$. Une action qui voudrait sortir de la grille laisse l'agent sur place.
#
# La dynamique est déterministe. Toute transition qui ne conduit pas à un état terminal donne la récompense $-1$ ; entrer dans un état terminal donne la récompense $0$. Une fois dans un terminal, le processus y reste avec une récompense nulle.
#
# Nous indexons les cases ligne par ligne. Pour une grille $n\times n$ sans murs, l'espace des états contient $n^2$ états. Chaque case est identifiée par ses coordonnées
#
# $$
# i,j\in\{0,\ldots,n-1\}.
# $$
#
# La case $(i,j)$ correspond à l'état d'indice
#
# $$
# s=i\,n+j\in\{0,\ldots,n^2-1\}.
# $$
#
# Le tableau `P` représente $P(s'\mid s,a)$ et a donc la forme `(nombre_etats, nombre_actions, nombre_etats)`. Le tableau `R`, de même forme, contient $r(s,a,s')$.
#
# *Pour approfondir : notes RL1, §2.7.*

# %% [markdown]
# ## Construction du MDP
#
# Nous allons maintenant implémenter le MDP décrit ci-dessus. La fonction `build_gridworld(...)` construit l'environnement et renvoie un dictionnaire Python, que nous appellerons `grid`. Ce dictionnaire rassemble :
#
# - la géométrie de la grille et les murs éventuels ;
# - les correspondances entre les indices des états et les coordonnées des cases ;
# - les états terminaux ;
# - le tenseur de transition `P` ;
# - le tenseur de récompense `R`.
#
# La liste `coords` associe à chaque indice d'état `s` les coordonnées de la case correspondante : `coords[s]` donne donc la case associée à l'état `s`. Le dictionnaire `coord_to_state` réalise la correspondance inverse. Les indices des états sont attribués avec `enumerate(coords)`, dans l'ordre naturel des cases, ligne par ligne. En l'absence de murs, cette numérotation correspond à la formule $s=i\,n_{\mathrm{cols}}+j$, où $n_{\mathrm{cols}}$ est le nombre de colonnes. En présence de murs, seules les cases accessibles sont numérotées et les indices restent consécutifs.
#
# Le paramètre `walls` décrit les murs éventuels sous la forme d'une collection de coordonnées. Les murs et les états terminaux sont convertis en ensembles Python (`set`), ce qui permet de vérifier rapidement si une coordonnée leur appartient. Jusqu'à la dernière partie consacrée au labyrinthe, nous utiliserons `walls=None` : toutes les cases de la grille seront donc accessibles.
#
# Prenez le temps d'identifier la signification des différentes composantes du dictionnaire `grid`. Elles représentent directement les éléments du MDP ainsi que les informations nécessaires à son affichage.

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


grid = build_gridworld(
    shape=(10, 10),
    terminal_coords={(0, 0), (9, 9)}
)

P, R = grid["P"], grid["R"]
print("MDP valide :", check_mdp(grid))
print("Nombre d'états :", len(grid["coords"]))
print("Forme de P :", P.shape)
print("Forme de R :", R.shape)
print("États terminaux :", grid["terminal_states"])


# %% [markdown]
# Le constructeur précédent réalise trois opérations principales :
#
# - il associe un indice à chaque case accessible ;
# - il détermine l'unique état suivant de chaque couple $(s,a)$ ;
# - il renseigne les tableaux `P` et `R` pour chaque transition possible.
#
# Ici, `P[s, a, :]` contient une seule valeur égale à `1`, car les transitions sont déterministes.
#
# ### Question
#
# Sans exécuter de nouveau code, quel est l'indice de la case $(3,7)$ ? Que doit faire l'action `haut` dans cette case ? Et l'action `haut` dans la case $(0,7)$ ?
#

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


print("Depuis (3, 7), action haut :", transition_summary(grid, (3, 7), "haut"))
print("Depuis (0, 7), action haut :", transition_summary(grid, (0, 7), "haut"))
print("Depuis (0, 1), action gauche :", transition_summary(grid, (0, 1), "gauche"))


# %% [markdown]
# -> Action 'haut' = 'diminuer indice de ligne' : (3,7) -haut-> (2,7) 
#
# (0,7) -haut-> (0,7)  (Il y a un mur au dessus de 0,7)

# %%

# %% [markdown]
# ## Visualisation des valeurs et de la politique
#
# Les deux fonctions suivantes servent uniquement à représenter les résultats sur la géométrie de la grille ; elles n'interviennent pas dans les calculs du MDP.
#
# La fonction `draw_values(V, env, ...)` associe chaque composante `V[s]` aux coordonnées de l'état `s`, puis affiche les valeurs sous la forme d'une carte colorée. Les états terminaux sont indiqués par la lettre `T` et, dans le labyrinthe, les murs apparaîtront en noir.
#
# La fonction `draw_policy(policy, env, ...)` représente dans chaque état les actions auxquelles la politique attribue une probabilité non nulle. Les flèches sont placées selon leur direction géométrique : gauche, haut, droite ou bas. Plusieurs flèches dans une même case indiquent donc que plusieurs actions peuvent y être choisies.
#
# Le paramètre optionnel `ax` permet d'insérer la représentation dans une figure Matplotlib existante, notamment pour afficher côte à côte une fonction de valeur et la politique associée.

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


uniform_policy = np.full((len(grid["coords"]), N_ACTIONS), 1 / N_ACTIONS)
draw_policy(uniform_policy, grid, "Politique uniforme")
plt.show()

# %% [markdown]
# ### Question facultative — changer la géométrie
#
# Construisez une grille $6\times8$ avec un seul terminal en $(5,7)$. Vérifiez la forme de `P`, puis inspectez une transition au bord de la grille avec `transition_summary`. Utilisez un autre nom, par exemple `grid_test`, pour ne pas affecter l'environnement du parcours principal du TP.
#
# Cette expérience ne modifie aucune fonction : seules les données passées à `build_gridworld` changent.

# %%
grid_test = build_gridworld(
    shape=(6, 8),
    terminal_coords={(5, 7)}
)

# %%
P_test, R_test = grid_test["P"], grid_test["R"]
print("MDP valide :", check_mdp(grid_test))
print("Nombre d'états :", len(grid_test["coords"]))
print("Forme de P :", P_test.shape)
print("Forme de R :", R_test.shape)
print("États terminaux :", grid_test["terminal_states"])


print("Depuis (3, 7), action droite :", transition_summary(grid_test, (3, 7), "droite"))
print("Depuis (0, 7), action haut :", transition_summary(grid_test, (0, 7), "haut"))
print("Depuis (5, 7), action droite :", transition_summary(grid_test, (5, 7), "droite"))

# %%

# %%
uniform_policy_test = np.full((len(grid_test["coords"]), N_ACTIONS), 1 / N_ACTIONS)
draw_policy(uniform_policy_test, grid_test, "Politique uniforme")
plt.show()

# %%
uniform_policy_test.shape # (48, 4) ~ (6x8) states with 4 actions probability for each
uniform_policy_test[0] # uniform probability for each actions, given S = s0

# %%
uniform_policy_test = np.full((len(grid_test["coords"]), N_ACTIONS), 1 / N_ACTIONS)
draw_policy(uniform_policy_test, grid_test, "Politique uniforme")
plt.show()

# %% [markdown]
# ## Visualiser les quatre matrices de transition
#
# Le tableau `P`, de forme `(100, 4, 100)`, contient une matrice de transition par action. Pour une action fixée $a$,
#
# $$
# P^a=
# \bigl(P(s'\mid s,a)\bigr)_{s,s'}
# \in\mathbb R^{100\times100}.
# $$
#
# En Python, cette matrice s’obtient avec `P[:, a, :]`.
#
# Les lignes sont indexées par l’état actuel $s$ et les colonnes par l’état suivant $s'$. La cellule suivante utilise `spy` : chaque point représente un coefficient non nul de la matrice. Dans ce GridWorld déterministe, chaque ligne contient exactement un coefficient égal à $1$.
# La diagonale principale $s'=s$ est tracée en gris clair comme repère visuel.

# %%
n_states = P.shape[0]
ticks = np.arange(0, n_states, 20)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for a, (action_name, ax) in enumerate(zip(ACTION_NAMES, axes.flat)):
    ax.spy(P[:, a, :], markersize=2)

    # Diagonale principale : s' = s
    ax.plot(
        [-0.5, n_states - 0.5],
        [-0.5, n_states - 0.5],
        color="lightgray",
        linewidth=1,
        zorder=0,
    )

    ax.set_title(rf"$P^{{\mathrm{{{action_name}}}}}$")
    ax.set_xlabel("État suivant $s'$")
    ax.set_ylabel("État actuel $s$")

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(
        axis="x",
        bottom=True,
        labelbottom=True,
        top=False,
        labeltop=False,
    )

fig.suptitle(
    "Matrices de transition associées aux quatre actions",
    fontsize=14,
)
plt.tight_layout()
plt.show()


# %% [markdown]
#

# %% [markdown]
# ### Questions de compréhension
#
# Avant de passer à l'évaluation d'une politique, observez attentivement les quatre matrices et essayez de répondre aux questions suivantes.
#
# 1. Pourquoi chaque matrice $P^a$ est-elle de taille $100\times100$, et non $10\times10$ 
#
# R1:
#
# La matrices de transition à autant de lignes et colonne que d'états
# Un état est déterminé par sa ligne et sa colonne dans le grid: 10 lignes x 10 colonnes : 100 états
#
# 2. Que représentent respectivement ses lignes et ses colonnes ?
#
# R2:
#
# Chaque ligne représente la distrubition de probabilité du ième état vers jième. le somme de la ligen est donc 1.
# Pour des transition déterministe, une seule colonne vaut 1 (pour une action choisie, une seule destination possbile).
#
# 3. Pourquoi chaque ligne contient-elle exactement un seul coefficient non nul ?
#
# R3: 
#
# Pour une action choisie une seule destination cnadidate possible
#
# 5. Pourquoi ce coefficient vaut-il $1$ ?
#
# R4: 
#
# Déterminisme : la somme de ligne doit être 1 (ligne distribution de probabilité)
#
#
# 7. Pourquoi voit-on certains coefficients sur la diagonale principale ?
#
#  Coeff sur diagonale principale désgine une transition d'un état sur lui même : Tout les états en bordure avec action vers la bordure.
# Ainsi les 10 premiers lignes de Phaut sont à 1 sur la diagonale car les 10 premières transitions vers le haut cognent le mur  
#  
# 9. Pourquoi les coefficients non diagonaux forment-ils des bandes décalées de $1$ pour les déplacements horizontaux et de $10$ pour les déplacements verticaux ?
#
# R9 :
# Déplacements horizontaux : variation d'incide d'état de +/- 1
# Déplacement verticaux : variation d'incide d'état de +/- 10 (Saut de ligne de longueur 10)
#
#     
# 11. Une colonne doit-elle, elle aussi, contenir exactement un seul $1$ ?
#
# R11 :
#
# La colonne k inquique depuis quels etats l'état k est accessible (pour une direction de mouvement donnée)
# Les états en bordures de grid sonr accessible depuis leur voisin et depuis eux même. leur colonnes comporte donc 2 valeurs à 1
#
# Ces questions portent uniquement sur la structure de $P$. Le tableau `R` possède la même forme `(100, 4, 100)`, mais il contient les récompenses $r(s,a,s')$, et non des probabilités.

# %% [markdown]
# # Partie 2 — Évaluation d'une politique
#
# Pour une politique fixée $\pi$, l'équation de Bellman est
#
# $$
# v_\pi(s)=
# \sum_a \pi(a\mid s)
# \sum_{s'}P(s'\mid s,a)
# \left[r(s,a,s')+\gamma v_\pi(s')\right].
# $$
#
# L'évaluation itérative part d'un vecteur quelconque $v_0$ et applique le membre de droite jusqu'à stabilisation. Nous utilisons des mises à jour **synchrones** : toutes les composantes de $v_{k+1}$ sont calculées à partir du même vecteur $v_k$.
#
# Nous arrêtons lorsque
#
# $$
# \Delta_k=\lVert v_{k+1}-v_k\rVert_\infty<\theta.
# $$
#
# *Pour approfondir : notes RL1, §3.2.*
#
# ### Explication du code
#
# Les deux fonctions suivantes traduisent directement l'équation de Bellman en opérations NumPy.
#
# La fonction `action_values(V, P, R, gamma)` calcule, pour chaque couple $(s,a)$, la récompense immédiate augmentée de la valeur actualisée de l'état suivant, puis en prend l'espérance selon $P(s'\mid s,a)$ :
#
# $$
# q_V(s,a)
# =
# \sum_{s'}P(s'\mid s,a)
# \left[r(s,a,s')+\gamma V(s')\right].
# $$
#
# Le tableau `V[None, None, :]` a la forme `(1, 1, nombre_etats)` : ses deux nouveaux axes permettent à NumPy d'utiliser `V[s_next]` pour tous les états actuels et toutes les actions. L'argument `axis=2` demande ensuite de sommer selon le troisième axe, celui des états suivants $s'$. Le résultat possède donc la forme `(nombre_etats, nombre_actions)`. Lorsque $V=v_\pi$, les valeurs obtenues sont précisément les $q_\pi(s,a)$.
#
# La fonction `policy_evaluation(...)` applique ensuite itérativement l'opérateur de Bellman associé à une politique fixée. À chaque balayage, elle calcule les valeurs des actions à partir du vecteur courant `V`, puis effectue leur moyenne selon les probabilités `policy[s, a]` :
#
# $$
# V_{\mathrm{new}}(s)
# =
# \sum_a\pi(a\mid s)\,q_V(s,a).
# $$
#
# Les mises à jour sont synchrones : toutes les composantes de `V_new` sont calculées à partir du même ancien vecteur `V`. L'itération s'arrête lorsque l'écart maximal entre deux vecteurs successifs devient inférieur à `theta`. La fonction renvoie alors la valeur approchée de la politique, le nombre de balayages effectués et l'historique des écarts `deltas`.
#
#
# Nous utilisons $\gamma=0{,}9$. Puisque $\gamma<1$ et que les récompenses sont bornées, l'évaluation itérative converge même pour une politique qui n'atteint pas nécessairement un état terminal.

# %% [markdown]
# ### Prédiction
#
# Avant d'exécuter l'évaluation, répondez mentalement :
# 1. quel signe auront les valeurs des états non terminaux ?
# 2. quelles symétries attendez-vous dans la carte des valeurs ?
# 3. pourquoi les deux terminaux doivent-ils garder la valeur $0$ ?
#
# 1 - Signe négatif, comme accumulation de valeurs négatives ou nulles
# 2 - Puisque nous avons un symétrie dans la dynamique et ainsi que dans la polique nous attendont une symétrie dans la carte des valeurs
# 3 - Car les états terminaux sont absordant : une fois atteints on n'en sort plus. La récompense associée à ces transitions répétées sur un état terminal s'accumule dans le rendu futur mais ne doit pas le modifier (donc 0).
#

# %%
def action_values(V, P, R, gamma):
    # Calcule q(s,a) pour tous les états et toutes les actions.
    return np.sum(P * (R + gamma * V[None, None, :]), axis=2)


def policy_evaluation(policy, P, R, gamma, terminal_states,
                      theta=1e-8, max_sweeps=100_000):
    # Évaluation itérative synchrone d'une politique stochastique.
    V = np.zeros(P.shape[0])
    deltas = []

    for sweep in range(1, max_sweeps + 1):
        Q = action_values(V, P, R, gamma)
        V_new = np.sum(policy * Q, axis=1)
        V_new[terminal_states] = 0.0

        delta = np.max(np.abs(V_new - V))
        deltas.append(delta)
        V = V_new

        if delta < theta:
            return V, sweep, np.array(deltas)

    raise RuntimeError("L'évaluation n'a pas convergé.")


gamma = 0.9
V_uniform, n_sweeps, deltas = policy_evaluation(
    uniform_policy, P, R, gamma, grid["terminal_states"]
)

print(f"Convergence en {n_sweeps} balayages.")
print(f"Dernier écart maximal : {deltas[-1]:.2e}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
draw_values(V_uniform, grid, rf"$v_\pi$ pour la politique uniforme ($\gamma={gamma}$)", axes[0])
axes[1].semilogy(deltas)
axes[1].set_title("Convergence de l'évaluation")
axes[1].set_xlabel("Balayage")
axes[1].set_ylabel(r"$\|v_{k+1}-v_k\|_\infty$")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Question
#
# Les valeurs observées respectent-elles vos prédictions ? Pourquoi les cases proches d'un terminal ont-elles une valeur plus élevée, c'est-à-dire moins négative ?

# %% [markdown]
# ### Expérience facultative — rôle de $\gamma$
#
# Prédisez d'abord l'effet d'un passage de $\gamma=0{,}9$ à $\gamma=0{,}5$ sur les états éloignés des terminaux. Exécutez ensuite la cellule suivante et comparez les deux cartes.

# %%
V_uniform_05, n_sweeps_05, _ = policy_evaluation(
    uniform_policy, P, R, gamma=0.5, terminal_states=grid["terminal_states"]
)

# Échelle de couleurs commune aux deux cartes.
vmin = min(V_uniform.min(), V_uniform_05.min())
vmax = max(V_uniform.max(), V_uniform_05.max())

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

shown_09 = draw_values(V_uniform, grid, r"Politique uniforme : $\gamma=0{,}9$", axes[0])
shown_05 = draw_values(V_uniform_05, grid, r"Politique uniforme : $\gamma=0{,}5$", axes[1])

shown_09.set_clim(vmin, vmax)
shown_05.set_clim(vmin, vmax)

fig.colorbar(shown_09, ax=axes, shrink=0.85, label=r"Valeur $v_\pi(s)$")

plt.show()

print("Balayages pour gamma=0.5 :", n_sweeps_05)

# %% [markdown]
# ### Question d'approfondissement — résolution exacte
#
# *Note :* Cette partie est utile, mais pas nécessaire pour la suite du TP.
#
# Dans un petit MDP, on peut aussi former $P_\pi$ et $\bar r_\pi$, puis résoudre
#
# $$
# (I-\gamma P_\pi)v_\pi=\bar r_\pi.
# $$
#
# La cellule suivante vérifie que cette résolution et l'algorithme itératif donnent des résultats concordants, à la précision de l'évaluation itérative près.
#
# #### Explication du code
#
# La fonction `np.einsum` permet d'écrire directement des sommes sur des indices, selon la convention de sommation d'Einstein. Chaque lettre désigne un axe des tableaux manipulés. Les indices présents à gauche de `->` mais absents à droite sont automatiquement sommés.
#
# Ici, nous utilisons :
#
# - `s` pour l'état actuel ;
# - `a` pour l'action ;
# - `n` pour l'état suivant $s'$ — car `einsum` utilise des lettres simples pour nommer les axes.
#
# Considérons la première expression :
#
# `P_pi = np.einsum("sa,san->sn", uniform_policy, P)`
#
# Le tableau `uniform_policy` possède les axes `(s, a)`, tandis que `P` possède les axes `(s, a, n)`. Le résultat conserve seulement les axes `(s, n)`. L'indice `a`, absent après la flèche, est donc sommé :
#
# $$ P_\pi(s,n) = \sum_a \pi(a\mid s)P(n\mid s,a). $$
#
# Le résultat `P_pi` est ainsi une matrice de forme `(nombre_etats, nombre_etats)`. Elle décrit les transitions entre états lorsque les actions sont choisies selon la politique $\pi$.
#
# La seconde expression est :
#
# `r_pi = np.einsum("sa,san,san->s", uniform_policy, P, R)`
#
# Les trois tableaux ont respectivement les axes `(s,a)`, `(s,a,n)` et `(s,a,n)`. Le résultat conserve uniquement `s` : les indices `a` et `n` sont donc sommés. On obtient
#
# $$ \bar r_\pi(s) = \sum_a \pi(a\mid s) \sum_n P(n\mid s,a)\,r(s,a,n). $$
#
# Le vecteur `r_pi` contient ainsi la récompense immédiate espérée dans chaque état lorsque l'on suit la politique uniforme.

# %%
P_pi = np.einsum("sa,san->sn", uniform_policy, P)
r_pi = np.einsum("sa,san,san->s", uniform_policy, P, R)
# `np.eye` construit la matrice identité et 
# `np.linalg.solve` résout le système linéaire sans calculer explicitement son inverse.
V_exact = np.linalg.solve(np.eye(P.shape[0]) - gamma * P_pi, r_pi)

print("Écart maximal entre les deux méthodes :",
      np.max(np.abs(V_uniform - V_exact)))


# %% [markdown]
# # Partie 3 — Amélioration d'une politique
#
# Après avoir évalué $\pi$, on calcule pour chaque action
#
# $$
# q_\pi(s,a)=
# \sum_{s'}P(s'\mid s,a)
# \left[r(s,a,s')+\gamma v_\pi(s')\right].
# $$
#
# Une politique gloutonne choisit une action de $\arg\max_a q_\pi(s,a)$. Lorsque plusieurs actions sont maximisantes, nous les affichons toutes et leur attribuons ici des probabilités égales.
#
# *Pour approfondir : notes RL1, fin du §3.2 et début du §3.3.*
#
# ### Explication du code
#
# La fonction `greedy_stochastic_policy` construit une politique gloutonne à partir du tableau `Q`, dont chaque ligne contient les valeurs des actions disponibles dans un état.
#
# Pour chaque état, `np.isclose` repère toutes les actions dont la valeur est égale, à la tolérance `atol` près, à la valeur maximale de la ligne. Cette tolérance évite que de petites erreurs numériques empêchent de reconnaître une égalité. L'option `keepdims=True` conserve une dimension de taille 1 afin que NumPy puisse effectuer les comparaisons et divisions ligne par ligne.
#
# Si $k$ actions sont maximisantes dans l'état $s$, la fonction attribue à chacune la probabilité $1/k$ :
#
# $$
# \pi'(a\mid s)=
# \begin{cases}
# 1/k & \text{si }a\in\arg\max_{a'}Q(s,a'),\\
# 0   & \text{sinon}.
# \end{cases}
# $$
#
# Les lignes de la politique correspondant aux états terminaux sont mises à zéro, puisqu'aucune décision n'y est nécessaire. La valeur de ces états est fixée séparément à zéro.
#
# #### Question
#
# Pour chacune des quatre cases centrales $(4,4)$, $(4,5)$, $(5,4)$ et $(5,5)$, quelles directions pensez-vous que la politique améliorée privilégiera ?
#

# %%
def greedy_stochastic_policy(Q, terminal_states, atol=1e-10):
    # Répartit uniformément la probabilité entre les actions maximisantes.
    best = np.isclose(Q, Q.max(axis=1, keepdims=True), atol=atol, rtol=0)
    policy = best / best.sum(axis=1, keepdims=True)
    policy[terminal_states] = 0.0
    return policy


Q_uniform = action_values(V_uniform, P, R, gamma)
improved_policy = greedy_stochastic_policy(
    Q_uniform, grid["terminal_states"]
)

draw_policy(improved_policy, grid, r"Politique gloutonne par rapport à $v_\pi$")
plt.show()

# %% [markdown]
# Inspectons localement le calcul de la politique gloutonne dans la case `(4,4)`.

# %%
coord = (4, 4)
s = state_of(grid, coord)
print("Case", coord, "— indice", s)
for name, value in zip(ACTION_NAMES, Q_uniform[s]):
    print(f"  q_pi({name:7s}) = {value: .4f}")


# %% [markdown]
# On voit donc concrètement pourquoi la politique gloutonne affiche deux flèches dans la case $(4,4)$ :
#
# $$ q_\pi(s,\text{gauche}) = q_\pi(s,\text{haut}) > q_\pi(s,\text{droite}) = q_\pi(s,\text{bas}). $$
#
# Les actions `gauche` et `haut` sont toutes les deux maximisantes ; la politique leur attribue donc à chacune une probabilité $1/2$.
#
# ### Question
#
# La politique gloutonne a été obtenue par un seul regard en avant : pour calculer $q_\pi(s,a)$, on évalue l'action $a$ en supposant qu'elle sera suivie de la politique uniforme. La nouvelle politique choisit ensuite une action maximisante dans chaque état. Pourquoi suivre cette nouvelle politique à chaque pas garantit-il une valeur au moins aussi élevée ?
#
# Le résultat général est le **théorème d'amélioration de politique** présenté dans RL1, §3.3.6.
#
# ### Question facultative — prédire puis vérifier une égalité
#
# Choisissez une autre case pour laquelle les symétries du GridWorld suggèrent que deux actions sont également avantageuses.
#
# Avant d'afficher les valeurs, prédisez quelles seront les deux actions maximisantes et expliquez brièvement votre raisonnement géométrique. Calculez ensuite l'indice `s` de cette case et affichez les quatre valeurs `Q_uniform[s]` pour vérifier votre prédiction.

# %% [markdown]
# # Partie 4 — Policy Iteration
#
# Une seule amélioration ne suffit pas en général. *Policy Iteration* procède ainsi :
#
# 1. évaluation de la politique courante jusqu'à convergence ;
# 2. amélioration gloutonne ;
# 3. répétition des étapes précédentes, ou arrêt lorsque la politique ne change plus.
#
# Nous utilisons maintenant des politiques déterministes, représentées par un tableau `actions` : `actions[s]` est l'indice de l'action choisie dans l'état $s$.
#
# En cas d'égalité, nous appliquons la convention de RL1 : si l'action courante est encore maximisante, on la conserve. Cela évite des changements artificiels entre actions de même valeur.
#
# *Pour approfondir : notes RL1, §3.3, notamment l'algorithme du §3.3.5.*
#
# Les trois fonctions suivantes implémentent les différentes étapes de *Policy Iteration* pour des politiques déterministes.
#
# La fonction `deterministic_policy(actions)` convertit une politique représentée par un simple vecteur d'actions en un tableau de probabilités de forme `(nombre_etats, nombre_actions)`. Dans chaque état, l'action choisie reçoit la probabilité $1$ et toutes les autres la probabilité $0$. Cette conversion permet de réutiliser directement la fonction `policy_evaluation`, qui accepte des politiques stochastiques aussi bien que déterministes.
#
# La fonction `greedy_deterministic_actions(Q, ...)` construit une politique déterministe gloutonne à partir des valeurs `Q[s, a]`. Dans chaque état, elle choisit une action maximisante. En cas d'égalité, elle conserve l'action de la politique courante si celle-ci est encore maximisante, conformément à la convention utilisée dans RL1. La tolérance `atol` permet de reconnaître les égalités numériques approchées.
#
# Enfin, `policy_iteration(...)` alterne les deux étapes fondamentales de l'algorithme :
#
# 1. évaluer la politique courante jusqu’à ce que l’écart maximal entre deux balayages successifs soit inférieur à `theta` ;
# 2. construire une nouvelle politique gloutonne par rapport à la valeur obtenue.
#
# L'algorithme s'arrête lorsqu'aucun état ne change d'action. Il renvoie la fonction de valeur finale, la politique déterministe obtenue et un journal `log` contenant, pour chaque étape, le nombre de balayages nécessaires à l'évaluation et le nombre d'états dont l'action a été modifiée. Le paramètre `max_improvements` constitue seulement une limite de sécurité.

# %%
def deterministic_policy(actions, n_actions=N_ACTIONS):
    # Convertit un tableau d'actions en politique au format (état, action).
    policy = np.zeros((len(actions), n_actions))
    policy[np.arange(len(actions)), actions] = 1.0
    return policy


def greedy_deterministic_actions(Q, terminal_states, current_actions=None,
                                 atol=1e-10):
    # Politique gloutonne déterministe, avec conservation des ex aequo.
    actions = np.zeros(Q.shape[0], dtype=int)
    terminal_set = set(terminal_states)

    for s in range(Q.shape[0]):
        if s in terminal_set:
            actions[s] = 0 if current_actions is None else current_actions[s]
            continue

        best_actions = np.flatnonzero(
            np.isclose(Q[s], Q[s].max(), atol=atol, rtol=0)
        )
        if current_actions is not None and current_actions[s] in best_actions:
            actions[s] = current_actions[s]
        else:
            actions[s] = best_actions[0]

    return actions


def policy_iteration(P, R, gamma, terminal_states, initial_actions,
                     theta=1e-8, max_improvements=1_000):
    actions = initial_actions.copy()
    log = []

    for improvement in range(max_improvements):
        policy = deterministic_policy(actions)
        V, evaluation_sweeps, _ = policy_evaluation(
            policy, P, R, gamma, terminal_states, theta=theta
        )
        Q = action_values(V, P, R, gamma)
        new_actions = greedy_deterministic_actions(
            Q, terminal_states, current_actions=actions
        )
        changed_states = int(np.count_nonzero(new_actions != actions))
        log.append((evaluation_sweeps, changed_states))

        if changed_states == 0:
            return V, actions, log
        actions = new_actions

    raise RuntimeError("Policy Iteration ne s'est pas stabilisée.")


# %% [markdown]
# Nous partons volontairement d'une politique médiocre qui demande toujours d'aller vers la droite.
#
# ### Question
#
# Cette politique n'explore pas la grille et se bloque sur le bord droit depuis la plupart des états. Pourquoi *Policy Iteration* peut-elle néanmoins découvrir de meilleures actions ?
#

# %%
RIGHT = ACTION_NAMES.index("droite")
initial_actions = np.full(P.shape[0], RIGHT, dtype=int)

V_pi, actions_pi, log_pi = policy_iteration(
    P, R, gamma, grid["terminal_states"], initial_actions
)
policy_pi = deterministic_policy(actions_pi)

print("Nombre d'étapes d'amélioration :", len(log_pi))
print("(balayages d'évaluation, états modifiés) à chaque étape :")
for k, item in enumerate(log_pi, start=1):
    print(f"  étape {k}: {item}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
draw_values(V_pi, grid, "Valeur finale de Policy Iteration", axes[0])
draw_policy(policy_pi, grid, "Politique finale", axes[1])
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Question
#
# Observez le journal de l'algorithme. Pourquoi certaines évaluations demandent-elles beaucoup plus de balayages que d'autres ? Pourquoi la dernière étape modifie-t-elle exactement zéro état ?
#
# ### Expérience facultative — autre politique initiale
#
# Relancez *Policy Iteration* avec des actions initiales aléatoires :
#
# ```python
# rng = np.random.default_rng(0)
# random_actions = rng.integers(0, N_ACTIONS, size=P.shape[0])
# ```
#
# Comparez le nombre d'étapes d'amélioration et la politique finale. Deux politiques finales différentes peuvent-elles être toutes les deux optimales ?

# %% [markdown]
# # Partie 5 — Value Iteration
#
# *Value Iteration* vise directement la valeur optimale :
#
# $$
# v_{k+1}(s)=
# \max_a \sum_{s'}P(s'\mid s,a)
# \left[r(s,a,s')+\gamma v_k(s')\right].
# $$
#
# L'évaluation et l'amélioration ne sont plus deux phases séparées : la maximisation intervient à chaque balayage. Nous conservons quelques valeurs intermédiaires afin de visualiser la propagation de l'information.
#
# *Pour approfondir : notes RL1, §3.4 ; pour la comparaison avec Policy Iteration, §3.5.*
#
# ### Prédiction
#
# On initialise $v_0=0$. Après le premier balayage, quelles cases peuvent déjà « savoir » qu'un terminal est proche ? À quelle vitesse cette information peut-elle traverser la grille ?
#

# %%
def value_iteration(P, R, gamma, terminal_states,
                    theta=1e-8, max_sweeps=100_000, keep_history=False):
    V = np.zeros(P.shape[0])
    history = [V.copy()] if keep_history else None
    deltas = []

    for sweep in range(1, max_sweeps + 1):
        Q = action_values(V, P, R, gamma)
        V_new = Q.max(axis=1)
        V_new[terminal_states] = 0.0

        delta = np.max(np.abs(V_new - V))
        deltas.append(delta)
        V = V_new

        if keep_history:
            history.append(V.copy())

        if delta < theta:
            return V, sweep, np.array(deltas), history

    raise RuntimeError("Value Iteration n'a pas convergé.")


V_vi, n_sweeps_vi, deltas_vi, history_vi = value_iteration(
    P, R, gamma, grid["terminal_states"], keep_history=True
)

Q_vi = action_values(V_vi, P, R, gamma)
policy_vi = greedy_stochastic_policy(
    Q_vi, grid["terminal_states"]
)

print("Nombre de balayages :", n_sweeps_vi)

indices = sorted({0, 1, 2, 4, len(history_vi) - 1})

# Échelle de couleurs commune à tous les balayages.
vmin = min(V_k.min() for V_k in history_vi)
vmax = max(V_k.max() for V_k in history_vi)

fig, axes = plt.subplots(1, len(indices), 
                         figsize=(4 * len(indices), 4), constrained_layout=True)

for ax, k in zip(axes, indices):
    shown = draw_values(history_vi[k], grid, f"Balayage {k}", ax)
    shown.set_clim(vmin, vmax)

fig.colorbar(shown, ax=axes, shrink=0.8, label=r"Valeur $v_k(s)$")

plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
draw_values(V_vi, grid, "Valeur obtenue par Value Iteration", axes[0])
draw_policy(policy_vi, grid, "Politique gloutonne associée", axes[1])
plt.tight_layout()
plt.show()

best_vi = np.isclose(Q_vi, Q_vi.max(axis=1, keepdims=True), atol=1e-8, rtol=0)
nonterminal = np.ones(P.shape[0], dtype=bool)
nonterminal[grid["terminal_states"]] = False
pi_actions_are_optimal = np.all(
    best_vi[np.arange(P.shape[0])[nonterminal], actions_pi[nonterminal]]
)

print("Écart maximal entre les valeurs finales :",
      np.max(np.abs(V_pi - V_vi)))
print("Chaque action de la politique PI est optimale selon VI :",
      pi_actions_are_optimal)

# %% [markdown]
# ### Question
#
# Les valeurs finales de *Policy Iteration* et *Value Iteration* coïncident numériquement. Les figures de politiques ne sont pourtant pas identiques partout : *Policy Iteration* conserve une seule action, tandis que notre affichage de *Value Iteration* montre toutes les actions maximisantes. Expliquez pourquoi il n'y a pas de contradiction.
#
# ### Expérience facultative — seuil d'arrêt
#
# Comparez `theta=1e-2`, `theta=1e-6` et `theta=1e-10`. Mesurez :
#
# - le nombre de balayages ;
# - l'écart avec la valeur obtenue pour `theta=1e-10` ;
# - l'éventuel changement de politique gloutonne ;
# - si les résultats sont identiques pour les trois seuils, expliquez pourquoi en vous appuyant sur la propagation des valeurs.
#
# ### Question d'approfondissement — mises à jour asynchrones
#
# Notre code calcule $v_{k+1}$ entièrement à partir de $v_k$. Écrivez une variante qui modifie `V[s]` immédiatement au cours du balayage. Comparez sa vitesse de convergence et reliez cette variante aux méthodes asynchrones évoquées dans RL1, §3.5.3.

# %% [markdown]
# # Partie 6 — Labyrinthe
#
# Nous remplaçons maintenant la grille vide par un labyrinthe. Les cases égales à `1` sont des murs ; elles ne font pas partie de l'espace d'états. Une tentative d'entrer dans un mur laisse l'agent sur place.
#
# Le départ est en $(0,0)$ et le terminal en $(10,10)$.
#
# Le point essentiel est le suivant : **le MDP change, mais les algorithmes ne changent pas**. Nous réutiliserons directement `value_iteration`, `action_values` et les fonctions de construction de politique.

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

walls = set(map(tuple, np.argwhere(maze_array == 1)))
maze = build_gridworld(
    shape=maze_array.shape,
    terminal_coords={(10, 10)},
    walls=walls,
)

print("Nombre de cases accessibles :", len(maze["coords"]))
print("Forme de P pour le labyrinthe :", maze["P"].shape)
print("MDP valide :", check_mdp(maze))

empty_policy = np.zeros((len(maze["coords"]), N_ACTIONS))
draw_policy(empty_policy, maze, "Le labyrinthe : murs en noir, terminal T")
plt.show()

# %% [markdown]
# ### Prédiction
#
# Regardez le labyrinthe avant de lancer l'algorithme. Par quels passages la valeur du terminal devra-t-elle remonter ? Les valeurs de deux cases géométriquement proches seront-elles nécessairement proches ?
#

# %%
# Même algorithme que dans la partie 5 : seuls P, R et les terminaux changent.
V_maze, n_sweeps_maze, _, _ = value_iteration(
    maze["P"], maze["R"], gamma=0.9,
    terminal_states=maze["terminal_states"],
    keep_history=False,
)
Q_maze = action_values(V_maze, maze["P"], maze["R"], gamma=0.9)
policy_maze = greedy_stochastic_policy(Q_maze, maze["terminal_states"])

print("Convergence en", n_sweeps_maze, "balayages.")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
draw_values(V_maze, maze, "Valeur optimale dans le labyrinthe", axes[0])
draw_policy(policy_maze, maze, "Actions optimales", axes[1])
plt.tight_layout()
plt.show()


# %% [markdown]
# Dans la suite, la fonction `follow_deterministic_policy(...)` construit une trajectoire en suivant une politique déterministe à partir de l'état initial `start_state`. Dans chaque état `s`, elle sélectionne l'action `actions[s]`, puis détermine l'unique état suivant à partir de la ligne `P[s, actions[s], :]`. Comme la dynamique est déterministe, cette ligne contient un seul coefficient égal à $1$ ; sa position est récupérée avec `np.argmax`.
#
# Les états successivement visités sont ajoutés à la liste `path` jusqu'à ce qu'un état terminal soit atteint. Cette liste permet ensuite de représenter le chemin suivi dans le labyrinthe. Le paramètre `max_steps` est une limite de sécurité : il borne le nombre de déplacements ; une erreur est signalée si aucun terminal n’a été atteint dans cette limite.
#
# Cette fonction ne calcule pas la politique optimale ; elle sert uniquement à simuler et visualiser la politique déterministe obtenue précédemment. Elle repose sur l'hypothèse d'une dynamique déterministe : dans un environnement stochastique, il faudrait tirer l'état suivant selon la distribution $P(\cdot\mid s,a)$.

# %%
def follow_deterministic_policy(P, actions, start_state, terminal_states,
                                max_steps=1_000):
    # Suit une politique dans un environnement déterministe.
    terminal_set = set(terminal_states)
    path = [start_state]
    s = start_state

    if s in terminal_set:
        return path

    for _ in range(max_steps):
        s = int(np.argmax(P[s, actions[s]]))
        path.append(s)

        if s in terminal_set:
            return path

    raise RuntimeError(f"Le terminal n'a pas été atteint en {max_steps} déplacements.")


actions_maze = greedy_deterministic_actions(Q_maze, maze["terminal_states"])
start_state = state_of(maze, (0, 0))
path = follow_deterministic_policy(
    maze["P"], actions_maze, start_state, maze["terminal_states"]
)
path_coords = np.array([maze["coords"][s] for s in path])

background = np.where(maze_array == 1, 1.0, 0.0)
plt.figure(figsize=(7, 7))
plt.imshow(background, cmap="Greys", vmin=0, vmax=1)
plt.plot(path_coords[:, 1], path_coords[:, 0], "o-", color="tab:red", markersize=4)
plt.scatter(path_coords[0, 1], path_coords[0, 0], s=100, color="tab:blue", label="départ")
plt.scatter(path_coords[-1, 1], path_coords[-1, 0], s=100, color="tab:green", label="terminal")
plt.xticks(range(maze_array.shape[1]))
plt.yticks(range(maze_array.shape[0]))
plt.title(f"Un chemin optimal — {len(path) - 1} déplacements")
plt.legend()
plt.show()

# %% [markdown]
# ### Question de synthèse
#
# Établissez deux listes courtes :
#
# - ce qui a changé entre le GridWorld vide et le labyrinthe ;
# - ce qui est resté strictement identique.
#
# La distinction attendue est celle entre **le modèle de l'environnement** $(\mathcal S,\mathcal A,P,r)$ et **l'algorithme de planification** qui exploite ce modèle.
#
# ### Mini-projet facultatif — modifier le labyrinthe
#
# Modifiez le labyrinthe en ouvrant ou en fermant une case (sans bloquer les cases de départ et d'arrivée), reconstruisez le MDP, puis exécutez à nouveau les mêmes cellules du notebook.
#
# Avant l'exécution, prédisez si le chemin optimal deviendra plus court, plus long, inchangé ou impossible.
#
# ### Mini-projet facultatif — transitions stochastiques
#
# Modifiez le constructeur pour que l'action demandée soit exécutée avec probabilité $0{,}8$ et que l'agent dévie vers chacune des deux directions latérales avec probabilité $0{,}1$. Si une déviation rencontre un mur, l'agent reste sur place. Renseignez aussi `R` pour les nouvelles transitions possibles, conservez les terminaux absorbants et appliquez la règle de maintien sur place aux bords. Attention : la simulation précédente avec `argmax` n’est plus adaptée à cette extension.
#
# Vérifiez d'abord que
#
# $$
# \sum_{s'}P(s'\mid s,a)=1
# $$
#
# pour tout $(s,a)$, puis réutilisez *Value Iteration*. Les actions optimales sont-elles les mêmes que dans le cas déterministe ? La politique optimale choisit-elle toujours le chemin géométriquement le plus court ?
#
# <details>
# <summary>Indication</summary>
#
# Pour une action donnée, ajoutez trois contributions dans `P[s, a, :]` au lieu d'une seule. Si deux issues conduisent au même état — par exemple à cause d'un mur — leurs probabilités doivent s'additionner.
#
# </details>

# %% [markdown]
# # Bilan
#
# Nous avons suivi la chaîne complète
#
# $$
# (\mathcal S,\mathcal A,P,r)
# \longrightarrow v_\pi
# \longrightarrow q_\pi
# \longrightarrow \text{amélioration gloutonne}
# \longrightarrow \pi_*.
# $$
#
# *Policy Iteration* sépare nettement évaluation et amélioration. *Value Iteration* les entremêle dans chaque mise à jour de Bellman. Le labyrinthe confirme enfin le message central du TP : lorsque le modèle change, les mêmes algorithmes de programmation dynamique continuent de s'appliquer.
#
# Pour replacer ces deux méthodes dans un cadre commun, revoir **RL1, §3.5 — Generalized Policy Iteration**.
#

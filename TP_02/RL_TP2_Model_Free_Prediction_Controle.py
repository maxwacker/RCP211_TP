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
# # TP 2 — Apprentissage par renforcement sans modèle : prédiction et contrôle
#
# **Durée indicative : 2 h**  
# **Environnement :** Python, NumPy et Matplotlib
#
# ## Objectifs
#
# Ce TP met en pratique les méthodes étudiées dans les notes **RL2 — Prédiction et contrôle model-free**. Dans le TP1, l'algorithme connaissait le modèle $(P,r)$ et calculait des espérances exactes. Ici, l'agent ne connaît plus rien du modèle : il agit, observe ce qui se passe, et corrige ses estimations au fil de l'expérience. Nous allons :
#
# 1. construire un environnement dont l'agent ne voit que les états et les récompenses observés ;
# 2. estimer la fonction de valeur d'une politique fixée avec **Monte Carlo**, puis **TD(0)** (prédiction) ;
# 3. apprendre une politique avec **SARSA**, puis **Q-learning** (contrôle) ;
# 4. comparer *on-policy* et *off-policy* sur le problème de la **falaise** ;
# 5. réutiliser **sans les modifier** ces algorithmes dans le **labyrinthe** du TP1 ;
# 6. (facultatif, hors séance) reproduire l'exemple du **Random Walk** de Sutton et Barto.
#
# Comme au TP1, le parcours principal est entièrement codé : on peut lire, exécuter et interpréter le notebook du début à la fin sans être bloqué par une question de programmation.
#
# > **À propos des questions facultatives et des questions d'approfondissement.**  
# > Elles sont intéressantes et méritent le détour, mais elles ne sont pas nécessaires pour comprendre la suite. Si le temps manque, poursuivez le parcours principal et revenez-y lors d'une deuxième lecture.
#
# Les questions ordinaires sont courtes : essayez d'y répondre mentalement ou en quelques phrases avant d'exécuter la cellule suivante. Quand une cellule contient un bloc **« À MODIFIER »**, changez les valeurs, relancez la cellule et observez : c'est la meilleure façon de comprendre le rôle des paramètres.
#
# ### Le fil conducteur en quelques lignes
#
# Dans le TP1, la programmation dynamique calculait les valeurs en résolvant les équations de Bellman avec le modèle connu. Ici, les objets à estimer restent les mêmes ($v_\pi$, $q_\pi$, $q_*$), mais les espérances exactes sont remplacées par des **corrections à partir d'échantillons**. Toutes les méthodes du TP reposent sur le même schéma (notes RL2, Remarque 2.4) :
#
# $$
# \text{nouvelle estimation} = \text{ancienne estimation} + \text{pas} \times \text{erreur}.
# $$
#
# Elles ne diffèrent que par la **cible** vers laquelle on corrige l'estimation :
#
# | Méthode | Cible | Problème |
# |---|---|---|
# | Monte Carlo | retour complet $G_t$ | prédiction |
# | TD(0) | $R_{t+1}+\gamma V(S_{t+1})$ | prédiction |
# | SARSA | $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$ | contrôle, *on-policy* |
# | Q-learning | $R_{t+1}+\gamma \max_a Q(S_{t+1},a)$ | contrôle, *off-policy* |
#
# ### Carte du TP
#
# | Partie | Contenu | Durée indicative |
# |---|---|---|
# | 1 | L'agent ne voit plus le modèle : l'environnement `reset` / `step` | 10 min |
# | 2 | Prédiction : Monte Carlo et TD(0) sur le GridWorld | 40 min |
# | 3 | Contrôle : SARSA et Q-learning sur le GridWorld | 35 min |
# | 4 | La falaise : *on-policy* contre *off-policy* | 20 min |
# | 5 | Le labyrinthe du TP1 | 10 min |
# | 6 | Mini-projet facultatif : le Random Walk | hors séance |
#
# ### Une règle du jeu
#
# Les tableaux `P` et `R` du TP1 existent toujours, mais l'**agent n'a pas le droit de les lire**. Ils ne servent qu'à deux choses : simuler le monde (à l'intérieur de l'environnement) et calculer une **vérité terrain** avec les outils du TP1, uniquement pour mesurer l'erreur des estimations.
#
# ### Références dans RL2
#
# - §1.1–1.2 : pourquoi le *model-free*, cadre général ;
# - §2.1 : Monte Carlo (*first-visit*, moyenne incrémentale) ;
# - §2.2 : TD(0), comparaison avec Monte Carlo (§2.2.6) ;
# - §3.1 : contrôle, *Generalized Policy Iteration*, politique $\varepsilon$-gloutonne et GLIE ;
# - §3.2 : SARSA ; §3.3 : Q-learning, exemple de la falaise (§3.3.7) ;
# - Exemple 2.9 : le Random Walk.
#
# Les références vers RL1 (§3.2 évaluation de politique, §3.4 *Value Iteration*) servent à rappeler les outils du TP1 utilisés comme vérité terrain.

# %% [markdown]
# ## 0 — Imports et conventions
#
# Nous reprenons exactement les conventions du TP1 : NumPy et Matplotlib seulement, et les quatre actions toujours rangées dans l'ordre
#
# $$
# \mathcal A=\{\text{gauche},\text{haut},\text{droite},\text{bas}\}.
# $$
#
# Les cases sont repérées par $(i,j)$ : $i$ est l'indice de **ligne** (il augmente vers le bas), $j$ l'indice de **colonne**.

# %%
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
plt.rcParams["figure.figsize"] = (8, 6)

ACTION_NAMES = ("gauche", "haut", "droite", "bas")

# MOVES[a] = (di, dj) : déplacement de la case (i, j) pour l'action a.
MOVES = ((0, -1), (-1, 0), (0, 1), (1, 0))
N_ACTIONS = len(ACTION_NAMES)


# %% [markdown]
# ### Les outils du TP1
#
# La cellule suivante rassemble, **sans modification**, les outils du TP1 dont nous avons besoin : la construction du GridWorld (`build_gridworld`), les deux fonctions d'affichage (`draw_values`, `draw_policy`), l'évaluation de politique et *Value Iteration*. Nous les utiliserons pour deux choses seulement : construire les environnements et calculer la vérité terrain. Il suffit d'exécuter la cellule ; rien de nouveau n'est à lire ici.

# %% jupyter={"source_hidden": true}
# ---------- Outils du TP1 (inchangés) ---------

def build_gridworld(shape, terminal_coords, walls=None):
    # Construit P(s'|s,a) et r(s,a,s') pour une grille déterministe.
    n_rows, n_cols = shape
    walls = set() if walls is None else set(walls)
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


def state_of(env, coord):
    return env["coord_to_state"][coord]


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


def action_values(V, P, R, gamma):
    # q(s,a) = somme_s' P(s'|s,a) [ r(s,a,s') + gamma V(s') ]
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


def greedy_stochastic_policy(Q, terminal_states, atol=1e-10):
    # Répartit uniformément la probabilité entre les actions maximisantes.
    best = np.isclose(Q, Q.max(axis=1, keepdims=True), atol=atol, rtol=0)
    policy = best / best.sum(axis=1, keepdims=True)
    policy[terminal_states] = 0.0
    return policy


# %% [markdown]
# # Partie 1 — L'agent ne voit plus le modèle
#
# *Pour approfondir : notes RL2, §1.1 et §1.2.*
#
# Dans le TP1, les algorithmes lisaient directement les tableaux `P` et `R`. Dans ce TP, nous changeons de point de vue :
#
# - l'**environnement** garde `P` et `R` pour lui. Il sait *simuler* : étant donné un état $s$ et une action $a$, il tire l'état suivant $S'\sim P(\cdot\mid s,a)$ et renvoie la récompense $r(s,a,S')$ ;
# - l'**agent** n'a accès qu'à deux fonctions : `env_reset` (début d'un épisode) et `env_step` (exécuter une action).
#
# Les données d'apprentissage sont donc des **trajectoires**
#
# $$
# S_0,\;A_0,\;R_1,\;S_1,\;A_1,\;R_2,\;S_2,\;\ldots
# $$
#
# avec la convention de RL1 : $R_{t+1}=r(S_t,A_t,S_{t+1})$.
#
# Nous gardons le GridWorld $10\times10$ du TP1 : deux terminaux en $(0,0)$ et $(9,9)$, une récompense de $-1$ pour toute transition qui ne mène pas à un terminal, et $0$ quand on entre dans un terminal.
#
# **Point de départ des épisodes.** Le TP1 n'avait pas d'état initial : les valeurs étaient calculées partout à la fois. Ici, un épisode doit commencer quelque part. Pour le GridWorld, nous tirons l'état de départ **uniformément parmi les états non terminaux**, afin que toutes les cases soient visitées. Pour la falaise et le labyrinthe, le départ sera fixe.
#
# #### Explication du code
#
# - `prepare_env` complète le dictionnaire de l'environnement avec la liste des états de départ possibles et un tableau de booléens `is_terminal`. Elle ajoute aussi un paramètre `slip` (probabilité que l'environnement remplace l'action choisie par une action tirée au hasard) ; il vaut $0$ partout, sauf dans l'expérience facultative de la falaise glissante de la Partie 4.
# - `sample_index(probs, rng)` tire un indice $i$ avec la probabilité `probs[i]` : on tire un nombre $u$ uniforme dans $[0,1)$ et l'on s'arrête au premier indice où la somme cumulée des probabilités dépasse $u$. Le générateur `rng` est un générateur pseudo-aléatoire NumPy : fixer sa graine (`np.random.default_rng(0)`) rend les résultats reproductibles.
# - `env_reset` renvoie un état de départ. `env_step` renvoie la récompense, l'état suivant et un booléen `done` qui vaut `True` si l'état suivant est terminal. C'est **tout** ce que voit l'agent.

# %%
def prepare_env(env, start_states):
    # Ajoute à l'environnement ce qu'il faut pour le simuler.
    env["start_states"] = list(start_states)
    is_terminal = [False] * len(env["coords"])
    for s in env["terminal_states"]:
        is_terminal[s] = True
    env["is_terminal"] = is_terminal
    env["slip"] = 0.0
    return env


def sample_index(probs, rng):
    # Tire un indice i avec la probabilité probs[i].
    u = rng.random()
    total = 0.0
    last = len(probs) - 1
    for i in range(last):
        total += probs[i]
        if u < total:
            return i
    return last


def env_reset(env, rng):
    # Début d'un épisode : choisit un état de départ.
    starts = env["start_states"]
    return starts[int(rng.integers(len(starts)))]


def env_step(env, s, a, rng):
    # L'environnement exécute l'action a dans l'état s.
    # Il renvoie (récompense, état suivant, épisode terminé ?).
    if rng.random() < env["slip"]:
        a = int(rng.integers(N_ACTIONS))      # l'environnement "glisse"
    s_next = sample_index(env["P"][s, a], rng)   # seul l'environnement lit P
    r = float(env["R"][s, a, s_next])            # ... et R
    done = env["is_terminal"][s_next]
    return r, s_next, done


# %%
# Checking understanding on grid 4x4 

grid_4x4 = build_gridworld(
    shape=(4, 4),
    terminal_coords={(0, 0), (3, 3)}
)
check_mdp(grid_4x4)
terminal_set_4x4 = set(grid_4x4["terminal_states"])
print("terminal_set_4x4 :", terminal_set_4x4 ) # 0, 15
nonterminal_4x4 = [s for s in range(len(grid_4x4["coords"])) if s not in terminal_set_4x4]
print("nonterminal_4x4 :", nonterminal_4x4 ) # 1 ... 14

# %%
# STUDY of prepare_env(env, start_states):
env_4x4 = grid_4x4
start_states_4x4=nonterminal_4x4
env_4x4["start_states"] = list(start_states_4x4)
is_terminal_4x4 = [False] * len(env_4x4["coords"])
print("is_terminal_4x4 init : ", is_terminal_4x4)
for s in env_4x4["terminal_states"]:
        is_terminal_4x4[s] = True
print("is_terminal_4x4 set up : ", is_terminal_4x4)



# %%
# STUDY env_reset(env, rng) 
# env : pour starts(=nonterminal), rng
# rng : (pseudo) random number generator
# Studying expression : starts[int(rng.integers(len(starts)))]
print("starts on grid_4x4 : ", start_states_4x4) #Départs possibles d'un épsode : les non terminaux
rng42 = np.random.default_rng(42)
start_states_4x4[int(rng42.integers(len(start_states_4x4)))] # select randomly (uniform law) an index, then getting value : [index] ?
# How it work : 1) select index randomly, then getting associted value :
some_sparse_starts = [4, 8, 16, 32]
ramdom_index = int(rng42.integers(len(some_sparse_starts)))
some_sparse_starts[ramdom_index]

# %%
prepare_env(grid_4x4, start_states_4x4)
env_4x4 = grid_4x4
env_4x4["slip"]

# %%
# STUDY sample_index(probs, rng) 
# Tire un indice i avec la probabilité probs[i].
# Invoqué depuis env_step, par 
#s_next = sample_index(env["P"][s, a], rng)
probs = grid_4x4["P"][1, 2] # Ici : s=1 (2e case), a = 2 (droite) 
# -> [0, 0, 1, 0, 0, 0, ...] : Politque déterministe 2e case --droite--> 3e case
print("probs : ",  probs)

# sample_index(probs, rng), sur un exemple plus concis de probs (et non déterministe)
small_probs = [0.2, 0.0, 0.3, 0.5]
rng0 = np.random.default_rng(0)
# Tire un indice i avec la probabilité probs[i].
def sample_index_study(probs, rng):
    # Tire un indice i avec la probabilité probs[i].
    u = rng.random()
    print("u:", u)
    total = 0.0
    last = len(probs) - 1
    for i in range(last):
        total += probs[i]
        print("iter: ", i, "cumulated probs", total)
        if u < total:
            print("iter: ", i, "cumulated probs", total)
            return i
    return last
print("small_probs : ",  small_probs)
sample_index_study(small_probs, rng0)

# %%
# Vérification de "Tire un indice i avec la probabilité probs[i]."
# Si fait un grand nombre de tirage on doit trouvé des gréquence qui correspondent au probabilités initiales
small_probs = [0.2, 0.1, 0.3, 0.4]
N = 1000

freq = np.zeros(len(small_probs))
for t in range(N-1):
    i = sample_index(small_probs, rng0)
    freq[i]+=1
freq / N # C'est bien ça. 

# Explication géométrique de l'algo
# On découpe le segment [0,1] en segments de la taille des proba
# La probabilité que u tombe dans ce segment est bien la taille de ce segment
# .2 .1 .3  .4
#|--|-|---|----|
#0             1
# On itère donc sur les bornes des segments pour identifier dans lequel u "est tombé" 

# %%
# Etude de env_step(env, s, a, rng): 
# Renvoie la récompense, l'état suivant et un booléen `done` qui vaut `True` si l'état suivant est terminal. 
# C'est **tout** ce que voit l'agent.
def env_step_study(env, s, a, rng):
    # L'environnement exécute l'action a dans l'état s.
    # Il renvoie (récompense, état suivant, épisode terminé ?).
    if rng.random() < env["slip"]:
        a = int(rng.integers(N_ACTIONS))      # l'environnement "glisse"
    s_next = sample_index(env["P"][s, a], rng)   # seul l'environnement lit P
    r = float(env["R"][s, a, s_next])            # ... et R
    done = env["is_terminal"][s_next]
    return r, s_next, done

# Sur ex grid_44
env_4x4["slip"] # probabilité que l'environnement remplace l'action choisie par une action tirée au hasard (0.0 sauf pour ex falaise)
# Quand slip = 0, rng.random() n'est jamais vrai. Si slip > 0, on peut parfois changer l'action aléatoirement
# Donc :
# - On tire l'état suivant s_next par sample_dev sur P[s,a] (dont le résultat peut être non déterministe)
# - On obtient la récompense associée à cette transition par R[s, a, s_next]
# - On détermine si on a atteint un état terminal : done True|False

# %%
grid = build_gridworld(
    shape=(10, 10),
    terminal_coords={(0, 0), (9, 9)}
)
check_mdp(grid)

terminal_set = set(grid["terminal_states"])
nonterminal = [s for s in range(len(grid["coords"])) if s not in terminal_set]
prepare_env(grid, start_states=nonterminal)

# Une trajectoire obtenue en choisissant les actions au hasard.
rng = np.random.default_rng(0)
s = env_reset(grid, rng)

print("  t | S_t      | A_t    | R_{t+1} | S_{t+1}")
for t in range(8):
    a = int(rng.integers(N_ACTIONS))
    r, s_next, done = env_step(grid, s, a, rng)
    print(f"{t:3d} | {str(grid['coords'][s]):8s} | {ACTION_NAMES[a]:6s} | {r:6.1f}  | {grid['coords'][s_next]}")
    s = s_next
    if done:
        print("Épisode terminé.")
        break

# %% [markdown]
# ### Questions
#
# 1. Après chaque action, qu'**observe** l'agent ? Qu'est-ce qui lui reste **caché** ?
#
# <font color='blue'>
#     L'agent récupère le nouvel état issu de cette action issu de l'action, la récopense et s'il s'agit d'un état terminal.
#     L'agent ne sait rien d'autre. Ni les autres actions qui aurraient été possibles, ni leurs récompense, leur valeur (Q), ni la valeur du nouvel états (V), 
# </font>
#
# 2. Si l'on relance la cellule précédente, obtient-on la même trajectoire ? Que faudrait-il changer pour en obtenir une autre ?
#
# <font color='blue'>
#     Le random number generator (rng) est en fait pseudo random : Il génrère la même série de nombre pour un sel (salt) donné.
#     Pour obtenir une autre série de nombre, il faut changer de salt à l'initialisation du rng
# </font>
#
# ### À retenir
#
# - L'agent interagit avec l'environnement par `env_step` et ne voit ni `P` ni `R`.
# - Les données sont des trajectoires $(S_t,A_t,R_{t+1},S_{t+1})$, produites par la politique que l'agent exécute.
# - Fixer la graine du générateur pseudo-aléatoire rend une expérience reproductible ; la changer donne une autre réalisation.

# %% [markdown]
# # Partie 2 — Prédiction : estimer $v_\pi$ à partir d'épisodes
#
# *Pour approfondir : notes RL2, §2.1 (Monte Carlo) et §2.2 (TD(0)).*
#
# Dans un problème de **prédiction**, la politique $\pi$ est fixée et l'on veut estimer sa fonction de valeur sans connaître le modèle. Ici, $\pi$ est la **politique uniforme** du TP1 (chacune des quatre actions avec probabilité $1/4$) et $\gamma=0{,}9$. Rappelons que
#
# $$
# v_\pi(s)=\mathbb E_\pi\!\left[G_t\mid S_t=s\right],
# \qquad
# G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
# $$
#
# Deux idées, deux cibles pour corriger l'estimation $V(s)$ :
#
# - **Monte Carlo** attend la fin de l'épisode et utilise le retour complet $G_t$ ;
# - **TD(0)** n'attend pas : dès qu'une transition est observée, il utilise $R_{t+1}+\gamma V(S_{t+1})$, c'est-à-dire *une estimation déjà disponible* de la valeur de l'état suivant. On dit qu'il **bootstrappe**.
#
# ## Vérité terrain
#
# Pour savoir si nos estimations sont bonnes, nous calculons la valeur exacte $v_\pi$ avec `policy_evaluation` du TP1. **C'est le seul endroit où nous utilisons `P` et `R` dans cette partie** : l'agent, lui, ne verra jamais ce résultat.

# %%
gamma = 0.9
uniform_policy = np.full((len(grid["coords"]), N_ACTIONS), 1 / N_ACTIONS)

# Vérité terrain : utilise le modèle, uniquement pour mesurer l'erreur.
V_uniform, _, _ = policy_evaluation(
    uniform_policy, grid["P"], grid["R"], gamma, grid["terminal_states"]
)

draw_values(V_uniform, grid, r"Vérité terrain $v_\pi$ (politique uniforme, $\gamma=0{,}9$)")
plt.show()


# %% [markdown]
# ## 2.1 Jouer un épisode et calculer les retours
#
# *Notes RL2, §2.1.1 et §2.1.2.*
#
# La fonction `generate_episode` joue un épisode avec la politique donnée. L'épisode s'arrête dès qu'un état terminal est atteint, ou au plus tard après `max_steps` pas : on dit alors que l'épisode est **coupé**. Elle renvoie trois listes, les états $S_t$, les actions $A_t$ et les récompenses (rangées de sorte que `rewards[t]` contienne $R_{t+1}$), ainsi qu'un booléen `terminated` qui vaut `True` si l'épisode s'est terminé dans un état terminal et `False` s'il a été coupé.
#
# Dans les cellules d'expérience, la longueur maximale des épisodes est réglée par le paramètre `EP_SIZE`. Avec la valeur par défaut `EP_SIZE = 1000`, environ 99 % des épisodes atteignent un terminal avant la limite, et la coupure n'a pratiquement aucun effet. La section 2.6 étudie ce qui se passe avec des épisodes beaucoup plus courts.
#
# Une fois l'épisode terminé, on calcule tous les retours avec la récurrence
#
# $$
# G_t=R_{t+1}+\gamma\,G_{t+1},
# \qquad
# G_T=0,
# $$
#
# **de la fin vers le début** : c'est ce que fait `compute_returns`.

# %%
def generate_episode(env, policy, rng, max_steps=10_000):
    # Joue un épisode : il s'arrête dans un état terminal, ou au plus tard après max_steps pas.
    # rewards[t] contient R_{t+1}.
    # terminated vaut True si un état terminal a été atteint, False si l'épisode a été coupé.
    s = env_reset(env, rng)
    states, actions, rewards = [], [], []
    terminated = False

    for _ in range(max_steps):
        a = sample_index(policy[s], rng)
        r, s_next, done = env_step(env, s, a, rng)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s_next
        if done:
            terminated = True
            break

    return states, actions, rewards, terminated 
# On a la liste des états, des actions des transitions et leurs récompenses

def compute_returns(rewards, gamma):
    # G_t = R_{t+1} + gamma * G_{t+1}, calculé de la fin vers le début.
    returns = [0.0] * len(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


# %%
rng = np.random.default_rng(1)
states, actions, rewards, terminated = generate_episode(grid, uniform_policy, rng)
returns = compute_returns(rewards, gamma) # Somme fin->début

T = len(rewards)
G0_direct = sum(gamma**k * rewards[k] for k in range(T)) # G_0 : Somme de récompenses avec "amenuissement" gamma sur les rewards

print("Longueur de l'épisode T :", T, "| terminé dans un état terminal :", terminated)
print("Premier état :", grid["coords"][states[0]])
print(f"G_0 par la récurrence : {returns[0]:.6f}")
print(f"G_0 par la somme directe : {G0_direct:.6f}")

# Longueur moyenne des épisodes sous la politique uniforme.
lengths = [len(generate_episode(grid, uniform_policy, rng)[0]) for _ in range(200)]
print("Longueur moyenne sur 200 épisodes :", np.mean(lengths))


# %% [markdown]
# ### Questions
#
# 1. Pourquoi calcule-t-on les retours de la **fin vers le début** ? Combien d'opérations cela coûte-t-il, comparé au calcul séparé de chaque $G_t$ par la somme directe ?
#
# <font color='blue'>
# On calcule le retour de la fin vers le début pour profiter de la récurence dans le calcul G_t = R_t+1 + gamma * G_t+1. Cela nécessite un seul parcours des rewards : O(T)
#
# La somme directe est une somme sur les T rewards, dont chaque terme est un 'sous-parcours' des rewards de de taille (T-1),(T-2), ... , 0 : Soit O(T**2).
#     
# </font>
#
#
# 2. La longueur moyenne d'un épisode est de l'ordre de deux cents pas. Qu'est-ce que cela annonce pour les retours Monte Carlo ? (Pensez à la variance, notes §2.1.5.)
#
# <font color='blue'>
# Variance / Monte Carlo : chaque étape d'un épisode est issu d'un tirage aléaoite. Les épisodes sur un même point de départ peuvent donc suivrent des chemins différents et les résulats obtenu d'estimation de V peuvent donc montrer une forte variance, d'autant plus grande que le chemin est long. La variance des résults augmente donc selon la taille des chemins. Il faut donc une grand quantité d'épisodes pour obtenir une estimation correcte
# </font>

# %% [markdown]
# ## 2.2 Monte Carlo *first-visit*
#
# *Notes RL2, Définition 2.1, §2.1.2 et §2.1.4.*
#
# Pour chaque état $s$, l'évaluation Monte Carlo *first-visit* ne retient, dans chaque épisode, que le retour observé après la **première** visite de $s$. Au lieu de stocker tous les retours pour en faire la moyenne, nous utilisons la forme incrémentale avec un pas constant $\alpha$ :
#
# $$
# V(s)\leftarrow V(s)+\alpha\bigl(G_t-V(s)\bigr).
# $$
#
# C'est exactement le schéma « nouvelle estimation = ancienne estimation + pas × erreur » du début du TP. Un pas constant oublie progressivement les retours anciens, contrairement à la moyenne empirique ($\alpha=1/N(s)$).
#
# Pour mesurer l'erreur, nous utiliserons l'erreur quadratique moyenne (RMS) sur les états non terminaux :
#
# $$
# \mathrm{RMS}(V)=\sqrt{\frac1{|\mathcal S_{\text{nt}}|}\sum_{s\in\mathcal S_{\text{nt}}}\bigl(V(s)-v_\pi(s)\bigr)^2}.
# $$
#
# #### Explication du code
#
# - `mc_prediction` joue `n_episodes` épisodes. Pour chacun, elle calcule les retours, puis parcourt l'épisode **dans l'ordre** et ne met à jour un état que s'il n'a pas déjà été vu dans cet épisode (ensemble `seen`).
# - La table `V` est initialisée à `v_init` (par défaut $0$) ; les terminaux valent toujours $0$.
# - Le paramètre `max_steps` est transmis à `generate_episode`. Si un épisode est coupé, les retours sont calculés avec les seules récompenses observées : tout se passe comme si le dernier état atteint était terminal ($G_T=0$).
# - Après chaque épisode, elle enregistre l'erreur RMS : la fonction renvoie la table `V` finale et la courbe d'erreur.

# %%
def rms_error(X, X_ref, states):
    # Erreur quadratique moyenne entre X et X_ref, restreinte aux états donnés.
    diff = X[states] - X_ref[states]
    return float(np.sqrt(np.mean(diff ** 2)))


def mc_prediction(env, policy, gamma, alpha, n_episodes, rng,
                  V_ref, nonterminal, v_init=0.0, max_steps=10_000):
    # Monte Carlo first-visit, avec pas constant alpha.
    V = np.full(len(env["coords"]), v_init)
    V[env["terminal_states"]] = 0.0
    errors = []

    for k in range(n_episodes):
        ## On gérère un épisode : En collectant la séquences des états visités 
        ## les actions et récompenses des transitions, et l'info de terminal atteint
        states, actions, rewards, terminated = generate_episode(env, policy, rng, max_steps)
        # Si l'épisode a été coupé, le retour s'arrête à la coupure (comme si G_T = 0).
        returns = compute_returns(rewards, gamma) ## Gt cummulés sur chaque transition

        seen = set()
        for t in range(len(states)):
            s = states[t]
            if s not in seen:                        # première visite de s
                seen.add(s)
                V[s] += alpha * (returns[t] - V[s]) ## On 'corrige' l'estimation précédante  SUR CHAQUE ETAT VISITE, MAIS 1 seul fois

        errors.append(rms_error(V, V_ref, nonterminal))

    return V, np.array(errors)


# %% [markdown]
# ### Question
#
# Dans `mc_prediction`, pourquoi faut-il attendre la **fin de l'épisode** avant de mettre `V` à jour ? Que se passerait-il sur une tâche continue, sans état terminal ? (Notes §2.1.5.)
#
# ## 2.3 TD(0)
#
# *Notes RL2, Définitions 2.5 et 2.6, §2.2.2 et §2.2.3.*
#
# TD(0) corrige $V(S_t)$ **à chaque transition**, sans attendre la fin de l'épisode. La cible est
#
# $$
# Y_t=R_{t+1}+\gamma V(S_{t+1}),
# $$
#
# l'erreur TD est $\delta_t=Y_t-V(S_t)$, et la mise à jour est
#
# $$
# V(S_t)\leftarrow V(S_t)+\alpha\,\delta_t.
# $$
#
# Si $S_{t+1}$ est terminal, il ne reste plus rien à recevoir : on pose $V(S_{t+1})=0$, et la cible vaut simplement $R_{t+1}$ (notes, « Actualisation et terminaison »).
#
# #### Explication du code
#
# La boucle principale reprend la structure de l'environnement : choisir une action avec la politique, observer `r, s_next, done`, former la cible, corriger `V[s]`, puis passer à l'état suivant. Contrairement à Monte Carlo, **aucune liste d'épisode n'est conservée** : la mise à jour n'utilise que la transition qui vient d'être observée.
#
# Si l'épisode est coupé après `max_steps` pas, rien de particulier n'est à faire : la dernière mise à jour a utilisé la cible $R_{t+1}+\gamma V(S_{t+1})$, avec l'estimation courante de $V(S_{t+1})$, et l'épisode suivant commence.

# %%
def td0_prediction(env, policy, gamma, alpha, n_episodes, rng,
                   V_ref, nonterminal, v_init=0.0, max_steps=10_000):
    # TD(0) en ligne : une mise à jour à chaque transition.
    V = np.full(len(env["coords"]), v_init)
    V[env["terminal_states"]] = 0.0
    errors = []

    for k in range(n_episodes):
        s = env_reset(env, rng)

        for t in range(max_steps):
            a = sample_index(policy[s], rng)
            r, s_next, done = env_step(env, s, a, rng)

            if done:
                target = r                          # pas de valeur future
            else:
                target = r + gamma * V[s_next]      # bootstrap sur V(S_{t+1})

            V[s] += alpha * (target - V[s])         # alpha * delta_t
            s = s_next
            if done:
                break
        # Si la boucle s'arrête après max_steps pas, l'épisode est simplement coupé.

        errors.append(rms_error(V, V_ref, nonterminal))

    return V, np.array(errors)


# %% [markdown]
# ### Questions
#
# 1. Pourquoi la cible vaut-elle $R_{t+1}$ seul quand l'état suivant est terminal ?
# 2. Au début de l'apprentissage, $V\equiv0$ alors que les vraies valeurs sont proches de $-10$ loin des terminaux. Que vaut la cible TD d'un état éloigné des terminaux ? Est-elle « juste » ? Quel effet cela a-t-il sur les premières mises à jour ? (Notes §2.2.6 : *biais*.)
#
# ## 2.4 Premier essai
#
# ### Prédiction
#
# Les deux méthodes démarrent avec $V\equiv 0$, reçoivent le même nombre d'épisodes et le même pas $\alpha$. Avant de lancer la cellule suivante : **après 100 épisodes**, laquelle des deux estimations sera la plus proche de la vérité terrain ? Sur quelles cases l'erreur restera-t-elle la plus grande ? Notez votre pari.

# %%
# >>> À MODIFIER (essayez d'autres valeurs, puis relancez la cellule)
ALPHA = 0.1          # pas d'apprentissage
N_EPISODES = 100     # nombre d'épisodes
EP_SIZE = 1000       # longueur maximale d'un épisode (voir la section 2.6)
SEED = 0             # graine du générateur pseudo-aléatoire
# <<<

rng = np.random.default_rng(SEED)
V_mc, errors_mc = mc_prediction(
    grid, uniform_policy, gamma, ALPHA, N_EPISODES, rng, V_uniform, nonterminal,
    max_steps=EP_SIZE
)
rng = np.random.default_rng(SEED)
V_td, errors_td = td0_prediction(
    grid, uniform_policy, gamma, ALPHA, N_EPISODES, rng, V_uniform, nonterminal,
    max_steps=EP_SIZE
)

maps = [
    (V_uniform, r"Vérité terrain $v_\pi$"),
    (V_mc, f"Monte Carlo après {N_EPISODES} épisodes"),
    (V_td, f"TD(0) après {N_EPISODES} épisodes"),
]
vmin = min(values.min() for values, _ in maps)
vmax = max(values.max() for values, _ in maps)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
for ax, (values, title) in zip(axes, maps):
    shown = draw_values(values, grid, title, ax)
    shown.set_clim(vmin, vmax)
fig.colorbar(shown, ax=axes, shrink=0.8, label="Valeur")
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(errors_mc, label="Monte Carlo")
plt.plot(errors_td, label="TD(0)")
plt.xlabel("Épisode")
plt.ylabel("Erreur RMS")
plt.title(rf"Une seule exécution ($\alpha={ALPHA}$)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"Erreur RMS finale : MC = {errors_mc[-1]:.2f}, TD(0) = {errors_td[-1]:.2f}")

# %% [markdown]
# ### Questions
#
# 1. Votre pari était-il juste ? Décrivez ce que vous voyez sur les cartes : quelles cases sont les mieux estimées ? Les moins bien estimées ?
# 2. Une seule exécution dépend beaucoup de la graine. Changez `SEED` et relancez : le classement des deux méthodes change-t-il ?
#
# ## 2.5 Moyenner sur plusieurs exécutions
#
# Une seule courbe est trop bruitée pour conclure. Nous répétons donc l'expérience avec $10$ graines différentes et nous traçons l'erreur **moyenne**, pour deux valeurs du pas : un petit pas ($\alpha=0{,}05$) et un grand pas ($\alpha=0{,}2$). La cellule prend une dizaine de secondes.

# %%
# >>> À MODIFIER
ALPHAS = (0.05, 0.2)    # les pas comparés
N_RUNS = 10             # nombre d'exécutions moyennées
N_EPISODES = 200        # épisodes par exécution
EP_SIZE = 1000          # longueur maximale d'un épisode (voir la section 2.6)
# <<<

fig, axes = plt.subplots(1, len(ALPHAS), figsize=(12, 4), sharey=True)

for ax, alpha in zip(axes, ALPHAS):
    curves_mc, curves_td = [], []
    for run in range(N_RUNS):
        rng = np.random.default_rng(run)
        _, errors = mc_prediction(
            grid, uniform_policy, gamma, alpha, N_EPISODES, rng, V_uniform, nonterminal,
            max_steps=EP_SIZE
        )
        curves_mc.append(errors)

        rng = np.random.default_rng(run)
        _, errors = td0_prediction(
            grid, uniform_policy, gamma, alpha, N_EPISODES, rng, V_uniform, nonterminal,
            max_steps=EP_SIZE
        )
        curves_td.append(errors)

    ax.plot(np.mean(curves_mc, axis=0), label="Monte Carlo")
    ax.plot(np.mean(curves_td, axis=0), label="TD(0)")
    ax.set_title(rf"$\alpha={alpha}$ (moyenne de {N_RUNS} exécutions)")
    ax.set_xlabel("Épisode")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Erreur RMS")
axes[0].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Questions
#
# 1. Dans la première phase de l'apprentissage, quelle méthode réduit l'erreur le plus vite ? Laquelle semble atteindre le plateau le plus bas avec le grand pas ?
# 2. Pourquoi un pas $\alpha$ plus grand accélère-t-il le début de l'apprentissage, mais relève-t-il le niveau de bruit final ?
# 3. Conclure ici « TD(0) est meilleur que Monte Carlo » serait-il justifié ? Relisez le tableau de la §2.2.6 des notes (biais, variance), et gardez cette question en tête pour la section 2.6 et pour le mini-projet de la Partie 6.
#
# ### Expérience facultative — le rôle du pas et des épisodes
#
# Modifiez `ALPHAS` (par exemple `(0.02, 0.5)`) puis `N_EPISODES`, et relancez la cellule précédente. Prédisez d'abord : que devient l'écart entre les deux méthodes avec un très petit pas ? Avec un très grand pas ?

# %% [markdown]
# ## 2.6 Épisodes coupés : quand TD(0) devient indispensable
#
# *Notes RL2, §2.1.5 et §2.2.6.*
#
# Jusqu'ici, presque tous les épisodes atteignaient un état terminal. Dans beaucoup de problèmes réels, ce n'est pas le cas : l'espace d'états est grand, les terminaux sont difficiles à atteindre, ou la tâche n'a tout simplement pas de fin. On doit alors **couper** les épisodes après un nombre fixé de pas, ici `EP_SIZE`.
#
# Les deux méthodes réagissent très différemment à la coupure :
#
# - **Monte Carlo** a besoin du retour complet $G_t$. Sur un épisode coupé, il ne dispose que des récompenses observées jusqu'à la coupure. Notre `mc_prediction` fait le choix le plus simple : elle calcule le retour avec ces seules récompenses, comme si le dernier état atteint était terminal ($G_T=0$).
# - **TD(0)** n'utilise qu'une transition à la fois. Couper l'épisode signifie seulement que certaines transitions ne sont pas jouées ; chaque mise à jour reste de la forme $R_{t+1}+\gamma V(S_{t+1})$.
#
# Pour que la comparaison soit équitable, les deux méthodes reçoivent **exactement les mêmes épisodes** (même graine à chaque exécution). Avec des épisodes courts, il en faut beaucoup plus : gardez le produit `N_EPISODES × EP_SIZE` autour de $40\,000$ pas au total, soit à peu près le nombre de pas de 200 épisodes complets. La cellule prend quelques secondes.
#
# ### Prédiction
#
# Avec `EP_SIZE = 20`, les épisodes sont environ dix fois plus courts qu'un épisode complet moyen. Laquelle des deux méthodes sera la plus gênée ? Son erreur va-t-elle continuer à diminuer si l'on augmente le nombre d'épisodes ? Notez votre pari.

# %%
# >>> À MODIFIER (essayez d'autres valeurs, puis relancez la cellule)
EP_SIZE = 20          # longueur maximale d'un épisode
N_EPISODES = 2000     # gardez N_EPISODES * EP_SIZE autour de 40 000
ALPHA = 0.2           # pas d'apprentissage
N_RUNS = 5            # nombre d'exécutions moyennées
# <<<

# Part des épisodes qui atteignent un état terminal avant la coupure.
rng = np.random.default_rng(0)
n_terminated = 0
for _ in range(1000):
    _, _, _, terminated = generate_episode(grid, uniform_policy, rng, EP_SIZE)
    if terminated:
        n_terminated += 1
print(f"Avec EP_SIZE = {EP_SIZE}, {n_terminated / 10:.0f} % des épisodes atteignent un état terminal.")

curves_mc, curves_td = [], []
for run in range(N_RUNS):
    rng = np.random.default_rng(run)
    _, errors = mc_prediction(
        grid, uniform_policy, gamma, ALPHA, N_EPISODES, rng, V_uniform, nonterminal,
        max_steps=EP_SIZE
    )
    curves_mc.append(errors)

    rng = np.random.default_rng(run)
    _, errors = td0_prediction(
        grid, uniform_policy, gamma, ALPHA, N_EPISODES, rng, V_uniform, nonterminal,
        max_steps=EP_SIZE
    )
    curves_td.append(errors)

mean_mc = np.mean(curves_mc, axis=0)
mean_td = np.mean(curves_td, axis=0)

plt.figure(figsize=(8, 4.5))
plt.plot(mean_mc, label="Monte Carlo")
plt.plot(mean_td, label="TD(0)")
plt.xlabel("Épisode")
plt.ylabel("Erreur RMS")
plt.title(rf"Épisodes coupés à {EP_SIZE} pas ($\alpha={ALPHA}$, moyenne de {N_RUNS} exécutions)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"Erreur RMS finale : MC = {mean_mc[-1]:.2f}, TD(0) = {mean_td[-1]:.2f}")

# %% [markdown]
# ### Questions
#
# 1. Votre pari était-il juste ? Laquelle des deux courbes cesse de descendre, et à quel niveau ?
# 2. Considérons un état loin des terminaux, dont la vraie valeur est proche de $-10$, et visité pour la première fois $k$ pas avant la coupure. Montrez que le retour tronqué utilisé par Monte Carlo vaut $-(1+\gamma+\cdots+\gamma^{k-1})=-10\,(1-0{,}9^{k})$. Que vaut-il pour $k=5$ ? Pourquoi cette erreur ne diminue-t-elle pas quand on ajoute des épisodes ? S'agit-il d'un problème de biais ou de variance ?
# 3. Pourquoi TD(0) n'est-il pas gêné par la coupure ? Que « suppose » sa dernière cible sur la suite de l'épisode, comparée à celle de Monte Carlo ?
# 4. Relancez la cellule avec `EP_SIZE = 50` (`N_EPISODES = 800`), `EP_SIZE = 100` (`N_EPISODES = 400`), puis `EP_SIZE = 200` (`N_EPISODES = 200`). **Prédisez d'abord** comment évolue le niveau final de Monte Carlo. Le biais disparaît-il complètement dès que les épisodes sont assez longs ? Quels états restent concernés ?
# 5. D'après ces expériences, dans quelles situations TD(0) est-il préférable à Monte Carlo, voire indispensable ? Pensez à la longueur des épisodes, à l'accessibilité des terminaux et aux tâches sans fin.
# 6. Dans cette expérience, les épisodes partent d'une case tirée au hasard : certains partent donc près d'un terminal. Que se passerait-il, pour les deux méthodes, si tous les épisodes partaient de la case $(4,4)$ avec `EP_SIZE = 5` ?

# %% [markdown]
# ### Question d'approfondissement — pourquoi TD est-il moins bruité ?
#
# Une cible de Monte Carlo est un retour complet, somme de centaines de récompenses aléatoires ; une cible TD ne contient qu'**une** récompense et une valeur déjà estimée. La cellule suivante mesure la dispersion des deux cibles pour la case $(4,4)$, en donnant à TD la **vraie** valeur $v_\pi(S_{t+1})$ : on isole ainsi l'effet de la variance, sans celui du biais. (Elle prend quelques secondes.)

# %%
s_test = state_of(grid, (4, 4))
rng = np.random.default_rng(0)
mc_targets, td_targets = [], []

for _ in range(2000):
    # Cible Monte Carlo : un retour complet depuis s_test.
    s = s_test
    G = 0.0
    discount = 1.0
    while True:
        a = sample_index(uniform_policy[s], rng)
        r, s_next, done = env_step(grid, s, a, rng)
        G += discount * r
        discount *= gamma
        s = s_next
        if done:
            break
    mc_targets.append(G)

    # Cible TD(0) : une seule transition, puis la vraie valeur de l'état suivant.
    a = sample_index(uniform_policy[s_test], rng)
    r, s_next, done = env_step(grid, s_test, a, rng)
    td_targets.append(r if done else r + gamma * V_uniform[s_next])

print(f"v_pi(4,4) exacte        : {V_uniform[s_test]:.3f}")
print(f"Cible MC  : moyenne {np.mean(mc_targets):.3f}, écart-type {np.std(mc_targets):.3f}")
print(f"Cible TD  : moyenne {np.mean(td_targets):.3f}, écart-type {np.std(td_targets):.3f}")

# %% [markdown]
# Les deux moyennes coïncident-elles avec $v_\pi(4,4)$ ? Comparez les écarts-types. Que cela implique-t-il pour le bruit des mises à jour ? Pourquoi, malgré cela, TD(0) n'était-il pas gagnant d'emblée dans l'expérience de la section 2.5 ?
#
# ### Micro-projet facultatif — *every-visit*
#
# La variante **every-visit** (notes, Définition 2.3) met à jour l'état à **chaque** visite dans l'épisode, pas seulement à la première.
#
# 1. Écrivez `mc_every_visit_prediction`, avec la même signature que `mc_prediction`. Elle ne diffère que par quelques lignes.
# 2. Exécutez la cellule de contrôle ci-dessous : elle compare les deux variantes en moyenne sur plusieurs exécutions.
# 3. **Prédisez d'abord** : laquelle démarre le plus vite ? Laquelle a le plateau le plus bas ? Pourquoi les retours des visites successives d'un même état dans un même épisode sont-ils *corrélés* (notes §2.1.4) ?
#
# <details>
# <summary>Indication</summary>
#
# Copiez `mc_prediction` et supprimez le test qui vérifie si `s` est déjà dans `seen` : toutes les visites mettent alors à jour `V[s]`. L'ensemble `seen` devient inutile.
#
# </details>

# %%
# Point de contrôle (à exécuter après avoir défini mc_every_visit_prediction)
if "mc_every_visit_prediction" not in globals():
    print("Définissez d'abord mc_every_visit_prediction (voir l'indication).")
else:
    ALPHA, N_RUNS, N_EPISODES = 0.1, 10, 200
    curves_first, curves_every = [], []
    for run in range(N_RUNS):
        rng = np.random.default_rng(run)
        _, errors = mc_prediction(
            grid, uniform_policy, gamma, ALPHA, N_EPISODES, rng, V_uniform, nonterminal
        )
        curves_first.append(errors)
        rng = np.random.default_rng(run)
        _, errors = mc_every_visit_prediction(
            grid, uniform_policy, gamma, ALPHA, N_EPISODES, rng, V_uniform, nonterminal
        )
        curves_every.append(errors)

    plt.figure(figsize=(7, 4))
    plt.plot(np.mean(curves_first, axis=0), label="first-visit")
    plt.plot(np.mean(curves_every, axis=0), label="every-visit")
    plt.xlabel("Épisode")
    plt.ylabel("Erreur RMS")
    plt.title(rf"Monte Carlo : first-visit et every-visit ($\alpha={ALPHA}$)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# %% [markdown]
# ### À retenir
#
# - **Monte Carlo** utilise le retour complet $G_t$ : cible sans biais, mais très bruitée, et mise à jour seulement en fin d'épisode.
# - **TD(0)** utilise $R_{t+1}+\gamma V(S_{t+1})$ : mise à jour à chaque transition et cible beaucoup moins dispersée, mais qui s'appuie sur une estimation encore fausse au début (*biais* transitoire).
# - Il n'y a pas de gagnant universel. Ici, l'initialisation $V\equiv0$ est loin de la vérité et les épisodes sont longs : le comportement dépend du problème, de $\alpha$ et de l'initialisation.
# - Monte Carlo a besoin d'épisodes qui se terminent : sur des épisodes coupés, ses retours sont tronqués et ses estimations restent biaisées, quel que soit le nombre d'épisodes. TD(0) apprend à partir de chaque transition et ne souffre pas de la coupure. C'est là qu'il devient indispensable : épisodes très longs, terminaux rarement atteints, tâches sans fin.
# - Le pas $\alpha$ règle un compromis entre vitesse d'apprentissage et bruit résiduel.

# %% [markdown]
# # Partie 3 — Contrôle : SARSA et Q-learning
#
# *Pour approfondir : notes RL2, §3.1 (contrôle, GPI, $\varepsilon$-gloutonne), §3.2 (SARSA), §3.3.5 (Q-learning).*
#
# En **contrôle**, la politique n'est plus donnée : on veut la construire. Sans modèle, évaluer une action en regardant l'état suivant est impossible ; on estime donc directement les valeurs d'**état-action** $Q(s,a)$, ce qui permet de comparer les actions sans connaître $P$.
#
# Le schéma est celui de la *Generalized Policy Iteration* du TP1 (RL1, §3.5), mais approximé et fondé sur des échantillons : on **agit** avec une politique déduite de $Q$, on **met à jour** $Q$ à chaque transition, et la politique change en même temps que $Q$.
#
# **Quelle politique pour collecter les données ?** Une politique purement gloutonne risquerait de ne jamais réessayer une action sous-estimée au départ. On utilise donc une politique $\varepsilon$-gloutonne (notes §3.1.2) : avec probabilité $1-\varepsilon$ on choisit une action de plus grande valeur estimée, et avec probabilité $\varepsilon$ une action tirée uniformément parmi toutes les actions. Au début, $Q\equiv 0$ : toutes les actions sont à égalité, et la politique déduite de $Q$ est donc une politique **aléatoire**, qui devient peu à peu meilleure au fil des mises à jour.
#
# Pour simplifier, nous prendrons $\varepsilon$ **constant** (ce qui sort du cadre strict de GLIE, notes §3.1.3, mais suffit pour comprendre).
#
# ## Vérité terrain
#
# Comme en Partie 2, nous calculons la solution exacte avec les outils du TP1, **uniquement pour mesurer l'erreur** : `value_iteration` donne $v_*$, et `action_values` en déduit $q_*$.

# %%
V_star, _, _, _ = value_iteration(
    grid["P"], grid["R"], gamma, grid["terminal_states"]
)
Q_star = action_values(V_star, grid["P"], grid["R"], gamma)

print("Forme de Q_star :", Q_star.shape, "(états, actions)")


# %% [markdown]
# ## Quelques outils de mesure
#
# Ces trois petites fonctions servent à analyser les résultats, pas à apprendre :
#
# - `moving_average` lisse une courbe (moyenne sur `window` valeurs consécutives) ;
# - `greedy_is_optimal` donne la fraction des états où l'action gloutonne de $Q$ est une action optimale pour la référence `Q_ref` ;
# - `greedy_path` suit la politique gloutonne de $Q$ depuis un état, en lisant `P` : c'est un outil d'**affichage**, l'agent ne s'en sert pas.

# %%
def moving_average(x, window):
    # Moyenne glissante sur `window` valeurs consécutives.
    window = min(window, len(x))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def greedy_is_optimal(Q, Q_ref, states, atol=1e-8):
    # Fraction des états où l'action gloutonne de Q est optimale pour Q_ref.
    count = 0
    for s in states:
        a = int(np.argmax(Q[s]))
        if Q_ref[s, a] >= Q_ref[s].max() - atol:
            count += 1
    return count / len(states)


def greedy_path(env, Q, start, max_steps=500):
    # Liste des états visités en suivant l'action gloutonne de Q (outil d'affichage).
    path = [start]
    s = start
    for _ in range(max_steps):
        if env["is_terminal"][s]:
            break
        a = int(np.argmax(Q[s]))
        s = int(np.argmax(env["P"][s, a]))
        path.append(s)
    return path


# %% [markdown]
# ## 3.1 La politique $\varepsilon$-gloutonne
#
# `epsilon_greedy(Q, s, epsilon, rng)` renvoie l'action à exécuter dans l'état `s` :
#
# - avec probabilité `epsilon`, une action tirée au hasard parmi toutes les actions ;
# - sinon, une action de valeur maximale. **S'il y a des ex aequo, on tire au hasard parmi eux.**

# %%
def epsilon_greedy(Q, s, epsilon, rng):
    n_actions = Q.shape[1]
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))            # exploration

    q_max = max(Q[s])
    best = [a for a in range(n_actions) if Q[s, a] >= q_max - 1e-10]
    return best[int(rng.integers(len(best)))]          # exploitation, ex aequo tirés au hasard


# %% [markdown]
# ### Questions
#
# 1. Pourquoi tirer au hasard parmi les actions ex aequo, au lieu de prendre simplement la première (ce que ferait `np.argmax`) ? Que se passerait-il au début, quand $Q\equiv0$ ?
# 2. Dans ce GridWorld, toutes les récompenses sont $\le 0$ et $Q$ est initialisé à $0$. Quand l'agent vient d'essayer une action dans un état, sa valeur devient négative, tandis que les actions non essayées restent à $0$. Quel effet cela a-t-il sur l'exploration, même avec $\varepsilon=0$ ? (On parle d'*initialisation optimiste*.)
#
# ## 3.2 SARSA
#
# *Notes RL2, Définition 3.1 et §3.2.2–3.2.3.*
#
# SARSA est l'analogue de TD(0) pour les valeurs d'état-action. Son nom vient des cinq éléments utilisés dans la mise à jour : $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$.
#
# $$
# Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\,\delta_t,
# \qquad
# \delta_t=R_{t+1}+\gamma\,Q(S_{t+1},A_{t+1})-Q(S_t,A_t),
# $$
#
# où $A_{t+1}$ est l'action **effectivement choisie** par la politique courante (ici $\varepsilon$-gloutonne). Si $S_{t+1}$ est terminal, $\delta_t=R_{t+1}-Q(S_t,A_t)$ : pas d'action suivante.
#
# C'est une méthode **on-policy** : la cible contient l'action réellement choisie, exploration comprise. SARSA apprend donc la valeur de la politique $\varepsilon$-gloutonne *qu'il suit*.
#
# ## 3.3 Q-learning
#
# *Notes RL2, Définition 3.2 et §3.3.5.*
#
# Q-learning ne change qu'**un seul terme** : à la place de l'action suivante réellement choisie, il utilise la meilleure action possible dans l'état suivant,
#
# $$
# \delta_t=R_{t+1}+\gamma\,\max_{a}Q(S_{t+1},a)-Q(S_t,A_t).
# $$
#
# Si $S_{t+1}$ est terminal, $\delta_t=R_{t+1}-Q(S_t,A_t)$ : pas de maximum. La cible est la version échantillonnée de l'équation de Bellman optimale pour $q_*$. C'est une méthode **off-policy** : les données viennent d'une politique exploratoire, mais la quantité apprise correspond à la politique gloutonne.
#
# | | SARSA | Q-learning |
# |---|---|---|
# | Cible | $R_{t+1}+\gamma\,Q(S_{t+1},A_{t+1})$ | $R_{t+1}+\gamma\,\max_a Q(S_{t+1},a)$ |
# | Ligne de code | `target = r + gamma * Q[s_next, a_next]` | `target = r + gamma * max(Q[s_next])` |
# | Apprend | la valeur de la politique suivie | la valeur de la politique gloutonne |
#
# #### Explication du code
#
# Les deux fonctions ont la même structure : une boucle sur les épisodes, une boucle sur les pas, la mise à jour de `Q[s, a]`. Elles renvoient la table `Q`, la somme des récompenses de chaque épisode, le nombre de pas de chaque épisode et, si l'on fournit une référence `Q_ref`, l'erreur RMS sur $Q$ après chaque épisode.
#
# La seule subtilité de SARSA est l'ordre des opérations : il faut choisir $A_{t+1}$ **avant** la mise à jour (puisque la cible l'utilise) et l'exécuter **ensuite**. Q-learning, lui, choisit son action au début de chaque pas.

# %%
def sarsa(env, gamma, alpha, epsilon, n_episodes, rng,
          Q_ref=None, nonterminal=None, max_steps=10_000):
    Q = np.zeros((len(env["coords"]), N_ACTIONS))
    episode_rewards, episode_steps, errors = [], [], []

    for k in range(n_episodes):
        s = env_reset(env, rng)
        a = epsilon_greedy(Q, s, epsilon, rng)
        total = 0.0

        for t in range(max_steps):
            r, s_next, done = env_step(env, s, a, rng)
            total += r

            if done:
                target = r                                   # pas d'action suivante
            else:
                a_next = epsilon_greedy(Q, s_next, epsilon, rng)
                target = r + gamma * Q[s_next, a_next]       # action réellement choisie

            Q[s, a] += alpha * (target - Q[s, a])
            if done:
                break
            s, a = s_next, a_next                            # A_{t+1} sera bien exécutée

        episode_rewards.append(total)
        episode_steps.append(t + 1)
        if Q_ref is not None:
            errors.append(rms_error(Q, Q_ref, nonterminal))

    return Q, np.array(episode_rewards), np.array(episode_steps), np.array(errors)


def q_learning(env, gamma, alpha, epsilon, n_episodes, rng,
               Q_ref=None, nonterminal=None, max_steps=10_000):
    Q = np.zeros((len(env["coords"]), N_ACTIONS))
    episode_rewards, episode_steps, errors = [], [], []

    for k in range(n_episodes):
        s = env_reset(env, rng)
        total = 0.0

        for t in range(max_steps):
            a = epsilon_greedy(Q, s, epsilon, rng)
            r, s_next, done = env_step(env, s, a, rng)
            total += r

            if done:
                target = r                                   # pas de maximum
            else:
                target = r + gamma * max(Q[s_next])          # meilleure action possible

            Q[s, a] += alpha * (target - Q[s, a])
            if done:
                break
            s = s_next

        episode_rewards.append(total)
        episode_steps.append(t + 1)
        if Q_ref is not None:
            errors.append(rms_error(Q, Q_ref, nonterminal))

    return Q, np.array(episode_rewards), np.array(episode_steps), np.array(errors)


# %% [markdown]
# ### Questions
#
# 1. Dans les deux fonctions, `Q` est initialisé à $0$ pour **tous** les états, terminaux compris, et les terminaux ne sont jamais mis à jour. Pourquoi est-ce cohérent avec la convention $V(\text{terminal})=0$ ?
# 2. Dans SARSA, que se passerait-il si l'on choisissait `a_next` **après** la mise à jour, avec le `Q` modifié, plutôt qu'avant ?
#
# ## 3.4 Premier essai sur le GridWorld
#
# Nous entraînons les deux algorithmes avec **beaucoup d'exploration** ($\varepsilon=0{,}3$), exprès pour rendre visible ce qui les distingue. Les épisodes partent d'un état non terminal tiré au hasard. Nous suivons trois choses : le nombre de pas par épisode (un agent qui apprend doit l'approcher de la longueur optimale), l'erreur RMS de $Q$ par rapport à $q_*$, et la qualité de la politique gloutonne finale.
#
# ### Prédiction
#
# Les deux algorithmes vont-ils apprendre la **même** fonction $Q$ ? Vont-ils trouver la **même** politique gloutonne ? Dans quel cas l'erreur par rapport à $q_*$ tendra-t-elle vers $0$ ? Notez votre pari avant de lancer la cellule.

# %%
# >>> À MODIFIER (essayez d'autres valeurs, puis relancez la cellule)
ALPHA = 0.2          # pas d'apprentissage
EPSILON = 0.3        # probabilité d'exploration
N_EPISODES = 3000    # nombre d'épisodes
SEED = 0             # graine du générateur pseudo-aléatoire
# <<<

rng = np.random.default_rng(SEED)
Q_sarsa, _, steps_sarsa, errors_sarsa = sarsa(
    grid, gamma, ALPHA, EPSILON, N_EPISODES, rng, Q_ref=Q_star, nonterminal=nonterminal
)
rng = np.random.default_rng(SEED)
Q_qlearn, _, steps_qlearn, errors_qlearn = q_learning(
    grid, gamma, ALPHA, EPSILON, N_EPISODES, rng, Q_ref=Q_star, nonterminal=nonterminal
)

# Longueur moyenne d'un épisode optimal, pour un départ tiré au hasard.
optimal_steps = np.mean([len(greedy_path(grid, Q_star, s)) - 1 for s in nonterminal])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(moving_average(steps_sarsa, 50), label="SARSA")
axes[0].plot(moving_average(steps_qlearn, 50), label="Q-learning")
axes[0].axhline(optimal_steps, color="gray", linestyle="--", label="longueur optimale moyenne")
axes[0].set_yscale("log")
axes[0].set_yticks([5, 10, 20, 50, 100])
axes[0].set_yticklabels(["5", "10", "20", "50", "100"])
axes[0].minorticks_off()
axes[0].set_xlabel("Épisode")
axes[0].set_ylabel("Pas par épisode (moyenne glissante sur 50)")
axes[0].set_title("Vitesse d'apprentissage")

axes[1].plot(errors_sarsa, label="SARSA")
axes[1].plot(errors_qlearn, label="Q-learning")
axes[1].set_xlabel("Épisode")
axes[1].set_ylabel(r"Erreur RMS sur $Q$ par rapport à $q_*$")
axes[1].set_title(r"Distance à $q_*$")

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.show()

print("Part des états où l'action gloutonne est optimale :")
print(f"  SARSA      : {greedy_is_optimal(Q_sarsa, Q_star, nonterminal):.0%}")
print(f"  Q-learning : {greedy_is_optimal(Q_qlearn, Q_star, nonterminal):.0%}")
print(f"Erreur RMS finale sur Q : SARSA = {errors_sarsa[-1]:.2f}, Q-learning = {errors_qlearn[-1]:.2f}")

# %%
# Valeur de la meilleure action, max sur a de Q(s,a), à comparer avec v_* du TP1.
maps = [
    (V_star, r"Optimal $v_*$ (TP1)"),
    (Q_qlearn.max(axis=1), "Q-learning : valeur de la meilleure action"),
    (Q_sarsa.max(axis=1), "SARSA : valeur de la meilleure action"),
]
vmin = min(values.min() for values, _ in maps)
vmax = max(values.max() for values, _ in maps)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
for ax, (values, title) in zip(axes, maps):
    shown = draw_values(values, grid, title, ax)
    shown.set_clim(vmin, vmax)
fig.colorbar(shown, ax=axes, shrink=0.8, label="Valeur")
plt.show()

# Politiques gloutonnes (plusieurs flèches = actions à égalité).
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
draw_policy(greedy_stochastic_policy(Q_star, grid["terminal_states"]), grid,
            r"Optimale (d'après $q_*$)", axes[0])
draw_policy(greedy_stochastic_policy(Q_sarsa, grid["terminal_states"]), grid,
            "SARSA : politique gloutonne", axes[1])
draw_policy(greedy_stochastic_policy(Q_qlearn, grid["terminal_states"]), grid,
            "Q-learning : politique gloutonne", axes[2])
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Questions
#
# 1. Votre pari était-il juste ? Les deux politiques gloutonnes sont-elles proches de la politique optimale ? Les deux fonctions $Q$ sont-elles proches de $q_*$ ?
# 2. Pourquoi l'erreur de SARSA par rapport à $q_*$ baisse-t-elle d'abord, puis **remonte-t-elle et se stabilise-t-elle** à un niveau élevé, alors que celle de Q-learning continue de descendre ? Quelle fonction SARSA est-il en train d'apprendre ? (Notes §3.2.3.)
# 3. Comment les deux faits suivants sont-ils compatibles : des fonctions $Q$ différentes, et pourtant des politiques gloutonnes quasi identiques ?
# 4. Les données de Q-learning viennent d'une politique très exploratoire ($\varepsilon=0{,}3$), et pourtant il apprend $q_*$. Quelle propriété de sa cible le permet ? (Notes §3.3.5.)
#
# ### Expérience facultative — $\varepsilon$ et $\alpha$
#
# Modifiez `EPSILON` (par exemple $0{,}05$, puis $0{,}6$) et `ALPHA` (par exemple $0{,}05$, puis $0{,}5$), puis relancez les deux cellules. **Prédisez d'abord** l'effet de $\varepsilon$ sur la longueur des épisodes pendant l'apprentissage, et sur l'écart entre SARSA et Q-learning. Quel est l'effet d'un pas $\alpha$ trop petit sur le nombre d'épisodes nécessaire ?
#
# ### Question d'approfondissement — ce que SARSA estime vraiment
#
# Nous avons dit que SARSA estime la valeur de la politique $\varepsilon$-gloutonne qu'il suit. Les outils du TP1 permettent de le **vérifier**. La cellule suivante construit la politique $\varepsilon$-gloutonne associée au $Q$ final de SARSA, calcule sa valeur exacte avec `policy_evaluation`, puis compare :
#
# - la valeur d'état implicite de SARSA, $\sum_a \pi_\varepsilon(a\mid s)\,Q(s,a)$, à la valeur exacte $v_{\pi_\varepsilon}$ ;
# - cette même valeur implicite à $v_*$.

# %%
def epsilon_greedy_policy(Q, epsilon, terminal_states):
    # Politique epsilon-gloutonne associée à Q, sous forme de tableau (états, actions).
    n_states, n_actions = Q.shape
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        q_max = max(Q[s])
        best = [a for a in range(n_actions) if Q[s, a] >= q_max - 1e-10]
        for a in range(n_actions):
            policy[s, a] = epsilon / n_actions
        for a in best:
            policy[s, a] += (1 - epsilon) / len(best)
    policy[terminal_states] = 0.0
    return policy


policy_eps = epsilon_greedy_policy(Q_sarsa, EPSILON, grid["terminal_states"])

# Valeur exacte de la politique epsilon-gloutonne (utilise le modèle : vérification).
V_eps, _, _ = policy_evaluation(
    policy_eps, grid["P"], grid["R"], gamma, grid["terminal_states"]
)

# Valeur d'état implicite dans le Q de SARSA : moyenne de Q selon policy_eps.
V_sarsa = np.sum(policy_eps * Q_sarsa, axis=1)

print(f"Écart RMS entre V_sarsa et v_(pi_eps), valeur exacte de la politique suivie : {rms_error(V_sarsa, V_eps, nonterminal):.2f}")
print(f"Écart RMS entre V_sarsa et v_*, valeur optimale                          : {rms_error(V_sarsa, V_star, nonterminal):.2f}")

# %% [markdown]
# De quelle fonction $V_{\text{sarsa}}$ est-elle la plus proche ? Que cela confirme-t-il sur SARSA ? (Les écarts ne sont pas nuls : $Q$ est encore bruité par le pas constant $\alpha$.)
#
# ### Micro-projet facultatif — $\varepsilon$ décroissant
#
# Avec $\varepsilon$ constant, SARSA ne peut pas atteindre $q_*$ : il suit toujours une politique qui explore. Les notes (§3.1.2, §3.2.4) indiquent que la garantie de convergence vers $q_*$ demande que la politique devienne **gloutonne à la limite**, tout en explorant assez longtemps (condition GLIE).
#
# 1. Écrivez `sarsa_glie(env, gamma, alpha, n_episodes, rng, Q_ref, nonterminal)`, une variante de `sarsa` où $\varepsilon$ diminue d'un épisode à l'autre :
# $$
# \varepsilon_k=\frac{0{,}5}{1+k/50},\qquad k=0,1,2,\ldots
# $$
# Elle doit renvoyer `Q, errors` (la table finale et l'erreur RMS par épisode).
# 2. Exécutez la cellule de contrôle : elle compare l'erreur de votre version à celle de SARSA avec $\varepsilon=0{,}3$ constant.
# 3. **Prédisez** : votre courbe finira-t-elle au-dessus ou en dessous de celle de SARSA à $\varepsilon$ constant ? Atteindra-t-elle exactement $0$ ? Pourquoi un $\varepsilon$ qui décroît **trop vite** poserait-il problème ?
#
# <details>
# <summary>Indication</summary>
#
# Copiez `sarsa`, supprimez l'argument `epsilon`, et calculez `epsilon = 0.5 / (1 + k / 50)` au début de chaque épisode, avant le premier appel à `epsilon_greedy`. Gardez `Q_ref` et `nonterminal` pour enregistrer l'erreur. Il suffit de renvoyer `Q, np.array(errors)`.
#
# </details>

# %%
# Point de contrôle (à exécuter après avoir défini sarsa_glie)
if "sarsa_glie" not in globals():
    print("Définissez d'abord sarsa_glie (voir l'énoncé et l'indication).")
else:
    rng = np.random.default_rng(0)
    Q_glie, errors_glie = sarsa_glie(grid, gamma, 0.2, 3000, rng, Q_star, nonterminal)

    rng = np.random.default_rng(0)
    _, _, _, errors_const = sarsa(
        grid, gamma, 0.2, 0.3, 3000, rng, Q_ref=Q_star, nonterminal=nonterminal
    )

    plt.figure(figsize=(7, 4))
    plt.plot(errors_const, label=r"SARSA, $\varepsilon=0{,}3$ constant")
    plt.plot(errors_glie, label=r"SARSA, $\varepsilon$ décroissant")
    plt.xlabel("Épisode")
    plt.ylabel(r"Erreur RMS sur $Q$ par rapport à $q_*$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    print(f"Erreur finale : constant = {errors_const[-1]:.2f}, décroissant = {errors_glie[-1]:.2f}")


# %% [markdown]
# ### À retenir
#
# - Le contrôle *model-free* est une **GPI approximée** : on estime $Q$ par TD et l'on agit avec une politique $\varepsilon$-gloutonne déduite de $Q$.
# - SARSA et Q-learning ne diffèrent que par la **cible** : l'action suivante réellement choisie pour SARSA, le maximum pour Q-learning.
# - SARSA (*on-policy*) apprend la valeur de la politique qu'il suit, exploration comprise : avec beaucoup d'exploration, ses valeurs sont plus basses que $q_*$. Q-learning (*off-policy*) apprend $q_*$ malgré un comportement exploratoire.
# - Sur ce problème sans danger, les deux trouvent une politique gloutonne quasi optimale : la différence se voit dans les valeurs, pas dans la politique. Elle deviendra visible dans les décisions dès que l'exploration peut coûter cher : c'est la falaise.

# %% [markdown]
# # Partie 4 — La falaise : *on-policy* contre *off-policy*
#
# *Pour approfondir : notes RL2, §3.3.7 (exemple de la falaise, d'après Sutton et Barto, exemple 6.6).*
#
# Sur le GridWorld, SARSA et Q-learning trouvent des politiques presque identiques. Pour voir apparaître une vraie différence de **comportement**, il faut un environnement où l'exploration peut coûter très cher.
#
# Le problème de la **falaise** : une grille $4\times12$. L'agent part de la case en bas à gauche et doit rejoindre la case but en bas à droite. Les cases de la bande inférieure situées entre les deux forment la falaise.
#
# - chaque déplacement ordinaire donne la récompense $-1$ ;
# - entrer dans la falaise donne $-100$ et renvoie l'agent au départ, **sans terminer l'épisode** ;
# - seule l'arrivée au but termine l'épisode (récompense $0$, comme dans le TP1) ;
# - la tâche est épisodique et non actualisée : $\gamma=1$ (notes, « Actualisation et terminaison »).
#
# Nous utilisons $\varepsilon=0{,}1$ **constant** : pendant l'apprentissage, l'agent tire de temps en temps une action au hasard, et cette action peut le faire tomber.
#
# #### Explication du code
#
# `build_cliff` réutilise `build_gridworld` du TP1. Les cases de la falaise sont déclarées comme des « murs » : elles ne sont pas des états. On modifie ensuite les transitions : toute action qui entrerait dans la falaise renvoie au départ avec la récompense $-100$. **Aucun algorithme n'est modifié** : seul le MDP change.

# %%
def build_cliff(n_rows=4, n_cols=12):
    cliff_cells = {(n_rows - 1, j) for j in range(1, n_cols - 1)}
    start_coord = (n_rows - 1, 0)
    goal_coord = (n_rows - 1, n_cols - 1)

    # Les cases de la falaise sont traitées comme des "murs" : ce ne sont pas des états.
    env = build_gridworld((n_rows, n_cols), {goal_coord}, walls=cliff_cells)
    s_start = env["coord_to_state"][start_coord]
    terminal_set = set(env["terminal_states"])

    # Toute action qui entrerait dans la falaise renvoie au départ avec la récompense -100.
    for s, (i, j) in enumerate(env["coords"]):
        if s in terminal_set:
            continue
        for a, (di, dj) in enumerate(MOVES):
            if (i + di, j + dj) in cliff_cells:
                env["P"][s, a, :] = 0.0
                env["R"][s, a, :] = 0.0
                env["P"][s, a, s_start] = 1.0
                env["R"][s, a, s_start] = -100.0

    return env, s_start


def draw_path(env, path, title, ax):
    # Affiche un chemin (liste d'états) sur la grille : murs en noir.
    background = np.zeros(env["shape"])
    for wall in env["walls"]:
        background[wall] = 1
    ax.imshow(background, cmap="Greys", vmin=0, vmax=1)

    coords = np.array([env["coords"][s] for s in path])
    ax.plot(coords[:, 1], coords[:, 0], "o-", color="tab:red", markersize=4)
    ax.scatter(coords[0, 1], coords[0, 0], s=120, color="tab:blue", zorder=3, label="départ")
    for s in env["terminal_states"]:
        i, j = env["coords"][s]
        ax.scatter(j, i, s=120, color="tab:green", zorder=3, label="but")

    ax.set_title(title)
    ax.set_xticks(range(env["shape"][1]))
    ax.set_yticks(range(env["shape"][0]))
    ax.set_xticks(np.arange(-0.5, env["shape"][1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env["shape"][0], 1), minor=True)
    ax.grid(which="minor", color="tab:gray", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)


def path_title(env, path, name):
    # Titre d'un chemin glouton : longueur, ou signalement d'une boucle.
    if env["is_terminal"][path[-1]]:
        return f"{name} : {len(path) - 1} pas"
    return f"{name} : le but n'est pas atteint (la politique boucle)"


cliff, s_start = build_cliff()
check_mdp(cliff)
prepare_env(cliff, start_states=[s_start])

print("Nombre d'états :", len(cliff["coords"]))
print("Case de départ :", cliff["coords"][s_start])

# Solution exacte avec Value Iteration (gamma = 1) : uniquement pour référence.
V_cliff, _, _, _ = value_iteration(cliff["P"], cliff["R"], 1.0, cliff["terminal_states"])
Q_cliff = action_values(V_cliff, cliff["P"], cliff["R"], 1.0)
optimal_path = greedy_path(cliff, Q_cliff, s_start)

print("Retour optimal depuis le départ :", V_cliff[s_start])
print("Longueur du chemin optimal :", len(optimal_path) - 1, "pas")

fig, ax = plt.subplots(figsize=(10, 3.5))
draw_path(cliff, optimal_path, "Le chemin optimal (Value Iteration) ; la falaise est en noir", ax)
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# ### Prédiction
#
# Le chemin optimal longe la falaise. Les deux algorithmes vont-ils l'apprendre ? Plus précisément :
#
# 1. Quel algorithme va, selon vous, apprendre à **passer le plus près du bord** ? Lequel va préférer un détour plus long et plus sûr ?
# 2. Pendant l'apprentissage, avec $\varepsilon=0{,}1$, lequel des deux obtiendra la **meilleure somme de récompenses par épisode** ? Est-ce forcément celui qui a trouvé le meilleur chemin ?
#
# Notez vos paris, puis lancez la cellule. Elle prend quelques secondes : elle entraîne chaque algorithme pour $10$ graines différentes.

# %%
# >>> À MODIFIER (essayez d'autres valeurs, puis relancez la cellule)
ALPHA = 0.1          # pas d'apprentissage
EPSILON = 0.1        # probabilité d'exploration
N_EPISODES = 500     # nombre d'épisodes
N_RUNS = 10          # nombre d'exécutions moyennées
SLIP = 0.0           # probabilité que l'environnement remplace l'action par une action au hasard
# <<<

cliff["slip"] = SLIP

curves_sarsa, curves_qlearn = [], []
for run in range(N_RUNS):
    rng = np.random.default_rng(run)
    Q, episode_rewards, _, _ = sarsa(cliff, 1.0, ALPHA, EPSILON, N_EPISODES, rng)
    curves_sarsa.append(episode_rewards)
    if run == 0:
        Q_sarsa_cliff = Q

    rng = np.random.default_rng(run)
    Q, episode_rewards, _, _ = q_learning(cliff, 1.0, ALPHA, EPSILON, N_EPISODES, rng)
    curves_qlearn.append(episode_rewards)
    if run == 0:
        Q_qlearn_cliff = Q

mean_sarsa = np.mean(curves_sarsa, axis=0)
mean_qlearn = np.mean(curves_qlearn, axis=0)

plt.figure(figsize=(8, 4.5))
plt.plot(moving_average(mean_sarsa, 10), label="SARSA")
plt.plot(moving_average(mean_qlearn, 10), label="Q-learning")
plt.ylim(-120, 0)
plt.xlabel("Épisode")
plt.ylabel("Somme des récompenses de l'épisode")
plt.title(f"Performance pendant l'apprentissage (moyenne de {N_RUNS} exécutions)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("Somme des récompenses, moyenne sur les 100 derniers épisodes :")
print(f"  SARSA      : {mean_sarsa[-100:].mean():.1f}")
print(f"  Q-learning : {mean_qlearn[-100:].mean():.1f}")

# Chemins suivis par la politique gloutonne finale (première exécution).
fig, axes = plt.subplots(2, 1, figsize=(10, 7))
path_sarsa = greedy_path(cliff, Q_sarsa_cliff, s_start, max_steps=60)
path_qlearn = greedy_path(cliff, Q_qlearn_cliff, s_start, max_steps=60)
draw_path(cliff, path_sarsa, path_title(cliff, path_sarsa, "SARSA, chemin glouton final"), axes[0])
draw_path(cliff, path_qlearn, path_title(cliff, path_qlearn, "Q-learning, chemin glouton final"), axes[1])
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Questions
#
# 1. Votre prédiction était-elle juste ? Quel chemin apprend chaque algorithme ?
# 2. Pourquoi SARSA préfère-t-il le détour ? Que « sait » sa cible $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$ que la cible de Q-learning ne sait pas ?
# 3. Q-learning a pourtant trouvé le chemin **optimal**. Pourquoi sa performance pendant l'apprentissage est-elle moins bonne que celle de SARSA ? Quelle est la différence entre la politique que l'on **apprend** et la politique que l'on **exécute** ?
# 4. Laquelle des deux performances vous paraît la plus pertinente si l'on déploie ensuite l'agent avec $\varepsilon=0$ ? Et si l'exploration doit rester active (par exemple un robot qui continue d'apprendre) ?
#
# ### Expérience facultative — l'effet de $\varepsilon$
#
# Relancez la cellule avec `EPSILON = 0.01`, puis `EPSILON = 0.3`. **Prédisez** : que devient l'écart de performance entre les deux algorithmes quand $\varepsilon$ est très petit ? Quand il est grand ? Les chemins appris changent-ils ? Comment cela relie-t-il ce que vous voyez à la phrase des notes : « la différence apparaît surtout lorsque l'exploration reste présente » ?
#
# ### Expérience facultative — une falaise glissante
#
# Fixez `SLIP = 0.2` : l'environnement remplace maintenant, avec probabilité $0{,}2$, l'action choisie par une action tirée au hasard (l'agent ne le sait pas, il voit seulement ce qui arrive). Relancez la cellule, avec `EPSILON = 0.1`.
#
# **Prédisez** : Q-learning va-t-il toujours longer la falaise ? Pourquoi ? Relisez l'exemple 3.4 des notes (grille stochastique) : quelle est la source du risque cette fois, et que dit cet exemple sur ce que Q-learning « voit » dans ses transitions ?
#
# ### À retenir
#
# - Sur la falaise, **Q-learning** apprend le chemin optimal mais pas à l'abri d'une exploration malheureuse : sa performance *pendant l'apprentissage* est plus faible.
# - **SARSA** apprend la valeur du comportement réel, exploration comprise : il choisit un détour prudent et obtient de meilleurs épisodes tant que l'exploration reste active.
# - La différence vient de la **cible**, pas de l'environnement : le même code, les mêmes paramètres, seule une ligne change.
# - Il n'existe pas de règle universelle « SARSA dans les environnements dangereux, Q-learning ailleurs » : les deux apprennent à partir des transitions réellement observées, et la question est de savoir *quelle politique* on veut évaluer.

# %% [markdown]
# # Partie 5 — Le labyrinthe
#
# Dans le TP1, le labyrinthe illustrait le message « le MDP change, les algorithmes de planification ne changent pas ». Nous reprenons exactement le même labyrinthe, avec un message analogue et une nouveauté : **ce sont maintenant les algorithmes d'apprentissage qui restent inchangés**, et ils n'ont plus le droit de lire le modèle.
#
# Le départ est en $(0,0)$ (départ **fixe**), le terminal en $(10,10)$, $\gamma=0{,}9$ comme dans le TP1. Les murs ne sont pas des états ; une tentative d'entrer dans un mur laisse l'agent sur place.

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
check_mdp(maze)

maze_start = state_of(maze, (0, 0))
prepare_env(maze, start_states=[maze_start])

# Référence : la solution exacte du TP1 (Value Iteration), uniquement pour comparer.
V_maze, _, _, _ = value_iteration(maze["P"], maze["R"], gamma, maze["terminal_states"])
Q_maze = action_values(V_maze, maze["P"], maze["R"], gamma)
optimal_maze_path = greedy_path(maze, Q_maze, maze_start)

print("Nombre de cases accessibles :", len(maze["coords"]))
print("Longueur du chemin optimal :", len(optimal_maze_path) - 1, "pas")

# %% [markdown]
# ### Prédiction
#
# Au premier épisode, la table $Q$ est entièrement nulle : l'agent ne sait rien du labyrinthe. Combien de pas va durer, selon vous, un premier épisode : quelques dizaines, quelques centaines, quelques milliers ? Et combien d'épisodes faudra-t-il pour que la longueur d'un épisode approche celle du chemin optimal ?
#
# La cellule suivante entraîne les deux algorithmes avec les **mêmes** fonctions `sarsa` et `q_learning` que dans les parties précédentes.

# %%
# >>> À MODIFIER
ALPHA = 0.3
EPSILON = 0.1
N_EPISODES = 2000
SEED = 0
# <<<

rng = np.random.default_rng(SEED)
Q_sarsa_maze, _, steps_sarsa, _ = sarsa(maze, gamma, ALPHA, EPSILON, N_EPISODES, rng)
rng = np.random.default_rng(SEED)
Q_qlearn_maze, _, steps_qlearn, _ = q_learning(maze, gamma, ALPHA, EPSILON, N_EPISODES, rng)

print("Longueur des 5 premiers épisodes :")
print("  SARSA      :", steps_sarsa[:5])
print("  Q-learning :", steps_qlearn[:5])

plt.figure(figsize=(8, 4.5))
plt.plot(steps_sarsa, alpha=0.6, label="SARSA")
plt.plot(steps_qlearn, alpha=0.6, label="Q-learning")
plt.axhline(len(optimal_maze_path) - 1, color="gray", linestyle="--", label="chemin optimal")
plt.yscale("log")
plt.xlabel("Épisode")
plt.ylabel("Pas par épisode")
plt.title("Le labyrinthe : longueur des épisodes")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
path_sarsa = greedy_path(maze, Q_sarsa_maze, maze_start)
path_qlearn = greedy_path(maze, Q_qlearn_maze, maze_start)
draw_path(maze, path_sarsa, path_title(maze, path_sarsa, "SARSA, chemin glouton"), axes[0])
draw_path(maze, path_qlearn, path_title(maze, path_qlearn, "Q-learning, chemin glouton"), axes[1])
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Question de synthèse
#
# Établissez deux listes courtes :
#
# - ce qui a **changé** entre le GridWorld et le labyrinthe ;
# - ce qui est resté **strictement identique**.
#
# Ajoutez une troisième liste : ce qui **diffère** entre ce que faisait *Value Iteration* dans le TP1 sur ce même labyrinthe et ce que font SARSA et Q-learning aujourd'hui (pensez aux données utilisées, à l'exploration, au nombre d'itérations).
#
# ### Questions
#
# 1. Pourquoi les premiers épisodes sont-ils si longs ? Qu'est-ce qui fait qu'ils raccourcissent ensuite ?
# 2. Le chemin glouton appris atteint-il la longueur optimale ? Que se passerait-il avec beaucoup moins d'épisodes (essayez `N_EPISODES = 100`) ?
#
# ### Expérience facultative — modifier le labyrinthe
#
# Comme dans le TP1, modifiez `maze_array` en ouvrant ou en fermant une case (sans bloquer le départ ni le but), reconstruisez `maze` et relancez les cellules de cette partie. **Prédisez** si le chemin optimal devient plus court, plus long, inchangé ou impossible. Les algorithmes d'apprentissage ont-ils besoin d'être modifiés ?
#
# ### À retenir
#
# - Les mêmes fonctions `sarsa` et `q_learning` s'appliquent à un MDP très différent : seul l'environnement change.
# - L'agent découvre le labyrinthe par essais : les premiers épisodes sont très longs, car $Q\equiv0$ ne le guide pas encore, puis ils raccourcissent à mesure que l'information sur le but se propage en arrière, transition par transition.
# - Pour le même résultat, la programmation dynamique du TP1 lisait tout le modèle ; ici l'agent n'a que son expérience.

# %% [markdown]
# # Partie 6 — Mini-projet facultatif : le Random Walk
#
# *Hors séance. Notes RL2, Exemple 2.9 (Sutton et Barto, exemple 6.2) et Figure 2.2.*
#
# Ce mini-projet n'est pas nécessaire pour comprendre le reste du TP. Il s'adresse à celles et ceux qui veulent, chez eux, **reconstruire une expérience de référence** du cours à partir des outils de ce TP.
#
# ## Le problème
#
# Cinq états alignés $A,B,C,D,E$, flanqués de deux états terminaux. Chaque épisode commence en $C$. À chaque pas, le processus va à gauche ou à droite avec probabilité $1/2$. Toutes les récompenses sont nulles, sauf $+1$ lors de la sortie par la droite ; le problème n'est pas actualisé ($\gamma=1$).
#
# ```
#    (terminal)  A   B   C   D   E  (terminal)
#       0        0   0   0   0   0      1
#                        départ
# ```
#
# Il n'y a **aucune décision à prendre** : c'est un *Markov reward process*, donc un MDP à **une seule action**. C'est pourquoi SARSA et Q-learning n'ont pas de sens ici : seule la *prédiction* est possible. Mais les fonctions `mc_prediction` et `td0_prediction` de la Partie 2 s'appliquent **telles quelles**, à condition de construire le bon environnement.
#
# Les vraies valeurs sont les probabilités d'atteindre le terminal de droite :
#
# $$
# v(A),\ldots,v(E)=\tfrac16,\ \tfrac26,\ \tfrac36,\ \tfrac46,\ \tfrac56.
# $$
#
# **But :** reproduire le panneau de droite de la Figure 2.2 des notes : l'erreur RMS de Monte Carlo (pas constant) et de TD(0), en fonction du nombre d'épisodes, pour plusieurs valeurs de $\alpha$.
#
# ## Étape 1 — Construire l'environnement
#
# Numérotez les états de $0$ à $6$ : $0$ est le terminal de gauche, $1,\ldots,5$ sont $A,\ldots,E$, et $6$ est le terminal de droite. Construisez les tableaux `P_rw` et `R_rw` de forme `(7, 1, 7)` (une seule action), rangez-les dans un dictionnaire `rw` avec les clés `"coords"`, `"terminal_states"`, `"P"`, `"R"`, puis appelez `prepare_env(rw, start_states=[3])`. La politique est un tableau de forme `(7, 1)` rempli de $1$. Vérifiez votre construction avec `check_mdp(rw)`.
#
# <details>
# <summary>Indication</summary>
#
# ```python
# n_states = 7
# P_rw = np.zeros((n_states, 1, n_states))
# R_rw = np.zeros((n_states, 1, n_states))
#
# for s in range(1, 6):           # états non terminaux A, ..., E
#     P_rw[s, 0, s - 1] = 0.5
#     P_rw[s, 0, s + 1] = 0.5
# R_rw[5, 0, 6] = 1.0             # seule récompense non nulle : E -> terminal droit
#
# P_rw[0, 0, 0] = 1.0             # terminaux absorbants
# P_rw[6, 0, 6] = 1.0
#
# rw = {"coords": list(range(n_states)),
#       "terminal_states": np.array([0, 6]),
#       "P": P_rw, "R": R_rw}
# prepare_env(rw, start_states=[3])
# policy_rw = np.ones((n_states, 1))
# ```
#
# </details>
#
# ## Étape 2 — La vérité terrain
#
# Retrouvez les valeurs $1/6,\ldots,5/6$ de deux façons : en remarquant que la valeur de chaque état intérieur est la moyenne des valeurs de ses deux voisins (notes, Exemple 2.9), puis en appelant `policy_evaluation` du TP1 avec $\gamma=1$. Rangez le résultat dans un vecteur `V_rw` de taille $7$ (les terminaux valent $0$) et définissez `nonterminal_rw = [1, 2, 3, 4, 5]`.
#
# ## Étape 3 — Un premier essai
#
# Appelez `td0_prediction` puis `mc_prediction` avec `v_init=0.5` (valeur d'initialisation utilisée par Sutton et Barto), $\gamma=1$, $100$ épisodes. Comparez les tables `V` obtenues à `V_rw`.
#
# ## Étape 4 — Reproduire la figure
#
# Moyennez l'erreur RMS sur **100 exécutions indépendantes** de 100 épisodes chacune, pour :
#
# - Monte Carlo : $\alpha\in\{0{,}01;\,0{,}02;\,0{,}03;\,0{,}04\}$ ;
# - TD(0) : $\alpha\in\{0{,}05;\,0{,}1;\,0{,}15\}$.
#
# Tracez les sept courbes.
#
# <details>
# <summary>Indication</summary>
#
# Structure d'une boucle possible, pour un $\alpha$ donné :
#
# ```python
# curves = []
# for run in range(100):
#     rng = np.random.default_rng(run)
#     _, errors = td0_prediction(rw, policy_rw, 1.0, alpha, 100, rng,
#                                V_rw, nonterminal_rw, v_init=0.5)
#     curves.append(errors)
# mean_curve = np.mean(curves, axis=0)
# ```
#
# </details>
#
# **Points de contrôle.** Avant tout apprentissage, l'erreur RMS de $V\equiv0{,}5$ vaut environ $0{,}24$. Après 100 épisodes, avec vos courbes moyennées, TD(0) avec $\alpha=0{,}05$ doit se situer autour de $0{,}04$, et Monte Carlo avec $\alpha=0{,}04$ autour de $0{,}06$.
#
# ## Questions
#
# 1. Dans la Partie 2, sur le GridWorld, Monte Carlo démarrait plus vite que TD(0). Ici, c'est TD(0) qui gagne. Quelles différences entre les deux problèmes peuvent l'expliquer ? (Pensez à la longueur des épisodes, à la dispersion des retours qui ne valent ici que $0$ ou $1$, et à la valeur initiale $0{,}5$, qui n'est pas absurde.)
# 2. Avec un pas constant, les courbes ne convergent pas vers $0$ : elles se stabilisent, voire remontent pour les grands pas. Pourquoi ? Que faudrait-il pour une convergence exacte ? (Notes §3.2.4, conditions de Robbins–Monro.)
#
# ### Question d'approfondissement — l'exemple 2.10 sur papier
#
# Sans coder, traitez l'exemple 2.10 des notes (les huit épisodes observés avec les états $A$ et $B$) : quelles valeurs $V(A)$ et $V(B)$ obtiendriez-vous avec Monte Carlo ? Avec TD(0) à convergence ? Laquelle des deux réponses vous semble la meilleure pour prédire le retour d'un nouvel épisode partant de $A$, et pourquoi ? Comparez ensuite avec le texte des notes.

# %% [markdown]
# # Bilan
#
# Nous avons suivi la chaîne complète, sans jamais lire le modèle :
#
# $$
# \text{interaction}
# \longrightarrow
# \text{trajectoires}
# \longrightarrow
# \text{corrections de }V\text{ ou }Q
# \longrightarrow
# \text{politique }\varepsilon\text{-gloutonne}
# \longrightarrow
# \pi_*.
# $$
#
# | | Monte Carlo | TD(0) | SARSA | Q-learning |
# |---|---|---|---|---|
# | Cible | $G_t$ | $R_{t+1}+\gamma V(S_{t+1})$ | $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$ | $R_{t+1}+\gamma\max_a Q(S_{t+1},a)$ |
# | Bootstrap | non | oui | oui | oui |
# | Mise à jour | fin d'épisode | à chaque transition | à chaque transition | à chaque transition |
# | Estime | $v_\pi$ | $v_\pi$ | $q$ de la politique suivie | $q_*$ |
#
# Quatre messages à emporter :
#
# 1. Le *model-free* remplace les espérances exactes du TP1 par des **corrections à partir d'échantillons** : les objets à estimer ($v_\pi$, $q_\pi$, $q_*$) ne changent pas.
# 2. Les quatre méthodes suivent le même schéma « ancienne estimation + pas × erreur » ; seule la **cible** change.
# 3. Monte Carlo et TD(0) proposent un compromis entre absence de biais et faible variance ; aucun n'est gagnant partout.
# 4. *On-policy* ou *off-policy* n'est pas qu'une différence de formules : quand l'exploration coûte cher, cela change le **comportement appris**.
#
# Les outils du TP1 ont rempli un rôle précieux : ils ont fourni la **vérité terrain** qui permet de mesurer ce que les algorithmes apprennent.
#
# Pour replacer ces méthodes dans le cadre général : revoir **RL2, §1.2**. Pour la suite, quand l'espace d'états est trop grand pour une table, on remplace $Q(s,a)$ par une fonction paramétrée : c'est l'approximation de fonction et le *deep RL* (notes RL2, §3.3.9).

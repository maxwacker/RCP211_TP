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
#
# <a id='chap-tpintrogeneration'></a>

# %% [markdown]
# # Travaux pratiques : introduction aux modèles génératifs
#
# [Cahier Jupyter](jupyter/Bloc_GenIA/session1_practical.ipynb)
#
# L’objectif de cette séance de TP est de réaliser un bref tour d’horizon
# de trois types de modèles génératifs simples:
#
# - modèles de mélanges gaussiens,  
# - chaînes de Markov,  
# - autoencodeurs.  
#
#
# La séance est divisée en 3 exercices indépendants pouvant être traités
# dans le désordre.

# %% [markdown]
# ## Exercice 1 : Modèles de mélanges gaussiens
#
# Ce premier exercice est une application directe des modèles de mélanges
# gaussiens. L’objectif est d’illustrer le fonctionnement de
# l’échantillonnage de nouvelles données à partir de la densité estimée.

# %% hide-output=false
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# Pour cet exercice, nous allons travailler sur un nuage de points
# synthétique qui est généré à partir d’une sinusoïde. Les observations
# $ X $ sont bi-dimensionnelles avec:
#
# $$
# x_2 = \sin(x_1) + \epsilon
# $$
#
# avec:
#
# - $ x_1 $ une variable aléatoire tirée uniformément dans $ [0, 10] $,  
# - $ \epsilon $ un bruit aléatoire tiré d’une loi normale $ \mathcal{N}(0, 0.1) $.  

# %% hide-output=false
n_samples = 300
x1 = 10 * np.random.rand(n_samples)
x2 = np.sin(x1) + 0.1*np.random.randn(n_samples)
X = np.vstack((x1, x2))
X = np.transpose(X, (1, 0))

# Affichage du nuage de point
plt.scatter(X[:,0], X[:,1], s=10)
plt.title
plt.show()

# %%
# Previous Cell Study
sx1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) #10 * np.random.rand(n_samples)
sx2 = sx1 * 10 
print("sx1 :\n",sx1)
print("sx2 :\n",sx2)
sX = np.vstack((sx1, sx2))
print("sX :\n",sX)
sX = np.transpose(sX, (1, 0))
print("np.transpose(sX, (1, 0)) :\n",sX)

# %% [markdown]
# ## Question
#
# Appliquer un [modèle de mélange gaussien](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) à la matrice d’observations $ X $. On choisira de façon empirique un nombre de composantes qui semble adapté au nuage de points.

# %%
#Compute bounds of datas for density plot range

# min & max values for x & y 
min_x = np.min(X[:,0])
max_x = np.max(X[:,0])
min_y = np.min(X[:,1])
max_y = np.max(X[:,1])
#Ranges for x & y
rx = max_x - min_x
ry = max_y - min_y

# Add some margins
margin = 0.1
mmin_x = min_x - margin
mmax_x = max_x + margin
mmin_y = min_y - margin
mmax_y = max_y + margin

mrx = mmax_x - mmin_x
mry = mmax_y - mmin_y

# %%
# Réponse:
# On cherche à générer des élément de(s) classe(s) portée par le dataset X (sans label de  supervision, sans connaissance du nombre de classe)
# Hypothèse : 1 seule classe = Les points x,y qui sont dans le nuage Oui ou Non
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# préparer les données où on calculera la densité
grid_size = 100 # n_samples =300 
#Gx = np.arange(-1.0, 11.0, 10 / grid_size)
#Gy = np.arange(-1.5, 1.5, 10 / grid_size)

Gx = np.arange(mmin_x, mmax_x, mrx/grid_size) 
Gy = np.arange(mmin_y, mmax_y, mry/grid_size)

Gx, Gy = np.meshgrid(Gx, Gy)

# estimation par mélange gaussien
gmm = GaussianMixture(n_components=3,n_init=3).fit(X)

# calcul de la densité pour les données de la grille
density = np.exp(gmm.score_samples(np.hstack(
    (
        (Gx.reshape(grid_size*grid_size))[:,np.newaxis],
        (Gy.reshape(grid_size*grid_size)[:,np.newaxis])
    )
)))

# affichage : données et estimation
# %matplotlib widget
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(Gx, Gy, density.reshape(grid_size,grid_size), rstride=1,
                    cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
ax.scatter(X[:,0], X[:,1], -0.025)
plt.show()

# %% [markdown]
# ## Correction
#
# L’idéal serait d’utiliser le critère AIC ou BIC pour déterminer le nombre de composantes. En première approximation, considérons 6 composantes:

# %% hide-output=false
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=6)
gmm.fit(X)


# %% hide-output=false
def make_ellipses(gmm, ax):
    colors = ['navy', 'firebrick', 'darkorange', 'dimgrey', 'forestgreen',
              'purple', 'turquoise', 'yellow', 'hotpink', 'springgreen']

    for n, color in enumerate(colors[:gmm.n_components]):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.text(*(gmm.means_[n, :2]+0.1), str(n), fontsize=14)


# %% [markdown]
# Le code ci-dessous permet d’afficher les nuage de points et les ellipses
# correspondants aux gaussiennes (centrées sur la moyenne et d’axes égaux
# aux covariances) en surimpression.
#
# **Note**: il n’est pas indispensable de comprendre les détails de la
# fonction `make_ellipses` pour ce TP, celle-ci faisant appel à des
# fonctionnalités avancées de la bibliothèque graphique `matplotlib`.

# %% hide-output=false
ax = plt.axes()
make_ellipses(gmm, ax)
ax.scatter(X[:, 0], X[:, 1], s=3)
ax.set_xlim(0, 10)
ax.set_ylim((-1.5, 1.5))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Question
#
# Tirer aléatoirement 200 points dans le modèle de mélange. Afficher le résultat.
#
# **Indice**: utiliser une des méthodes de l’objet [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).

# %% [markdown]
# ## Correction
#
# `sklearn` intègre la méthode `sample()` pour échantillonner dans le mélange :

# %% hide-output=false
X_samples = gmm.sample(200)[0]
plt.scatter(X[:, 0], X[:, 1], s=3, alpha=0.5)
plt.scatter(X_samples[:, 0], X_samples[:, 1], marker='x', s=20)

# %% [markdown]
# ## Question
#
# Tirer aléatoirement 100 points dans la première composante du mélange. Afficher le résultat.
#
# **Indice**: la moyenne et la matrice de variance-covariance de la composante numéro $ i $ du mélange sont stockées respectivement dans `gmm.means_[i]` et `gmm.covariances_[i]`. La fonction `multivariate_normal` de [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html) peut être utile.

# %% [markdown]
# ## Correction
#
# Chaque composante du mélange est une gaussienne multivariée dont la moyenne et la matrice de covariance sont estimées lors de l’apprentissage. Pour échantillonner selon une composante précise, il suffit donc de tirer selon cette gaussienne :

# %% hide-output=false
i = 0 # numéro de la composante souhaitée
X_samples = np.random.multivariate_normal(gmm.means_[i], gmm.covariances_[i], size=100)
plt.scatter(X[:, 0], X[:, 1], s=3, alpha=0.5)
plt.scatter(X_samples[:, 0], X_samples[:, 1], marker='x', s=20)

# %% [markdown]
# ## Exercice 2 : Autoencodeurs et génération d’images
#
# Cet exercice présente l’utilisation des autoencodeurs avec PyTorch. Nous
# allons utiliser un modèle simple, entièrement connecté, permettant de
# compresser une image en une représentation vectorielle.
#
# Cet exercice utilise MNIST (ou FashionMNIST) comme jeu de données.

# %% hide-output=false
# Imports liés à torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Matplotlib
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm, trange

# %% hide-output=false
mnist = torchvision.datasets.FashionMNIST("data/", download=True, transform=torchvision.transforms.ToTensor())

# %% [markdown]
# ## Question
#
# Visualiser les premières images du jeu de données.

# %% [markdown]
# ## Correction

# %% hide-output=false
fig = plt.figure()
for idx, (img, label) in enumerate(mnist):
    if idx > 4:
        break
    fig.add_subplot(1,5,idx+1)
    plt.imshow(img[0].numpy())
    plt.axis('off')
    plt.title(label)
plt.show()

# %% [markdown]
# ## Question
#
# En utilisant l’interface `nn.Sequential` de PyTorch, écrire le code qui définit un modèle autoencodeur entièrement connecté.
#
# Ce modèle prend en entrée une image de dimensions $ 28\times28 $ sous forme d’un vecteur aplati de longeur $ 784 $. On définira une variable `hidden_state` qui permet de contrôler la taille du code $ z $ en sortie de l’encodeur.
#
# Le décodeur devra prendre en entrée un code $ z $ de longueur `hidden_state` et produire un vecteur aplati de longueur $ 28\times28 = 784 $ (identique à l’image). On choisira une valeur raisonnable pour la dimension du code (par exemple, entre 30 et 250).

# %% [markdown]
# ## Correction
#
# On se borne ici à un auto-encodeur « simple », c’est-à-dire une combinaison de deux perceptrons multi-couches. On pourrait aussi utiliser un encodeur convolutif.

# %% hide-output=false
hidden_state = 128

encoder = nn.Sequential(
    nn.Linear(28*28, 1024),
    nn.ReLU(),
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, hidden_state)
)

decoder = nn.Sequential(
    nn.Linear(hidden_state, 256),
    nn.ReLU(),
    nn.Linear(256, 1024),
    nn.ReLU(),
    nn.Linear(1024, 28*28),
    nn.Sigmoid(),
)

autoencoder = nn.Sequential(encoder, decoder)


# %% [markdown]
# ## Question
#
# Y a-t-il un intérêt à utiliser une fonction de non-linéarité en sortie du décodeur? Justifier.

# %% [markdown]
# ## Correction
#
# Lorsque l’espace de sortie (et donc l’espace d’entrée) n’est pas bornée, il n’est généralement contre-productif d’appliquer une non-linéarité en sortie du décodeur. Par exemple, si l’on souhaite reconstruire des valeurs réelles positives et négatives, il ne faut surtout pas appliquer de ReLU (qui va interdire les valeurs négatives).
#
# Dans notre cas, les valeurs des pixels sont cependant bornées entre 0 et 1. Il est possible que le décodeur produise des valeurs légèrement négatives ou légèrement supérieures à 1, ce qui produira des artefacts dans l’image si on ne les enlève pas (*clipping*). Une solution élégante est d’appliquer une non-linéarité sigmoïde qui va contraindre les valeurs des pixels dans la plage (0,1) désirée.

# %% hide-output=false
def train(model, dataset, epochs=20, device="cpu"):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())
    flatten = nn.Flatten()
    model = model.to(device)
    model.train()

    t = trange(1, epochs + 1, desc="Entraînement du modèle")
    for e in t:
        avg_loss = 0.
        for batch, _ in tqdm(dataloader, leave=False):
            batch = batch.to(device)
            model.zero_grad()
            batch = flatten(batch)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        avg_loss /= len(dataloader)
        t.set_description(f"Epoch {e}: loss = {avg_loss:.3f}")
    return model


# %% [markdown]
# ## Question
#
# Pourquoi privilégie-t-on dans ce cas précis la fonction de perte $ L_1 $ (*mean absolute error*) plutôt que de la fonction de perte $ L_2 $ (*mean squared error*) ?

# %% [markdown]
# ## Correction
#
# L’erreur quadratique moyenne « écrase » les petites valeurs de l’erreur (inférieures à 1) à cause du carré. Dans notre cas, les valeurs des pixels sont entre 0 et 1. Pour avoir un gradient plus élevé même pour des erreurs faibles, on utilise donc plutôt l’erreur absolue.

# %% [markdown]
# ## Question
#
# En utilisant la fonction `train()` définie ci-dessus, réaliser l’apprentissage de l’autoencodeur défini précédemment.

# %% [markdown]
# ## Correction

# %% hide-output=false
train(autoencoder, mnist, epochs=10, device="cpu")

# %% [markdown]
# ## Question
#
# Pour quelques images du jeu de données, visualiser l’image originale et sa reconstruction.

# %% [markdown]
# ## Correction

# %% hide-output=false
data, _ = mnist[444]
plt.imshow(data.numpy()[0]) and plt.show()
with torch.no_grad():
    reconst = autoencoder(data.reshape(1, -1)).numpy()[0].reshape(28, 28)
plt.imshow(reconst) and plt.show()

# %% [markdown]
# ## Question
#
# Choisir deux images du jeu de données **de classes différentes**. Calculer leurs codes $ z_1 $ et $ z_2 $ à l’aide de l’encodeur. Afficher la reconstruction du code moyen $ 0,5 z_1 + 0,5 z_2 $. Que constatez-vous ?

# %% [markdown]
# ## Correction

# %% hide-output=false
flatten = nn.Flatten()
c1 = encoder(flatten(mnist[444][0]))
c2 = encoder(flatten(mnist[3][0]))
fig = plt.figure(figsize=(12, 6))
alphas = np.linspace(0, 1, 10)
for idx, alpha in enumerate(alphas):
    fig.add_subplot(1, len(alphas), idx+1)
    c = alpha * c1 + (1 - alpha) * c2
    with torch.no_grad():
        img = decoder(c)
        img = img.numpy()[0].reshape(28, 28)
        plt.imshow(img)
plt.show()

# %% [markdown]
# La reconstruction du code interpolé n’est pas particulièrement réaliste et correspond à un mélange flou des deux images.

# %% [markdown]
# ## Question
#
# Afficher les résultats des reconstructions de l’interpolation entre $ z_1 $ et $ z_2 $ pour différentes valeurs du facteur d’interpolation $ \alpha $.

# %% [markdown]
# ## Correction

# %% hide-output=false
c1 = encoder(flatten(mnist[6][0]))
c2 = encoder(flatten(mnist[8][0]))
fig = plt.figure(figsize=(12, 6))
alphas = np.linspace(0, 1, 10)
for idx, alpha in enumerate(alphas):
    fig.add_subplot(1, len(alphas), idx+1)
    c = alpha * c1 + (1 - alpha) * c2
    with torch.no_grad():
        img = decoder(c)
        img = img.numpy()[0].reshape(28, 28)
        plt.imshow(img)
plt.show()

# %% [markdown]
# On remarque que l’interpolation donne des résultats peu plausibles. Lorsque l’on s’éloigne des petites valeurs de $ \alpha $. L’espace latent de l’auto-encodeur n’est pas aussi régulier qu’espéré.

# %% [markdown]
# ## Question
#
# Construire un nouveau jeu de données contenant:
#
# - une matrice `codes`, qui contient pour chaque ligne $ i $ le code $ z_i $ obtenu en encodant l’image $ i $ du jeu de données `mnist`,  
# - une liste `labels` dont l’élément $ i $ représente l’étiquette de l’image $ i $ du jeu de données `mnist`.  
#
#
# Appliquer une réduction de dimension non-linéaire par l’algorithme [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) sur cette matrice de codes de sorte à pouvoir afficher un nuage de points dans le plan. On prendra soin de colorer chaque point en fonction de son étiquette.
#
# Que constatez-vous ?

# %% [markdown]
# ## Correction
#
# Sans aucune supervision, l’espace latent de l’auto-encodeur permet de relativement bien séparer les différentes classes d’objets avec une dimensionalité nettement inférieure à celle des données d’origine.

# %% hide-output=false
codes, labels = [], []
for data, label in tqdm(mnist):
    with torch.no_grad():
        code = encoder(flatten(data))[0]
    codes.append(code.numpy())
    labels.append(label)

codes = np.array(codes)
labels = np.array(labels)

# %% hide-output=false
from sklearn.manifold import TSNE
proj = TSNE(n_components=2)
code_projections = proj.fit_transform(codes)

# %% hide-output=false
fig, ax = plt.subplots(figsize=(12, 12))
scatter = plt.scatter(code_projections[:, 0], code_projections[:, 1], c=labels, cmap=plt.cm.Set1)
legend1 = ax.legend(*scatter.legend_elements(),
                      loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.show()

# %% [markdown]
# ## Question
#
# *Approfondissement* (ne faites cette question qu’à la fin si vous avez encore du temps)
#
# Réaliser la même visualisation par t-SNE en utilisant non pas le code intermédiaire mais directement l’image aplatie. Que constatez-vous? Pourquoi ne peut-on pas utiliser cette approche dans le cas général?

# %% [markdown]
# ## Correction
#
# Le t-SNE appliqué directement sur l’image sépare encore mieux les données. On peut supposer que la réduction de dimension dans l’auto-encodeur nous a fait perdre un peu d’information. Dans le cas général, ce n’est pas possible d’appliquer t-SNE sur les données brutes : la dimensionalité des données est bien trop élevée. Cela ne fonctionne ici que parce que les images sont petites ($ 28 \times 28 $).

# %% hide-output=false
codes, labels = [], []
for data, label in tqdm(mnist):
    codes.append(data.reshape(-1).numpy())
    labels.append(label)

codes = np.array(codes)
labels = np.array(labels)

# %% hide-output=false
from sklearn.manifold import TSNE
proj = TSNE(n_components=2)
code_projections = proj.fit_transform(codes)

# %% hide-output=false
fig, ax = plt.subplots(figsize=(12, 12))
scatter = plt.scatter(code_projections[:, 0], code_projections[:, 1], c=labels, cmap=plt.cm.Set1)
legend1 = ax.legend(*scatter.legend_elements(),
                      loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.show()

# %% [markdown]
# ## Exercice 3 : chaîne de Markov pour la génération de texte
#
# L’objectif de cet exercice est d’implémenter en Python pur une chaîne de
# Markov simple. Rappelons qu’une chaîne de Markov représente une séquence
# $ (X_t) $ où l’on s’intéresse à modéliser les probabilités de
# transitions qui permettent de passer d’un état $ i $ à un état
# $ j $, c’est-à-dire à la matrice:
#
# $$
# p_{i,j} = P(X_{t+1} = j | X_t = i)
# $$
#
# Pour cet exercice, nous allons utiliser un corpus textuel correspondant
# à la traduction française du roman *Frankenstein ou le Prométhée
# moderne* de l’écrivaine [Mary
# Shelley](https://fr.wikipedia.org/wiki/Mary_Shelley) (1797-1851).
#
# La modélisation du problème est la suivante:
#
# - un mot représente un état $ i $ de la chaîne,  
# - on considère la fréquence des
#   [bi-grammes](https://fr.wikipedia.org/wiki/N-gramme) de mots comme
#   probabilités de transition.  
#
#
# Pour être plus précis, on considérera que l’ensemble $ E $ des états
# possibles correspond à un dictionnaire (fini) de mots. Pour chaque
# bi-gramme, c’est-à-dire chaque paire de mots
# $ (X_t = i, X_{t+1} = j) $, nous allons calculer son nombre
# d’occurrences $ N_{i,j} $.
#
# Pour un mot donné $ i $, sa probabilité de transition vers le mot
# suivant $ j $ correspond à la probabilité d’occurrence du bigramme
# $ (i,j) $, c’est-à-dire :
#
# $$
# p_{i,j} = \frac{N_{i,j}}{\sum_k N_{i,k}}
# $$
#
# Observons que le cas des chaînes de Markov d’ordre 2 et supérieur
# correspond à s’intéresser aux n-grammes pour $ n>2 $.
#
# **Note**: les mots du dictionnaire $ E $ sont à prendre au sens
# large. En particulier, on considérera que les symboles de ponctuation
# (?, !, …, etc.) constituent des mots à part entière dans le
# dictionnaire.
#
# Afin de débuter cet exercice, commençons par examiner le corpus et créer
# le dictionnaire d’états $ E $. Pour ce faire, nous transformer la
# chaîne de caractères qui constitue l’intégralité du texte en une
# séquence de mots. Cette opération s’appelle la *tokenization* (ou
# segmentation). Elle s’appuie sur des règles de grammaire propres à
# chaque langage.
#
# **Note**: Le fonctionnement précis de ce processus est hors-sujet pour
# ce cours, se référer au module de traitement automatique du langage de
# RCP217 pour plus de détails.
#
# Nous allons utiliser la bibliothèque de traitement du langage `nltk`
# pour réaliser la *tokenization*:

# %% hide-output=false
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# %% hide-output=false
import requests
# Téléchargement du fichier
r = requests.get("https://cedric.cnam.fr/vertigo/Cours/RCP211/data/frankenstein.txt")
with open("frankenstein.txt", "wb") as f:
    f.write(r.content)
with open("frankenstein.txt") as f:
    corpus = word_tokenize(f.read(), language='french')

# %% [markdown]
# Nous pouvons désormais afficher le début du corpus:

# %% hide-output=false
" ".join(corpus[:50])

# %% [markdown]
# ## Question
#
# Combien de mots uniques ce corpus contient-il ?

# %% [markdown]
# ## Correction

# %% hide-output=false
words = np.unique(corpus)
word_index = {word: idx for idx, word in enumerate(words)}
len(words)


# %% [markdown]
# Afin de créer une chaîne de Markov d’ordre 1 sur ces données, il nous
# faut produire la liste de tous les bi-grammes contenus dans le texte.
# Nous pouvons les générer avec la fonction suivante :

# %% hide-output=false
def make_pairs(corpus):
    for i in range(len(corpus) - 1):
        yield (corpus[i], corpus[i+1])


# %% [markdown]
# Nous allons utiliser l’algorithme suivant pour construire la matrice de
# transition:

# %% hide-output=false
Créer un dictionnaire d vide
Pour chaque paire (i, j) du corpus:
      Si d n'a pas d'entrée pour i:
          d[i] = dictionnaire vide
      Si d[i] n'a pas d'entrée pour j:
          d[i][j] = 1
      Sinon:
          d[i][j] += 1
Renvoyer d


# %% [markdown]
# Remarquons que cet algorithme ne construit pas exactement la matrice de
# transition. Elle produit un double dictionnaire (c’est-à-dire un
# dictionnaire dont les éléments sont également des dictionnaires) tel que
# `d[mot1][mot2]` renvoie le nombre d’occurrences du bi-gramme (mot1,
# mot2) dans le corpus.

# %% [markdown]
# ## Question
#
# Écrire une fonction `build_transition_matrix` qui implémente cet algorithme. Les paires sont calculées à l’aide de la fonction `make_pairs`.

# %% [markdown]
# ## Correction

# %% hide-output=false
def build_transition_matrix(corpus):
    """
        Construit un double dictionnaire d tel que:
            d[mot1][mot2] = nombre d'occurrences du bi-gramme (mot1, mot2) dans le corpus

        Arguments
        ---------
        corpus (str list): une liste de mots

        Renvoie
        -------
        dict: double dictionnaire des nombres d'occurrences des bi-grammes
    """
    pairs = make_pairs(corpus)
    transition_matrix = {}

    for word1, word2 in pairs:
        if word1 not in transition_matrix.keys():
            transition_matrix[word1] = {}
        if word2 not in transition_matrix[word1].keys():
            transition_matrix[word1][word2] = 1
        else:
            transition_matrix[word1][word2] += 1
    return transition_matrix


# %% [markdown]
# ## Question
#
# Pourquoi utiliser un dictionnaire plutôt qu’une matrice (dense) de transition ?

# %% [markdown]
# ## Correction
#
# La matrice dense de transition devrait contenir une entrée pour chaque bi-gramme, c’est-à-dire $ N\times N $ entrées. Vu le nombre de mots unique du dictionnaire, cette matrice occuperait beaucoup de mémoire. L’utilisation du dictionnaire permet de ne stocker réellement que les bi-grammes qui existent (et donc d’ignorer tous les 0 de la matrice de transition).
#
# Nous pouvons désormais calculer la “matrice de transition”:

# %% hide-output=false
transition_matrix = build_transition_matrix(corpus)

# %% [markdown]
# ## Question
#
# Afficher les 10 bi-grammes les plus fréquents.

# %% [markdown]
# ## Correction

# %% hide-output=false
from collections import defaultdict

bigrams = defaultdict(lambda: 0)
for w1, d in transition_matrix.items():
    for w2, c in d.items():
        bigrams[w1, w2] = c

sorted(bigrams.items(), key=lambda x: -x[1])

# %% [markdown]
# ## Question
#
# Afficher les probabilités de transition pour tous les bi-grammes commençant par “une”.

# %% [markdown]
# ## Correction

# %% hide-output=false
transition_matrix["une"]


# %% [markdown]
# ## Question
#
# Écrire une fonction `sample_next_word(word, transition_matrix)` qui, à partir d’un mot `word` et de la matrice de transition tire au hasard le mot suivant en respectant les probabilités de transition $ p_{i,j} $.
#
# **Indice**: on pourra utiliser à bon escient la fonction `np.random.choice` de [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html).

# %% [markdown]
# ## Correction

# %% hide-output=false
def sample_next_word(word, transition_matrix):
    """
       Renvoie un mot au hasard complétant la séquence en suivant les probabilités de transition

       Arguments
       ---------
        - word (str): un mot (première moitié du bi-gramme)
        - transition_matrix (dict): double dictionnaire du nombre d'occurrences

       Renvoie
       -------
         - str: un mot complétant le bi-gramme
    """
    # Dictionnaire des transitions pour "word"
    transitions = transition_matrix[word]
    # Bi-grammes possibles
    candidates = np.array(list(transitions.keys()))
    # Nombre d'occurrence des bi-grammes commençant par "word"
    weights = np.array(list(transitions.values()))
    # Tirage aléatoire
    next_word = np.random.choice(candidates,
                                 p=weights/sum(weights))
    return next_word


# %% [markdown]
# ## Question
#
# Utiliser cette fonction pour générer une phrase commençant par “Une”.

# %% [markdown]
# ## Correction

# %% hide-output=false
w = "Une"
sentence = [w]
while w != ".":
    w = sample_next_word(w, transition_matrix)
    sentence.append(w)
print(" ".join(sentence))


# %% [markdown]
# ## Question
#
# *Approfondissement* (ne faites cette question que si vous avez le temps)
#
# Réaliser le même travail mais dans le cas de la chaîne de Markov d’ordre 2 (raisonner sur des triplets plutôt que des pairs).
#
# **Indice**: on pourra utiliser [defaultdict](https://docs.python.org/fr/3/library/collections.html#collections.defaultdict) pour écrire plus simplement la fonction de construction de la matrice de transition.

# %% [markdown]
# ## Correction

# %% hide-output=false
def make_triplets(corpus):
    for i in range(len(corpus) - 2):
        yield (corpus[i], corpus[i+1], corpus[i+2])


# %% hide-output=false
from collections import defaultdict

def build_transition_matrix_triplets(triplets):
    triplets = make_triplets(corpus)
    transition_matrix = defaultdict(lambda: defaultdict(lambda: 0))
    for word1, word2, word3 in triplets:
        transition_matrix[word1, word2][word3] += 1
    return transition_matrix


# %% hide-output=false
triplet_transition_matrix = build_transition_matrix_triplets(corpus)


# %% hide-output=false
def sample_next_word_triplets(word1, word2, transition_matrix):
    # Dictionnaire des transitions pour "word1", "word2"
    transitions = transition_matrix[word1, word2]
    # Bi-grammes possibles
    candidates = np.array(list(transitions.keys()))
    # Nombre d'occurrence des tri-grammes commençant par "word1", "word2"
    weights = np.array(list(transitions.values()))
    # Tirage aléatoire
    next_word = np.random.choice(candidates,
                                 p=weights/sum(weights))
    return next_word


# %% hide-output=false
w1, w2 = "un", "homme"
sentence = [w1, w2]

while w2 != ".":
    w = sample_next_word_triplets(w1, w2, triplet_transition_matrix)
    w1 = w2
    w2 = w
    sentence.append(w2)

print(" ".join(sentence))

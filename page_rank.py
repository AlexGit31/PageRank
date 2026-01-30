import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as nla

# ==========================================
# 1. Gestion des données et Mapping
# ==========================================


def get_mappings_and_pages(filepath):
    """
    Lit le fichier pour identifier toutes les pages uniques.
    Retourne :
        - page_to_idx : dictionnaire {nom_page: index}
        - pages_list : liste où pages_list[i] donne le nom de la page à l'index i
    """
    unique_pages = set()

    with open(filepath, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for ligne in reader:
            # On vérifie que la ligne a bien le bon format
            if len(ligne) > 3:
                consult = ligne[3].split(";")
                for page in consult:
                    if page != "<":
                        unique_pages.add(page)

    # On trie pour avoir un ordre déterministe
    pages_list = sorted(list(unique_pages))

    # Compréhension de dictionnaire pour créer le mapping inverse
    page_to_idx = {page: i for i, page in enumerate(pages_list)}

    return page_to_idx, pages_list


# ==========================================
# 2. Construction de la Matrice (Logique Pile)
# ==========================================


def build_adjacency_matrix(filepath, page_to_idx, N):
    """
    Construit la matrice d'adjacence L en utilisant la méthode de la PILE.
    Respecte strictement la logique originale.
    """
    L = np.zeros((N, N), dtype=np.float64)

    with open(filepath, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")

        for ligne in reader:
            if len(ligne) <= 3:
                continue

            consult = ligne[3].split(";")
            stack = []  # Initialisation de la pile pour ce chemin

            for token in consult:
                if token == "<":
                    # Retour arrière : on dépile
                    if stack:
                        stack.pop()
                else:
                    # C'est une page visitée
                    page_actuelle = token

                    # Si la pile n'est pas vide, le sommet est la page précédente (source)
                    if stack:
                        page_precedente = stack[-1]

                        i_source = page_to_idx[page_precedente]
                        j_target = page_to_idx[page_actuelle]

                        L[i_source][j_target] = 1

                    # On empile la page actuelle (elle devient potentielle source)
                    stack.append(page_actuelle)
    return L


def normalize_matrix(L, N):
    """
    Gère les 'dangling nodes' (lignes de zéros) et normalise par ligne.
    """
    # 1. Gestion des culs-de-sac (Dangling nodes)
    # Si une page ne pointe vers rien, on suppose qu'elle pointe vers tout le monde (surf aléatoire)
    for i in range(N):
        if np.sum(L[i]) == 0:
            L[i, :] = 1.0 / N

    # 2. Normalisation stochastique (la somme de chaque ligne doit faire 1)
    # On divise chaque ligne par sa somme
    row_sums = L.sum(axis=1)
    # astuce numpy : on divise la matrice par le vecteur colonne des sommes
    L = L / row_sums[:, np.newaxis]

    return L


# ==========================================
# 3. Algorithme PageRank
# ==========================================


def compute_pagerank(beta, epsilon, P, v, s, N, max_iter=1000):
    """
    Calcule le vecteur PageRank q.
    P : Transposée de la matrice d'adjacence normalisée (L.T)
    v : Vecteur de personnalisation
    s : Somme de v (ou facteur de normalisation de v)
    """

    # Démarrage du chronomètre
    start_time = time.perf_counter()

    # Initialisation uniforme
    q = np.ones(N) / N
    q = q / nla.norm(q, 1)

    iterations = 0
    diff = epsilon + 1.0  # Pour entrer dans la boucle

    while diff > epsilon and iterations < max_iter:
        q_old = q.copy()
        sum_q = np.sum(q)

        # Formule originale préservée :
        # q = beta * P * q + terme de téléportation
        q = beta * (P @ q)
        q = q + ((1 - beta) / s * sum_q * v)

        # Renormalisation (sécurité numérique)
        q = q / nla.norm(q, 1)

        # Calcul de la différence pour la convergence
        diff = nla.norm(q - q_old, 1)
        iterations += 1

    # Arrêt du chronomètre
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    return q, iterations, execution_time


# ==========================================
# 4. Fonctions de Visualisation
# ==========================================


def plot_scan_and_add(q):
    """Trace la courbe cumulée (Lorenz curve) des scores PageRank triés."""
    # Tri décroissant
    q_sorted = np.sort(q)[::-1]
    # q_sorted = np.sort(q)
    # q_sorted = np.append(q_sorted, [0])
    # q_sorted = q_sorted[::-1]
    # # Somme cumulée
    q_cumsum = np.cumsum(q_sorted)

    x = np.arange(len(q))

    plt.figure(figsize=(10, 6))
    plt.plot(x, q_cumsum, label="Scan & Add (Cumsum)")
    plt.plot(
        x,
        np.cumsum(q),
        label="Distribution brute (non-triée)",
        alpha=0.5,
        linestyle="--",
    )
    plt.title("Scan & Add : Concentration du PageRank")
    plt.xlabel("Index des pages (triées par importance)")
    plt.ylabel("Probabilité cumulée")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()


def analyze_convergence_beta(L_transposed, v, s, N):
    """Trace le nombre d'itérations nécessaires avant convergence selon Beta."""
    betas = np.linspace(0.1, 0.99, 20)
    iters = []
    fixed_epsilon = 1e-6

    for b in betas:
        _, it, _ = compute_pagerank(b, fixed_epsilon, L_transposed, v, s, N)
        iters.append(it)

    plt.figure(figsize=(8, 5))
    plt.plot(betas, iters, "o-", color="purple")
    plt.title(
        f"Vitesse de convergence en fonction de Beta ($\epsilon={fixed_epsilon}$)"
    )
    plt.xlabel("Beta (Facteur d'amortissement)")
    plt.ylabel("Nombre d'itérations")
    plt.grid(True)
    plt.show()


def analyze_convergence_epsilon(L_transposed, v, s, N):
    """Trace le nombre d'itérations nécessaires selon la précision Epsilon (échelle log)."""
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    iters = []
    fixed_beta = 0.85

    for eps in epsilons:
        _, it, _ = compute_pagerank(fixed_beta, eps, L_transposed, v, s, N)
        iters.append(it)

    plt.figure(figsize=(8, 5))
    plt.semilogx(epsilons, iters, "s-", color="green")  # Echelle log en X
    plt.gca().invert_xaxis()  # On inverse l'axe pour aller du plus grand epsilon au plus petit
    plt.title(f"Vitesse de convergence en fonction de Epsilon (Beta={fixed_beta})")
    plt.xlabel("Epsilon (Précision demandée)")
    plt.ylabel("Nombre d'itérations")
    plt.grid(True)
    plt.show()


def analyze_time_beta(L_transposed, v, s, N):
    """
    Trace le temps de calcul et le nombre d'itérations en fonction de Beta.
    """
    betas = np.linspace(0.1, 0.99, 15)  # On teste 15 valeurs de beta
    times = []
    iters = []
    fixed_epsilon = 1e-6

    print(f"Analyse de la performance pour {len(betas)} valeurs de beta...")

    for b in betas:
        # On ignore le vecteur q (le _) car on veut juste les stats
        _, it, duration = compute_pagerank(b, fixed_epsilon, L_transposed, v, s, N)
        times.append(duration)
        iters.append(it)

    # Création du graphique avec deux axes y (Temps et Itérations)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:red"
    ax1.set_xlabel("Beta")
    ax1.set_ylabel("Temps de calcul (secondes)", color=color)
    ax1.plot(betas, times, "o-", color=color, label="Temps")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Deuxième axe pour les itérations
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Nombre d'itérations", color=color)
    ax2.plot(betas, iters, "s--", color=color, alpha=0.6, label="Itérations")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Impact de Beta sur le temps de convergence et les itérations")
    fig.tight_layout()
    plt.show()


# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    FILE_PATH = "./paths_finished.tsv"

    print("1. Chargement et Mapping...")
    # On récupère le dictionnaire (nom->idx) et la liste (idx->nom)
    page_to_idx, idx_to_page = get_mappings_and_pages(FILE_PATH)
    # print(page_to_idx["Socialism"])
    # print(page_to_idx["Communism"])
    N = len(pages_list := idx_to_page)  # Syntaxe walrus (python 3.8+)
    print(f"   Nombre de pages uniques : {N}")

    print("2. Construction de la matrice L...")
    L = build_adjacency_matrix(FILE_PATH, page_to_idx, N)

    print("3. Normalisation...")
    L = normalize_matrix(L, N)

    # Préparation pour PageRank
    # On utilise la transposée pour l'équation P @ q
    P = L.T

    # Configuration du vecteur de personnalisation (v)
    v = np.ones(N)  # = np.zeros(N) dans le PPR
    indices_cles = [3237, 3422, 919]  # Vos indices spécifiques

    # Vérification que les indices existent (pour éviter crash si fichier différent)
    valid_indices = [k for k in indices_cles if k < N]
    if valid_indices:
        v[valid_indices] = 1
    else:
        # Fallback si indices hors limites, on met tout à 1
        v = np.ones(N)

    s = np.sum(v)  # Somme pour la normalisation dans la formule
    if s == 0:
        s = 1  # Sécurité division par zéro

    # --- Calcul Principal ---
    print("4. Calcul du PageRank...")
    beta_val = 0.85
    epsilon_val = 0.000001

    final_q, iterations, duration = compute_pagerank(beta_val, epsilon_val, P, v, s, N)

    print(f"   Convergence en {iterations} itérations et {duration:.4f} secondes.")

    # --- Affichage des résultats (Top 10) ---
    print("\n--- TOP 10 PAGES ---")
    # On associe score et index, on trie, puis on récupère le nom
    top_indices = np.argsort(final_q)[::-1]  # [:50]
    i = 1
    for idx in top_indices:
        # print(f"Score: {final_q[idx]:.6f} | Page: {idx_to_page[idx]} (Index: {idx})")
        # print(f"{i} & {idx_to_page[idx]}  &  {final_q[idx]:.6f}  \\ ")

        # if idx in [3237, 3422, 919]:
        #     print(f"{i} & {idx_to_page[idx]}  &  {final_q[idx]:.6f}  \\ ")

        if idx_to_page[idx] in ["Soviet_Union", "Marxism", "Vladimir_Lenin"]:
            print(f"{i} & {idx_to_page[idx]}  &  {final_q[idx]:.6f}  \\ ")

        i += 1

    # --- Plots ---
    print("\nGénération des graphiques...")

    # 1. Scan & Add
    plot_scan_and_add(final_q)

    # 2. Courbe d'apprentissage vs Beta
    analyze_convergence_beta(P, v, s, N)

    # 3. Courbe d'apprentissage vs Epsilon
    analyze_convergence_epsilon(P, v, s, N)

    # Le nouveau plot de temps vs Beta
    analyze_time_beta(P, v, s, N)

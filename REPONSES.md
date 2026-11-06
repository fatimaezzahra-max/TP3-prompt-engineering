# TP3 — Prompt Engineering : Classification, Sentiment Analysis


---

## 1. Environnement de travail

- Projet créé avec **uv** (`uv init`, `uv venv`) dans VS Code, environnement virtuel `.venv`.
- Bibliothèques : `langchain`, `langchain-ollama`, `langchain-openai`, `python-dotenv`, `pandas`, `numpy`, `scikit-learn`, `tqdm`, `matplotlib`.
- **LLM utilisé pour toute l'évaluation :** `llama3.2:3b`, en local avec Ollama (`ChatOllama`, `temperature=0`).
- Les clés API (utilisées seulement pour les essais avec Groq) sont stockées dans un fichier `.env`, exclu du dépôt via `.gitignore`.

> Capture : `screenshots/05_ollama.png` (modèle `llama3.2:3b` installé : commande `ollama list`)

---

## 2. Cas d'utilisation : Sentiment Analysis (IMDB)

### Step 1 — Objectifs et métriques
Chaque avis doit recevoir l'étiquette `positive` ou `negative`. La métrique est le **micro-F1** (`sklearn.metrics.f1_score(average="micro")`), calculé en agrégeant TP, FP et FN sur toutes les classes ; elle est plus nuancée que l'accuracy simple et adaptée à un jeu équilibré. Vérification sur l'exemple du cours (TP = 50, FP = 10, FN = 5) : précision 0,833, rappel 0,909, **micro-F1 = 0,870**.

### Step 2 — Données
- Fichier : `IMDB Dataset.csv`, colonnes `review` et `sentiment` ; **50 000 avis** (25 000 positifs / 25 000 négatifs).
- Pour limiter la taille des prompts, seuls les avis de **800 caractères ou moins** sont conservés (les balises `<br />` sont remplacées par des espaces), puis le jeu est rééquilibré : **9 221 positifs / 9 221 négatifs**, soit 18 442 avis.
- Découpage (`train_test_split`, `test_size=0.2`, `random_state=42`) : **14 753** avis comme réservoir d'exemples few-shot et **3 689** comme réservoir gold. Les deux jeux sont sauvegardés en JSON.
- Échantillon gold utilisé pour l'évaluation : **20 avis** (`sample(20, random_state=42)`), comme dans le cours.

> Capture : `screenshots/01_donnees.png`

### Step 3 — Prompts
1. **Zero-shot** : message système seul (classer en `positive` / `negative`, sans explication, avis délimité par des triples backticks).
2. **Few-shot** : même message système + **8 exemples** (4 positifs et 4 négatifs) générés par `create_examples` : l'équilibrage et le mélange aléatoire limitent le biais de l'étiquette majoritaire et le biais de récence.
3. **Chain-of-Thought (CoT)** : mêmes exemples, message système enrichi d'instructions en deux étapes (lire attentivement l'avis, estimer la probabilité qu'il soit positif).

### Step 4 — Évaluation
La fonction `evaluate_prompt` appelle le LLM sur les 20 avis gold, extrait le label (`positive` / `negative`) de la réponse et calcule le micro-F1. Comme le few-shot et le CoT dépendent du tirage des exemples, ils sont évalués **10 fois** avec de nouveaux exemples à chaque run.

**a) Première évaluation (un seul tirage d'exemples)**

| Prompt | micro-F1 |
|---|---|
| Zero-shot | 0,90 |
| Few-shot | 0,95 |
| CoT | 0,95 |

**b) Variabilité sur 10 runs**

| Prompt | Moyenne | Écart-type | Min | Max |
|---|---|---|---|---|
| Few-shot | **0,940** | 0,021 | 0,90 | 0,95 |
| CoT | **0,935** | 0,024 | 0,90 | 0,95 |
| Zero-shot (1 évaluation) | 0,900 | — | — | — |

Scores des 10 runs (`results_micro_f1.csv`) :
- Few-shot : 0,90 / 0,95 / 0,95 / 0,95 / 0,95 / 0,90 / 0,95 / 0,95 / 0,95 / 0,95
- CoT : 0,90 / 0,95 / 0,95 / 0,95 / 0,90 / 0,90 / 0,95 / 0,95 / 0,95 / 0,95

> Captures : `screenshots/02_scores_f1.png`, `screenshots/03_matrices_confusion.png`, `screenshots/04_boxplot_10_runs.png`

---

## 3. Analyse et conclusion

1. **Le few-shot (0,940) et le CoT (0,935) obtiennent des scores très proches, légèrement au-dessus du zero-shot (0,900).** Avec 20 avis gold, un score de 0,95 correspond à une seule erreur et 0,90 à deux erreurs : l'écart entre prompts est inférieur à une erreur en moyenne. Les exemples semblent aider un peu, mais l'écart est trop petit pour conclure.
2. **L'écart entre few-shot et CoT (0,005) est très inférieur à l'écart-type (≈ 0,02).** On ne peut pas dire qu'un des deux est meilleur. Le CoT utilisé ici demande toujours de ne pas expliquer la réponse : il ne déclenche pas de vrai raisonnement visible, ce qui peut expliquer l'absence de gain.
3. **La variabilité due au tirage des exemples est visible mais faible** (écart-type d'environ 0,02 sur 10 runs) : c'est pourquoi le choix d'un prompt doit s'appuyer sur plusieurs runs plutôt que sur une seule évaluation. Le zero-shot n'a été évalué qu'une fois, car il ne dépend pas du tirage d'exemples.
4. **Limites :**
   - seulement 20 avis gold : chaque erreur coûte 0,05 de micro-F1, l'estimation est peu robuste ;
   - avis limités à 800 caractères et petit modèle local (3 milliards de paramètres) : les résultats peuvent différer avec des avis plus longs ou un modèle plus grand ;
   - un seul modèle évalué ; une comparaison avec Groq (`openai/gpt-oss-20b`) n'a pas pu être menée à terme à cause des limites de débit de l'offre gratuite.
5. **Améliorations possibles :** augmenter le nombre d'avis gold (100 ou plus), tester un vrai CoT avec raisonnement avant l'étiquette, comparer plusieurs modèles.

---

## 4. Difficultés rencontrées et solutions

| Problème | Cause | Solution |
|---|---|---|
| `FileNotFoundError: IMDB Dataset.csv` | Le CSV n'était pas dans le dossier du notebook | Recherche automatique du fichier (dossier `data/`) |
| Ollama : 401 puis 410 puis 402 avec `qwen3.5:cloud` | Compte non connecté, modèle cloud retiré, puis crédits requis | Abandon des modèles cloud et usage d'un modèle **local** (`llama3.2:3b`) |
| `qwen3:4b` écrit sa réflexion dans la réponse | Les mots « positive » / « negative » du raisonnement faussent l'extraction du label | Modèle `llama3.2:3b` et fonction `clean_response` qui retire le bloc `<think>` |
| Groq : 404 `model_not_found` (`llama-3.1-8b-instant`) | Modèle indisponible pour la clé | Liste des modèles via l'API (`/models`), choix de `openai/gpt-oss-20b` |
| Groq : 429 (limite de débit) | Prompts few-shot longs × nombreux appels sur l'offre gratuite | Avis limités à 800 caractères, reprise automatique avec pause, et évaluation complète faite en local avec Ollama (sans quota) |

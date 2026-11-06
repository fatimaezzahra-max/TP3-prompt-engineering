# TP3 — Prompt Engineering : Analyse de sentiment

Master AIDC — FST Béni Mellal, Université Sultan Moulay Slimane (USMS)

Classification d'avis de films IMDB (`positive` / `negative`) avec un LLM : comparaison de trois prompts (zero-shot, few-shot, Chain-of-Thought) évalués avec le micro-F1.

**Le document de réponses est [`REPONSES.md`](REPONSES.md)** (démarche, résultats, analyse, difficultés).

## Contenu du dépôt

| Fichier / dossier | Description |
|---|---|
| `TP3_sentiment_analysis.ipynb` | Notebook du cas d'utilisation Sentiment Analysis (Steps 1 à 4), avec les sorties exécutées |
| `REPONSES.md` | Réponses, résultats et analyse |
| `results_micro_f1.csv` | micro-F1 des 10 runs (few-shot et CoT) |
| `screenshots/` | Captures d'écran prouvant la réalisation des tâches |
| `pyproject.toml`, `uv.lock`, `requirements.txt` | Environnement `uv` du projet |

## Résultats (Ollama `llama3.2:3b`, 20 avis gold, 10 runs)

| Prompt | micro-F1 |
|---|---|
| Zero-shot | 0,900 |
| Few-shot | 0,940 ± 0,021 |
| CoT | 0,935 ± 0,024 |

## Reproduire

```bash
uv sync
ollama pull llama3.2:3b      # Ollama doit être lancé
# placer "IMDB Dataset.csv" dans data/ puis exécuter TP3_sentiment_analysis.ipynb
```

Le fichier `.env` (clés API) n'est pas versionné ; il n'est nécessaire que pour Groq (`GROQ_API_KEY`).

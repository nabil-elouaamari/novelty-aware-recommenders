# Novelty-Aware Recommendations on Steam  
Studying the tradeoff between accuracy and novelty using EASE re ranking.
The research plan can be found [here](data/docs/research-plan.md).

## 1. Research Questions

**Main RQ**  
How do re ranking EASE recommendations using genre-based item to history cosine novelty affect the accuracy–novelty trade off?

**Sub RQ**  
Can user-specific novelty weighting, where users are grouped based on profile characteristics, improve the accuracy–novelty trade off compared to a single global novelty weight?

---

## 2. Motivation

Traditional recommenders focus on pure accuracy. That often leads to:

- Repetition of the same popular items
- Over personalisation and filter bubbles
- Low diversity in what users see
- Lower long-term satisfaction, especially for users who like to explore

This project looks at novelty aware re ranking on top of EASE (Embarrassingly Simple Autoencoder). 
The idea is to:

- Start from a strong and simple baseline recommender
- Measure how far recommended games are from what the user already plays
- Explicitly trade off accuracy versus novelty instead of only chasing accuracy

There is also a personal angle. As a player I often ignore the big mainstream titles and look for niche, experimental games. 
Practical recommenders tend to keep pushing the same popular AAA games. 
The project is a way to understand how to tune a system so that it still recommends relevant games but also surfaces more unusual options for users who want that.

---

## 3. Dataset

The project uses a Steam video game dataset with implicit feedback.

### Files

- `bundles.csv`  
  Bundle to game relationships.

- `games.csv`  
  Game level metadata such as genres, publisher, price, basic popularity signals.

- `extended_games.csv`  
  Extra game information such as platform and average playtime.

- `item_reviews.csv`  
  Aggregated review statistics per game.

- `user_reviews.csv`  
  Aggregated review statistics per user.

- `train_interactions.csv`  
  Implicit feedback for training. Each row is `(user_id, item_id)` meaning the user owns or played the game.

- `test_interactions_in.csv`  
  Fold in interactions for the public test users, in the Codabench format.

### Notes and constraints

- Interactions are implicit, there are no explicit ratings.
- There are no timestamps, so chronological splits are not possible. Offline evaluation uses a random holdout per user instead.
- Users have very different history lengths, from very short to very long profiles.
- Genre information is used to build a genre-based distance matrix between items, which is later used to define novelty.

---

## 4. Goals and Scope

### Core goals

- Implement EASE as the main collaborative filtering baseline.
- Tune EASE regularization and popularity blending for the given dataset.
- Define an item to history novelty score based on genre distance.
- Re rank EASE recommendations with:
  - A single global novelty weight
  - User-specific novelty weights based on profile characteristics
- Evaluate the accuracy versus novelty tradeoff and answer:
  - How global novelty weights move us along an accuracy–novelty curve.
  - Whether simple user grouping schemes can improve this tradeoff.

### Out of scope

- Deep models or sequence-based models.
- Rich side information beyond genres for novelty (for example, text embeddings).
- Real user studies. All evaluation is offline and on Codabench.

---

## 5. Project Structure

The repository is organized as follows:

```text
├── data/
│   ├── raw/                            # Original CSV files (not versioned in git)
│   ├── processed/                      # Preprocessed matrices (genre_similarity.npy, genre_distance.npy, etc.)
│   ├── docs/                           # Research plans, evaluation forms, presentation slides
│   └── README.md                       # Short notes on where the data comes from and how to download it
├── src/
│   ├── config.py                       # Global constants (lambda, top K, number of users, pop alpha, seeds)
│   ├── data/
│   │   ├── loader.py                   # load_interactions, load_games, etc.
│   │   └── preprocess.py               # Filters, ID encoding
│   ├── models/
│   │   ├── base_model.py               # BaseRecommender interface
│   │   └── ease.py                     # EASE implementation with popularity blending
│   ├── novelty/
│   │   ├── distance.py                 # build_genre_similarity_matrix, build_genre_distance_matrix
│   │   ├── rerank.py                   # rerank_with_novelty (global novelty weight)
│   │   └── user_groups.py              # user specific novelty weights (profile, genre entropy, combined)
│   ├── evaluation/
│   │   ├── splitter.py                 # train in / train out split per user
│   │   ├── metrics.py                  # NDCG, recall, novelty, intra list similarity, coverage
│   │   └── evaluator.py                # offline evaluate_model function
│   ├── experiments/
│   │   ├── ease_sweep.py               # offline sweep over EASE lambda
│   │   ├── novelty_sweep.py            # global novelty lambda sweep
│   │   └── user_specific_sweep.py      # user specific novelty experiments
│   └── pipelines/
│       ├── recommend.py                # generate_recommendations for offline and Codabench
│       └── save.py                     # save_submission helper for Codabench submissions
├── tests/                              # Unit tests
├── submissions/                        # Codabench ready made submissions (*.zip)
├── notebooks/
│   ├── results/                        # results of offline experiments (image plots, table sweeps)
│   ├── 01_eda.ipynb                    # basic data exploration
│   ├── 02_ease_baseline.ipynb          # EASE implementation
│   ├── 03_ease_tuning.ipynb            # EASE tuning
│   ├── 04_feature_engineering.ipynb    # genre distance matrix and novelty score
│   └── 05_novelty_reranking.ipynb      # novelty re ranking experiments
└── README.md
```

## 6. Setup and Installation

### Requirements
- Python 3.10 or later
- Recommended: a virtual environment (venv, conda, or similar)

### Install dependencies

From the project root:
```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Place the data
The dataset is not stored in git.
You can find more information on how to download the dataset [here](data/README.md).

## 7. Workflow via notebooks
The numbered notebooks in `notebooks/` are meant to be a step by step blueprint of the full pipeline.
You can both read them to understand the project and run them to reproduce the main results.

Suggested order:
1. `01_eda.ipynb`
   - Quick overview of the dataset.
   - Basic checks on users, items, sparsity, and genre information.
2. `02_ease_baseline.ipynb`
    - Builds the offline train in / train out split.
    - Trains a first EASE model.
    - Evaluates NDCG, recall, novelty, and diversity metrics.
    - Contains the code used to create the first Codabench style recommendations.
3. `03_ease_tuning.ipynb`
    - Calls `src/experiments/ease_sweep.py`.
    - Sweeps over different regularization values for EASE.
    - Plots NDCG, recall, and diversity versus lambda.
    - Documents how the final lambda for the baseline was chosen.
4. `04_feature_engineering.ipynb`
    - Uses `games.csv` to build genre vectors for each game.
    - Computes item-item genre similarity and distance matrices.
    - Saves `genre_similarity.npy` and `genre_distance.npy` to `data/processed/`.
    - These matrices are later used both in the novelty metric and in re ranking.
5. `05_novelty_reranking.ipynb`
    - Runs the global novelty sweep via `novelty_sweep.py`.
    - Plots metric versus novelty weight and the accuracy–novelty tradeoff curve.
    - Runs the user-specific grouping experiments via `user_specific_sweep.py`.
    - Creates the scatter plot that compares:
      - Baseline EASE
      - Global novelty configurations
      - User profile, user genre, and combined grouping
    - Contains the code used to generate novelty-aware Codabench submissions (for analysis, not for leaderboard optimization).

If you want to understand the project conceptually, reading these notebooks in order is usually enough.
If you want to reuse the code in your own project, the notebooks show how the pieces in `src/` are wired together.

## 8. Documentation and Comments
Most core functions and classes are documented at two levels:
- **High level**: notebooks explain the purpose of each experiment and how the pipeline fits together.
- **Code level**: key parts such as `EASE`, `rerank_with_novelty`, `user_specific_rerank`, and `evaluate_model` have docstrings that describe inputs, outputs, and the logic.

If you are reading the code for the first time, a suggested path is:
1. `src/models/ease.py`
2. `src/pipelines/recommend.py`
3. `src/evaluation/evaluator.py`
4. `src/novelty/rerank.py` and `src/novelty/user_groups.py`
5. The experiment scripts in `src/experiments/`

## 9. Academic Material
The `docs/` folder contains the material used for grading:
- `Scoresheet Code.pdf`
- `Scoresheet Report.pdf` 
- `Scoresheet Presentation.pptx`
- `Final Presentation.pptx`
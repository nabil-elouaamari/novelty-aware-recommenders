# Novelty-Accuracy Trade-offs in Recommender Systems (EASE + Re-Ranking)

## Research Question

### RQ1:
_How does re-ranking EASE recommendations using genre-based item-to-history cosine novelty affect the accuracy–novelty trade-off?_

### RQ1a:
_Can user-specific novelty weighting, where users are grouped based on profile characteristics, improve the accuracy–novelty trade-off compared to a single global novelty weight?_

## Project Motivation
Traditional recommender systems optimize accuracy, but this often leads to:
- Repetition of popular items
- Over-personalization ("filter bubbles")
- Low content diversity
- Poor user satisfaction over time

This project explores **novelty-aware re-ranking** on top of the EASE model (Embarrassingly Simple Autoencoder) to understand how we can:
- Improve exploratory recommendations
- Increase long-term engagement
- Personalize novelty according to user behavior

I also chose this topic because of my own experience as a gamer.
I'm a very picky player, I rarely enjoy the mainstream, overly popular titles that most people love.
Instead, I'm always looking for odd, niche, experimental games that fit very specific tastes.
Recommendation systems often fail for people like me: they keep pushing the same big AAA games everyone already knows.

So part of the motivation behind this project is personal:
I want to understand how recommender systems can better balance discovery and relevance, especially for players who want fresh, unusual, or under-the-radar experiences instead of the usual top sellers.

## Dataset Description
This project uses a Steam video game interaction dataset.
The dataset consists of six CSV files:
- **bundles.csv** – bundle–to–game relationships
- **extended_games.csv** – deeper info (platform, average playtime)
- **games.csv** – game metadata (genre, publisher, price, popularity)
- **item_reviews.csv** – aggregated review statistics per game
- **interactions** – implicit feedback: which user owns/plays which game
    - **test_interactions_in.csv** – test set interactions
    - **train_interactions.csv** – training set interactions
- **user_reviews.csv** – aggregated review statistics per user


### Important Notes About the Dataset
- No timestamps for interactions → chronological splits are impossible.
- Users have widely varying history lengths.
- Raw data requires preprocessing: filtering, merging, normalizing, and leakage-free feature engineering.


## Project Goals
### Core Research Goals

- Implement EASE as the baseline recommender.
- Add a modular post-processing re-ranking pipeline.
- Evaluate how novelty weighting changes the accuracy-novelty Pareto front.
- Test personalization strategies for user-specific novelty.

## Project Structure
```
├── data/
├── notebooks/
└── src/
```


## Where is the data?
The dataset used in this project is part of the "datasets.zip" package provided through the course materials.
Because the dataset is ~3–4 GB, it is not included in this repository.

To run the code:
1. Download the [dataset](data/raw/README.md).
2. Extract all CSV files into:
    ```
    data/raw/
    ```
   
## Project Documentation
All academic materials related to this project are stored under:
```
docs/
```
These include:
- Scoresheet (Report)
- Scoresheet (Code)
- Scoresheet (Presentation + Q&A)

They are provided for transparency and to clarify the evaluation standards of the project.
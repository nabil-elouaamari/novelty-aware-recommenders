# Research Plan - Nabil El Ouaamari

The goal of this assignment is to create a first version of the research plan that you will execute for your Artificial Intelligence Project,
which you will continue to iterate on and refine.
To do so, you will fill out this MarkDown (md) file.

Your task is to fill out all the sections that have a [TODO] tag, following the instructions provided to guide you through each step. 
By the end of this assignment, you will have a first research plan, ready to be executed and (peer) reviewed. 

## Identifying the Problem

The first step to developing a research plan (and designing an offline evaluation experiment) is to identify the problem that we will be addressing. 
We can also refer to the problem as the evaluation objective, as it is our goal to evaluate how well our solution addresses the problem(s) we care about.

An evaluation objective is typically formulated from the perspective of one (or more) stakeholders, for example, the platform, users, and item providers. 
Each of these stakeholders has goals, that can be achieved by designing a recommender system that has certain desired properties.

Our offline evaluation experiment then evaluates whether the solution (algorithm) has more or less of the desired properties than a representative selection of other algorithms (the baselines).
If this is the case, the solution (algorithm) is better suited to our goal than any other algorithm. 

### Stakeholders

#### Task
Answer the following questions in the context of the research questions/directions you had in mind for your project:
- Who are the stakeholders of your system?
- Which stakeholder(s) do you care about?
- Is one stakeholder more important than the others?

#### Answer

The Stakeholders of my system are:
- Users (players) : They want to play relevant yet engaging game recommendations.
- Item Providers (publishers/developers) : They want fair exposure of their games to the right audience.
- Platform (Steam) : They want to maximize retention and revenue by providing a good user experience.

The stakeholder(s) I care about:
Primarily users and secondly item providers.

Is one stakeholder more important than the others?
Yes, users are the most important stakeholders because their satisfaction directly impacts the platform's success.

### Stakeholder Goals

#### Task
Now that you have listed your stakeholder(s), please answer the following questions in the context of the research questions/directions you had in mind for your project:
- What are these stakeholder(s) goals?
- Do these goals overlap or are they at odds with each other?
- Which stakeholder goals will you pursue in your project? 

#### Answer

The goals of the stakeholders are:
- Users: Find games that match their interests while discovering new and varied experiences.
- Item Providers: Fair exposure to their games and reaching the right audience.
- Platform: Increase user retention

Do these goals overlap or are they at odds with each other?
There is some overlap, as improving user experience can also benefit item providers and the platform.
But increasing novelty can decrease accuracy, which may conflict with user satisfaction.

Which stakeholder goals will you pursue in your project?
- Primary: improve user-side experience by enhancing novelty without major accuracy loss.
- Secondary: Check whether personalized novelty-weight can improve the balance between user satisfaction and item provider exposure.

### Desired System Properties

#### Task
Using your stakeholder goals, think about how these goals can be translated to desired properties of a recommendation algorithm. 
Please answer the following questions in the context of the research questions/directions you had in mind for your project:
- What desired properties should the algorithm have to achieve these goals?
- Can it have all these properties at the same time?

#### Answer

The desired properties the algorithm should have to achieve these goals are:
- Reasonable Accuracy: NDCG and Recall remain close to baseline values.
- High Novelty: The algorithm should recommend items that are new, less familiar, or further from what the user has already interacted with. (item-to-history distance)
- Balance between novelty and accuracy: The algorithm shouldn't over-prioritize either goal. It should find a 'sweet spot' where both novelty and accuracy are optimized.
- Personalization of novelty: The algorithm should adapt the level of novelty based on individual user preferences and behaviors.
  - explorers (e.g. players who frequently try new games) may prefer higher novelty
  - loyal players (e.g. players who stick to familiar genres) may prefer lower novelty

Can it have all these properties at the same time?
Not entirely, as there is a trade-off between accuracy and novelty.
The aim is to locate a Pareto-optimal region where accuracy loss is minimal and novelty gain is substantial.


## Coming Up with a Solution

After defining the problem, and thus, the objective of our offline evaluation experiment, we can move on to designing possible solutions. 
For example, we could argue that "ItemKNN will outperform deep neural networks in a news context, because the frequency with which a recommendation model is updated is more important for its online success in the news domain than its offline ranking accuracy and ItemKNN can be updated more frequently than deep neural networks".
This is an example of a hypothesis.
The best hypotheses are rooted in theory.
This can be either a proven theory, but may also be based in observation or intuition.  
In general, any theory is preferable to no theory.
Returning to our earlier example, we could base this hypothesis on our observations of the production news recommender system at Froomle: In A/B tests, simple models that could be (and were) updated more frequently, consistently outperformed more complex models that required more training time. 
We could also base this hypothesis on a combination of proven facts and intuition: It is a proven fact that news relevance changes rapidly as news items age. As such, a recommendation algorithm that can rapidly and frequently incorporate the latest news items should outperform algorithms that cannot. Finally (and ideally), we could base this hypothesis on proven theory: Verachtert, Jeunen, and Goethals [REF] have shown that models quickly grow stale in the context of news recommendation. As such, a recommendation algorithm that lends itself to frequent and rapid updates should perform better in a news context. 

If we are unable to come up with some theory for why a hypothesis may be true, it is usually best to phrase it as a research question instead.
For example: "Is update frequency more important that offline ranking accuracy in the news domain? If so, will ItemKNN outperform deep neural networks?"

### Research Question & Hypothesis

#### Task

Please answer the following questions in the context of the evaluation objectives you had in mind for your project:
- What is my hypothesis/research question? 
- Do I have a clear hypothesis, rooted in theory, or should I pose a research question instead?
- What theory is my hypothesis based on? 
- Is this theory based on observations, intuition or is it a proven theory? 

#### Answer

<b>Hypothesis/Research Question:</b></br>
"How does adjusting the novelty weight in a post-processing re-ranking algorithm affect the trade-off between accuracy and novelty, 
and can personalize novelty weighting based on user behavior improve this balance?"
Hypothesis: 
Increasing the novelty weight (λ) in the re-ranking step will raise the system's novelty (as measured by Item-to-History Distance) while slightly lowering accuracy (NDCG, Recall).
However, when novelty weighting is personalized by user behavior, for example, giving higher novelty weights to "explorer" users and lower ones to "loyal" players, the overall trade-off will improve.

<b>Do I have a clear hypothesis, rooted in theory, or should I pose a research question instead?</b></br>
Yes, I have a clear hypothesis, since it builds on established theory in recommender systems about the accuracy-novelty trade-off and the effect of personalization on recommendation relevance.

<b>What theory is my hypothesis based on?</b></br>
This hypothesis is based on two theoretical foundations in recommender system research:
1. Accuracy-Novelty Trade-off Theory: There is a well-documented trade-off between accuracy and novelty in recommender systems. Increasing novelty often leads to a decrease in accuracy, as more novel items may be less relevant to the user's established preferences. Examples are <b>Kaminskas, M., & Bridge, D. (2016). Diversity, serendipity, novelty, and coverage: A survey and empirical analysis of beyond-accuracy objectives in recommender systems.</b> and <b>Vargas, S., & Castells, P. (2011). Rank and relevance in novelty and diversity metrics for recommender systems.</b>
2. Personalization and user heterogeneity theory: Different users have different tolerance levels for novelty and exploration. "Explorers" are more open to new content, while "loyal" users prefer recommendations similar to what they already like. Personalized weighting can therefore better match individual user preferences, potentially improving satisfaction and maintaining accuracy. <b>Steck, H. (2018). Calibrated recommendations.</b> and <b>Zhang, Y. C., Séaghdha, D. Ó., Quercia, D., & Jambor, T. (2012). Auralist: Introducing serendipity into music recommendation. In Proceedings of the 5th ACM International Conference on Web Search and Data Mining (WSDM '12), pp. 13–22.</b>

<b>Is this theory based on observations, intuition or is it a proven theory?</b></br>
This theory is based on a combination of proven theories and empirical observations:
- The <b>accuracy-novelty trade-off</b> and the impact of personalization on recommendation relevance are well-established in recommender system literature, supported by numerous studies and experiments.
- The <b>user-behavior personalization</b> part is intuitive, derived from practical observations in recommender systems, users differ in how much "exploration" they appreciate.


## Designing an Offline Evaluation Experiment

Now that we have identified the problem and objectives of our evaluation, as well as defined our research question or hypothesis,
we can start designing an offline evaluation experiment that is capable of validating our hypothesis or answering our research question. 
The design of a typical offline evaluation evaluation experiment consists of seven steps, guided by two main principles.

### Principles of Offline Evaluation 

The two main principles of offline evaluation design are validity and reliability.

#### Validity 

There are different types of validity.  
In the context of your artificial intelligence project, the type of validity we care about most is internal validity: 
Ruling out alternative explanations for the results of our offline evaluation experiment.
To maximize internal validity, we have to consider how we might be unintentionally biasing our results.
There are three major threats to internal validity: confounding, selection bias, and researcher bias.
Confounding is usually the result of treating the new recommendation algorithm and the baselines differently.
Selection bias is a result of differences between the sample used in the evaluation and the population it was sampled from.
The final (and most dangerous) threat is researcher bias: The fact that researchers are generally flexible regarding their experimental designs in terms of baselines, datasets, evaluation protocol, and performance measures may easily lead to researcher bias, where algorithm designers may only look for evidence that supports the superiority of their own method. 

To safeguard the internal validity of an experiment it is important to consider the following questions:
-  How might we treat the new recommendation algorithm and the other algorithms differently? 
-  How can we equalize the treatment of all algorithms under comparison (as much as possible)? 
-  Is our preprocessed evaluation dataset representative of the larger population? 
-  Are our experimental design choices in line with our research questions and hypotheses? 
-  How are we (unintentionally) biasing our results?

#### Task

Consider how you might introduce selection bias, confounding, and researcher bias into your experiment, and describe
what countermeasures you will take in the design of your experiment to avoid it.

#### Answer
##### 1. How I could introduce selection bias (Is my evaluation sample representative?):
- Over-filtering the dataset removes casual/short-history users and will inflate accuracy metrics.

Countermeasure:
- Minimal filtering (as low as feasible); report exact filter
- Analyze and report dataset characteristics before and after filtering.

##### 2. How I could introduce confounding (Are methods treated differently?):
- Using different data splits for baselines and my method.
- Personalization rule based on test behavior

Countermeasure:
- Classify users as explorer/loyal based only on their training interactions (for example, the number of different genres they played before the test period).
- The experiment must treat both the baseline (EASE) and your novelty-augmented EASE: with the same input data, same filtering rules, same evaluation protocol, and same tuning procedure.

##### 3. How I could introduce researcher bias (Am I (unintentionally) nudging results?):
- Cherry-picking a λ that looks best on test; hiding λ settings that underperform.
- Choosing baselines that are weak or not well-suited to the dataset.
- Tuning hyperparameters of my method more extensively than baselines.
- Ignoring negative or inconclusive results

Countermeasure:
- Plot the full accuracy-novelty trade-off curve for a wide range of λ values, not just the "best" one. 
- Predefine all parameters and metrics before running experiments. For example, decide in advance:
  - your λ grid,
  - the definition of “explorer” vs “loyal” users,
  - primary metrics (NDCG, Item-to-History Distance).


#### Reliability

The reliability of an offline evaluation experiment encompasses many aspects.
Firstly, it means the experiment is demonstrably and reasonably free of errors and inconsistencies. 
However, truly reliable experimental code should also be re-runnable, repeatable, reproducible, reusable, and replicable.  
An experiment is re-runnable if it is executable, or in other words, if a "restart and run all" does not produce errors -- as many JuPyTer notebooks do when executed from top to bottom.
An experiment is repeatable if it produces the same results each time it is run. 
An experiment is reproducible if someone else can take the code, run it, and reobtain the reported results. For example, when your lecturers run your JuPyTer notebooks at the end of the class to see whether it produces the results you describe in your paper.
An experiment reusable when it is easy to understand and modify, thus, when it is well-documented and simple. 
An experiment is replicable when the description of the code in the report is sufficient to write the code from scratch and obtain similar results.  

#### Task

Consider how you can make your experiment reliable, re-runnable, repeatable, reproducible, reusable, and replicable and describe
what measures you will take in the design of your experiment to ensure it is.

#### Answer

I would ensure it by doing the following:
1) **Error-free & re-runnable**
   1) Single entrypoint for both the script and notebook.
   2) Restart-and-run-all works without errors.
   3) Use virtual environments (e.g. conda) to manage dependencies.
   4) Assertions and sanity checks throughout the code.
2) **Repeatable**
   1) Fixed seeds for all random processes (data splits, model initialization).
   2) Deterministic data processing and model training.
3) **Reproducible** 
   1) Comprehensive README with setup and execution instructions.
   2) Data versioning: store checksums or hashes of datasets used.
   3) Share code via GitHub with clear commit history.
4) **Reusable**
   1) Modular code structure with functions and classes.
   2) Inline comments and docstrings for clarity.
   3) Configuration files for easy parameter adjustments.
5) **Replicable**
   1) Detailed methodology section in the report.
   2) Clear explanation of algorithms, data processing, and evaluation metrics.
   3) Share datasets or provide links to public datasets used.
   4) Include pseudocode for key algorithms and processes.
   5) Table of hyperparameters and their values.

### Aspects of Offline Experimental Design

Finally, we can move on to the seven aspects of offline experimental design:
- Data Selection
- Data Preprocessing
- Data Splitting
- Selection of Baselines
- Hyperparameter Tuning
- Metric Selection

Our goal is to make experimental design decisions that will increase the validity of the result to support our evaluation objective. 

### Dataset Selection

Although the dataset has already been selected for your artificial intelligence project, it is still helpful to think about its properties
and what parts of the dataset you will use. 

#### Task 
Please answer the following questions in the context of the dataset you will use for this project:
- When was the dataset collected?
- Over which timespan?
- In which region?
- What sampling/filtering strategy was applied?
- How did users find the items in the dataset?
- What are the dataset statistics?

#### Answer

The dataset I will use for this project is a preprocessed version of Julian McAuley’s Steam dataset. 
It contains purchase and playtime logs plus rich item metadata for games on the Steam platform. 
The original data consists of purchase histories, reviews, and game features for Australian Steam users, collected between October 2010 and January 2018.

- When was the dataset collected?  
  The underlying data was collected between October 2010 and January 2018.

- Over which timespan?  
  Roughly 7 years of user activity on Steam, from late 2010 to early 2018.

- In which region?  
  This version of the dataset focuses on Australian Steam users.

- What sampling/filtering strategy was applied?  
  The public UCSD dataset was first constructed from Steam logs. Then the course version I use was further merged with an additional metadata source and downscaled to a more manageable size by the lecturer (for example, by subsetting users/items and keeping the most informative features). So my project starts from a preprocessed subset rather than the full raw Steam logs.

- How did users find the items in the dataset?  
  The dataset does not contain this directly, but on Steam users typically arrive at games through a mix of: browsing the store, searching by name or tag, reading reviews, curated lists, and personalized recommendations, plus sales and promotional pages.

- What are the dataset statistics?  
  In the course version, the interactions form a very sparse user item matrix: there are many users and many games, but each user only interacts with a small fraction of all available games. In the report I will summarize this with the number of users, number of games, total interactions and the resulting density (interactions divided by users times items).



### Data Preprocessing

#### Task

Please answer the following questions in the context of the hypothesis/research question you had in mind for your project:
- Why do you want to filter, sample, or otherwise preprocess the datasets?
- What types of filters/feature engineering do you want to apply?
- What is the best way to apply it?
- How does this filtering affect the dataset characteristics?
- How could my feature engineering introduce leakages?
- Is the sampled/filtered/processed dataset still representative of the original dataset? 

#### Answer

- Why do you want to filter, sample, or otherwise preprocess the datasets?
  - Remove users or games with too few interactions (to ensure statistical reliability).
  - Reduce computational cost so that re-ranking experiments can be run multiple times with different novelty weights (λ).
- What types of filters/feature engineering do you want to apply?
  - Minimum interaction threshold: Keep users with at least x interactions and games with at least y interactions. (e.g. 5-10)
  - Implicit feedback conversion: Convert certain interaction types (e.g. playtime) into binary interactions.
  - User behavior classification: Classify users as "explorers" or "loyal" based on their interaction diversity.
- What is the best way to apply it?
  - Load and merge relevant data files.
  - Apply filtering in a single preprocessing step before data splitting to avoid data leakage.
  - Compute per-user and per-item statistics (e.g. # interactions, genres, tags).
  - Save the cleaned dataset as Parquet/CSV for reproducibility.
  - Compute user-type and novelty scores after splitting (using only training data to prevent leakage).
- How does this filtering affect the dataset characteristics?
  - Reduces the number of users and items, potentially increasing average interactions per user/item.
  - May skew the distribution towards more active users and popular games.
- How could my feature engineering introduce leakages?
  - User segmentation leakage: If the explorer/loyal classification uses validation or test interactions, the personalization step would "know" future user behavior.
  - Novelty computation leakage: If Item-to-History Distance includes games from the test set, the re-ranking would unfairly benefit from future information.
- Is the sampled/filtered/processed dataset still representative of the original dataset?
  - To some extent, yes. While filtering may exclude casual users and niche games, the remaining dataset should still reflect the core user base and popular games on Steam.
  - To maintain transparency, I will:
    - Report dataset statistics before and after filtering.
    - Discuss representativeness as a limitation (e.g. novelty effects may differ for very casual users not included after filtering).

### Data Splitting

#### Task 
Please answer the following questions in the context of the hypothesis/research question you had in mind for your project:
- Do we have enough users in the datasets to split users into different sets?
- How large a timeframe spans our datasets?
- Can I split it in a time-aware fashion? 
- Is the order of interactions likely important in this application domain?
- Are there large differences in the lengths of users' interaction histories in the datasets?
- Do all users have reasonably long interaction histories, such that we may assume their preferences are well-known?
- In what way will I split the dataset? 

#### Answer

- Do we have enough users in the datasets to split users into different sets?
  - Yes, the Steam dataset has a large user base, allowing for a robust train/validation/test split.
- How large a timeframe spans our datasets?
  - The dataset includes user game interactions collected over multiple years, between October 2010 and January 2018, for Australian Steam users.
- Can I split it in a time-aware fashion?
  - Not directly. Since timestamps are not included, a true chronological split (training on older interactions and testing on newer ones) is not possible. Instead, I will apply a per-user random leave-one-out split, which is commonly used in implicit-feedback datasets without time information. This still preserves internal validity because each user contributes data to all subsets.
- Is the order of interactions likely important in this application domain?
    - In principle, yes, gaming interests can evolve over time as users explore new genres or trends. However, without timestamps, the order cannot be modeled explicitly. Therefore, for this experiment, I assume that the user's preferences are relatively stable across their recorded interactions.
- Are there large differences in the lengths of users' interaction histories in the datasets?
  - Some users have long histories (dozens or hundreds of games), while others have very short ones (just a few games).
- Do all users have reasonably long interaction histories, such that we may assume their preferences are well-known?
  - No, not all users have long histories. To ensure reliable evaluation, I will filter out users with fewer than a minimum number of interactions (e.g. 5-10).
- In what way will I split the dataset?
  - For each user, randomly hold out one item for validation and one item for testing. All remaining interactions will form the training set. The random seed will be fixed for reproducibility. 

### Selection of Baselines

#### Task 
Please answer the following questions in the context of the hypothesis/research question you had in mind for your project:
- What other recommendation algorithms have been published recently that address the desired property and goal? 
- What general-purpose baselines have been shown to have good performance, and reasonable computational load, on this dataset, or, in this application domain?

#### Answer

In the context of novelty-aware recommendation, several approaches have been proposed in recent years to balance accuracy and novelty through re-ranking or multi-objective optimization:
- Calibrated recommendation models (Steck, 2018): These aim to adjust recommendation lists so that they better match user preference distributions while improving novelty and diversity.
- Serendipity and novelty-enhancing re-ranking (Vargas & Castells, 2011; Kaminskas & Bridge, 2016): These studies introduce post-processing algorithms that reward less familiar or less popular items to increase user discovery without drastically reducing accuracy.
- Popularity-based penalization methods (Abdollahpouri et al., 2019): These modify ranking scores to down-weight overly popular items, indirectly promoting novelty and fairness.
- Graph-based and hybrid neural models (LightGCN, NGCF, or Multi-VAE): These models achieve high accuracy and can be extended with side objectives (diversity or novelty), but they are significantly more complex and computationally demanding.

For baselines, I will consider:
- EASE (Embarrassingly Shallow Autoencoders) (main baseline)
  - Known for strong performance and interpretability on implicit data.
  - Chosen because it produces stable item relevance scores, which can be modified consistently using novelty weights (λ).
  - Computationally efficient, allowing for multiple re-ranking experiments.
- ItemKNN (secondary baseline - optional):
  - A classic neighborhood-based method that is simple and interpretable.
  - Provides a solid benchmark for comparison, especially in terms of accuracy.


### Hyperparameter Tuning

#### Task 
Please answer the following questions in the context of the hypothesis/research question you had in mind for your project:
- Which hyperparameters do the recommendation algorithms in this offline evaluation experiment have? 
- What is a reasonable range of values they may take, and we should explore? 
- Should we use random search or TPEs to optimize hyperparameters? 
- What target metric should we set? 
- How sensitive is the recommendation algorithm to (small) changes in the hyperparameter values?

#### Answer

- Which hyperparameters do the recommendation algorithms in this offline evaluation experiment have? 
  - EASE (base model):
    - <code>alpha</code> (regularization strength)
  - Novelty-augmented EASE (my method):
    - <code>λ_global</code> (novelty weight for global re-ranking)
    - <code>λ_personalized</code> (novelty weight for personalized re-ranking based on user type)
    - User classification threshold (how we classify users; e.g. threshold on genre entropy or #unique genres in train).
  - (optional) itemKNN:
    - <code>k</code> (number of neighbors)
    - <code>similarity_metric</code> (e.g. cosine, jaccard)
    - <code>shrinkage</code> (regularization for similarity scores)

- What is a reasonable range of values they may take, and we should explore?
  - EASE:
    - <code>alpha</code>: [1e-6, 1e2] (log scale)
  - Novelty-augmented EASE:
    - <code>λ_global</code>: [0.0, 1.0] (linear scale)
    - <code>λ_personalized</code>:
      - For **explorers**: [0.5, 1.0]
      - For **loyal players**: [0.0, 0.5]
  - Segmentation rule (computed on train only):
    - **Measure**: genre entropy or #unique genres played. 
    - **Thresholds**: quantile-based, e.g. explorers = top 40%, loyal = bottom 40% (middle 20% use λ_global).
  - (optional) itemKNN:
    - <code>k</code>: [10, 200] (linear scale)
    - <code>similarity_metric</code>: {cosine, jaccard}
    - <code>shrinkage</code>: [0, 1000] (linear scale)

- Should we use random search or TPEs to optimize hyperparameters?
  - Given the relatively low number of hyperparameters and the need to explore a wide range of values (especially for λ), I will use a grid search approach for λ values combined with random search for EASE's α. This allows systematic exploration of the novelty weights while efficiently tuning the base model.

- What target metric should we set? 
  - Maximize Item-to-History Distance subject to NDCG@10 ≥ NDCG_baseline@10 − δ, where δ is a small allowed drop (e.g., 5% relative or 0.01 absolute, pre-declared).  
  - Report the full accuracy-novelty trade-off curve across λ values.

- How sensitive is the recommendation algorithm to (small) changes in the hyperparameter values?
  - EASE <code>alpha</code>: typically low-to-moderate sensitivity; performance often flat across a band of α, then degrades when α is too small (overfit) or too large (over-smoothed).
  - Novelty weights:
    - <code>λ_global</code>: moderate sensitivity; small increases can lead to noticeable novelty gains but also accuracy drops.
    - <code>λ_personalized</code>: potentially higher sensitivity; misclassification of user types can lead to suboptimal recommendations.
  - (optional) itemKNN:
    - <code>k</code>: moderate sensitivity; too low k can lead to noisy recommendations, too high k can dilute relevance.
    - <code>similarity_metric</code>: choice can significantly affect performance depending on data sparsity.
    - <code>shrinkage</code>: low-to-moderate sensitivity; helps stabilize similarity estimates for low-interaction items.


### Metric Selection

#### Task 
Please answer the following questions in the context of the hypothesis/research question you had in mind for your project:
- Which metric should I use to measure my desired properties?
- Does this metric truly measure the desired property related to our evaluation goal? 
- Has this metric been validated and shown to be a good proxy? 
- What will we optimize the hyperparameters of all recommendation algorithms under comparison for?

#### Answer

- Which metric should I use to measure my desired properties?
  - Accuracy: <code>NDCG@10</code> and <code>Recall@10</code>
  - Novelty: <code>Item-to-History Distance</code> (average distance between recommended items and user's historical interactions)

- Does this metric truly measure the desired property related to our evaluation goal?
  - Yes, NDCG@10 and Recall@10 are standard metrics for evaluating recommendation accuracy, reflecting how well the recommended items match user preferences.
  - Item-to-History Distance effectively captures novelty by quantifying how different recommended items are from what the user has already interacted with.

- Has this metric been validated and shown to be a good proxy?
  - Yes, NDCG and Recall are widely used and validated in recommender system research.
  - Item-to-History Distance has been used in prior studies (e.g., Kaminskas & Bridge, 2016) as a reliable measure of novelty.

- What will we optimize the hyperparameters of all recommendation algorithms under comparison for?
  - For EASE and itemKNN baselines: Optimize for maximum NDCG@10 on the validation set.
  - For Novelty-augmented EASE: Analyze the accuracy-novelty trade-off by varying λ values, aiming to maximize Item-to-History Distance while keeping NDCG@10 within an acceptable range of the baseline.




## Congratulations! You've completed the assignment.

Please upload your research plan to GitHub Classroom.

# Literature collection v6 (Project frontalis)

Scope: whether an individual's experience history can leave a stable, dominant imprint on a learned evaluation, given that training-path noise (data order, SGD randomness) currently explains about 80-90% of persona differences.

Verification levels: "full text read" = not used this round. "abstract only" = abstract fetched and checked this round. "search result snippet only" = seen only in search summaries, not opened.

## Table

| # | Title | Authors | Year | Venue | URL | Theme | Relevance to THE PROBLEM | Verification |
|---|---|---|---|---|---|---|---|---|
| 1 | On the Reproducibility of Neural Network Predictions | Bhojanapalli, Wilber, Veit, Rawat, Kim, Menon, Kumar | 2021 | arXiv preprint | https://arxiv.org/abs/2102.03349 | 1 | Measures churn (disagreement between independently trained models) and names initialization, batch order and augmentation as sources. Abstract does not split seed vs. order; that split is unverified. | abstract only |
| 2 | Nondeterminism and Instability in Neural Network Optimization | Summers, Dinneen | 2021 | ICML 2021 (PMLR 139) | https://arxiv.org/abs/2103.04514 | 1 | Reports that all nondeterminism sources act alike and that end-to-end training instability is the key determinant. Suggests path noise is intrinsic, not one fixable source. | search result snippet only |
| 3 | Accounting for Variance in Machine Learning Benchmarks | Bouthillier et al. (16 authors) | 2021 | MLSys 2021 | https://arxiv.org/abs/2103.03098 | 1 | Models variance from data sampling, initialization and hyperparameters. Proposes a decision rule based on the probability that one method beats another in a single run. Useful for setting a noise bar before claiming a history effect. | search result snippet only |
| 4 | On the Variance of Neural Network Training with respect to Test Sets and Distributions | Keller Jordan | 2023 | arXiv (rev. 2024) | https://arxiv.org/abs/2304.01910 | 1 | Argues run-to-run test variance comes from roughly independent errors and class calibration. Search summaries attributed an init-dominates claim to this paper; the fetched abstract does not say that, so the claim is not used here. | abstract only |
| 5 | Stochastic Weight Averaging Revisited | Guo, Jin, Liu | 2022 | arXiv | https://arxiv.org/abs/2201.00519 | 1 | Frames weight averaging as variance reduction, dependent on how converged the SGD run is. Averaging may suppress path noise, but it may also suppress history signal; a caution. | search result snippet only |
| 6 | Overcoming catastrophic forgetting in neural networks (EWC) | Kirkpatrick et al. | 2017 | PNAS (arXiv 1612.00796) | https://arxiv.org/abs/1612.00796 | 2 | Protects weights important to earlier experience with a quadratic penalty, so early learning leaves a lasting trace. Designed against forgetting across tasks, not to amplify an individual's imprint. | search result snippet only |
| 7 | On Quadratic Penalties in Elastic Weight Consolidation | Huszar | 2017 | arXiv | https://arxiv.org/abs/1712.03847 | 2 | Argues EWC's quadratic penalty is weakly justified and can double-count earlier data. Relevant if consolidation is used to make history stick. | search result snippet only |
| 8 | DualNet: Continual Learning, Fast and Slow | Pham, Liu, Hoi | 2021 | arXiv 2110.00175 (NeurIPS 2021) | https://arxiv.org/abs/2110.00175 | 2 | Complementary-learning-systems design: a fast learner for specific experience and a slow learner for general representation. Gives a concrete split between history-specific and shared weights. | search result snippet only |
| 9 | Profile-to-PEFT: instant personalized LLM adaptation via hypernetwork | Tan et al. | 2025 | arXiv 2510.16282 (ACL 2026) | https://arxiv.org/abs/2510.16282 | 2, 3 | A hypernetwork generates per-user adapters from a profile built from the user's history. Makes history an input to the weights, so the imprint is structural rather than a side effect of the SGD path. Targets LLMs, not small MLPs. | search result snippet only |
| 10 | The Emergence of Individuality (EOI) | Jiang, Lu | 2021 | ICML 2021 (arXiv 2006.05842) | https://arxiv.org/abs/2006.05842 | 3 | Individuality emerges from an intrinsic reward for being identifiable from one's own observations, creating a feedback loop. Agents are cooperative multi-agent, not history-driven, so it is an analogy only. | search result snippet only |
| 11 | Homeostatic reinforcement learning for integrating reward collection and physiological stability | Keramati, Gutkin | 2014 | eLife 3:e04811 | https://elifesciences.org/articles/04811 | 4 | Defines reward as satisfaction of physiological needs, so value is tied to internal state trajectories rather than hand-chosen features. Still hand-specifies the homeostatic variables. | search result snippet only |
| 12 | Homeostatically Regulated Reinforcement Learning (HRRL) perspective | Yoshida, Sprekeler, Gutkin | 2025 | arXiv 2507.04998 | https://arxiv.org/abs/2507.04998 | 4 | Extends the homeostatic framework to deep RL, with behaviors such as risk aversion and anticipatory regulation. Relevant to an internal-state-driven value that depends on history. | search result snippet only |
| 13 | Successor Features for Transfer in Reinforcement Learning | Barreto et al. | 2017 | NeurIPS 2017 (arXiv 1606.05312) | https://arxiv.org/abs/1606.05312 | 4 | Separates environment dynamics (learned from experience) from reward weights. This separation could let history shape the dynamics part while reward weights stay shared. | search result snippet only |
| 14 | Understanding Black-box Predictions via Influence Functions | Koh, Liang | 2017 | ICML 2017 (arXiv 1703.04730) | https://arxiv.org/abs/1703.04730 | 5 | Attributes predictions to training points via influence functions, usable on non-convex nets with approximation. Can separate history-driven from path-driven effects, but approximations may be noisy on deep nets. | search result snippet only |
| 15 | Datamodels: Predicting Predictions from Training Data | Ilyas, Park, Engstrom, Leclerc, Madry | 2022 | arXiv 2202.00622 | https://arxiv.org/abs/2202.00622 | 5 | Predicts a trained model's output from the subset of training data used, and checks by counterfactual retraining. Gives a direct way to measure how much output depends on which history was used, with seeds held fixed. | abstract only |

## What this suggests for design

- Measure the noise floor first, as in rows 2 and 3: repeat same-history runs across seeds and set a decision criterion (probability of exceeding path noise) before claiming any history effect.
- Make history an explicit input to the weights (row 9, hypernetwork from a history summary), so the imprint is built in and not left to the SGD path. This needs a check that the generator does not just memorize run noise.
- Split weights into a history-specific part and a shared part (rows 8 and 13), so the history effect is concentrated in a small set of parameters where it can be measured.
- Tie value to internal state trajectories, not hand-chosen features (rows 11 and 12), which keeps the value tied to experience. Note that the homeostatic variables are still hand-specified.
- Use datamodel-style counterfactual retraining with fixed seeds (row 15) and influence-style attribution (row 14) as the causal test: history effect should exceed path noise under this measure.
- Treat consolidation methods (rows 6 and 7) as optional. They protect early weights but may also freeze path noise; test them against the noise floor.

## What we could not find

- No paper found on fixed data order (or fixed seed) reducing variance across independent runs as a dedicated result. The seed-vs-order split in row 1 is unverified.
- No paper found on emergent personality or idiosyncratic value functions in single-agent RL from differing experience histories (row 10 is multi-agent, cooperative).
- The Kaplanis paper on fast/slow weights in continual RL was not found in search results.
- No paper found on "interoceptive reward" as a phrase. Row 11 and 12 use homeostatic framing instead.
- No paper found combining continual learning with hypernetwork per-user adapters for small MLP learners; row 9 is LLM-based.
- Search summaries attributed claims to some papers (e.g. init-dominates in Jordan 2023, a single-weight churn figure, Fort et al. 2019) that were not verified in the abstract. They are excluded from this table.

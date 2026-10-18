# Project frontalis

**Can an AI grow a personality through experience?** This is a research project that tests that question with experiments.

Outside a conversational AI (an LLM), we place a small neural network, the **Neural Personality Network (NPN)**, that learns from experience what to like and what to avoid. We study whether it becomes a subject with its own individuality. It is intended as the personality base of an AI companion character.

日本語版: [README.md](README.md)

> **Status (2026-10-18): research in progress. We do not yet have a result that shows a personality has formed.** What we obtained, and what does not work yet, are stated plainly in sections 3 to 5.

---

## 1. What this research aims at

Today's conversational AIs can write personality-like text, but they may only be *playing* a personality. We want to know whether, **as a result of accumulated experience, the AI's own evaluation (what it counts as good or bad) is formed, and changes how it takes in the next experience.**

The loop we want to build:

1. The current personality evaluates an experience.
2. That evaluation and the actual outcome update the personality.
3. The updated personality takes in the next experience differently from before.
4. Over time this accumulates into **value judgements specific to that individual, which the designer did not specify one by one.**

Division of roles:

| Component | Role |
| --- | --- |
| LLM (Claude etc.) | An "interpreter" for language understanding and expression. Swappable. |
| NPN | The evaluating subject that is formed by experience. The personality lives here. |
| Experience record | What happened, how it was interpreted at the time, and what was corrected later. |

### Principles (summary)

The governing document is [`FRONTALIS_PHILOSOPHY.md`](https://github.com/sakri9616-coder/Frontalis/blob/4fecbad/FRONTALIS_PHILOSOPHY.md) (pinned commit `4fecbad`; written in Japanese). The principles we hold to:

- **Formation is not expression.** An LLM speaking in a personality-like way is not evidence that the NPN formed one. We separate the two by keeping the LLM and prompt fixed and swapping only the NPN's state: does the judgement change?
- **Individuality is not a fault.** Persistence, lasting aversion and resistance to refutation can be part of a personality. Only computational breakdowns (divergence, saturation) count as faults.
- **Criteria are not moved to fit results.** What counts as success is fixed and recorded before the experiment. Tuning to pass is prohibited.
- **Changing a learning rate or a coefficient alone is not evidence that a personality formed.**

## 2. The central question (redefined on 2026-10-17)

Our early implementations replaced the question we care about with an easier one. If the developer fixes the kinds of values (five hand-designed desires) and only updates their weights with a formula, then **an evaluative viewpoint the designer never thought of cannot arise from experience, by construction.**

We therefore redefined the central question:

> Study mechanisms by which experience lets a system acquire evaluative features, relations and criteria that the developer did not specify in advance, and by which those acquired items change how the next experience is interpreted and learned from.

To say "acquired", three things must be shown:

1. The acquired features **work in new situations** (they are not memorised).
2. **Removing or swapping** them changes later interpretation and learning.
3. Individuals with different experiences **actually diverge** in their evaluations.

## 3. Current progress

### 3.1 At a glance

| What we want to show | Status |
| --- | --- |
| Expectations (predictions of an action's outcome) learned from experience work in new situations | **Shown**, and reproduced across evaluations |
| Differences in experience show up as differences in evaluation | **Only partly.** Mostly buried under random training noise (challenge 1) |
| An individual's evaluation state changes its own next learning | **Shown in how we measure it, but not attributable to our mechanism** (challenge 2) |
| Learned evaluations make individuals into different personalities | **Not shown yet** |
| One NPN state makes different LLMs agree on the direction of a stance | Designed, not run |
| Human evaluation | Not done |

### 3.2 Experiments and results

Each evaluation fixed its criteria first (**preregistration**) and used evaluation random seeds **once only**. Results, good or bad, are recorded as they came.

| Stage | What we did | Result |
| --- | --- | --- |
| Comparison baseline (v3.4 "Y-series") | An older design with fixed value kinds; evaluated re-learning after a correction (40 seeds, once) | Swapping the evaluation state did not change the direction of learning. In a fixed value space the intended loop could not be built. No faults. This is the basis for the redefinition above. |
| Draft 4 (light NN, one evaluation) | A NN that learns its evaluation; 40 seeds | "Acquisition" is not supported. But **learned expectations transfer to unseen contexts and beat a control whose encoder was frozen before training** (40 of 40 seeds). |
| v5 (one evaluation) | Added a mechanism that uses the evaluation state as a learning coefficient (wgate); 40 fresh seeds | Two claims passed (below), one with an important caveat. |
| Heavy NN (NN-REL) | A NN with relational attention, tried on development seeds | No improvement over the light NN. No reason to scale it up. |
| v6 pilot | A design change meant to leave a lasting imprint of each history, development seeds | See challenge 1. Not preregistered yet. |

**The two v5 passes, and how to read them**

- Claim V1, "the evaluation state changes what is learned next": passed. But we later found that **a network trained without our mechanism passes the same measure.** The pass therefore comes mostly from **how we measure**: multiplying the evaluation state into the learning target changes what is learned, which is close to a tautology.
- Claim V2, "learned expectations transfer to unseen contexts": passed. This concerns predictions about the world, not the content of evaluation. It is not evidence of a personality by itself.

**One more finding.** Each persona's "evaluation state" (how heavily it weighs each channel, such as aversion) differs between histories about 15 times more than between retries of the same history. But this is a number accumulated mechanically from outcomes, **not a learned individuality.**

## 4. Challenges

In order of importance:

1. **The imprint of history is hidden under random training noise.** If we re-run a persona with the same history and only shuffle the order of training data, it differs almost as much as personas with different histories. Only about 10 to 20 percent of the difference between personas is due to history, for both light and heavy NNs. More compute does not fix this.
   - In a pilot, a design with an encoder shared by everyone (the part that turns input into an internal representation) plus an individual head, averaged over four heads, made the effect of history about 3.6 times the random noise (development seeds). However the absolute effect is small, and the design was chosen after seeing earlier results, before preregistration. We cannot say an individuality has formed.
2. **We must separate the effect of the mechanism from the effect of the measurement.** We need a measure that can say the evaluation state changes learning *because of* the mechanism (our pilot swaps the state in the middle of training and compares with an identical-history twin).
3. **In our current world, the path from evaluation to interpretation is closed.** The interpretation of "what is happening now" is already 99.9 percent determined by evidence, leaving no room for evaluation to matter. We need worlds where the situation is ambiguous.
4. **Our worlds are a few structures designed by the developer.** We use two kinds of artificial worlds. Whether the findings hold in other structures is unknown.
5. **Only aversive value is tested.** The direction of "coming to like something" is untested.
6. **The LLM-mediated test and human evaluation are not done.**

## 5. How the experiments are run (rigor)

- Criteria are fixed before implementation and recorded with SHA-256 (a file fingerprint). They are not moved after seeing results. Any change is logged as a separate condition and the original result is kept.
- Random seeds are partitioned by role (development, diagnostic, detection-power test, evaluation, reserved). Evaluation seeds are used once.
- The checks themselves are tested by injecting bugs on purpose (mutation testing).
- A reviewer separate from the designers runs a critique that **questions the premises**, before each freeze.
- Literature is recorded with the depth actually read (full text / abstract only / snippet only).
- Lost comparisons, failed criteria and wrong inferences are recorded, not deleted.

## 6. Technology and compute

- **Language and numerics**: Python 3.12, numpy (v3.x and the Y-series), PyTorch 2.7+ (GPU learners).
- **Compute**: a local GPU (NVIDIA RTX 3050, 8 GB VRAM, shared) plus, **within free tiers only**, GPUs on Google Colab, Kaggle and Modal. Nothing that incurs cost beyond a free budget is done without human approval. Before using another machine we check that the same computation gives results within the retry noise of the local run.
- **Memory limits**: GPU memory (VRAM) and host RAM are treated as separate constraints.
- **LLM use**: Claude models and free models via OpenCode (DeepSeek etc.) for design, critique, implementation and literature collection. No paid API is used.
- **Workflow**: git worktrees for isolation.

## 7. Branches and documents

| Location | Content |
| --- | --- |
| `main` | Legacy Phase 3-D/3-E/3-F code and archives (**legacy**; current work is on the branches below) |
| [`claude/project-thread-1vq8ai`](https://github.com/sakri9616-coder/Frontalis/blob/4fecbad/FRONTALIS_PHILOSOPHY.md) | Philosophy `FRONTALIS_PHILOSOPHY.md` |
| [`claude/charming-galileo-wawr7u`](https://github.com/sakri9616-coder/Frontalis/tree/claude/charming-galileo-wawr7u/frontalis-v3-lab) | Minimal NPN implementation v3.0 to v3.3 (PR #2) |
| [`claude/v3.4-prereg`](https://github.com/sakri9616-coder/Frontalis/tree/claude/v3.4-prereg/frontalis-v3-lab) | v3.4 Y-series: preregistration, implementation, evaluation (comparison baseline) |
| [`claude/value-representation-design`](https://github.com/sakri9616-coder/Frontalis/tree/claude/value-representation-design/frontalis-v3-lab/proposals) | **The main research.** Design (`DESIGN.md`), preregistrations (`PREREG-DRAFT.md`, `PREREG-v5-DRAFT.md`), freeze records (`FREEZE.md`, `FREEZE-v5.md`), independent critiques, development log (`probes/LOG-dev.md`), GPU learners |

## 8. Reproduction

v3.x and the Y-series run on numpy alone (checked with Python 3.12).

```bash
cd frontalis-v3-lab
python -B -m unittest discover -s tests -p "test_ra_mechanisms.py"
```

The GPU experiments of the main research are in `frontalis-v3-lab/proposals/probes/` on the `claude/value-representation-design` branch. Evaluation seeds can be used only once; see `FREEZE-v5.md` and `probes/LOG-dev.md`.

## 9. Limitations

- The worlds are a few structures designed by the developer. Conclusions always hold "for those worlds".
- The v3.x results are mechanism checks, not experimental results. They include probability and effect tables written by the designer.
- Most cited literature was verified at abstract or snippet level only.
- This work does not claim to reproduce mental illness or subjective consciousness.

## 10. License

| Kind | License |
| --- | --- |
| Source code, trained NN weights | [Apache License 2.0](LICENSE) |
| Research documents and experiment results | [CC BY 4.0](LICENSE-CC-BY-4.0) |
| Training data | Not yet specified (rights are being checked; no reuse is granted until then) |

See [LICENSING.md](LICENSING.md). Copyright 2026 Sakuri9616.

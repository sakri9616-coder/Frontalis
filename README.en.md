# Project frontalis (English overview)

日本語版: [README.md](README.md)

Project frontalis studies a **Neural Personality Network (NPN)**: a neural network that lives *outside* a swappable large language model (LLM) and forms and changes a personality through experience. It is built for an AI companion character.

> Status as of 2026-10-17: research in progress. **We make no claim that personality formation has been demonstrated.** Negative results are reported below.

## 1. Research purpose

The goal is not an AI that writes human-like text, nor one that plays a consistent character. The goal is to test whether the following loop can be built:

1. An existing personality evaluates an experience.
2. That evaluation and the outcome update the personality.
3. The updated personality takes in the next experience differently from before.
4. Over time this loop accumulates into value judgements that the designer did not specify one by one.

Division of roles:

| Component | Role |
| --- | --- |
| LLM | Cognitive and language interface: interprets text, queries the NPN, verbalises its judgements. Swappable. |
| NPN | The evaluating subject whose state is formed by experience. Holds evaluation tendencies and learns from experience. |
| Experience record | History of what happened, how it was interpreted at the time, and what was later corrected. |

Two distinctions matter:

- **Formation is not expression.** An LLM speaking in a personality-like way is not evidence that the NPN formed one. We test this by keeping the LLM and prompt fixed and swapping only the NPN state.
- **Individuality is not a mechanism fault.** Persistence, lasting aversion, and resistance to refutation are not faults. Only computational breakdowns are: divergence, saturation, or collapse of unrelated values.

The governing document is [`FRONTALIS_PHILOSOPHY.md`](https://github.com/sakri9616-coder/Frontalis/blob/4fecbad/FRONTALIS_PHILOSOPHY.md) (pinned commit `4fecbad`; written in Japanese).

## 2. Current direction (redefined on 2026-10-17)

Earlier implementations replaced the research question with an easier one: the value space was fixed by the developer (five hand-designed desires and an effect table), and value learning was a linear Kalman update. Inside such a space the system cannot, by construction, acquire an evaluative feature that was not designed in.

The core question is therefore now:

> Can a network acquire, through experience, evaluative features, relations and criteria that the developer did not specify, and do these change how later experiences are interpreted and learned from?

Acquisition is only claimed if (1) the features **transfer to structurally new experiences** (not memorisation), (2) **removing or swapping** them changes later learning and interpretation, and (3) agents with different histories **diverge**. The advantage of a neural network is *not* assumed: linear models, fixed-desire models, kernel regression, hierarchical Bayes and random features serve as **negative controls**, and a neural network that loses is reported as such. The earlier fixed-space mechanisms (the "Y-series" below) are kept as a **comparison baseline**.

## 3. Experiment structure

| Generation | Where | What | Status |
| --- | --- | --- | --- |
| Phase 3-D/3-E/3-F | [`main`](https://github.com/sakri9616-coder/Frontalis/tree/main) | Legacy implementation (time, mood, forgetting, LLM links). | Legacy code; results are not premises of current work. |
| v3.0–v3.3 | [`claude/charming-galileo-wawr7u`](https://github.com/sakri9616-coder/Frontalis/tree/claude/charming-galileo-wawr7u/frontalis-v3-lab) ([PR #2](https://github.com/sakri9616-coder/Frontalis/pull/2)) | Minimal NPN: five desires, effect table, hierarchical Bayesian belief, Kalman value net. | Mechanism checks, not experimental results. |
| v3.4 Y-series | [`claude/v3.4-prereg`](https://github.com/sakri9616-coder/Frontalis/tree/claude/v3.4-prereg/frontalis-v3-lab) | Re-evaluation and re-learning after corrections (M1 absorption, M2 source credibility, M3 guilt/self-trust). | Evaluated; comparison baseline. |
| Value-representation acquisition | [`claude/value-representation-design`](https://github.com/sakri9616-coder/Frontalis/tree/claude/value-representation-design/frontalis-v3-lab/proposals) | The main research question; GPU experiments. | Draft 4, not frozen. |

### 3.1 Value-representation study (draft 4, GPU)

- The **world** returns no value; it returns changes in an agent's multi-channel body state. What is good or bad is formed by the agent's own evaluation.
- **Innate signals** stay multi-channel: homeostasis (leaving a tolerated range is aversive) and prediction error / learning progress are compared as co-primary conditions.
- The **loop** is in the main condition: the evaluation made at the time of an experience becomes the next learning target of the evaluation head.
- **Learners**: a set-based MLP (NN-SET) and a relational attention network (NN-REL); controls: linear, fixed-desire, kernel regression, hierarchical Bayes, random features, an untrained network.
- **Judgements** use no ground-truth value function. They look at divergence between histories, the loop's contribution, interventions on learned subspaces, swapping the *evaluation state*, and transfer to new structure. Each contrast is calibrated on null worlds and planted networks first.
- **LLM tests** check whether several LLMs reading the same NPN state agree on the *direction* of a stance. Differences in strength or style are tolerated; if the direction splits, we first suspect the NPN output was not rich enough (a pre-registered richness ladder).

### 3.2 Discipline

- Criteria and thresholds are fixed and hash-recorded *before* implementation; old criteria and failures are kept; they are not tuned to pass.
- Seeds are partitioned (development / diagnostic / mutation test / evaluation / reserved); evaluation seeds are used once.
- Detection power is checked by mutation testing (8 of 8 injected faults detected in the Y-series).
- Independent critiques that may attack the *premises* gate every freeze; the design critique has already stopped one freeze.
- Literature is recorded with the depth actually read (full text / abstract only / not opened).

## 4. Why a GPU

The main experiment trains **tens of thousands of small networks ("personas") in parallel** across conditions, 40 independent random worlds and several learners. A batched implementation reaches about 46,000 persona-steps per second on our current GPU; the relational-attention learner is about ten times slower (about 2,850 persona-steps per second). Early estimates put the main relational-attention condition alone at roughly 300 GPU-hours and the whole programme at 4–6 weeks (the estimate is being updated with draft-4 measurements).

We keep two different memory limits apart:

- **GPU memory (VRAM)**: the current GPU is an NVIDIA RTX 3050 with 8 GB, shared with other jobs. Running several jobs at once caused CUDA out-of-memory failures, so jobs run one at a time with reduced batch size. Larger VRAM would allow larger persona batches and the planned capacity scaling (from about 50k up to about 1M parameters).
- **Host RAM**: the machine has 32 GB, shared. At production scale (32,000 experiences per persona) one data-generation step tried to allocate an 11 GiB array in host RAM (NumPy) and failed. That is not a GPU problem; it is handled by generating data in chunks.

## 5. Results so far (honest summary)

- **v3.x mechanisms** work as designed, but they are mechanism checks. Open problems remain (for example, extinction and context-dependent return).
- **Y-series evaluation** (evaluation seeds, n = 40, run once):
  - Different personalities showed different *directions* of learning after the same correction in some settings, but the same differences appeared in the older design and with the M3 mechanism switched off, so they are **not attributable to M3**.
  - **Swapping the evaluation state did not change the direction** of the correction's effect (similarity 0.99–1.0): the direction came from beliefs and the experience record, not from the evaluation state. In other words, in a fixed value space this mechanism did not produce the "evaluation state → next learning" loop the philosophy asks for.
  - The self-trust variable moved only slightly (|log τ| ≤ 0.028). No mechanism faults were found. Over-correction after many misattributions was reduced from 76% to 13% of individuals going to the opposite side, but not removed.
  - None of the three primary criteria passed in full: the evaluation-state criterion failed, and the other two passed only partially. This result supports the redefinition above.
- **Value-representation study**: development probes were largely negative at first (history-dependent divergence was as large as retry noise); a mechanism redesign fixed non-finite values and several noise sources, and two independent critiques led to a rewrite. Nothing has been run on the evaluation seeds.

## 6. Unverified hypotheses

None of the following is currently supported.

1. A neural network acquires feature interactions that fixed-axis models cannot express.
2. Acquired features transfer to unseen experiences rather than being memorised.
3. Removing or swapping learned features changes later interpretation and learning.
4. The evaluation state just before an experience changes the direction of the next learning, beyond self-reinforcement of the initial evaluation.
5. The same correction leads agents with different histories to learn in different directions.
6. One persistent external state makes several different LLMs agree on the direction of a stance (we found no supporting literature).
7. Acquisition memory can be kept while allowing both extinction and context-dependent return.
8. The Y-series mechanisms match human psychology (not yet checked against the literature).
9. A neural network beats linear, kernel and hierarchical-Bayes controls on value representation and transfer. **This is not assumed**; losses are reported.

## 7. Technology

- Python 3.12, NumPy (v3.x and Y-series), `unittest`; PyTorch 2.7 with CUDA 12.8 for the batched GPU learners.
- LLM tooling: Claude Code models for design, critique and implementation; OpenCode CLI with free-tier DeepSeek v4.1 Flash and MiMo v2.6 Flash for expression tests (MiMo as a tool-less reader only). No paid API is used.
- Work is isolated with git worktrees; frozen pre-registrations are recorded with SHA-256.

## 8. Limitations

- The worlds are structure families chosen by the developer; conclusions hold only "for that family".
- Most cited literature was verified at abstract level only.
- No human evaluation has been done.
- This work does not claim to reproduce mental illness or subjective consciousness.

## 9. License

Source code and trained network weights: [Apache License 2.0](LICENSE). Research documents and experiment results: [CC BY 4.0](LICENSE-CC-BY-4.0). Training data: not yet specified (rights are being checked). Details: [LICENSING.md](LICENSING.md). Copyright 2026 Sakuri9616.

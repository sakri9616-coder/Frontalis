# ds2 audit: sim4.run vs measure4.swap_update, cells.planted_views

Line references: `sim4.py` (run 471-594, NNPop 169-268, RidgePop 271-326), `measure4.py`, `cells.py`.
Notation for the intra-step chain (all in `sim4.run`, `t` = current step):
`bnew = 0.7*b + db` (510), `d = clamp(|bnew|-TOL,0)^2` (511), `pmix` (512), `err` (513),
`Ef/Es` (514-516), `lp = (Es-Ef)/(Es+1e-3)` (517), `app = gain*(1-clamp(sat,0,1))` (518-519),
`raw/inn` (520, 527-530), `f_k = inn_k + CYC*ref*(inn_k - vvec.sum(1))` (532), `f = sum_k f_k` (533),
`fscale` (534), `fs = clamp(fscale,1e-3)` (535), `f/f_k -> f/fs, f_k/fs` (536),
`kap` (537), `a_ = sign(f)*v/fs` (538), `prior_logit = kap*a_` (539), `r_on = softmax(ll+prior_logit)` (540),
`r` (545), `sal` (560), `TVt = r*f_k/fs` (561), stored (562), ridge `add` (565), NN losses (581-583), choice (486).

## A. Paths of each evaluative-state quantity

1. `w_use` (init 389; set 525-526). Live only if `on & (PW>0)` (521, 525); otherwise exactly `ones` (525).
   - (1) TV: 527-530 -> `inn_k` -> 532 `f_k` -> 536 /fs -> 561 `TVt = r*f_k`; and via `f` (533) -> `a_` (538) -> `r_on` (540) -> `r` (545) -> 561.
   - (2) SAL: 533 `f = f_k.sum(1)` -> 560 `clamp(|f|,max=sal_max)` (only the `sign(f_k.sum)`/|.| enters, not per-channel magnitudes).
   - (3) NN belief (set/nn0): none (see 12).
   - (4) krr belief: only through `r` = weight `wb` at 565, and `w_use` enters `r` only through `sign(f)` in 538 and `fs` in 538/536.
   - (5) choice: no direct term; only indirectly via trained params (TV/SAL -> 582 -> `h.Wv`,`v.*`).
2. `cum` (init 389; 522). Enters nothing except `wS/wH` (523-524) -> `w_use` (525) and only when `pw_on` (522). So all of `cum`'s paths are item 1's paths, and `cum` is dead for `CYC=0` or `PW=0` (342-353: `pw="none"`).
3. `kappa` (init 387; 537-542, 553-555).
   - Dead for `KM==0` (`kap=0` at 537 makes `prior_logit=0`, so `r_on = softmax(ll)`).
   - (1) TV: 537 `kap` -> 539 `prior_logit` -> 540 `r_on` -> 545 `r` -> 561.
   - (2) SAL: none (`sal` uses `f` only; `f` has no `kappa` term).
   - (3) NN belief: none.
   - (4) krr belief: via `r` -> `wb` (565).
   - (5) choice: none directly.
   - Self-reference: 542 updates `kappa` using `r_on`, `prior` (541) and `a_` computed with the old `kappa`.
4. `fscale` (init 385; updated 534).
   - (1) TV: 535-536 divides `f_k` (so `TVt = r*f_k/fs`, 561); also 538 `a_` divides `v` by `fs` -> `r` -> 561.
   - (2) SAL: 536 `f/fs` -> 560 `clamp(|f|,max)`.
   - (3) NN belief: none.
   - (4) krr belief: via `fs` in 538 -> `r` -> `wb` (565).
   - (5) choice: direct, 486 `val = v4.sum(-1)... / st["fscale"][:,None]`.
5. `sig2` (init 386; updated 543-544).
   - (1) TV: 507 `ll` -> 512 `pmix` -> 513 `err` -> 515-516 `Ef/Es` -> 517 `lp` -> 528/529 -> 530 `inn_k` -> 532 `f_k` -> 536/561; also 507 `ll` -> 540 `r_on` -> 545 `r` -> 561.
   - (2) SAL: via `lp` -> `inn_k` -> 532 `f_k` -> 533 `f` -> 560.
   - (3) NN belief: none.
   - (4) krr belief: via `ll` -> `r_on` -> `r` -> `wb` (565).
   - (5) choice: none directly.
6. `Ef`, `Es` (init 388; 514-516). Live only for `INN==I0,IHE` (via `lp`, used at 528/529; for `IH` `lp` is computed at 517 but never used). Paths identical to `sig2`'s except no path through `ll`/`r_on` in 540: (1) TV and (2) SAL through 517 -> `inn_k` -> `f_k`/`f`; also through `sign(f)` in 538 -> `a_` -> `r` -> TV and (4) krr `wb`. (3) none. (5) none directly.
7. `sat` (init 389; 518). Live only for `IHE` (519 `app` used only in 520's third branch and 529). (1) TV: 519 `app` -> 529 -> 530 -> 532 -> 536/561, plus `sign(f)` -> 538 -> `r` -> 561. (2) SAL via `f` (533/560). (3) none. (4) krr `wb` via `a_`/`r`. (5) none directly.
8. `b` (init 385; 510, 531). Live only for `IH` and `IHE` (via `d`, 511, used in 527 and 529). **Dead for `I0`**: the `I0` branch (520, 528) uses only `w_use*lp`, and `lp` (517) contains no `b`, so for `I0` personalities `b` is a pure accumulator with no path to TV, SAL, `r`, or choice. Paths for IH/IHE: 510 `bnew` -> 511 `d` -> 527/529 -> 530 -> 532 -> 536/561 (TV); -> 533 -> 560 (SAL); -> 538 `sign(f)` -> 540 -> 545 `r` -> 561 and 565 (krr `wb`). (3) none. (5) none directly.
9. Cross-condition dead state (from 1-8): for `IH`, `Ef/Es` and the `lp` term are dead; for `I0`, `b` and `sat` are dead (`sat` is also dead for `IH`); for `KM==0`, `kappa` is dead.
10. Indirect path to choice (all quantities): any quantity reaching TV (561) or SAL (560) is stored (562) and re-read in the value loss `lv = (sw*(vv-tv)^2)` (577, 582), which updates the value head for NN learners (584) or the ridge value block (565); the next step's choice (482-490) uses that network. No evaluative quantity other than `fscale` touches line 486 directly.
11. Choice direct dependency in full: 486 uses only `v4` (network value, 485), `st["fscale"]` (486), `beta`, `ceps`, `CU` (487-488), `leader` (420, 489) and `ch` (489). So `fscale` is the only evaluative-state quantity that changes `p_alt` without a learning step.
12. **Does the NN belief loss use `r`? No.** Lines 581-583: `mix` (581) is built from `dbb` (= `buf["DB"]`, 577) and `pbb` (= `net.belief`, 579) only, with fixed scale 0.2; `lv` (582) uses `sw` (= `buf["SAL"]`) and `tv` (= `buf["TV"]`). `r` appears only inside `TVt` (561), i.e. only in `lv`. Since the belief branch (`b.*`, `h.Wb`) and the value branch (`v.*`, `h.Wv`) are disjoint parameter sets (177-190, 255-265), the value loss cannot change belief learning and no evaluative-state quantity (including `r`) reaches set/nn0 belief learning: the only belief supervision is `dbb` against `pbb` (581). For `set`/`nn0` the belief learner is therefore purely a predictor of the observed `db`, and the two learner families get different belief targets: NN = logsumexp soft-min over the 2 candidates (581), ridge = the raw observation `db` on both rows weighted by `r` (565, `add` signature 318-322).

## B. sim4.run target (506-562) vs measure4.swap_update target (141-173)

Baseline form is the same in both: `TV = r * f_k / fs`, `f_k = inn + ref*(inn - V)`, `r = softmax(ll + kappa*a_)`, `a_ = sign(f)*V_sum/fs`.

13. Whose channel weights multiply the innate signal. run: `w_use` is the **receiver's own** (525-530; 159 has no analogue). swap_update: `wv = stt["w"][None]` is the **donor's** `w` (159) and multiplies `d`/`lp`/`app` in 161-165. Direction: this is the single biggest difference — it *creates* the dependence of the target on the donor's `w` (`d tgt / d w_donor` proportional to `d`, `lp`, `clamp(dbt[:,4],0)`); run's target has exactly the same functional dependence but on the receiver's own `w`, giving zero donor dependence.
14. Hinge state. run: `bnew = DECAY*st["b"] + db` then `d = clamp(|bnew|-TOL,0)^2` (510-511) with `DECAY=0.7` (37). swap_update: `d = clamp(|dbt|-TOL,0)^2` on `dbt` alone (151) — there is no `b` state at all (`View.state`, measure4 28-29, has `w, fscale, kappa, sig2, Ef, Es` only). Direction: the donor-`w` coefficient in swap is `d(db)`; run's own-`w` coefficient is `d(0.7b+db)`. Since `|0.7b+db| >= |db|` iff `b·(0.7b+2db) >= 0` and `b` is an EWMA of past `db` (531), run's hinge is typically (not always) the larger one, so the swap's dependence on `w` is typically *smaller* and vanishes when the b-state cancels.
15. Whose value enters the `ref` cycle term. run: `vvec.sum(1)` = the receiver's own network value (532). swap_update: `v_d = donor.V(...)` (149), used at 167 (`inn - v_d.sum(1)`) and 169 (`a_`). Direction: adds a donor-`V` dependent term of magnitude `ref*|v_d.sum(1)|` inside the brackets, so a donor with large `|V|` changes both the magnitude and the sign of `f` (and hence `r`, 170).
16. Whose `fscale` normalises. run: receiver's `fs` (535-536). swap_update: donor's `fs = max(stt["fscale"],1e-3)` (168), a Python float, applied to `f_k` and to `a_` (168-169). Direction: the whole target scales as `1/fscale_donor` (and `a_` too), so a donor with larger `|sum_k f_k|` shrinks the swap target and the `kappa` logit; it also makes the donor-`w` coefficient scale with `1/fscale_donor`.
17. Whose `kappa`. run: own `kappa`, with `kap = kappa + eps_k` for `KM==3` and `0` for `KM==0` (537). swap_update: donor's `stt["kappa"]` directly (170), with no `eps_k` term and no `KM==0` zeroing and no `eps_k` field in `View.state`. Direction: donor-`kappa` dependence appears; a large `|kappa_donor|` sharpens `r` (170), increasing the between-candidate spread of the target.
18. Whose `Ef/Es`. run: own `Ef/Es`, updated in place and stored (514-516). swap_update: the **donor's** `Ef/Es` (156), indexed by the new contexts, updated only in local variables and never written back. Direction: donor-state dependence via `lp` (158) into the `0.5*wv*lp` term (165) and into `pmix`/`err` (154-155), while `pmix` is built from the *receiver's* belief — a belief/state mixture with no counterpart in run.
19. IHE appetitive term. run: `app = gain*(1-clamp(sat,0,1))` with `gain = clamp(db[:,4],0)` and a habituation state `sat` (518-519, 529). swap_update: `torch.clamp(dbt[:,4], min=0)` (165) — no `sat` exists in `View.state`. Direction: for `IHE` the swap coefficient on `w[4]` is `gain` while run's is `gain*(1-clamp(sat,0,1)) <= gain`, i.e. run's dependence on `w[4]` is attenuated by habituation and the swap's is not.
20. Belief/uncertainty inputs are the same role (the learner's): run uses own `pb` and own `sig2` (507); swap uses `receiver.PB` (148) and `receiver.state["sig2"]` (152). Difference: run's `sig2` is live/updated (544) while swap's is a frozen snapshot from the end of training. Direction: neutral for donor-`w` dependence except that a frozen larger/smaller `sig2` rescales `ll` and hence `r`.
21. Objective and parameter set (outside the target but part of the same difference). run: value loss weighted by `sw = SAL` (577, 582) and summed with `0.1*mix` belief loss (583); gradient touches only `v.*`/`h.Wv` for `set`/`nn0`. swap_update: unweighted `((value_vec(X,q)-tgt)**2).sum(-1).mean()` (182) for 20 plain-SGD steps, and the optimizer at 180 includes every parameter for kinds other than `nn0`/`rf`, so the belief encoder `b.*` is trained by the value target. Direction: enlarges the set of receiver parameters that inherit donor dependence, and removes the salience re-weighting that run applies to the same `TV` value.
22. Scope/rows: run's target is stored per step (562) and later sampled (573); swap's target is applied in-sample to the same 30 experiences (144, 182) and, in the ridge branch, added with weight 1 (`Gv = net.Gv[i] + phi.T@phi`, 194) whereas run's ridge `add` weights phi and RHS by `sal` (565, 320-322). Direction: the ridge swap gives each of the 30 rows equal influence on the donor-`w` dependence, while run weights by the salience computed from `|sum_k f_k|` (560).
23. Constants that do match: `TOL` (sim4 37, imported at measure4 12) and `ref=0.5` default (141); `DECAY=0.7` is used only in run (510).

## C. swap_update: donor vs receiver inputs to the target

24. Donor (`stt = donor.state`, 150; `v_d = donor.V`, 149) enters the target (173) through:
   - `stt["w"]` -> `wv` (159) -> `inn` (161/163/165) -> `f_k` (167) -> target; also `wv` -> `f` -> `sign(f)` -> `a_` (169) -> `r` (170) -> target.
   - `stt["kappa"]` -> `stt["kappa"]*a_` inside `r = softmax(ll + stt["kappa"]*a_, 1)` (170).
   - `stt["fscale"]` -> `fs` (168) -> divides `f_k` (168) and `a_` (169).
   - `stt["Ef"]`, `stt["Es"]` -> `lp` (156-158) -> `inn` for `I0`/`IHE` (163, 165).
   - donor's value `V`: `v_d` (149) -> `inn - v_d.sum(1)` (167) and `v_d.sum(-1)` in `a_` (169).
   - Not available/never used: `b`, `cum`, `sat`, `eps_k` (absent from `View.state`, 28-29).
25. Receiver enters the target only through: `receiver.PB` -> `pb_r` (148) -> `ll` (152) and `pmix` (154); `receiver.state["sig2"]` -> `ll` (152); and, in the non-cycle branch only, `receiver.state["fscale"]` (172). The receiver's own `w`, `kappa`, `Ef`, `Es` and `V` are **not** used for the target — the target's evaluative content is entirely the donor's (plus the receiver's belief/uncertainty). The receiver's network (`receiver.p`, 176) is what is trained (182/194).
26. `r` in swap is 0.5 uniformly when `cyc` is false (172); in run `r` is 0.5 when `on` is false (545) and `on = CYC>0` with `CYC` per personality (413, 452).

## D. planted_views: donor `PL-state` vs donor `PL-state0`

27. `PL-state` (cells 74-78): `perm = rng([seed,74]).permutation(KB)`; `Bs` is a fresh `View` object built with `__new__` that copies every entry of `A3.__dict__` except `cache` (76), so `Bs.net`, `Bs.p`, `Bs.i`, `Bs.wi` are the *same objects/values* as `A3`'s; then `Bs.state = dict(A3.state)` (a shallow copy) and `Bs.state["w"] = A3.state["w"][perm_tensor]` (77) — only the channel-weight vector is permuted.
28. `PL-state0` (79): `"B": A3` — the donor is exactly the `A3` View, no copy, unpermuted `state["w"]`.
29. Therefore the **only** difference between the two donors is `state["w"][k] = A3.state["w"][perm[k]]`, i.e. channel `k`'s weight carries the value that `A3` held for channel `perm[k]`. Everything else (network weights `p`, `i`, `net`, `fscale`, `kappa`, `sig2`, `Ef`, `Es`, and hence `donor.V`, 149) is identical, and `A`, `A2`, `B2` (`vw["B2"]`) are the same objects in both dicts. `Bs` is a plain `View`, not a `Planted`, so `isinstance(donor, Planted)` at 149 is False and `donor.V` is used. In `swap_update` this single difference acts only through `wv` (159 -> 161-165) and through `sign(f)` in `a_` (167-170); `A3.state` is not mutated (77 writes into the copy). Because the receiver at 299 is always `B2`, `PL-state` vs `PL-state0` isolates the effect of the donor's channel-weight permutation on the same receiver.

## E. Bugs and inconsistencies (with lines)

30. **Axis inconsistency in the cycle term.** `sim4.py:532` `f_k = inn_k + CYC*ref*(inn_k - vvec.sum(1))` reduces `vvec` (P,2,KB) over the **2-candidate axis**, producing a per-channel reference that adds the self-acting and the partner-acting row; the per-candidate total used for attribution is `v = vvec.sum(-1)` (505) and for choice `v4.sum(-1)` (486). Two different reductions of the same tensor are used in the same step. Mirrored in `measure4.py:169` (`v_d.sum(-1)`) vs `167` (`v_d.sum(1)`), so the swap probe inherits the same choice and cannot discriminate.
31. **Normalisation mismatch.** `TVt = r*f_k/fs` (536, 561) and the choice logit `val/fs` (486) are both divided by `fscale`, which is an EWMA of `|f| = |sum_k f_k|` (533-534), i.e. the scale of the *innate signal's channel-sum*, not of the values. Consequences: (a) the value targets are normalised by a quantity unrelated to the values being predicted; (b) `1/|sum_k f_k|` grows without bound when channels cancel; (c) the effective temperature of the choice softmax is `beta/fs` (487) and varies with the innate condition (`IH` vs `I0` vs `IHE` have different `|sum_k f|`), so `beta` is not comparable across conditions.
32. **`C-off` is not "off".** With `CYC=0` (342) the channel weighting is neutralised (`pw_on=False` at 521-525 gives `w_use=ones`) but the innate signal is still injected: `f_k = inn_k` (532), `r = 0.5` (545), so `TVt = 0.5*inn_k/fs != 0` (561) and `sal = 1` (560). The no-cycle control therefore trains the value head on a non-zero evaluative target, whose only removed ingredient is the channel-specific weighting (`w_use`).
33. **`r` vs `r_on` in the same step.** Line 543 updates `sig2` from `res` computed with `r_on` (the posterior), while line 545 sets the target's weight to `r = 0.5` when `on` is False. The posterior is thus used for the uncertainty update even in the condition where the target deliberately ignores it.
34. **Unconditional state updates.** `Ef/Es` (514-516), `sig2` (544) and `fscale` (534) are updated on every step regardless of `on`/`pw_on`, so "cycle off" personalities still accumulate learning-progress and scale state that then enters `f_k` (via 517/528-530) and the choice (via 486).
35. **Accumulator vs signal mismatch.** For `I0`/`IHE` the accumulated weight uses `lp.abs()` (`raw`, 520) while the signal it weights is the signed `lp` (528, 529), so `w_use` tracks the magnitude of the learning progress while multiplying a signed term.
36. **Dead state (waste, not a crash).** For `IH`, `Ef/Es` and `lp` (517) are computed and stored but never used (527 uses `d` only); for `I0`, `b` is a pure accumulator with no path (520/528 use `lp` only) and `sat` is dead for both `IH` and `I0` (`app` is used only in the `IHE` branch, 520/529); for `KM==0`, `kappa` is dead (`kap=0`, 537) yet the field is still carried and hashed (`state_hash`, 632).
37. **`View.state` cannot express the training rule.** `measure4.py:28-29` stores `w, fscale, kappa, sig2, Ef, Es`; `b`, `cum` and `sat` are absent, so `swap_update` computes `d` on `db` alone (151) instead of `0.7b+db` (510-511) and replaces `app` by `clamp(dbt[:,4],0)` (165). J3's probe therefore evaluates a rule that differs from the one the personalities were trained under, and the discrepancy scales exactly the donor-`w` coefficient (items 14, 19).
38. **`swap_update` trains the belief encoder with the value loss.** `measure4.py:180` keeps every parameter in the optimizer for kinds other than `nn0`/`rf`, so `b.*`/`h.Wb` receive gradient from `loss` (182), which contains only value targets. In `sim4.run` the value loss is disjoint from the belief parameters (581 vs 582-583). The swap probe thus moves parameters that the audited training rule never moves with value targets; it also drops the `0.1*mix` term and the `sw` weighting (182 vs 582-583).
39. **Two different belief objectives.** NN belief is supervised as a fixed-scale (0.2) logsumexp soft-min over the two candidates (`sim4.py:581`), while the ridge belief target is the raw observation `db` on both candidate rows with weights `r` (`sim4.py:565`, `add` 318-322). The attribution temperature (fixed 0.2) also disagrees with the evaluator's calibrated `sig2` (507); the two learner families therefore do not learn the same belief function that the evaluator's `r` assumes.
40. **`ghat_resid` basis does not match the generator.** `measure4.py:111-112` uses `clip(|pb_s|-TOL,0)^2` and `clip(|pb_p|-TOL,0)^2`, i.e. hinges on the *belief predictions* of both roles, whereas `run` computes `d` only on the actor's body state `bnew = 0.7*b + db` (510-511) and never on the partner's `b`. The `rho` residual (119) is thus a residual against a rule that `run` does not implement.
41. **Calibration scale uses a different probe set than the standardisation.** `cells.py:56-58` standardises `phi` with `P_tr` statistics, but `cells.py:66-68` measures `ph` (and hence `alpha`) on `P_un`. The plant applied to `tr` and `nc` (`Planted.get`, measure4 79-88) therefore uses a scale calibrated on `un`.
42. **Mixed types in the view cache.** `measure4.py:62-64` stores `V` and `PB` as numpy arrays but `H` as a `torch` tensor; `Planted.get` (84-87) and `persistence` (245-256) rely on that. Not a numerical error, but the cache is not type-uniform.
43. **`loss` at 590.** `if not torch.isfinite(loss)` tests only the last inner iteration's loss (`j = SPN-1`); with the default `spn=1` (332) this is the only iteration, but for `spn>1` (prod, `cells.py:22-23`) it under-reports non-finite minibatches.
44. **No index error found in the probe/pair plumbing.** Checked and consistent: `probe_set` row order (`s = repeat(ents, 2*len(ctxs))`, `p = repeat(part.reshape(-1), len(ctxs))`, `c = tile(ctxs, 2*len(ents))`, sim4 138); `interaction_share_vec` reshape `(160, 12)` against `n_rows = 80*2` (measure4 322-323, 1920 rows = 160x12); `sb.flip(-1)` partner construction (sim4 575) and `sb[:, ::-1]` (measure4 243); `ID_DIMS` masking (36, 217); hb/fixk feature sizes vs `lam` (284-288, 310-311); `M2` einsum axes (496 vs 145). The one reduction that is inconsistent is item 30.


## Checked by the design lead (2026-10-18)
- Confirmed: 12 (NN belief loss does not use r), 14 (the swap's hinge uses db alone), 15-17, 27-29, 39.
- Wrong: 38. `value_vec` uses only `v.*` and `h.Wv`/`h.bv`; the belief parameters receive no gradient (grad None) and SGD does not move them.
- Not a bug: 30. `vvec.sum(1)` (sum over the two candidates) is the network's prediction of the whole event's evaluation f_k, because the stored targets are r_j * f_k with sum_j r_j = 1; `v = vvec.sum(-1)` is the per-candidate total used for attribution. The two reductions answer different questions.
- Design choices, not bugs: 31, 32 (C-off keeps the innate signal by design; only the cycle is removed), 34, 40, 41.

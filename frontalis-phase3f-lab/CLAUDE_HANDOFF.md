# Phase 3-F handoff

## Start here

This is an isolated research lab. Read REPORT.md, PREREGISTRATION.md, reports/final-verification.json, then the relevant code. Preserve Phase 3-E, all earlier labs, ASIST, Nephia, real memory, and settings. No body integration or Git publishing has been performed. The user authorized continued API use for this Phase 3-F work; credentials are environment-only. Prefer artifacts and runnable validation over additional prose.

## Implemented

- npnf/core.py: manually learned RBF centers and normalized LMS heads, two domains, numerical seven-axis appraisal, information-update clock, transient affect, mood, coefficient forgetting, consolidation trace, append-only episodes with facts versus interpretation and correction/recall links. No LLM sentiment is a training target. Long-term inference does not consult episodes.
- npnf/dual_view.py: separately identified post hoc architecture with equal cue/observed logit weighting. New affect uses that appraisal. Underlying learned weights match the observed-only control; architecture is included in the personality hash. A .view.json is REQUIRED for dual snapshots.
- npnf/transport.py: measured JSON schema and exact eight-axis lossless object-to-array conversion. Never fill missing facts. tools/evaluate.py and tools/evaluate_dual.py are JSON stdin/stdout read-only boundaries with model-ID allowlists.
- playground/server.py + index.html, local port 8905: editable interpretation, explicit confirmation, time/recall/correction, initial tendencies, both architectures, language, immutable snapshots, ZIP exports, old-session restoration, CSRF, save-before-RAM commit. start-workbench.ps1 starts it using the existing runtime.

## Evidence and limitations

Primary sources were frozen before numerical runs (reference/preregistration.json). Every added condition has a distinct reference/*-freeze.json and explanation. Do not change frozen files to improve results.
Primary history criterion FAILED 0/60, while temporal separation passed 240/240 and plasticity/locality passed 60/60. Dual-view P0 additional criterion passed 15/15; do not merge this with the primary failure. Main history uses only 102 active parameters at every reserved capacity, so high-capacity complexity is NOT demonstrated. Fixed-teacher RBF generalization is worse than linear ridge. Structural comparison with 3-E changes both architecture and learning, so it is not an isolated architecture effect; its sourceHash refers to the imported 3-E preregistration, and structure-freeze.json binds the added code.
Original extraction had array/object ambiguity. Retain strict primary failures, lossless diagnosis, and schema-enforced results separately. Qwen engine errors are not proof of poor semantic capability. Actual NPN tool calls used gold meaning, so they test tool execution separately. The actual-tool verbalizer received structured meaning, not the original narrative; do not interpret missing narrative details as facts it was given.
Primary 72 utterances and additional 80 were generated, plus 10 real tool-mediated utterances and two six-stage workbench conversations. LLM audit is NOT formal human evaluation. Human forms have no human results. Experimental IDs contained state names; this may cue the verbalizer. Use opaque IDs in a future cleaner replication. Initial future choice is an exact tie resolved by input order. Some later margins are small. Do not claim robust human-like forgiveness or universal emotional semantics.

## Runtime and reproduction

Existing Python: ../frontalis-phase3d-lab/.venv/Scripts/python.exe, torch 2.7.0+cu128, numpy 2.5.4. Use -B and PYTHONUTF8=1; do not install into or modify the preserved environment. Node D:/node.exe. OpenCode executable is C:/Users/sakur/AppData/Local/Programs/@opencodedesktop/resources/opencode-cli.exe; exact free ID opencode/deepseek-v4.1-flash-free. LM Studio URL http://127.0.0.1:1234. Never print keys or dump environment variables.

From this lab:
```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$researchPython='..\frontalis-phase3d-lab\.venv\Scripts\python.exe'
& $researchPython -B -m unittest discover -s tests -v
& $researchPython -B scripts/reproduction_copy.py another-run
```
In the fresh adjacent frontalis-phase3f-repro-another-run directory, run experiments/run.py all and optionally experiments/performance.py. Do NOT rerun prepare.py or freeze.py over copied immutable inputs. API scripts perform paid/free live calls if their cache is absent; offline tests/demo do not. Main numeric conditions are resumable by verified cache; many additional experiments require a fresh copy because snapshots are immutable. structure.py requires the preserved sibling Phase 3-E lab.
repro-smoke verifies five capacities offline and repeated device measurements. The final fresh copy verifies current source/tests and offline restoration. Neither is a second complete execution of every numeric trajectory.

## Additional study completed after the follow-up request

See ADDITIONAL_RESULTS.md and experiments/replication_longrun.py. Its separate freeze uses unseen seeds 7901/7902/7903. Sixty history groups, 240 past series and 960 current branches replicate dual readout across all four initial tendencies: before-current criterion 60/60, but after eight current experiences 0/60 (mean RMS .00694). Do not report the first finding without the second.

Sixty long-run series provide 120 checkpoints (32/1024 repetitions). After 1024 idle updates, retention is .0000393 for uniform and .16249 for consolidation after long formation. Contrary-outcome learning vs a matched decay-only control passes 30/30 per mode/repetition condition. No capacity separation or saturated cue outputs was demonstrated; active parameters remain 102. Post hoc inspection of the saved seed7901/profile0 trajectory finds target-logit drift only .0001675 over 1024 repetitions. Broad self-amplifying personality growth remains unproven.

Five further immutable models are under models/replication-longrun/, separate from the original 265 states. reports/replication-longrun.json contains the frozen-condition outputs; reports/replication-analysis.json identifies the later descriptive target-drift inspection. Source/plan files remain unchanged. Execution was resumed from immutable complete caches and parallelized by disjoint capacities, then seeds for the large capacity; the scheduling adapters have their own freezes. Interrupted unsaved branches were rerun, with no condition or within-condition ordering changes. The original frozen script aggregated the full set.

To reproduce, use a fresh research copy and run `experiments/replication_longrun.py all`. The two scheduling scripts are optional execution adapters. No additional API calls were made for this study. The current workbench restores the dual-view personality at update43, with actual UI evidence in reports/ui-verification.json. Review-form import remains untested in the browser; export and persistence were exercised and all test ratings reset to zero.

## Next research, if requested

Test richer and held-out semantic contexts that use the available capacity; compare locality against transfer without relabeling preference as correctness. Study explicit unknown/missing masks, cue/observed geometry, fast history convergence and weak target feedback. The unseen-seed/initial-tendency replication above is complete; investigate its observed limits before broader claims. Resolve annotation ambiguity with independent human annotators, then conduct the prepared human evaluation. Keep the current primary failures as the canonical result. Decide Nephia integration separately with the user.

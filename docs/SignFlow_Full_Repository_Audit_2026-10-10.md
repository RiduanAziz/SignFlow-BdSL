# Full Repository Audit — SignFlow ASL Recognition

**Audit date:** 10 October 2026  
**Repository:** [RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition)  
**Branch audited:** `main`  
**Audited HEAD:** `9b2f4991dff8317b6cac681eaa976b63fd7f1035`  
**Latest CI run reviewed:** [Run #15](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/actions/runs/38043721844)

## 1. Executive assessment

The repository contains most of the planned research and demo components: WLASL metadata/manifests, MediaPipe Tasks landmark extraction, normalization and temporal-dynamics pipeline, multiple model families, a robust holistic-fusion classifier, training/evaluation/robustness scripts, FastAPI and WebSocket services, and the SignFlow React/Vite interface.

The principal problem is **integration and evidence quality**, not a lack of source files. At the audited HEAD, CI does not finish: frontend test startup fails before the Python test step. The WLASL-20 metrics are preliminary and not yet suitable as final thesis evidence because the test vocabulary coverage is incomplete, report artifacts disagree, and the robustness evaluator does not use the same mask-aware path as training/standard evaluation.

### Readiness by area

| Area | Assessment | Why |
|---|---|---|
| Repository layout | Good foundation | Clear `src/sign_language`, `scripts`, `configs`, `tests`, `app/frontend`, and `results` separation. |
| Dataset bookkeeping | Partial | A processed manifest report exists, but a separate inventory describes an older/different scope and original split overlap. Runtime availability of local landmark files cannot be verified from GitHub. |
| Model implementation | Implemented, needs stronger validation | Baseline and proposed architecture are present; saved weights are not in the public tree, and the end-to-end path needs exact config/preprocessing parity. |
| Standard evaluation | Partial | Metrics are reported, but the WLASL-20 test manifest lacks one configured class and summaries conflict. |
| Robustness evaluation | Not yet reliable | Mask propagation, perturbation semantics, config reconstruction and source report artifacts need correction. |
| Live inference/API | Partial | Several lifecycle and payload-handling issues remain; consecutive gestures can leave stale mask frames after a commit. |
| Frontend | Built as source; verification blocked | Latest CI uses Node 20 with a frontend dependency stack requiring newer Node. Robustness Lab feature acquisition is not reliably triggered when opening its tab. |
| Reproducibility/docs | Needs revision | Status and TODO documentation claim successful verification despite latest CI failure and conflicting result artifacts. |

## 2. P0 — Fix before further thesis experiments

### P0.1 — CI stops before tests run because the Node version is too old

**Evidence:** [.github/workflows/ci.yml](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/.github/workflows/ci.yml) sets Node `20`. The lockfile resolves Vitest `5.0.3`, jsdom `30.1.2`, and undici `8.11.2`, which declare newer Node requirements. The log for [CI run #15](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/actions/runs/38043721844) records `TypeError: webidl.util.markAsUncloneable is not a function` while Vitest starts its worker. The frontend step fails and the Python pytest step is skipped.

**Fix:** Upgrade `actions/setup-node` to Node 24 (or select compatible dependency versions and regenerate the lockfile for Node 20). Keep CI fail-fast, but do not report tests as passing unless the latest workflow actually completes.

**Acceptance:** `npm ci`, `npm test -- --run`, `npm run build`, and `pytest -q` all complete successfully on a clean runner. The final CI report must show all four steps, not just a configuration dry-run.

### P0.2 — WebSocket mask buffer is not cleared when a word is committed

**Evidence:** [API WebSocket handler](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/src/sign_language/api/main.py#L450-L500) clears `sequence_buffer` after manual confirmation but does not clear `mask_buffer`. The automatic commit path around [the guided recognition state machine](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/src/sign_language/api/main.py#L950-L1005) also clears the sequence buffer without clearing the mask buffer. The neutral-pause and reset branches clear both, but a new gesture can begin before a neutral pause occurs.

The sequence and mask deques can then have different lengths. Prediction preprocessing may fail on mask/sequence alignment, with the exception logged and skipped; the new gesture can go unrecognized until buffers become aligned again.

**Fix:** Whenever a gesture's sequence buffer is cleared, clear `mask_buffer` too. Review `candidate_history`, `wrist_history`, and `speed_history` for each transition so old movement cannot contaminate the next gesture.

**Acceptance:** Add WebSocket tests for auto-commit followed immediately by another gesture, manual confirm followed by another gesture, reject, pause, reset, and repeated-sign recognition. Assert sequence and mask lengths match before every inference call.

### P0.3 — Current results are not yet defensible as final WLASL-20 thesis results

**Evidence:**
- [`wlasl20_baseline_evaluation.json`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/results/wlasl20_baseline_evaluation.json) reports 78 test samples, 20 configured classes, Top-1 accuracy 10.26%, macro-F1 0.0768.
- [`wlasl20_proposed_evaluation.json`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/results/wlasl20_proposed_evaluation.json) reports 78 samples, Top-1 accuracy 19.23%, macro-F1 0.2015.
- Filtering the committed test manifest to the 20 configured glosses gives 78 items, but only **19 observed classes**; `go` has no test sample. Thus it is not evidence of complete held-out coverage for all 20 classes.
- [`research_summary.md`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/results/research_summary.md) reports a different experiment with 20 samples, 20% accuracy and macro-F1 0.0556. The older `evaluation_report.json` and `proposed_evaluation_report.json` also report that separate 20-sample/100-class run. `research_summary.txt` describes yet another run with 194 samples and a different loss/metric set.
- The comparative markdown claims 74 validation items, while the committed training metrics record `num_samples: 69` in validation. This difference must be explained by an explicit count of records actually loaded, especially missing local `.npz` files.

**Fix:** Do not manually edit output summaries. Re-run both model evaluations from the same configured WLASL-20 manifest and the exact same available sample IDs. Add a hard evaluation preflight that prints manifest rows, valid landmark files, records excluded by vocabulary, unknown labels, observed test classes, zero-support classes, signer overlap, and final evaluated sample IDs. Fail or clearly warn if a configured class has no test support. Calculate macro metrics over an explicit `labels=range(num_classes)` list and report both configured and observed class counts.

**Acceptance:** A single run manifest identifies the experiment/config, vocabulary order, seed, checkpoint file hash, preprocessing/schema version, signer lists, per-split manifest counts, actually loaded counts, missing files, test IDs, metrics and timestamp. Regenerated tables and summaries must be created only from these machine-readable outputs.

## 3. P1 — Research pipeline correctness

### P1.1 — Dedicated robustness evaluation is not using the same model/preprocessing path

**Evidence:** [`scripts/evaluate_robustness.py`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/scripts/evaluate_robustness.py) loads model weights and some checkpoint fields but does not restore the complete saved checkpoint configuration as the standard evaluator does. All clean/noisy evaluation loops call `pipeline.normalize_sequence(...)` and then `model(t_in)` without passing the normalized visibility mask. In contrast, [`scripts/evaluate.py`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/scripts/evaluate.py) restores saved model/dataset settings and passes masks into the model.

For the proposed fusion model, this means the dedicated robustness results are not necessarily evaluating the same mask-aware model path as training and standard evaluation. The default `configs/base.yaml` is the temporal transformer, so using it with a proposed checkpoint may also build the wrong architecture unless the user remembers to override the config.

**Fix:** Factor checkpoint loading, saved-config reconstruction, preprocessing, mask generation, and prediction into one shared utility used by standard evaluation, robustness evaluation, desktop realtime, and API inference. Use `normalize_sequence_with_mask()` and pass `mask=` consistently. Assert model name, input dimension, sequence length, class order and feature schema match the checkpoint metadata.

### P1.2 — Robustness operators are not consistently paired with masks or physical perturbations

**Evidence:** [`src/sign_language/robustness.py`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/src/sign_language/robustness.py):

- `translation_noise()` samples a separate offset for each flattened feature dimension (`size=(1, array.shape[-1])`). This is not a physical global translation of a 3D landmark skeleton. Use a shared `(x, y, z)` displacement across landmarks/frames (or precisely document another perturbation definition).
- `landmark_dropout()` can mutate a passed mask but returns only the modified coordinate array. The offline evaluator currently invokes spatial noise without passing a mask, then retains the original mask. Dropped landmarks can therefore still be treated as observed.
- The evaluator uses a fixed seed (`42`) for every sample and noise case, applying repeated random patterns across samples. Prefer a deterministic sample-derived seed and log it, or repeat independent seeds and report variability.
- `scale_noise()` multiplies raw coordinates uniformly; subsequent center/scale normalization can largely remove that change. The experiment may therefore measure normalization invariance rather than robustness to residual scale error. Define the perturbation stage and interpretation before reporting the result.

**Fix:** Have perturbation functions return `(sequence, mask)` when mask changes; align both through frame-drop, duplication, truncation and dropout. Add unit tests for severity zero, severity bounds, deterministic behavior, shape preservation, mask alignment and expected geometric effect. Decide whether perturbations are applied in raw-coordinate space or after normalization and record the choice.

### P1.3 — Missing masks can be converted into all-valid masks in preprocessing

**Evidence:** [`src/sign_language/landmarks/pipeline.py`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/src/sign_language/landmarks/pipeline.py) calls `normalize_landmarks(arr, mask=mask)` and subsequently `resample_sequence(..., mask=mask)`. When `mask=None`, normalization can infer visible points from coordinates, but the resampler receives `None` and creates an all-ones mask. Missing landmarks inferred as zeros can thus be labelled valid downstream. This matters for the `/predict` endpoint and live robustness endpoint because their request schemas do not carry an explicit mask.

**Fix:** Resolve a mask once at the preprocessing boundary and preserve it through normalization, interpolation, derivatives and inference. For a missing mask, use a clearly defined fallback derived from non-finite/zero coordinates only if that convention is valid for the selected representation; otherwise require an explicit mask. Do not infer visibility from coordinate magnitude alone if a true coordinate can legitimately be zero.

### P1.4 — Evaluation must report actual loaded data, not just manifest counts

**Evidence:** [`build_dataset_from_manifest`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/src/sign_language/data/dataset.py) silently skips entries whose corresponding landmark `.npz` file is absent. Therefore, manifest counts can differ from the dataset actually evaluated. Current training metrics reporting 69 validation records versus comparative documentation claiming 74 illustrates why these counts should be explicit.

**Fix:** Add an audit/preflight report listing missing `.npz` items and class/signers; support a strict flag for experiments that should stop on any expected missing input. Do not overwrite a previous result if the run failed or evaluated zero/partial data. Record sample IDs actually used.

### P1.5 — Conflicting data inventory reports need scope labels and regeneration

**Evidence:** [`data/manifests/dataset_report.json`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/data/manifests/dataset_report.json) describes a processed subset of 11,980 samples, 2,000 classes and 78 signers, with signer-disjoint processed manifests (5,723 train / 2,247 validation / 4,010 test). [`results/dataset_inventory.json`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/results/dataset_inventory.json) describes a broader original metadata inventory of 21,083 instances/119 signers and original split overlaps, with status text saying the signer-independent construction is incomplete.

These can describe different stages/scopes, but the repository does not make that distinction sufficiently prominent. Treating these numbers as one dataset split would be misleading.

**Fix:** Define canonical stages (source metadata inventory, local-video availability, extracted-feature inventory, final split, per-experiment vocabulary subset). Regenerate reports from data and make stage/scope, generation command, timestamp and schema version explicit. Keep the source inventory for context, but mark it as pre-filter/source metadata and do not use its original split labels for the final thesis evaluation.

## 4. P1 — Live API and frontend integration

### P1.1 — Robustness Lab may use an empty or stale feature buffer

**Evidence:** [`App.tsx`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/app/frontend/src/App.tsx#L550-L585) passes `replayFrames` into `RobustnessLab` as `recentFeatures`. The replay buffer is explicitly requested by the Gesture Replay tab handler, but switching to the Robustness Lab tab only changes the active tab. This means the lab can show an empty/stale sequence unless the replay buffer has first been requested elsewhere.

**Fix:** Maintain a dedicated rolling client-side feature buffer as frames arrive, or request replay data whenever entering Robustness Lab and show a loading/age indicator until fresh features arrive. Add an integration test proving the lab receives current features without navigating through Gesture Replay first.

### P1.2 — Robustness endpoint tests prediction consistency, not labelled robustness

**Evidence:** [`run_robustness_experiment()`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/src/sign_language/api/main.py#L260-L345) compares clean and perturbed prediction labels for one live sequence. Its note correctly says ground-truth accuracy belongs to offline labelled benchmarks.

**Fix:** Keep this feature named and presented as a **single-sequence consistency diagnostic**, not as measured research robustness. Use `scripts/evaluate_robustness.py` (after fixing it) for benchmark claims based on labelled test samples and matched sample IDs. The UI should clearly separate the two.

### P1.3 — WebSocket URLs are local-first rather than deployment-host aware

**Evidence:** [`DEFAULT_SETTINGS` and WebSocket initialization in `App.tsx`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/app/frontend/src/App.tsx#L1-L45) use localhost by default for HTTP deployments; the secure branch uses `wss://${window.location.hostname}:8000/ws/live`. This will be wrong for many deployments where the frontend and API share one host/port behind a reverse proxy, or where the API has a different hostname/port.

**Fix:** Build the WebSocket URL from an explicit environment setting or from the current origin plus a documented proxy path; choose `ws`/`wss` based on page protocol. Preserve an explicit localhost default only for local development. Add tests for local dev and same-origin production.

### P1.4 — `/predict` performs two inference calls for one request

**Evidence:** [`predict()` in `main.py`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/src/sign_language/api/main.py#L199-L235) calls `predict_label(...)` and then `predict_logits(...)` separately. `predict_label()` itself calls `predict_logits()`, so the same input is processed twice.

**Fix:** Add a single predictor method that returns logits and prediction details from one forward pass, or reuse logits to derive both outputs. Test that the model's forward path is called once per HTTP request.

### P1.5 — Rejected candidates leave buffer state ambiguous

**Evidence:** The `reject_candidate` path clears the candidate/history and sets `WAITING`, but does not clear `sequence_buffer`/`mask_buffer` (see [WebSocket handler](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/src/sign_language/api/main.py#L470-L488)). It can keep revisiting the same rejected gesture depending on motion/state conditions.

**Fix:** Specify rejection semantics. Either discard both sequence/mask buffers and motion histories on reject, or require a new neutral boundary before capture restarts. Exercise the transition in a protocol/state-machine test.

## 5. P1 — Model and metric interpretation risks

### P1.1 — Proposed model mask pooling is only partly modality-aware

**Evidence:** [`RobustHolisticFusionClassifier`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/src/sign_language/models/fusion.py#L160-L237) reduces a feature-level mask to one temporal validity mask by checking whether *any* feature is present at each frame, then uses that same temporal mask for all four modality pools. A frame with pose/face present but a hand absent can count as valid for the hand temporal stream too.

**Fix:** Consider computing per-modality frame-validity masks and using the relevant mask for each modality's temporal pooling/gating. If partial-modality frames are intended to contribute as zeros, document that design and ablate it against per-modality masking. Add tests where one modality is absent in some or all frames.

### P1.2 — Checkpoint choice is implicit and potentially surprising

**Evidence:** The API startup candidate list in [`main.py`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/src/sign_language/api/main.py#L35-L75) prioritizes `models/temporal_transformer_trained.pt`, then `temporal_transformer_best.pt`, and only then `robust_holistic_fusion_best.pt`. If more than one local checkpoint exists, startup may silently use the older/default model rather than the proposed thesis model.

**Fix:** In thesis/demo runs, set `MODEL_PATH` and `CONFIG_PATH` explicitly. Log the checkpoint path, checkpoint hash, model name, class count, vocabulary and preprocessing schema in startup logs and `/model/info`. Consider requiring a configured checkpoint instead of silently choosing from a priority list.

## 6. P2 — Data splitting, packaging, security and maintainability

### P2.1 — One signer split helper does not validate supplied overlap

**Evidence:** [`signer_aware_split()`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/src/sign_language/data/splitting.py#L101-L128) checks that all signers are assigned somewhere but does not reject overlaps between provided train/validation/test signer sets or unknown signers. The sibling `split_manifest_by_signer()` has stronger checks.

**Fix:** Reuse one shared signer-assignment validator for every split utility; test overlap, unknown signer, missing signer and empty splits.

### P2.2 — Dependency definitions disagree

**Evidence:** [`requirements.txt`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/requirements.txt) contains exact pins, while [`pyproject.toml`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/pyproject.toml) uses broad version ranges and [`environment.yml`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/environment.yml) floats most dependencies. CI installs from `pyproject.toml`, so it does not verify the exact `requirements.txt`/Conda environment users may follow.

**Fix:** Declare one tested installation path as canonical. If maintaining both pip and Conda, derive them from a documented supported set, record the tested OS/Python/framework versions, and run install smoke tests for each supported path. Avoid claiming platform compatibility that has not been tested.

### P2.3 — Dockerfile builds the backend but not the frontend bundle

**Evidence:** [`Dockerfile`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/Dockerfile) copies Python sources but does not build or copy `app/frontend/dist`. [`docker-compose.yml`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/docker-compose.yml) bind-mounts the host `dist` directory. A clean standalone image therefore does not automatically include the production UI.

**Fix:** Either document the host-built `dist` requirement explicitly or use a multi-stage Docker build that builds the frontend and copies its output into the runtime image. Add a container smoke test for `/` and `/health`.

### P2.4 — Manual training workflow assumes local-only data that a clean runner does not have

**Evidence:** [`training-validation.yml`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/.github/workflows/training-validation.yml) runs full training on GitHub-hosted infrastructure, while raw videos, extracted `data/landmarks/*.npz`, checkpoints and MediaPipe `.task` assets are absent from the public tree/ignored locally.

**Fix:** Make this workflow a synthetic-data smoke-training job or a clearly documented manual workflow that requires a secure, explicit data/artifact input. Do not commit the large dataset or checkpoint just to make CI pass.

### P2.5 — Local-data/checkpoint policy is mostly respected; document setup validation

**Evidence:** [`.gitignore`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/.gitignore) ignores raw/processed video folders, `data/landmarks`, features/frames, `*.pt`, `*.pth`, `*.ckpt`, `*.task` and training outputs. The public tree does not contain the landmark `.npz` corpus, raw clips, model checkpoints or the MediaPipe task asset.

**Recommendation:** Keep this policy. Add a setup/check command that validates local expected paths, counts usable landmark files, validates a sample `.npz` schema, checks the MediaPipe asset and checkpoint/config consistency, and reports missing items without printing secrets or requiring large files in GitHub.

### P2.6 — API deployment hardening is insufficient for public exposure

**Evidence:** [`main.py`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/src/sign_language/api/main.py#L108-L123) configures permissive CORS (`allow_origins=["*"]`, credentials enabled). Prediction payloads also lack explicit maximum frame/feature-count limits in the visible request model. This may be tolerable for a local research tool, but it should not be exposed directly to the public internet unchanged.

**Fix before public deployment:** Restrict CORS origins, validate maximum sequence length and feature dimension, cap WebSocket frame payload sizes, add rate limits/authentication where appropriate, and put the service behind a reverse proxy/TLS configuration. Keep loopback-only use for local development.

### P2.7 — Transcript edits and token deletion need a defined contract

**Evidence:** [`TranscriptWorkspace.tsx`](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/blob/main/app/frontend/src/components/TranscriptWorkspace.tsx#L35-L57) appends newly committed tokens to editable text, but the synchronization logic mainly reacts to token-count increases and a transition to zero tokens. Removing/replacing tokens without increasing token count may intentionally leave free-text edits untouched, but this behavior should be explicit and tested.

**Fix:** Decide whether the transcript textarea is a manual document independent of token chips or a projection of the token list. Then test word removal, undo, new session, manual correction, punctuation and new token append under that contract.

## 7. Result artefacts and documentation reconciliation

The following committed files do not currently form one coherent, reproducible experiment:

- `results/wlasl20_baseline_evaluation.json` and `results/wlasl20_proposed_evaluation.json`: WLASL-20 outputs, `n=78` each.
- `results/evaluation_report.json` and `results/proposed_evaluation_report.json`: older, separate experiment, `n=20`, `num_classes=100`.
- `results/research_summary.md`: summary for the 20-sample experiment with unchanged .20 accuracy in every robustness row.
- `results/research_summary.txt`: another distinct run with 194 evaluation samples.
- `results/wlasl20_comparative_study.md`: claims detailed 20-class comparison and robustness outcomes, but expects `results/robustness/wlasl20_baseline_report.json` and `...proposed_report.json`; those files are not present in the public Git tree.
- `scripts/generate_comparative_study.py` hardcodes the 122/74/78 protocol counts and assumes all robustness records exist; a clean clone cannot regenerate that comparison without those local JSON files.
- `docs/PROJECT_STATUS.md` claims all recent repairs and test suites are validated, despite CI run #15 failing before Python tests. `docs/MASTER_TODO.md` still refers to an earlier 34-test baseline and reports that are not in the public tree.

**Action:** Keep a lightweight, machine-readable experiment manifest for every run. Give each experiment a unique directory or run ID, and never overwrite a report for a different vocabulary or dataset stage. Generate markdown/figures only from raw JSON produced by that same run. Update `PROJECT_STATUS.md`, `PROJECT_AUDIT.md`, `MASTER_TODO.md`, and README only after verification, marking each item as **implemented**, **unit tested**, **validated on local real data**, or **reproduced end-to-end**—these are different levels of evidence.

## 8. Recommended repair order

1. **Fix CI first:** Node version/dependency compatibility; rerun frontend tests/build and pytest on current HEAD.
2. **Fix real-time state correctness:** clear sequence and mask buffers in all commit/reject/reset/boundary transitions; add sequential gesture integration tests.
3. **Unify model loading and preprocessing:** saved-config reconstruction, label order, feature dimension, sequence length, schema and masks must be the same across training, standard evaluation, robustness evaluation and live inference.
4. **Correct robustness transforms:** physically meaningful translation, dropout mask return/alignment, deterministic per-sample seeds, input validation and explicit perturbation stage.
5. **Audit the WLASL-20 split locally:** with local files only, report manifest rows versus valid `.npz` files, signer disjointness, vocabulary support per split, missing class coverage and exact IDs evaluated.
6. **Retrain or re-evaluate both models on identical samples:** preserve split/checkpoint metadata and record the best checkpoint selection rule.
7. **Regenerate every metric/report from the new raw results:** remove or clearly archive conflicting older summaries; do not carry over current robustness tables as thesis conclusions.
8. **Complete app integration validation:** Robustness Lab data freshness, mode switching, WebSocket reconnect/config behavior, consecutive gestures, browser camera permission/error paths and frontend production build/API paths.
9. **Update setup, Docker and documentation:** verify clean install and local-only data checks. Do not upload large raw data, extracted landmark arrays, checkpoints or `.task` files to GitHub.
10. **Only then prepare final thesis slides:** use exact sample counts, observed class coverage, final metrics, labelled robustness accuracy/F1 degradation and latency/FPS measured under stated hardware/software conditions.

## 9. Acceptance checklist for a thesis-ready repository

- [ ] Latest CI run passes Python tests, frontend tests and frontend production build.
- [ ] Clean setup can identify the required local dataset, landmark files, checkpoint and MediaPipe asset, and reports what is missing.
- [ ] All three final manifests are deterministic, validated, signer-disjoint and have documented vocabulary/class support.
- [ ] Actual evaluated sample IDs and missing-input counts are exported for every model/run.
- [ ] Baseline and proposed use the same test IDs, label mapping, preprocessing, seed policy and evaluation metric definitions.
- [ ] The configured WLASL-20 class set is fully represented in the test data or missing classes are explicitly excluded with a justified revised protocol.
- [ ] Offline evaluation and robustness evaluation share the same checkpoint reconstruction and mask-aware preprocessing code.
- [ ] Perturbation operators preserve mask alignment and their mathematical definitions match their names/descriptions.
- [ ] Raw robustness JSON exists for every reported baseline/proposed experiment and can regenerate the published table.
- [ ] Live app passes tests for commit/reject/reset/pause/repeated gestures and does not reuse stale sequence or masks.
- [ ] Local browser, production bundle and deployment WebSocket URLs are tested.
- [ ] The report and README accurately distinguish implemented, tested, locally validated and end-to-end reproduced features.
- [ ] Public repo contains no large local dataset, `.npz` corpus, checkpoints or `.task` model asset.

## 10. Scope and limitations of this audit

This audit inspected the repository tree, source files, configs, tracked result artefacts, documentation, and latest CI logs available on GitHub at the recorded HEAD. It did **not** execute training or inference on the user's local `data/landmarks` files or local model checkpoints, because those assets are intentionally not present in the public repository. Therefore, claims about the local corpus/checkpoints, current GPU training, webcam accuracy, or end-to-end runtime remain to be checked in the user's local environment. The reported findings distinguish source-level/CI-confirmed issues from local-data validation work that still needs execution.

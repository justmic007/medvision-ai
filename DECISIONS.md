# Decision log

Append-only record of architectural and scoping decisions. Each entry states
the decision, the reasoning, and (where relevant) what was rejected.

---

**D-01 — Docker-first backend.**
The app runs as a container (docker compose up) rather than a bare local
environment. Reproducibility and deploy-fidelity; also sidesteps local
interpreter/PATH conflicts. Accepted cost: a modest iteration tax on an Intel
Mac (image rebuilds, mount performance). Rejected: bare venv as the primary
run target.

**D-02 — Python 3.11.**
Pinned to 3.11 as the version where the full downstream stack
(torch/torchvision, MONAI, TorchXRayVision, MLflow) is unambiguously supported.
3.12 would likely work now but buys no needed feature while adding a small
dependency-resolution risk. TorchXRayVision itself requires only >=3.6, so the
constraint comes from torch, not it.

**D-03 — Public/synthetic data only; never committed.**
Only public or synthetic imaging data is used, and no image data or model
weights are committed to the repo (data/, weights/, *.pth, *.dcm are
gitignored). Keeps the repo clean and side-steps licensing/privacy exposure.

**D-04 — Deterministic core kept separate from the generative (VLM) layer.**
The classifier/threshold/heatmap/retrieval core is deterministic and is the
product for Phases 0–5. The optional VLM narrative layer (Phase 6) consumes the
core's structured output but never feeds back into it. This wall bounds
hallucination risk: generated prose can never alter a deterministic finding.

**D-05 — Problem statement; geographic angle framed qualitatively.**
MedVision AI is a decision-support research prototype for chest X-ray analysis,
designed with low-resource settings in mind — contexts where radiologist access
is scarce and a single specialist may serve a large population, delaying
interpretation of routine chest films. It pairs a multi-pathology classifier
with visual explainability (where a finding is grounded) and cited literature
retrieval (linking each finding to peer-reviewed evidence). Explicitly
non-diagnostic; clinician-in-the-loop ("it surfaces, the clinician decides").
The low-resource framing is stated as design intent, qualitatively, with no
fabricated statistics and no claim of deployment or validation. Rejected: a
quantitative framing citing specific radiologist-density figures (would require
sourcing we chose not to depend on) and a fully neutral capability-only framing
(less motivating for the intended audience).

**D-06 — Accept DICOM and PNG/JPG; normalize early.**
Input accepts both DICOM (the real clinical format) and PNG/JPG (convenient for
the demo). Both are normalized to a common tensor at the preprocessing boundary
so nothing downstream sees the format difference. DICOM support is a low-cost,
high-signal domain-seriousness cue. Rejected: PNG-only.

**D-07 — Published operating points; labeled subset is measurement-only.**
Thresholding uses TorchXRayVision's published operating points rather than
custom-tuned thresholds. The small labeled subset (a slice of NIH ChestX-ray14)
is reserved strictly for measurement (reporting ROC/AUC and operating-point
metrics), never for tuning — avoiding the tune-and-report-on-the-same-data
inflation trap. A held-out calibration split is held in reserve, to be used only
if the published operating points visibly misbehave on our data. Note: ROC/AUC
is threshold-independent, so the headline metric doesn't depend on this choice;
only operating-point metrics (precision/recall/F1 at the cut) do.

---

## Open items (not yet decided)

- Vector store for literature RAG (Phase 4): Chroma (simpler to run) vs
  FAISS (lighter-weight). Decide at Phase 4.
- Demo UI (Phase 5): Gradio mounted in FastAPI (minimal UI code, idiomatic
  for imaging demos) vs Next.js (consistency with a full-stack presentation).
  Decide at Phase 5.

## Implementation notes carried forward

- Phase 2 must expose the classifier's activations, not just a probability
  dict. GradCAM (Phase 3) hooks the final conv layer of the same forward
  pass that produced predictions. If Phase 2 discards everything but the numbers,
  Phase 3 has to re-run inference to recover activations. Design for this in
  Phase 2.

# MedVision AI — Concepts, in plain language

This explains the ideas behind MedVision in terms a non-specialist can follow.
It doubles as a reference for talking about the project — each entry says *what
it is*, *why MedVision uses it*, and *how to explain it simply*.

It deliberately skips general computing terms (Docker, APIs, databases) and
focuses on what is distinctive about this project: the imaging, the ML, and the
design decisions that make it a responsible medical-AI system rather than a
black box.

---

## The imaging and ML

### Chest X-ray (CXR)
**What:** A 2D radiograph of the chest — the most common medical imaging exam in
the world. MedVision works on frontal views (taken from front or back).
**Why:** Cheap and fast to take, but require a trained radiologist to interpret —
and radiologists are scarce in many places. That gap is the problem MedVision
addresses.
**Say it simply:** "The kind of chest scan you get for a suspected chest
infection."

### DICOM
**What:** The file format hospitals actually use for medical images (`.dcm`). It
holds the image plus metadata, and stores pixel data differently from ordinary
photos.
**Why:** MedVision accepts DICOM *and* ordinary PNG/JPG. Supporting the real
clinical format — not just web images — is a small piece of work that signals the
system is built for the actual medical world.
**Say it simply:** "The medical-world equivalent of a JPG — what the scanner
actually produces."

### Classifier
**What:** The neural network that looks at the X-ray and outputs a probability for
each possible finding. MedVision uses a pretrained model (TorchXRayVision's
DenseNet) rather than training its own.
**Why:** Training a strong medical classifier needs huge labelled datasets and
GPUs. Using a well-validated pretrained model lets MedVision focus its effort on
the parts that make it *trustworthy* (explainability, evidence), not on
reinventing the classifier.
**Say it simply:** "The part that actually reads the X-ray and says what it
thinks it sees — and I built around a proven one rather than training my own."

### Pathologies / findings
**What:** The 18 conditions the model can flag — e.g. effusion (fluid),
cardiomegaly (enlarged heart), pneumonia, nodule, fracture. This list comes fixed
with the model; MedVision inherits it.
**Why:** These define what the whole system can detect and report on. Everything
downstream (heatmaps, literature) is organised around this label set.
**Say it simply:** "The checklist of things it looks for on the X-ray."

### Probability and threshold (operating point)
**What:** For each finding the model outputs a number from 0 to 1 (how confident
it is). A *threshold* decides where that number counts as "present." MedVision
uses the model authors' own published thresholds — which are calibrated per
finding, not a naive 0.5.
**Why:** Using the published operating points (rather than tuning our own on a
tiny dataset) is the honest choice — it avoids inflating results, and the
thresholds come with the model's validation behind them.
**Say it simply:** "The model gives a confidence score; the threshold is the
cutoff for calling something 'present' — and I used the makers' validated cutoffs
rather than inventing my own."

### GradCAM (the heatmap)
**What:** A technique that produces a heatmap over the X-ray showing *where the
model was looking* when it made a prediction — warm colours where a region drove
the finding.
**Why:** A bare prediction is a black box. The heatmap lets a clinician check the
model's reasoning: if it flags an effusion and the heatmap lights up the right
part of the lung, that's corroborating; if it lights up the shoulder or a text
marker, the prediction is probably spurious and should be dismissed. This is the
interpretability half of MedVision's trust story.
**Say it simply:** "It shows where on the image the model was looking, so a doctor
can tell whether it focused on real anatomy or got fooled by an artifact."
**Honest caveat:** It shows *where* the model attended, not *why* clinically — a
useful audit aid, not a guarantee of correctness.

### Tensor
**What:** The numerical array a neural network actually consumes — the image
turned into numbers of a specific shape and value range.
**Why:** Preprocessing's whole job is turning any input image (DICOM or PNG) into
exactly the tensor this model expects — same size, same value range it was
trained on — so results are correct.
**Say it simply:** "The image converted into the exact grid of numbers the model
was trained to read."

---

## The evidence layer

### Literature RAG (retrieval-augmented grounding)
**What:** For each finding the model flags, MedVision retrieves real, cited
medical literature (PubMed abstracts) about it — so the output links to evidence,
not just a label.
**Why:** It bridges "the model saw X" and "here's what the medical literature says
about X," so a clinician doesn't have to leave the tool to find supporting
evidence. This is the evidence half of the trust story (the heatmap is the other
half).
**Say it simply:** "For every finding, it pulls up real published research about
that condition, with citations — so the result is backed by evidence you can
check."

### PubMed / NCBI E-utilities
**What:** PubMed is the standard database of medical research; NCBI E-utilities is
the official programmatic way to search it. MedVision queries it live.
**Why:** Using the authoritative medical literature source (rather than a
general web search) keeps the evidence credible and citable.
**Say it simply:** "The official medical research database — the same one doctors
and researchers actually use."

### Vector store and embeddings
**What:** A way of storing text so it can be searched by *meaning* rather than
exact keywords. Text is converted into numeric "embeddings"; similar meanings sit
close together, so retrieval finds relevant abstracts even without exact word
matches.
**Why:** It makes the literature retrieval find *relevant* evidence, not just
keyword matches.
**Say it simply:** "It searches research by meaning, not just matching words — so
it finds the relevant studies even if they use different terminology."

---

## The design decisions that make it responsible

### Non-diagnostic / clinician-in-the-loop
**What:** MedVision never diagnoses. It surfaces findings, shows its reasoning,
and links evidence — then a human clinician decides. The tool assists; it never
replaces the doctor.
**Why:** It's an unvalidated research prototype, and honestly scoping it that way
is both correct and the responsible-AI posture reviewers respect. Every claim the
project makes stays defensible because of this boundary.
**Say it simply:** "It's a second pair of eyes that flags things and shows its
working — the doctor always makes the call. It assists, it doesn't diagnose."

### Deterministic core vs. optional generative (VLM) layer
**What:** The core (classify, heatmap, evidence) is *deterministic* — same input,
same output, no invented content. A separate optional layer can later turn the
findings into a written narrative using a generative model, but it only *rewords*
the deterministic findings — it can never change them.
**Why:** Generative text is where AI hallucinates. Walling it off from the factual
core means the fluent prose can never alter or invent a finding — you get
readability without letting the generative part corrupt the facts.
**Say it simply:** "The part that decides what's in the X-ray is fixed and
trustworthy. The part that writes it up in nice prose comes later, is optional,
and is fenced off so it can never make up a finding."

### VLM (vision-language model)
**What:** A type of AI model that takes an image and produces text about it —
"vision" (it sees the image) plus "language" (it writes). In MedVision, a VLM is
the optional layer that could turn the structured findings into a written,
radiology-style narrative.
**Why:** It makes the output human-readable — prose instead of a list of labels
and numbers. But it is optional and fenced off (see the deterministic-vs-VLM
entry), because generative text is where AI invents things, and that risk must
never touch the factual core.
**Say it simply:** "An AI that looks at an image and writes about it — here, an
optional part that turns the findings into a readable report, kept walled off so
it can never invent a finding."

### Model pluggability (extensibility)
**What:** MedVision's core is built so that "a model" means anything that takes a
prepared image and returns findings. The chest X-ray model fills that slot today;
a different model — say brain-tumour MRI — could slot into the same interface
later without rewriting the system.
**Why:** It shows the architecture generalises beyond one use case. The chest
X-ray pipeline is shipped fully; other modalities are documented future work, not
built now — deliberately, to avoid over-engineering for models that do not yet
exist.
**Say it simply:** "I designed it so a different scan type — like brain MRI —
could be plugged in later without rebuilding everything, but I fully finished one
type rather than half-building several."

### Clinical workflow (admin, clinician, cases, audit trail)
**What:** Around the analysis engine, MedVision adds a real workflow: clinicians
have accounts and see their own past cases; an admin oversees everything; and an
audit trail records who did what, when.
**Why:** It turns a stateless analysis tool into something a clinic could actually
use, and demonstrates understanding that a model is only useful inside a
workflow — with accountability and provenance, which matter especially in
medicine.
**Say it simply:** "It's wrapped in the plumbing a real clinic needs — logins,
case history, and a record of who reviewed what."

### The compute split (local vs. cloud GPU)
**What:** Everything except the optional heavy generative model runs locally on an
ordinary CPU. Only that one big model, if built, runs on free cloud GPUs
(Colab/Kaggle).
**Why:** It keeps the whole core system runnable on a normal laptop — no GPU
needed for the parts that matter — while still allowing the one genuinely
GPU-hungry optional feature.
**Say it simply:** "The whole core runs on a normal laptop; only the one optional
heavyweight feature needs a cloud GPU."

### MLflow (experiment tracking)
**What:** A tool that logs each analysis run — which model version, which
thresholds, when — so results are reproducible and traceable.
**Why:** Reproducibility and provenance are part of doing ML responsibly; being
able to say exactly how a result was produced matters in a medical context.
**Say it simply:** "It keeps a logbook of every run so any result can be traced
back to exactly how it was produced."

---

*This glossary grows with the project. Concepts are added as each phase
introduces them.*

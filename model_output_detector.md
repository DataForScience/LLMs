<div style="width: 100%; overflow: hidden;">
    <div style="width: 150px; float: left;"> <img src="data/D4Sci_logo_ball.png" alt="Data For Science, Inc" align="left" border="0"> </div>
    <div style="float: left; margin-left: 10px;"> <h1>Classifiers</h1>
<h1>Model Output Detector</h1>
        <p>Bruno Gonçalves<br/>
        <a href="http://www.data4sci.com/">www.data4sci.com</a><br/>
            @bgoncalves, @data4sci</p></div>
</div>

```python
# One uv line installs everything. Torch and transformers serve the optional deep model only.
!uv pip install --system --quiet scikit-learn pandas numpy matplotlib joblib watermark torch transformers datasets accelerate huggingface_hub

import datetime
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from datasets import Dataset
from huggingface_hub import hf_hub_download
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

import watermark

%load_ext watermark
%matplotlib inline
```

We start by print out the versions of the libraries we're using for future reference

```python
%watermark -n -v -m -g -iv
```

Load default figure style

```python
plt.style.use('d4sci.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
```

# Start

This notebook closes the loop. The benchmark notebook generated paragraph-length answers to HC3 questions, one JSONL per model run. HC3 pairs every question with human-written answers and with ChatGPT answers from 2022. Here all three sources meet: your model generations, the human class, and the ChatGPT class, matched on the same prompts. A detector learns to name the author of each answer, and an analysis section takes the results apart: train against test, per-class scores, confusion structure, domain effects, confidence, calibration, embedding geometry, and a learning curve.

Two stages. A TF-IDF character n-gram model with logistic regression trains in seconds and sets the bar. An optional ModernBERT head trains on the same split when the linear model plateaus. Expectations: human vs any model should clear 95% at paragraph length, cross-family model pairs land above 90%, same-family siblings near 70-85%, and the Q4 vs Q8 quant twins probe the floor. A twin score near chance is a finding, not a failure.

## Parameters

Model labels come from the GGUF filename in each run's summary JSON, so quant twins stay separate classes. `ADD_HUMAN` and `ADD_CHATGPT` append the two HC3 classes. `HC3_FILE` must match the benchmark run, the prompt order depends on it. The split key is the prompt index: a prompt lands entirely in train or entirely in test across every class at once, which kills prompt leakage, the classic way detectors cheat.

```python
RESULTS_DIR       = "results"
HC3_FILE          = "all.jsonl"   # must match the benchmark notebook
MIN_CHARS         = 30            # drop junk fragments, paragraphs are the task now
ADD_HUMAN         = True          # HC3 human answers as the "human" class
ADD_CHATGPT       = True          # HC3 ChatGPT answers as the "chatgpt-2022" class
HUMAN_LABEL       = "human"
CHATGPT_LABEL     = "chatgpt-2022"
BALANCE           = True          # downsample every class to the smallest class
TEST_MOD          = 5             # prompt_idx % TEST_MOD == 0 goes to test, near 20%
SEED              = 0

TRAIN_TRANSFORMER = False         # flip on for the ModernBERT upgrade
TRF_MODEL         = "answerdotai/ModernBERT-base"
MAX_LEN           = 256           # covers most paragraph answers, doubles training speed
TRF_EPOCHS        = 3
```

## Load the generations

Every `throughput__*.generations.jsonl` pairs with a summary JSON of the same stem. The summary carries the authoritative model file and hash. Runs of the same model merge under one label.

```python
gen_files = sorted(Path(RESULTS_DIR).glob("throughput__*.generations.jsonl"))
assert gen_files, "no generations found, run the benchmark notebook first"

frames = []
for gf in gen_files:
    stem = gf.name.replace(".generations.jsonl", "")
    label, mhash = stem.split("__")[1], stem.split("__")[2]
    summ_path = gf.with_name(stem + ".json")
    if summ_path.exists():
        s = json.load(open(summ_path))
        label = Path(s["model"]["file"]).stem
        mhash = s["model"]["hash"]
    d = pd.read_json(gf, lines=True)
    d["label"], d["model_hash"], d["source_file"] = label, mhash, gf.name
    frames.append(d)

data = pd.concat(frames, ignore_index=True)
print(f"files: {len(gen_files)} | rows: {len(data):,} | classes: {data.label.nunique()}")
data.groupby("label").size().sort_values(ascending=False)
```

## Pair with HC3: domain map, human class, ChatGPT class

The loader below reproduces the benchmark's prompt order line for line: same file, same filters, same dedupe. That puts every HC3 answer on the same `prompt_idx` as the model answers to the same question. Human answers are Reddit and expert text, ChatGPT answers are the 2022 model, and both cap at the benchmarked prompt range so every class answers one shared pool.

```python
hc3_path = hf_hub_download(repo_id="Hello-SimpleAI/HC3",
                           filename=HC3_FILE, repo_type="dataset")
hd = pd.read_json(hc3_path, lines=True)
hd["question"] = hd.question.fillna("").str.strip()
hd = hd[(hd.question.str.len() > 0) & (hd.human_answers.str.len() > 0)]
pairs = hd.drop_duplicates(subset=["question"]).reset_index(drop=True)
pairs["prompt_idx"] = pairs.index

bench_max = int(data.prompt_idx.max())
pairs = pairs[pairs.prompt_idx <= bench_max]

# Why a domain map: HC3 spans five domains, and the analysis slices accuracy by each.
data["domain"] = data.prompt_idx.map(dict(zip(pairs.prompt_idx, pairs.source)))

def explode_answers(col, label, hash_tag):
    rows = pairs[["prompt_idx", "source", col]].explode(col).dropna(subset=[col])
    rows = rows[rows[col].str.strip().str.len() > 0]
    out = pd.DataFrame({
        "run_id": -1,
        "temperature": np.nan,        # temperature slices drop these rows on groupby
        "n_prompts": np.nan,
        "prompt_idx": rows.prompt_idx.values,
        "sample_idx": rows.groupby("prompt_idx").cumcount().values,
        "out_tokens": (rows[col].str.len() / 4).clip(lower=1).round().astype(int),
        "text": rows[col].values,
        "label": label,
        "model_hash": hash_tag,
        "source_file": f"HC3 {col}",
        "domain": rows.source.values,
    })
    return out

extra = []
if ADD_HUMAN:
    extra.append(explode_answers("human_answers", HUMAN_LABEL, "human"))
if ADD_CHATGPT:
    extra.append(explode_answers("chatgpt_answers", CHATGPT_LABEL, "hc3-chatgpt"))
if extra:
    data = pd.concat([data] + extra, ignore_index=True)

print(f"rows: {len(data):,} | classes: {data.label.nunique()}")
data.groupby("label").size()
```

## Clean, dedupe, balance

Greedy runs repeat across sweep configs, so exact duplicates get one row each. Balancing keeps the confusion matrix honest: a skewed class mix inflates accuracy for the wrong reason. The ChatGPT class holds one answer per prompt, so it usually sets the floor that every class downsamples to.

```python
data["text"] = data.text.fillna("").str.strip()
data = data[data.text.str.len() >= MIN_CHARS]
data = data.drop_duplicates(subset=["label", "prompt_idx", "temperature", "text"])

if BALANCE:
    n_min = int(data.groupby("label").size().min())
    data = pd.concat(
        [g.sample(n=n_min, random_state=SEED) for _, g in data.groupby("label")],
        ignore_index=True)
    print(f"balanced to {n_min:,} rows per class")

print(f"rows: {len(data):,} | classes: {data.label.nunique()}")
data.groupby("label").size()
```

## Split by prompt, never by completion

Every downstream number in this notebook comes from this split. The detector fits on train prompts and every chart reads from held-out test prompts, except the train bars shown for the generalization gap.

```python
# Why modulo on prompt_idx: the same prompts feed every class in the same order,
# so this one rule puts a prompt on one side of the split for all classes at once.
test_mask = data.prompt_idx % TEST_MOD == 0
train, test = data[~test_mask].copy(), data[test_mask].copy()
print(f"train: {len(train):,} rows on {train.prompt_idx.nunique():,} prompts | "
      f"test: {len(test):,} rows on {test.prompt_idx.nunique():,} prompts")
```

## Train the baseline

Character 3-5 grams catch tokenizer artifacts, punctuation habits, and stock phrases. Sublinear TF stops one repeated phrase from owning a document. The whole pipeline fits in seconds on CPU.

```python
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                              min_df=2, max_features=200_000, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=2000)),
])

t0 = time.time()
pipe.fit(train.text, train.label)
fit_s = time.time() - t0

pred = pipe.predict(test.text)
acc = accuracy_score(test.label, pred)
macro = f1_score(test.label, pred, average="macro")
print(f"fit {fit_s:.1f} s | test accuracy {acc:.3f} | macro F1 {macro:.3f}\n")
print(classification_report(test.label, pred, digits=3))
```

## Analyze the results

Ten views, all on the prompt-level split. Train bars appear once, in the gap panel. Everything after that reads test only.

```python
# Why score train too: the gap between the bars is the overfit measure, one chart, no guessing.
pred_train = pipe.predict(train.text)
acc_tr = accuracy_score(train.label, pred_train)
macro_tr = f1_score(train.label, pred_train, average="macro")

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(2)
ax.bar(x - 0.18, [acc_tr, acc], width=0.36, color=colors[0], label="accuracy")
ax.bar(x + 0.18, [macro_tr, macro], width=0.36, color=colors[1], label="macro F1")
ax.set_xticks(x)
ax.set_xticklabels(["train", "test"])
ax.set_ylim(0, 1.05)
ax.set_title("Generalization gap")
for xi, v in zip([-0.18, 0.82, 0.18, 1.18], [acc_tr, acc, macro_tr, macro]):
    ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.legend()
plt.tight_layout()
plt.show()
print(f"accuracy gap: {acc_tr - acc:+.3f} | macro F1 gap: {macro_tr - macro:+.3f}")
```

```python
# Why per-class F1: the average hides the one class the detector cannot see.
rep = classification_report(test.label, pred, output_dict=True)
per_class = (pd.DataFrame(rep).T
             .loc[sorted(test.label.unique()), ["precision", "recall", "f1-score"]]
             .sort_values("f1-score"))

fig, ax = plt.subplots(figsize=(8, 0.45 * len(per_class) + 1.5))
ax.barh(per_class.index, per_class["f1-score"], color=colors[0])
ax.set_xlim(0, 1.02)
ax.set_xlabel("F1 on test")
ax.set_title("Per-class F1, sorted")
for i, v in enumerate(per_class["f1-score"]):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout()
plt.show()
```

```python
# Why row-normalized colors with raw counts: shades compare classes fairly, numbers stay auditable.
labels_sorted = sorted(data.label.unique())
cm = confusion_matrix(test.label, pred, labels=labels_sorted)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

side = max(5, 1 + 0.8 * len(labels_sorted))
fig, ax = plt.subplots(figsize=(side, side))
im = ax.imshow(cm_norm, vmin=0, vmax=1)
ax.set_xticks(range(len(labels_sorted)))
ax.set_yticks(range(len(labels_sorted)))
ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(labels_sorted, fontsize=8)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("Confusion matrix, row-normalized shading, raw counts")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
fig.colorbar(im, fraction=0.046)
plt.tight_layout()
plt.show()

off = cm.astype(float).copy()
np.fill_diagonal(off, 0)
flat = [(labels_sorted[i], labels_sorted[j], int(off[i, j]))
        for i in range(len(labels_sorted)) for j in range(len(labels_sorted)) if off[i, j] > 0]
print("top confused pairs, true -> predicted:")
for a, b, n in sorted(flat, key=lambda x: -x[2])[:5]:
    print(f"  {a} -> {b}: {n}")
```

```python
test_res = test.assign(pred=pred, correct=lambda d: d.pred == d.label)

# Why three slices: temperature moves style variance, length moves evidence,
# domain moves register. HC3 rows carry no temperature, so the left panel skips them.
by_temp = test_res.groupby("temperature").correct.mean()
bins = pd.cut(test_res.out_tokens, [0, 32, 64, 128, 192, 260])
by_len = test_res.groupby(bins, observed=True).correct.mean()
by_dom = test_res.groupby("domain").correct.mean().sort_values()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar([f"{t:g}" for t in by_temp.index], by_temp.values, color=colors[0])
axes[0].set_xlabel("temperature")
axes[0].set_ylabel("accuracy")
axes[0].set_title("By temperature (model classes)")
axes[0].set_ylim(0, 1.02)

axes[1].bar([str(b) for b in by_len.index], by_len.values, color=colors[1])
axes[1].set_xlabel("output tokens")
axes[1].set_title("By answer length")
axes[1].set_ylim(0, 1.02)
axes[1].tick_params(axis="x", rotation=30)

axes[2].bar(by_dom.index, by_dom.values, color=colors[2])
axes[2].set_xlabel("HC3 domain")
axes[2].set_title("By domain")
axes[2].set_ylim(0, 1.02)
axes[2].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()
```

```python
# Why confidence views: they separate "wrong and loud" from "wrong and unsure".
P = pipe.predict_proba(test.text)
classes_arr = np.array(pipe.classes_)
maxp = P.max(axis=1)
correct = (pred == test.label.values)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
edges = np.linspace(1 / len(classes_arr), 1.0, 25)
axes[0].hist(maxp[correct], bins=edges, alpha=0.7, color=colors[0], label="correct")
axes[0].hist(maxp[~correct], bins=edges, alpha=0.7, color=colors[1], label="wrong")
axes[0].set_xlabel("top predicted probability")
axes[0].set_ylabel("test rows")
axes[0].set_title("Confidence by outcome")
axes[0].legend()

# Reliability: within each confidence bin, accuracy should track the diagonal.
bin_ids = np.clip(np.digitize(maxp, np.linspace(0, 1, 11)) - 1, 0, 9)
mids, obs = [], []
for b in range(10):
    m = bin_ids == b
    if m.sum() >= 20:
        mids.append(maxp[m].mean())
        obs.append(correct[m].mean())
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].plot(mids, obs, "o-", color=colors[2])
axes[1].set_xlabel("mean predicted probability")
axes[1].set_ylabel("observed accuracy")
axes[1].set_title("Reliability")
plt.tight_layout()
plt.show()
```

```python
# Why top-k: with many classes, second place carries real information for triage.
true_idx = np.searchsorted(classes_arr, test.label.values)
order = np.argsort(-P, axis=1)
rank_true = np.argmax(order == true_idx[:, None], axis=1)
topk = {k: float((rank_true < k).mean()) for k in (1, 2, 3)}

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar([f"top-{k}" for k in topk], list(topk.values()), color=colors[0])
ax.set_ylim(0, 1.05)
ax.set_title("Top-k accuracy")
for i, v in enumerate(topk.values()):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.show()
```

```python
# Why this heatmap: it names the temperature where each model becomes identifiable.
# The HC3 classes drop out here, they have no temperature axis.
heat = test_res.pivot_table(index="label", columns="temperature",
                            values="correct", aggfunc="mean")
fig, ax = plt.subplots(figsize=(2 + 1.2 * heat.shape[1], 0.5 * heat.shape[0] + 2))
im = ax.imshow(heat.values, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(heat.shape[1]))
ax.set_xticklabels([f"{t:g}" for t in heat.columns])
ax.set_yticks(range(heat.shape[0]))
ax.set_yticklabels(heat.index, fontsize=8)
ax.set_xlabel("temperature")
ax.set_title("Per-class accuracy by temperature")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=8)
fig.colorbar(im, fraction=0.03)
plt.tight_layout()
plt.show()
```

```python
# Why fit the projection on train only: the map obeys the same split as the detector.
vec = pipe.named_steps["tfidf"]
svd = TruncatedSVD(n_components=2, random_state=SEED).fit(vec.transform(train.text))

show = test if len(test) <= 2000 else test.sample(2000, random_state=SEED)
X2 = svd.transform(vec.transform(show.text))

fig, ax = plt.subplots(figsize=(8, 6))
for i, lab in enumerate(sorted(show.label.unique())):
    m = (show.label == lab).values
    ax.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.6,
               color=colors[i % len(colors)], label=lab)
ax.set_title("Test set in 2D TF-IDF space (train-fitted SVD)")
ax.set_xlabel("component 1")
ax.set_ylabel("component 2")
ax.legend(fontsize=7, markerscale=2, loc="best")
plt.tight_layout()
plt.show()
```

```python
# Why a learning curve: a flat test line says the signal saturated, more sweeps add nothing.
fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
curve = []
for f in fracs:
    sub = pd.concat([g.sample(frac=f, random_state=SEED)
                     for _, g in train.groupby("label")])
    m = clone(pipe).fit(sub.text, sub.label)
    curve.append({"n_train": len(sub),
                  "train_acc": accuracy_score(sub.label, m.predict(sub.text)),
                  "test_acc": accuracy_score(test.label, m.predict(test.text))})
curve = pd.DataFrame(curve)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(curve.n_train, curve.train_acc, "o-", color=colors[0], label="train")
ax.plot(curve.n_train, curve.test_acc, "o-", color=colors[1], label="test")
ax.set_xlabel("training rows")
ax.set_ylabel("accuracy")
ax.set_title("Learning curve")
ax.legend()
plt.tight_layout()
plt.show()
curve.round(3)
```

```python
# Why inspect features: the top n-grams name the habits the detector keys on.
clf = pipe.named_steps["clf"]
feats = vec.get_feature_names_out()
coefs = clf.coef_ if len(clf.classes_) > 2 else np.vstack([-clf.coef_[0], clf.coef_[0]])

for i, lab in enumerate(clf.classes_):
    top = [repr(feats[k]) for k in np.argsort(coefs[i])[-8:][::-1]]
    print(f"{lab}\n  {', '.join(top)}\n")
```

## Reading the results

The gap panel comes first for a reason: a train score far above test means memorized prompts or too few classes worth of data, and nothing downstream matters until it closes. The per-class bars and the confusion matrix then name the hard classes, family siblings and quant twins should own the off-diagonal. Human vs model at paragraph length should sit near the top of the F1 chart, HC3 human text carries typos, contractions, first-person asides, and Reddit register that no instruct model reproduces at temperature 0.8, let alone 0.

The domain panel earns its place with one question: does the detector transfer register, or memorize it. Accuracy that holds across eli5, finance, and medicine means style features, accuracy that collapses on one domain means topic features. The temperature panel usually rises for model classes, sampled text carries more habits than greedy text. The length panel should climb through the 32-128 token bins and flatten, past that point more words add no new evidence.

The confidence pair separates failure modes. Wrong-and-unsure errors cluster left and the reliability curve hugs the diagonal, so the probabilities work for triage. Wrong-and-loud errors signal near-duplicate classes, check the twins. Top-k tells you whether second place belongs in an interface. The 2D map gives the geometry: families form islands, twins overlap, the human cloud and the ChatGPT-2022 cloud should sit far apart, three model generations separate them. The learning curve closes the loop on data budget, a flat test line means the next benchmark sweep buys nothing for this detector.

## Optional: ModernBERT upgrade

The linear model reads characters. A transformer reads context. Flip `TRAIN_TRANSFORMER` when the baseline plateaus below the target. Same split, same labels, so the two scores compare cleanly.

```python
if TRAIN_TRANSFORMER:
    if torch.cuda.is_available():
        DEVICE, BS = "cuda", 32
    elif torch.backends.mps.is_available():
        DEVICE, BS = "mps", 8
    else:
        DEVICE, BS = "cpu", 4

    labels_list = sorted(data.label.unique())
    l2i = {l: i for i, l in enumerate(labels_list)}

    tok = AutoTokenizer.from_pretrained(TRF_MODEL)

    def enc(batch):
        out = tok(batch["text"], truncation=True, max_length=MAX_LEN)
        out["labels"] = [l2i[l] for l in batch["label"]]
        return out

    ds_tr = Dataset.from_pandas(train[["text", "label"]]).map(enc, batched=True)
    ds_te = Dataset.from_pandas(test[["text", "label"]]).map(enc, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        TRF_MODEL, num_labels=len(labels_list))

    def metrics_fn(p):
        y = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, y),
                "macro_f1": f1_score(p.label_ids, y, average="macro")}

    args = TrainingArguments(
        output_dir=f"{RESULTS_DIR}/detector_modernbert",
        per_device_train_batch_size=BS,
        per_device_eval_batch_size=BS * 2,
        num_train_epochs=TRF_EPOCHS,
        learning_rate=2e-5,
        bf16=(DEVICE == "cuda"),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr,
                      eval_dataset=ds_te, compute_metrics=metrics_fn)
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(f"{RESULTS_DIR}/detector_modernbert")
else:
    print("TRAIN_TRANSFORMER is False, the linear baseline stands")
```

## Save the detector and the report

```python
ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
out_dir = Path(RESULTS_DIR)

joblib.dump(pipe, out_dir / "detector_tfidf_logreg.joblib")

report = {
    "run_id": ts,
    "host": platform.node(),
    "sklearn": sklearn.__version__,
    "dataset": {
        "files": [f.name for f in gen_files],
        "hc3_file": HC3_FILE,
        "human_class": ADD_HUMAN,
        "chatgpt_class": ADD_CHATGPT,
        "rows": len(data),
        "classes": {l: int(n) for l, n in data.groupby("label").size().items()},
        "train_rows": len(train),
        "test_rows": len(test),
        "split": f"prompt_idx % {TEST_MOD} == 0 -> test",
        "balanced": BALANCE,
    },
    "baseline": {
        "model": "tfidf_char35_logreg",
        "fit_seconds": round(fit_s, 1),
        "train_accuracy": round(float(acc_tr), 4),
        "test_accuracy": round(float(acc), 4),
        "train_macro_f1": round(float(macro_tr), 4),
        "test_macro_f1": round(float(macro), 4),
        "top_k_accuracy": {str(k): round(v, 4) for k, v in topk.items()},
        "per_class": classification_report(test.label, pred, output_dict=True),
        "accuracy_by_temperature": {f"{k:g}": round(float(v), 4)
                                    for k, v in by_temp.items()},
        "accuracy_by_domain": {k: round(float(v), 4) for k, v in by_dom.items()},
        "learning_curve": curve.round(4).to_dict(orient="records"),
    },
}
with open(out_dir / f"detector_report__{ts}.json", "w") as f:
    json.dump(report, f, indent=2)
print("wrote", f"detector_report__{ts}.json", "and detector_tfidf_logreg.joblib")
```

## Use it

```python
def identify(text, k=3):
    proba = pipe.predict_proba([text])[0]
    order = np.argsort(proba)[::-1][:k]
    return [(pipe.classes_[i], round(float(proba[i]), 3)) for i in order]

sample = test.sample(1, random_state=SEED).iloc[0]
print("text:", sample.text[:200])
print("true:", sample.label)
print("top guesses:", identify(sample.text))
```

Retrain by rerunning top to bottom after every new benchmark sweep. New generations files join the dataset by filename, new models become new classes, the HC3 classes ride along on their flags, and the report JSON stamps each detector with the exact files behind it.

<center>
     <img src="data/D4Sci_logo_full.png" alt="Data For Science, Inc" align="center" border="0" width=300px> 
</center>

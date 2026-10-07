#!/usr/bin/env python
# coding: utf-8

# <div style="width: 100%; overflow: hidden;">
#     <div style="width: 150px; float: left;"> <img src="data/D4Sci_logo_ball.png" alt="Data For Science, Inc" align="left" border="0"> </div>
#     <div style="float: left; margin-left: 10px;"> <h1>Benchmarks</h1>
# <h1>GGUF Throughput: HC3 Prompts</h1>
#         <p>Bruno Gonçalves<br/>
#         <a href="http://www.data4sci.com/">www.data4sci.com</a><br/>
#             @bgoncalves, @data4sci</p></div>
# </div>

import datetime
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import torch
import transformers
from gguf import GGUFReader
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# vLLM exists only on the Linux install path. The try keeps the first-cell rule intact everywhere.
try:
    import vllm
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None

import watermark


# We start by print out the versions of the libraries we're using for future reference

# The API twin of `%watermark -n -v -m -g -iv`, so the CLI run logs the same provenance.
print(watermark.watermark(datename=True, python=True, machine=True,
                          githash=True, iversions=True, globals_=globals()))


# Load default figure style

plt.style.use('d4sci.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# # Start

# One question drives this script: how long does a given GGUF take to process N prompts on this
# machine? It times prompt batches from HC3, the Human ChatGPT Comparison Corpus, and it sweeps two
# more axes: repeats of the same prompt and a temperature range. Every generated sample lands in a
# log with a sequential run id and its temperature. HC3 questions come paired with human-written
# answers, so these generations feed the detector notebook directly, model classes from here, the
# human class from the corpus.
#
# The scan walks a list of models in the same size class, one full sweep each. Every artifact
# carries the model name in its filename, so the runs land side by side instead of on top of each
# other, and the leaderboard at the end compares like with like.
#
# Device ladder: CUDA runs vLLM. MPS and CPU fall back to transformers, which dequantizes the same
# GGUF on load. The fallback measures the device, not the quant kernels, so compare engines with
# that in mind.

# ## Parameters
#
# `MODELS` lists the scan: same size class, different vendors and architectures, all single-file
# Q4_K_M GGUFs from ungated repos, so an unattended run never stalls on a license wall. Each entry
# travels as repo, file, and tokenizer, the tokenizer pointing at the base HF repo that supplies
# the chat template. The gemma tokenizer is the one gated repo: the account holds Gemma access,
# and shells launched from the terminal carry HF_TOKEN. A `path` key wins over the download when
# it names an existing local file.
#
# The sweep runs every temperature against every prompt count. `N_SAMPLES` repeats each prompt
# within a run. Repeats at temperature 0 would return identical copies, so the code forces one
# sample there. Paragraph outputs at `MAX_NEW = 256` make this decode-heavy: the CUDA defaults
# produce near 10,000 generations per model in one to two hours on the Spark.
#
# `CHECKPOINT_EVERY` chunks each config so an interrupted run resumes instead of restarting, and it
# bounds what a crash can lose to one chunk of work. Checkpoints key on the model hash, so a scan
# killed mid-list resumes inside the model it was running and skips nothing that finished.
# `RESUME = False` discards stale checkpoints and starts each sweep over.

MODELS = [
    {"repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
     "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
     "tokenizer": "Qwen/Qwen2.5-7B-Instruct"},
    {"repo": "bartowski/gemma-2-9b-it-GGUF",
     "file": "gemma-2-9b-it-Q4_K_M.gguf",
     # Why this works gated: the account holds Gemma access, and HF_TOKEN travels
     # with every shell launched from the terminal.
     "tokenizer": "google/gemma-2-9b-it"},
    {"repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
     "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
     "tokenizer": "mistralai/Mistral-7B-Instruct-v0.3"},
    {"repo": "Qwen/Qwen3-8B-GGUF",
     "file": "Qwen3-8B-Q4_K_M.gguf",
     "tokenizer": "Qwen/Qwen3-8B",
     # Why thinking off: the template defaults to reasoning mode, and 256-token
     # answers would drown in <think> preambles.
     "template_kwargs": {"enable_thinking": False}},
    {"repo": "bartowski/Ministral-8B-Instruct-2410-GGUF",
     "file": "Ministral-8B-Instruct-2410-Q4_K_M.gguf",
     "tokenizer": "mistralai/Ministral-8B-Instruct-2410"},
    {"repo": "bartowski/Yi-1.5-9B-Chat-GGUF",
     "file": "Yi-1.5-9B-Chat-Q4_K_M.gguf",
     "tokenizer": "01-ai/Yi-1.5-9B-Chat"},
    {"repo": "bartowski/Falcon3-7B-Instruct-GGUF",
     "file": "Falcon3-7B-Instruct-Q4_K_M.gguf",
     "tokenizer": "tiiuae/Falcon3-7B-Instruct"},
]

HC3_FILE       = "all.jsonl"       # or reddit_eli5.jsonl, finance.jsonl, medicine.jsonl, open_qa.jsonl, wiki_csai.jsonl
PROMPT_COUNTS  = None              # None = per-device defaults, or a list like [100, 1000, 10000]
TEMPERATURES   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]   # the scanned range
N_SAMPLES      = 4                 # repeats per prompt, takes effect above temperature 0
SEED           = 0                 # fixed seed keeps sampled runs reproducible
MAX_NEW        = 256               # paragraph-length answers
MAX_MODEL_LEN  = 1024              # HC3 worst case is 660 tokens (404 prompt + 256 answer), and a
                                   # tight ceiling lets the scheduler admit deeper decode batches
GPU_UTIL       = 0.70              # unified memory on the Spark, leave headroom
FALLBACK_BATCH = 8                 # transformers batch size on mps and cpu
CHECKPOINT_EVERY = 1000            # prompts per chunk, bounds what an interruption can lose
RESUME         = True              # True picks an interrupted sweep back up, False starts over
RESULTS_DIR    = "results"


# ## Pick the device and the engine

# Why this order: vLLM beats transformers on CUDA by a wide margin, so it wins when present.
# MPS and CPU get transformers, since vLLM ships no kernels for either.
if torch.cuda.is_available() and vllm is not None:
    DEVICE, ENGINE = "cuda", "vllm"
elif torch.cuda.is_available():
    DEVICE, ENGINE = "cuda", "transformers"   # vLLM missing, still usable
elif torch.backends.mps.is_available():
    DEVICE, ENGINE = "mps", "transformers"
else:
    DEVICE, ENGINE = "cpu", "transformers"

# Why per-device defaults: temperatures, repeats, and 256-token outputs multiply the work,
# so the CUDA default stops at 1,000 prompts and the fallback devices stay tiny.
if PROMPT_COUNTS is None:
    PROMPT_COUNTS = {"cuda": [10000], #[100, 1000],
                     "mps":  [10000], #[25, 100],
                     "cpu":  [10000] #[10, 25]
                     }[DEVICE]

n_configs = len(TEMPERATURES) * len(PROMPT_COUNTS)
print(f"device: {DEVICE} | engine: {ENGINE} | counts: {PROMPT_COUNTS} "
      f"| temps: {TEMPERATURES} | samples: {N_SAMPLES} "
      f"| configs: {n_configs} x {len(MODELS)} models")


# ## Load HC3
#
# HC3 ships plain JSONL files, so the loader skips the dataset script and reads the file straight
# from the Hub. Rows keep only questions with at least one human answer, which guarantees the
# detector a paired human text for every prompt. The dedupe order defines `prompt_idx`, and the
# detector notebook reproduces it line for line. The questions load once; each model renders them
# through its own chat template inside the scan loop.

hc3_path = hf_hub_download(repo_id="Hello-SimpleAI/HC3",
                           filename=HC3_FILE, repo_type="dataset")
hd = pd.read_json(hc3_path, lines=True)
hd["question"] = hd.question.fillna("").str.strip()
hd = hd[(hd.question.str.len() > 0) & (hd.human_answers.str.len() > 0)]

pairs = hd.drop_duplicates(subset=["question"]).reset_index(drop=True)
n_max = min(max(PROMPT_COUNTS), len(pairs))
pairs = pairs.head(n_max)
print(f"rows: {len(hd):,} | unique questions: {len(pairs):,} used of {n_max:,} | "
      f"domains: {dict(pairs.source.value_counts())}")

INSTRUCTION = ("Answer the question in one clear paragraph of 4 to 8 sentences. "
               "Write for a general reader.")


# ## Model helpers
#
# The fast hash reads the first and last 16 MB plus the file size. That fingerprints a 5 GB file
# in under a second and stays stable across re-downloads. It keys the leaderboard together with
# the filename.

def resolve_model(spec):
    if spec.get("path") and Path(spec["path"]).exists():
        return Path(spec["path"])
    print(f"downloading {spec['file']} from {spec['repo']}")
    return Path(hf_hub_download(repo_id=spec["repo"], filename=spec["file"]))

def fast_hash(path, chunk=16 * 1024 * 1024):
    h = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(-chunk, 2)
            h.update(f.read(chunk))
    h.update(str(size).encode())
    return h.hexdigest()[:12]

def gguf_meta(path):
    out = {}
    try:
        r = GGUFReader(str(path), "r")
        for key in ("general.name", "general.architecture",
                    "general.size_label", "general.file_type"):
            fld = r.fields.get(key)
            if fld is None:
                continue
            v = fld.parts[fld.data[0]]
            if hasattr(v, "tobytes"):
                try:
                    v = v.tobytes().decode()
                except Exception:
                    v = v.tolist()
            out[key] = v
    except Exception as e:
        out["error"] = str(e)
    return out

def k_for(temperature):
    # Why clamp: greedy decoding returns identical copies, so repeats add nothing at temperature 0.
    return 1 if temperature == 0 else N_SAMPLES


# ## Scan the models
#
# One full pass per model: resolve the file, render the prompts through the model's chat template,
# start the engine, sweep with checkpointing, draw the per-model figures, write the per-model
# artifacts, then tear the engine down so the next model gets the memory back.
#
# Both engine branches expose one function: `run(prompts, temperature)` returns prompt tokens,
# output tokens, and one record per generated sample. The timing loop stays engine-blind. One
# accounting note: vLLM shares the prefill across the `n` samples of a prompt, transformers
# recomputes it per sample. Prompt tokens count once per prompt on both paths, so the workload
# definition matches, and the speed gap shows up where it belongs: in the clock.
#
# The sweep runs each config in chunks of `CHECKPOINT_EVERY` prompts. A finished chunk appends its
# samples to a checkpoint JSONL and then records its timing in a state file, in that order, so a
# crash between the two writes just redoes one chunk and a resume drops the orphaned duplicates.
# Sequential `run_id`s get assigned after the sweep, over the deterministic (temperature, prompt
# count, prompt, sample) order, so an interrupted-and-resumed run numbers its samples exactly like
# an uninterrupted one.

out_dir = Path(RESULTS_DIR)
out_dir.mkdir(exist_ok=True)
ckpt_dir = out_dir / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

for spec in MODELS:
    model_path = resolve_model(spec)
    model_hash = fast_hash(model_path)
    meta = gguf_meta(model_path)
    size_gb = model_path.stat().st_size / 1e9
    print(f"\n{'=' * 70}\n{model_path.name} | {size_gb:.2f} GB | hash {model_hash}\n{meta}")

    # ---- Prompts through this model's chat template ----
    tok = AutoTokenizer.from_pretrained(spec["tokenizer"])

    def build(question):
        # Why the cap: a few finance and medicine questions run very long, and 2,000
        # characters keeps every prompt inside MAX_MODEL_LEN with room for the answer.
        user = f"{INSTRUCTION}\n\nQuestion: {question[:2000]}"
        return tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
            **spec.get("template_kwargs", {}))

    prompt_texts = [build(q) for q in pairs.question]
    lens = [len(ids) for ids in tok(prompt_texts).input_ids]
    print(f"prompt tokens | max: {max(lens)} | mean: {sum(lens)//len(lens)} | total: {sum(lens):,}")
    assert max(lens) + MAX_NEW <= MAX_MODEL_LEN, "raise MAX_MODEL_LEN"

    # ---- Start the engine ----
    if ENGINE == "vllm":
        # Why prefix caching off: the sweep reuses nested prompt subsets,
        # and a warm cache would flatter every run after the first.
        llm = LLM(
            model=str(model_path),
            tokenizer=spec["tokenizer"],
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_UTIL,
            enable_prefix_caching=False,
        )

        def run(prompts, temperature):
            k = k_for(temperature)
            # Why a fixed seed: reruns of the same config reproduce the same samples.
            sp = SamplingParams(temperature=temperature, top_p=1.0,
                                max_tokens=MAX_NEW, n=k, seed=SEED)
            outs = llm.generate(prompts, sp)
            p_tok = sum(len(o.prompt_token_ids) for o in outs)
            o_tok, gens = 0, []
            for pi, o in enumerate(outs):
                for si, seq in enumerate(o.outputs):
                    o_tok += len(seq.token_ids)
                    gens.append((pi, si, len(seq.token_ids), seq.text.strip()))
            return p_tok, o_tok, gens

        engine_version = vllm.__version__

    else:
        # Why float16 off cpu: mps runs half precision well, cpu wants float32.
        dt = torch.float32 if DEVICE == "cpu" else torch.float16
        # transformers dequantizes the GGUF on load, so memory equals params times dtype size.
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path.parent), gguf_file=model_path.name, torch_dtype=dt
        ).to(DEVICE).eval()

        # Why left padding: decoder-only generation appends on the right,
        # so padding must sit on the left to keep every row aligned.
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        def run(prompts, temperature):
            k = k_for(temperature)
            torch.manual_seed(SEED)
            kwargs = dict(max_new_tokens=MAX_NEW, num_return_sequences=k,
                          pad_token_id=tok.pad_token_id)
            if temperature > 0:
                kwargs.update(do_sample=True, temperature=temperature, top_p=1.0)
            else:
                kwargs.update(do_sample=False)

            p_tok, o_tok, gens = 0, 0, []
            for i in range(0, len(prompts), FALLBACK_BATCH):
                batch = prompts[i:i + FALLBACK_BATCH]
                enc = tok(batch, return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    out = model.generate(**enc, **kwargs)
                p_tok += int(enc.attention_mask.sum())
                gen = out[:, enc.input_ids.shape[1]:]
                for j in range(gen.shape[0]):
                    pi, si = i + j // k, j % k
                    row = gen[j]
                    # Pad tokens after eos drop out of the count. Close enough for throughput.
                    n_t = int((row != tok.pad_token_id).sum())
                    o_tok += n_t
                    gens.append((pi, si, n_t, tok.decode(row, skip_special_tokens=True).strip()))
            return p_tok, o_tok, gens

        engine_version = transformers.__version__

    print(f"{ENGINE} {engine_version} ready on {DEVICE}")

    # ---- Warm up, then sweep with checkpointing ----
    run(prompt_texts[:min(8, len(prompt_texts))], TEMPERATURES[0])   # warmup, untimed

    # Why a fingerprint: same model, corpus, and sweep knobs mean the same run, so a
    # restart resumes it, and any knob change starts a fresh checkpoint on its own file.
    fingerprint = hashlib.sha256(json.dumps(
        [model_hash, HC3_FILE, TEMPERATURES, PROMPT_COUNTS, N_SAMPLES,
         SEED, MAX_NEW, MAX_MODEL_LEN, GPU_UTIL, ENGINE, DEVICE]).encode()).hexdigest()[:12]
    gens_path = ckpt_dir / f"ckpt__{fingerprint}.generations.jsonl"
    state_path = ckpt_dir / f"ckpt__{fingerprint}.state.json"

    if RESUME and state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"resuming checkpoint {fingerprint}: {len(state['chunks'])} chunks already done")
    else:
        state = {"chunks": {}}
        gens_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    # Records from finished chunks reload here. Keying by (temp, n, prompt, sample) with
    # last-write-wins drops duplicates left by a crash that landed generations but not state.
    done = {}
    if gens_path.exists():
        with open(gens_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:   # partial last line from an interrupted write
                    continue
                if r["chunk"] in state["chunks"]:
                    done[(r["temperature"], r["n_prompts"], r["prompt_idx"], r["sample_idx"])] = r

    def save_chunk(key, temp, n, start, gens, stats):
        # Why generations before state: a chunk only counts once its samples are on
        # disk, so a crash between the two writes costs one redone chunk, never data.
        with open(gens_path, "a") as f:
            for pi, si, n_t, text in gens:
                rec = {"chunk": key, "temperature": temp, "n_prompts": n,
                       "prompt_idx": start + pi, "sample_idx": si,
                       "out_tokens": n_t, "text": text}
                f.write(json.dumps(rec) + "\n")
                done[(temp, n, start + pi, si)] = rec
        state["chunks"][key] = stats
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(state_path)

    rows = []
    for temp in TEMPERATURES:
        for n in PROMPT_COUNTS:
            n = min(n, len(prompt_texts))
            chunk_stats = []
            for c, start in enumerate(range(0, n, CHECKPOINT_EVERY)):
                key = f"T{temp:g}|n{n}|c{c}"
                if key in state["chunks"]:
                    chunk_stats.append(state["chunks"][key])
                    continue
                t0 = time.time()
                p_tok, o_tok, gens = run(prompt_texts[start:min(start + CHECKPOINT_EVERY, n)], temp)
                wall = time.time() - t0
                stats = {"wall_s": wall, "prompt_tokens": p_tok,
                         "output_tokens": o_tok, "n_generations": len(gens)}
                save_chunk(key, temp, n, start, gens, stats)
                chunk_stats.append(stats)
                print(f"  chunk {key} | {wall:6.1f} s | {len(gens):,} generations saved")
            wall = sum(s["wall_s"] for s in chunk_stats)
            p_tok = sum(s["prompt_tokens"] for s in chunk_stats)
            o_tok = sum(s["output_tokens"] for s in chunk_stats)
            rows.append({
                "temperature": temp,
                "n_prompts": n,
                "n_samples": k_for(temp),
                "n_generations": sum(s["n_generations"] for s in chunk_stats),
                "wall_s": round(wall, 1),
                "prompt_tokens": p_tok,
                "output_tokens": o_tok,
                "prompt_tok_s": round(p_tok / wall),
                "output_tok_s": round(o_tok / wall),
                "prompts_per_s": round(n / wall, 2),
            })
            print(f"T={temp:.1f} | n={n:>6,} x{k_for(temp)} | {wall:7.1f} s "
                  f"| prompt {p_tok/wall:8,.0f} tok/s | output {o_tok/wall:7,.0f} tok/s")

    # Why run_id last: sequential ids over the deterministic (temperature, n, prompt,
    # sample) order come out identical whether the run was interrupted or not.
    ordered = sorted(done.values(), key=lambda r: (TEMPERATURES.index(r["temperature"]),
                                                   r["n_prompts"], r["prompt_idx"], r["sample_idx"]))
    gen_log = [{"run_id": i, "temperature": r["temperature"], "n_prompts": r["n_prompts"],
                "prompt_idx": r["prompt_idx"], "sample_idx": r["sample_idx"],
                "out_tokens": r["out_tokens"], "text": r["text"]} for i, r in enumerate(ordered)]

    bench = pd.DataFrame(rows)
    gen_df = pd.DataFrame(gen_log)
    print(f"generations logged: {len(gen_log):,}")

    # ---- Visualize the sweep, one set of figures per model ----
    # Why one line per temperature: scaling and sampling cost land on the same axes.
    fig, ax = plt.subplots()
    for ci, (temp, grp) in enumerate(bench.groupby("temperature")):
        c = colors[ci % len(colors)]
        ax.plot(grp.n_prompts, grp.output_tok_s, "o-", color=c, label=f"output, T={temp:g}")
        ax.plot(grp.n_prompts, grp.prompt_tok_s, "s--", color=c, alpha=0.5, label=f"prompt, T={temp:g}")
    ax.set_xscale("log")
    ax.set_xlabel("prompts in batch")
    ax.set_ylabel("aggregate tokens per second")
    ax.set_title(f"{model_path.stem} | {ENGINE} on {DEVICE}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'tokens_per_second_{model_path.stem}.png', dpi=300)
    plt.close()

    # Why the largest batch: steady state lives there, small batches measure startup overhead.
    big = bench[bench.n_prompts == bench.n_prompts.max()].sort_values("temperature")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(big.temperature, big.output_tok_s, "o-", color=colors[0])
    axes[0].set_xlabel("temperature")
    axes[0].set_ylabel("output tok/s")
    axes[0].set_title("Decode throughput vs temperature")

    # Why prompts/s here: repeats multiply decode work per prompt, and this axis shows the bill.
    axes[1].plot(big.temperature, big.prompts_per_s, "o-", color=colors[1])
    axes[1].set_xlabel("temperature")
    axes[1].set_ylabel("prompts per second")
    axes[1].set_title(f"Prompt completion rate (x{N_SAMPLES} samples above T=0)")
    plt.tight_layout()
    fig.savefig(f'throughput_vs_temperature_{model_path.stem}.png', dpi=300)
    plt.close()

    # Why length distributions: temperature moves output length, and length moves every tok/s number.
    temps = sorted(gen_df.temperature.unique())
    data = [gen_df.loc[gen_df.temperature == t, "out_tokens"] for t in temps]

    fig, ax = plt.subplots()
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(temps) + 1))
    ax.set_xticklabels([f"{t:g}" for t in temps])
    ax.set_xlabel("temperature")
    ax.set_ylabel("output tokens per sample")
    ax.set_title("Output length by temperature")
    plt.tight_layout()
    fig.savefig(f'length_by_temperature_{model_path.stem}.png', dpi=300)
    plt.close()

    print(gen_df.groupby("temperature").out_tokens.agg(["mean", "median", "std", "max"]).round(1))

    # Why repeat variation: this chart answers what temperature buys in output diversity.
    rep = gen_df[gen_df.temperature > 0]
    if N_SAMPLES > 1 and len(rep):
        g = rep.groupby(["temperature", "n_prompts", "prompt_idx"]).agg(
            uniq=("text", "nunique"), k=("text", "size"), tok_std=("out_tokens", "std"))
        g["uniq_ratio"] = g.uniq / g.k
        per_t = g.groupby(level="temperature").agg(
            uniq_ratio=("uniq_ratio", "mean"), tok_std=("tok_std", "mean"))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(per_t.index, per_t.uniq_ratio, "o-", color=colors[2])
        axes[0].set_ylim(0, 1.05)
        axes[0].set_xlabel("temperature")
        axes[0].set_ylabel(f"distinct samples per prompt / {N_SAMPLES}")
        axes[0].set_title("Sample diversity")

        axes[1].plot(per_t.index, per_t.tok_std, "o-", color=colors[3])
        axes[1].set_xlabel("temperature")
        axes[1].set_ylabel("std of output tokens per prompt")
        axes[1].set_title("Length spread across repeats")
        plt.tight_layout()
        fig.savefig(f'length_repeats_{model_path.stem}.png', dpi=300)
        plt.close()
    else:
        print("repeat variation needs N_SAMPLES above 1 and a temperature above 0")

    # ---- Write results, filenames keyed by the model name ----
    # Three artifacts per model: a summary JSON with the config and every sweep row, a generations
    # JSONL with one record per sample, and one leaderboard row per config. Once the artifacts
    # land, the checkpoint pair retires, so the next sweep starts clean instead of replaying this
    # run's timings.
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"throughput__{model_path.stem}__{model_hash}__{ts}"

    summary = {
        "run_id": ts,
        "model": {"file": model_path.name, "hash": model_hash,
                  "size_gb": round(size_gb, 2), "gguf": meta, "tokenizer": spec["tokenizer"]},
        "engine": {"name": ENGINE, "version": engine_version, "device": DEVICE,
                   "max_new_tokens": MAX_NEW, "max_model_len": MAX_MODEL_LEN,
                   "temperatures": TEMPERATURES, "n_samples": N_SAMPLES,
                   "seed": SEED, "host": platform.node()},
        "dataset": {"name": "Hello-SimpleAI/HC3", "file": HC3_FILE,
                    "unique_prompts_available": len(prompt_texts)},
        "sweep": rows,
    }
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(summary, f, indent=2)

    gen_df.to_json(out_dir / f"{stem}.generations.jsonl", orient="records", lines=True)

    lb = out_dir / "throughput_leaderboard_v2.csv"
    lb_rows = [{
        "run_id": ts,
        "model": model_path.name,
        "hash": model_hash,
        "engine": ENGINE,
        "engine_version": engine_version,
        "device": DEVICE,
        **r,
    } for r in rows]
    pd.DataFrame(lb_rows).to_csv(lb, mode="a", header=not lb.exists(), index=False)

    # The artifacts above now hold everything the checkpoint held, so it retires.
    gens_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    print("wrote", stem, "|", len(lb_rows), "leaderboard rows |", len(gen_log), "generations")

    # ---- Tear the engine down so the next model gets the memory back ----
    if ENGINE == "vllm":
        del llm
    else:
        del model
    del run, tok
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ## Compare runs
#
# Every model, engine, device, and temperature lands in one table. Sort by `output_tok_s` for
# decode speed, or filter on matching `n_prompts` and `temperature` to compare like with like.
# The chart below takes each setup's best decode number, so it compares peaks, not matched
# workloads.

board = pd.read_csv(out_dir / "throughput_leaderboard_v2.csv")
print(board.sort_values(["temperature", "n_prompts", "output_tok_s"],
                        ascending=[True, True, False]).to_string())

# Why best-row-per-setup: runs differ in counts and temps across machines, so this compares peaks only.
best = (board.sort_values("output_tok_s", ascending=False)
             .drop_duplicates(subset=["model", "engine", "device"])
             .sort_values("output_tok_s"))
labels = (best.model.str.replace(".gguf", "", regex=False)
          + " | " + best.engine + " | " + best.device)

fig, ax = plt.subplots(figsize=(9, 0.6 * len(best) + 1.5))
ax.barh(labels, best.output_tok_s, color=colors[0])
ax.set_xlabel("best observed output tok/s")
ax.set_title("Peak decode throughput per model, engine, device")
plt.tight_layout()
fig.savefig('peak_decode_leaderboard.png', dpi=300)
plt.close()

# The leaderboard accumulates across scans. The generations JSONLs feed the detector notebook
# directly: prompt index, sample index, temperature, and paragraph-length text per row, keyed by
# model hash, with HC3's human answers waiting on the same prompt indices.

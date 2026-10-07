#!/usr/bin/env python
# coding: utf-8

# <div style="width: 100%; overflow: hidden;">
#     <div style="width: 150px; float: left;"> <img src="data/D4Sci_logo_ball.png" alt="Data For Science, Inc" align="left" border="0"> </div>
#     <div style="float: left; margin-left: 10px;"> <h1>Benchmarks</h1>
# <h1>GGUF Generation: HC3 Prompts at Default Sampling</h1>
#         <p>Bruno Gonçalves<br/>
#         <a href="http://www.data4sci.com/">www.data4sci.com</a><br/>
#             @bgoncalves, @data4sci</p></div>
# </div>

import gc
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# vLLM exists only on the Linux install path. The try keeps the first-cell rule intact everywhere.
try:
    import vllm
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None

import watermark

print(watermark.watermark(datename=True, python=True, machine=True,
                          githash=True, iversions=True, globals_=globals()))


# # Start
#
# The fast sibling of `hc3_generation_benchmark.py`: one pass per model over the HC3 prompts, at
# each model's own default sampling settings, one sample per prompt. No temperature sweep, no
# repeats, no figures — the deliverable is a single JSONL with every generation tagged by model,
# feeding the detector notebook directly. `prompt_idx` follows the same dedupe order as the
# benchmark script, so the human answers pair up identically.

# ## Parameters
#
# Same model list as the benchmark. Each model's sampling comes from its own
# `generation_config.json` on the Hub (temperature, top_p, top_k where the vendor set them,
# vLLM defaults otherwise), so every model generates the way its vendor intended.
#
# `MAX_NEW = 1024` is headroom, not a target: the instruction asks for a 4-to-8-sentence
# paragraph, so nearly every answer ends at its EOS token long before the limit. The limit
# only keeps a rambling model from running a sequence to the `MAX_MODEL_LEN` ceiling.
#
# Memory stays small on purpose: the KV cache only needs to cover the sequences actually
# decoding at once, and `MAX_NUM_SEQS` caps that at 256. At `MAX_MODEL_LEN = 1536` that is
# roughly 22 GB of cache for the 7B GQA models, so `GPU_UTIL = 0.35` (~42 GB of the Spark's
# unified memory) leaves the rest of the machine alone. The one model that feels the cap is
# gemma-2-9b, whose fat KV layout fits ~68 concurrent sequences in that budget — still deep
# enough to keep decode near saturation.

# Why per-model engines: the vLLM GGUF plugin (0.0.5) loads only some architectures
# correctly on this stack. Verified 2026-08-17/18 with engine start plus a coherence check:
# only Qwen2.5 generates clean text under vLLM; gemma-2 and Qwen3 fail on their tied
# lm_head, Mistral and Ministral on an unknown model_type, and Falcon3 and Yi-1.5 load but
# generate prompt-ignoring gibberish (Yi's outputs repeat the same noise motifs across
# prompts — a token/embedding mapping mismatch, and it also floods the log with
# detokenizer "invalid prefix" resets). Those six run under transformers instead, which
# dequantizes the same GGUF — identical weights, well-tested modeling code — and batch 64
# on cuda decodes within ~15% of vLLM here, since batched decode is memory-bandwidth-bound
# on either engine.
MODELS = [
    {"repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
     "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
     "tokenizer": "Qwen/Qwen2.5-7B-Instruct"},
    {"repo": "bartowski/gemma-2-9b-it-GGUF",
     "file": "gemma-2-9b-it-Q4_K_M.gguf",
     # Why this works gated: the account holds Gemma access, and HF_TOKEN travels
     # with every shell launched from the terminal.
     "tokenizer": "google/gemma-2-9b-it",
     "engine": "transformers"},
    {"repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
     "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
     "tokenizer": "mistralai/Mistral-7B-Instruct-v0.3",
     "engine": "transformers"},
    {"repo": "Qwen/Qwen3-8B-GGUF",
     "file": "Qwen3-8B-Q4_K_M.gguf",
     "tokenizer": "Qwen/Qwen3-8B",
     # Why thinking off: the template defaults to reasoning mode, and the detector
     # wants plain paragraph answers, not <think> traces.
     "template_kwargs": {"enable_thinking": False},
     "engine": "transformers"},
    {"repo": "bartowski/Ministral-8B-Instruct-2410-GGUF",
     "file": "Ministral-8B-Instruct-2410-Q4_K_M.gguf",
     "tokenizer": "mistralai/Ministral-8B-Instruct-2410",
     "engine": "transformers"},
    {"repo": "bartowski/Yi-1.5-9B-Chat-GGUF",
     "file": "Yi-1.5-9B-Chat-Q4_K_M.gguf",
     "tokenizer": "01-ai/Yi-1.5-9B-Chat",
     "engine": "transformers"},
    {"repo": "bartowski/Falcon3-7B-Instruct-GGUF",
     "file": "Falcon3-7B-Instruct-Q4_K_M.gguf",
     "tokenizer": "tiiuae/Falcon3-7B-Instruct",
     "engine": "transformers"},
]

HC3_FILE       = "all.jsonl"
N_PROMPTS      = 10000
SEED           = 0                 # fixed seed keeps sampled runs reproducible
MAX_NEW        = 1024              # headroom, not a target — EOS ends answers well before this
MAX_MODEL_LEN  = 1536              # HC3 worst-case prompt is 404 tokens, + 1024 answer = 1428
GPU_UTIL       = 0.35              # small on purpose, see the note above
MAX_NUM_SEQS   = 256               # concurrency cap that sizes the KV cache actually used
FALLBACK_BATCH = None              # transformers batch size, None = 64 on cuda, 8 on mps and cpu
CHECKPOINT_EVERY = 1000            # prompts per chunk, bounds what an interruption can lose
RESUME         = True              # True picks an interrupted run back up, False starts over
RESULTS_DIR    = "results"
OUT_FILE       = "hc3_default_generations.jsonl"   # one combined file, model name on every row


# ## Pick the device and the engine

if torch.cuda.is_available() and vllm is not None:
    DEVICE, ENGINE = "cuda", "vllm"
elif torch.cuda.is_available():
    DEVICE, ENGINE = "cuda", "transformers"
elif torch.backends.mps.is_available():
    DEVICE, ENGINE = "mps", "transformers"
else:
    DEVICE, ENGINE = "cpu", "transformers"

if FALLBACK_BATCH is None:
    FALLBACK_BATCH = {"cuda": 64, "mps": 8, "cpu": 8}[DEVICE]

print(f"device: {DEVICE} | engine: {ENGINE} | prompts: {N_PROMPTS:,} | models: {len(MODELS)}")


# ## Load HC3
#
# Identical selection and dedupe to the benchmark script, so `prompt_idx` means the same thing
# in both output files and the detector pairs human answers without translation.

hc3_path = hf_hub_download(repo_id="Hello-SimpleAI/HC3",
                           filename=HC3_FILE, repo_type="dataset")
hd = pd.read_json(hc3_path, lines=True)
hd["question"] = hd.question.fillna("").str.strip()
hd = hd[(hd.question.str.len() > 0) & (hd.human_answers.str.len() > 0)]

pairs = hd.drop_duplicates(subset=["question"]).reset_index(drop=True)
pairs = pairs.head(min(N_PROMPTS, len(pairs)))
print(f"rows: {len(hd):,} | unique questions used: {len(pairs):,} | "
      f"domains: {dict(pairs.source.value_counts())}")

INSTRUCTION = ("Answer the question in one clear paragraph of 4 to 8 sentences. "
               "Write for a general reader.")


# ## Model helpers

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

def default_sampling(repo):
    # Why to_diff_dict: it holds only what the vendor wrote into generation_config.json,
    # so library fallbacks (like transformers' own top_k=50) never masquerade as a
    # model default. Anything unset stays at the neutral value.
    out = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
    try:
        cfg = GenerationConfig.from_pretrained(repo).to_diff_dict()
    except Exception as e:
        print(f"no generation config for {repo} ({e}), using temperature 1.0")
        return out
    for key in ("temperature", "top_p", "top_k"):
        if cfg.get(key) is not None:
            out[key] = cfg[key]
    return out


# ## Generate, one pass per model
#
# Every finished chunk appends its rows to a per-model checkpoint and then records itself in a
# state file, in that order, so a crash between the two writes costs one redone chunk, never
# data. When a model finishes, its deduped rows append to the combined output and the model's
# hash lands in the done-list, so a restart skips straight past it. The fingerprint leaves the
# memory knobs out on purpose: retuning GPU_UTIL or MAX_NUM_SEQS mid-scan must not throw away
# finished chunks, since neither changes what gets generated.

out_dir = Path(RESULTS_DIR)
out_dir.mkdir(exist_ok=True)
ckpt_dir = out_dir / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / OUT_FILE
done_path = out_dir / (OUT_FILE + ".done.json")
meta_path = out_dir / (OUT_FILE.replace(".jsonl", "") + ".meta.json")
done_models = set(json.loads(done_path.read_text())) if RESUME and done_path.exists() else set()
if not RESUME:
    out_path.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)

# Why a sidecar: the exact prompt wording shapes every generation, so it belongs in the
# artifacts — but repeating it on every row would bloat the JSONL with a constant.
meta_path.write_text(json.dumps({
    "instruction": INSTRUCTION,
    "prompt_template": "{INSTRUCTION}\\n\\nQuestion: {question[:2000]} -> chat template, add_generation_prompt=True",
    "dataset": {"name": "Hello-SimpleAI/HC3", "file": HC3_FILE,
                "n_prompts": len(pairs), "dedupe": "drop_duplicates(question), head"},
    "engine": ENGINE, "device": DEVICE, "seed": SEED,
    "max_new_tokens": MAX_NEW, "max_model_len": MAX_MODEL_LEN,
    "sampling": "per-model generation_config.json defaults, recorded on each row",
}, indent=2))

scan_t0 = time.time()
for spec in MODELS:
    model_path = resolve_model(spec)
    model_hash = fast_hash(model_path)
    model_name = model_path.stem
    if model_hash in done_models:
        print(f"skipping {model_name}: already in {out_path.name}")
        continue

    sampling = default_sampling(spec["tokenizer"])
    # A spec's engine override only means something where vLLM is an option at all.
    eng = "transformers" if ENGINE == "transformers" else spec.get("engine", ENGINE)
    print(f"\n{'=' * 70}\n{model_name} | hash {model_hash} | engine {eng} "
          f"| default sampling: {sampling}")

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
    assert max(lens) + MAX_NEW <= MAX_MODEL_LEN, "raise MAX_MODEL_LEN"

    # ---- Start the engine ----
    if eng == "vllm":
        llm = LLM(
            model=str(model_path),
            tokenizer=spec["tokenizer"],
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_UTIL,
            max_num_seqs=MAX_NUM_SEQS,
            # Why prefix caching on: this is generation, not a benchmark, and every
            # prompt shares the instruction prefix — a free head start on prefill.
            enable_prefix_caching=True,
        )

        def run(prompts):
            sp = SamplingParams(max_tokens=MAX_NEW, n=1, seed=SEED, **sampling)
            outs = llm.generate(prompts, sp)
            gens = []
            for pi, o in enumerate(outs):
                seq = o.outputs[0]
                gens.append((pi, len(seq.token_ids), seq.text.strip()))
            return gens

        engine_version = vllm.__version__

    else:
        # Why bfloat16 on cuda: gemma-2's logit soft-capping overflows float16.
        # mps runs float16 well, cpu wants float32.
        dt = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[DEVICE]
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path.parent), gguf_file=model_path.name, torch_dtype=dt
        ).to(DEVICE).eval()

        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        def run(prompts):
            torch.manual_seed(SEED)
            kwargs = dict(max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
            if sampling["temperature"] > 0:
                kwargs.update(do_sample=True, temperature=sampling["temperature"],
                              top_p=sampling["top_p"],
                              top_k=sampling["top_k"] if sampling["top_k"] > 0 else 0)
            else:
                kwargs.update(do_sample=False)

            gens = []
            # Why leave=False: one transient bar per chunk, so the chunk log stays clean.
            for i in tqdm(range(0, len(prompts), FALLBACK_BATCH),
                          desc=f"batches of {FALLBACK_BATCH}", leave=False):
                batch = prompts[i:i + FALLBACK_BATCH]
                enc = tok(batch, return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    out = model.generate(**enc, **kwargs)
                gen = out[:, enc.input_ids.shape[1]:]
                for j in range(gen.shape[0]):
                    row = gen[j]
                    n_t = int((row != tok.pad_token_id).sum())
                    gens.append((i + j, n_t, tok.decode(row, skip_special_tokens=True).strip()))
            return gens

        engine_version = transformers.__version__

    print(f"{eng} {engine_version} ready on {DEVICE}")
    run(prompt_texts[:min(8, len(prompt_texts))])   # warmup, untimed

    # Why INSTRUCTION is in here: a wording change changes every generation, so it must
    # start fresh checkpoints, exactly like a knob that alters what gets sampled.
    fingerprint = hashlib.sha256(json.dumps(
        [model_hash, HC3_FILE, N_PROMPTS, sampling, SEED, MAX_NEW,
         MAX_MODEL_LEN, eng, DEVICE, INSTRUCTION]).encode()).hexdigest()[:12]
    gens_path = ckpt_dir / f"ckpt_default__{fingerprint}.generations.jsonl"
    state_path = ckpt_dir / f"ckpt_default__{fingerprint}.state.json"

    if RESUME and state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"resuming checkpoint {fingerprint}: {len(state['chunks'])} chunks already done")
    else:
        state = {"chunks": {}}
        gens_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    # Records from finished chunks reload here, last-write-wins on prompt_idx to drop
    # duplicates left by a crash that landed generations but not state.
    done = {}
    if gens_path.exists():
        with open(gens_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:   # partial last line from an interrupted write
                    continue
                if r["chunk"] in state["chunks"]:
                    done[r["prompt_idx"]] = r

    model_t0 = time.time()
    for c, start in enumerate(range(0, len(prompt_texts), CHECKPOINT_EVERY)):
        key = f"c{c}"
        if key in state["chunks"]:
            continue
        t0 = time.time()
        gens = run(prompt_texts[start:start + CHECKPOINT_EVERY])
        wall = time.time() - t0
        with open(gens_path, "a") as f:
            for pi, n_t, text in gens:
                rec = {"chunk": key,
                       "model": model_name, "model_hash": model_hash, "engine": eng,
                       **sampling,
                       "prompt_idx": start + pi, "out_tokens": n_t, "text": text}
                f.write(json.dumps(rec) + "\n")
                done[start + pi] = rec
        state["chunks"][key] = {"wall_s": wall, "n_generations": len(gens)}
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(state_path)
        n_done = min(start + CHECKPOINT_EVERY, len(prompt_texts))
        print(f"  chunk {key} | {wall:6.1f} s | {n_done:,}/{len(prompt_texts):,} prompts")

    # ---- Append this model's rows to the combined output, then retire the checkpoint ----
    with open(out_path, "a") as f:
        for pi in sorted(done):
            r = done[pi]
            r.pop("chunk", None)
            f.write(json.dumps(r) + "\n")
    done_models.add(model_hash)
    tmp = done_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(done_models)))
    tmp.replace(done_path)
    gens_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)

    wall = time.time() - model_t0
    # Why a runtimes sidecar: chunk timings retire with the checkpoint, so this is the
    # only durable record of what each model cost. wall_s covers this process's share
    # only — chunks inherited from an interrupted run were paid for by that run.
    with open(out_dir / (OUT_FILE.replace(".jsonl", "") + ".runtimes.jsonl"), "a") as f:
        f.write(json.dumps({
            "model": model_name, "model_hash": model_hash, "engine": eng,
            "wall_s": round(wall, 1), "n_generations": len(done),
            "output_tokens": sum(r["out_tokens"] for r in done.values()),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(f"{model_name} done: {len(done):,} generations in {wall/60:.1f} min "
          f"-> {out_path.name}")

    # ---- Tear the engine down so the next model gets the memory back ----
    if eng == "vllm":
        del llm
    else:
        del model
    del run, tok
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print(f"\nscan complete: {len(done_models)}/{len(MODELS)} models "
      f"in {(time.time() - scan_t0)/3600:.1f} h | output: {out_path}")

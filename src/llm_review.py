"""
llm_review.py - GPT-6.1 (Azure Foundry) as a second pair of eyes; NEVER makes gate decisions
=============================================================================================
  python llm_review.py review --stage 1     # anomaly review of gate/fit results -> report/llm_review_stage1.md
  python llm_review.py triage --run NAME    # cheap log triage of one run (fast GPT-5 deployment if configured)
  python llm_review.py draft                # Results-section draft from results.json; every number is
                                            # verified against results.json and unverified ones are flagged
Secret: AZURE_OPENAI_KEY (env var or ~/.wdladder_secrets). Endpoint/deployments in config.FOUNDRY.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import config as C

SYSTEM = ("You are a meticulous ML-research reviewer for a scaling-law paper. You check experiment results "
          "for anomalies and threats to validity. Use ONLY numbers present in the provided JSON. Never invent "
          "results or citations. Be concise and concrete.")


def _client():
    from openai import AzureOpenAI
    return AzureOpenAI(azure_endpoint=C.FOUNDRY["endpoint"], api_key=C.secret("AZURE_OPENAI_KEY"),
                       api_version=C.FOUNDRY["api_version"])


def chat(prompt, fast=False, max_tokens=3000):
    dep = (C.FOUNDRY["deployment_fast"] if fast and C.FOUNDRY["deployment_fast"] else C.FOUNDRY["deployment_main"])
    cl = _client()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    try:
        r = cl.chat.completions.create(model=dep, messages=msgs, max_completion_tokens=max_tokens)
    except Exception as e:                       # older deployments only accept max_tokens
        if "max_completion_tokens" not in str(e) and "unsupported" not in str(e).lower():
            raise
        r = cl.chat.completions.create(model=dep, messages=msgs, max_tokens=max_tokens)
    return r.choices[0].message.content


def _results():
    return json.load(open(os.path.join(C.P["report"], "results.json")))


def review(stage):
    R = _results()
    prompt = f"""Stage {stage} results of a study: Muon with scaled weight decay (decay multiplied by eta_t/eta_max,
'Muon-SW') vs constant decoupled decay on an MAI-Base-2-style MoE scaling ladder, hyper-parameters expressed as
multipliers of MAI's laws  ln LR* = -1.51 + 0.175 ln TPP - 0.573 ln steps,  LR*WD* = 17.04/steps.
Pre-registered decision rules (already applied by code, do NOT override): {json.dumps(C.PREREG)}
Results JSON:
{json.dumps(R, default=str)[:60000]}

Tasks:
1. List anomalies (seed noise vs effect size, non-convex or edge optima, bootstrap failures, suspicious
   speed-ups, diverged runs, inconsistent TPP30 vs TPP120 signs).
2. List threats to validity for THIS stage and the cheapest extra run (if any) that would resolve each.
3. State whether the code's decision looks consistent with the numbers (yes/no + one sentence).
Max 350 words."""
    out = chat(prompt)
    fn = os.path.join(C.P["report"], f"llm_review_stage{stage}.md")
    open(fn, "w").write(out)
    print(out, "\n->", fn)


def triage(run):
    d = os.path.join(C.P["runs"], run)
    tail = open(os.path.join(d, "log.jsonl")).read().splitlines()[-120:]
    ev = open(os.path.join(d, "evals.jsonl")).read() if os.path.exists(os.path.join(d, "evals.jsonl")) else ""
    out = chat(f"Training log tail (jsonl) and evals of run {run}. Flag loss spikes, grad-norm blow-ups, fp16 "
               f"scale collapse, expert-load imbalance (max_load >> 1), stalls. 150 words max.\nLOG:\n"
               + "\n".join(tail) + "\nEVALS:\n" + ev[-8000:], fast=True, max_tokens=800)
    print(out)


def _numbers(text):
    return [float(x) for x in re.findall(r"(?<![\w.])-?\d+\.\d+(?![\w.])", text)]


def draft():
    R = _results()
    out = chat(f"""Write the Results section (LaTeX, ~600 words) of a paper titled around 'Scaled weight decay on an
MAI-style MoE scaling ladder: how much of the Muon-SW speed-up survives tuning?'. Use ONLY these numbers, round
to at most 4 significant digits, and say 'not measured' for missing items. Report: gate-1 untuned gain and
compute-equivalent speed-up; fitted optima in MAI multipliers; lambda ratio vs the three hypotheses (H_none=1,
H_int=1/r, H_end=eta_max/eta_T); tuned gap with CI; survival fraction; WSD and scale (L4, L6, L9) results; outcome.
JSON: {json.dumps(R, default=str)[:60000]}""", max_tokens=4000)
    flat = json.dumps(R, default=str)
    src = _numbers(flat)
    bad = [x for x in set(_numbers(out)) if not any(abs(x - y) <= max(5e-4, 5e-3 * abs(y)) for y in src)
           and x not in (17.04, 1.51, 0.175, 0.573, 0.526, 27.2)]
    fn = os.path.join(C.P["report"], "results_draft.tex")
    open(fn, "w").write(out + ("\n% UNVERIFIED NUMBERS (check by hand): " + ", ".join(map(str, sorted(bad))) if bad else ""))
    print(out[:3000], "\n...\n->", fn, "\nunverified numbers:", bad or "none")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["review", "triage", "draft"])
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--run", default="")
    a = ap.parse_args()
    {"review": lambda: review(a.stage), "triage": lambda: triage(a.run), "draft": draft}[a.cmd]()

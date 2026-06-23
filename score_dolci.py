#!/usr/bin/env python3
"""
score_dolci.py
Scores the prompts in Allenai-Dolci dataset by running them locally via Docker.
"""
import argparse
import ast
import json
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

# Per-completion wall-clock budget (seconds), for either harness.
COMPLETION_TIMEOUT = 20

# Escaped backticks so it doesn't break Markdown code blocks in the UI
_CODE_BLOCK = re.compile(r"\`\`\`(?:python)?\s*(.*?)\`\`\`", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


# ----------------------------- code extraction -----------------------------
def extract_code(completion: str) -> str:
    if not completion:
        return ""
    answer = _THINK_BLOCK.sub("", completion)
    blocks = _CODE_BLOCK.findall(answer)
    if blocks:
        return blocks[-1].strip()
    return ""


def syntax_ok(code: str) -> bool:
    if not code or not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ----------------------------- format detection ----------------------------
def detect_format(tests) -> str:
    if not tests:
        return "empty"
    first = tests[0]
    if isinstance(first, dict) and "input" in first and "output" in first:
        return "stdio"
    if isinstance(first, str):
        return "assert"
    return "unknown"


# ----------------------------- ASSERT harness ------------------------------
def _run_asserts_child(code: str, tests: list, q: mp.Queue):
    n_pass = 0
    n_total = len(tests)
    
    try:
        compiled_code = compile(code, "<string>", "exec")
    except Exception as e:  # noqa: BLE001
        q.put((0, n_total, f"compile_error:{type(e).__name__}"))
        return
        
    for t in tests:
        try:
            ns = {}
            exec(compiled_code, ns)  # noqa: S102
            exec(t, ns)  # noqa: S102
            n_pass += 1
        except AssertionError:
            pass
        except Exception:  # noqa: BLE001
            pass
            
    q.put((n_pass, n_total, None))


def score_asserts(code: str, tests: list, timeout: int) -> dict:
    result = {"n_pass": 0, "n_total": len(tests), "pass_fraction": 0.0,
              "error": None, "format": "assert"}
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_run_asserts_child, args=(code, tests, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(1)
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL); p.join(1)
        p.close()
        result["error"] = "timeout"
        return result
    try:
        n_pass, n_total, err = q.get(timeout=5)
    except Exception:  # noqa: BLE001
        p.close(); result["error"] = "no_result"; return result
    p.close()
    if err:
        result["error"] = err; return result
    result["n_pass"] = n_pass
    result["n_total"] = n_total
    result["pass_fraction"] = (n_pass / n_total) if n_total else 0.0
    return result


# ----------------------------- STDIO harness -------------------------------
def normalize_lines(s: str) -> list:
    if s is None:
        return []
    lines = s.replace("\r\n", "\n").split("\n")
    lines = [ln.rstrip() for ln in lines]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def outputs_match(got: str, expected: str, rel_tol: float = 1e-6,
                  abs_tol: float = 1e-6) -> bool:
    g_lines = normalize_lines(got)
    e_lines = normalize_lines(expected)
    
    if len(g_lines) != len(e_lines):
        return False

    for gl, el in zip(g_lines, e_lines):
        if gl == el:
            continue
            
        g_tok = gl.split()
        e_tok = el.split()
        if len(g_tok) != len(e_tok):
            return False

        for gt, et in zip(g_tok, e_tok):
            if gt == et:
                continue
            try:
                gf = float(gt)
                ef = float(et)
            except ValueError:
                return False
            diff = abs(gf - ef)
            if diff <= abs_tol or diff <= rel_tol * max(abs(gf), abs(ef)):
                continue
            return False
    return True


def score_stdio(code: str, tests: list, timeout: int, max_cases: int = 0) -> dict:
    result = {"n_pass": 0, "n_total": len(tests), "pass_fraction": 0.0,
              "error": None, "format": "stdio", "cases_run": 0}
    if not tests:
        result["error"] = "no_tests"; return result

    if max_cases and max_cases > 0:
        run_tests = tests[:max_cases]
    else:
        run_tests = tests
    result["n_total"] = len(run_tests)

    per_case = max(2, timeout // max(1, len(run_tests)))

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(code)
        script_path = tf.name

    n_pass = 0
    cases_run = 0
    try:
        for tc in run_tests:
            cases_run += 1
            stdin_data = tc.get("input", "")
            expected = tc.get("output", "")
            try:
                proc = subprocess.run(
                    [sys.executable, script_path],
                    input=stdin_data,
                    capture_output=True,
                    text=True,
                    timeout=per_case,
                )
            except subprocess.TimeoutExpired:
                continue 
            except Exception:  # noqa: BLE001
                continue
            if proc.returncode != 0:
                continue  
            if outputs_match(proc.stdout, expected):
                n_pass += 1
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    result["n_pass"] = n_pass
    result["cases_run"] = cases_run
    result["pass_fraction"] = (n_pass / result["n_total"]) if result["n_total"] else 0.0
    return result


# ----------------------------- dispatch ------------------------------------
def score_completion(code: str, tests: list, fmt: str, timeout: int,
                     max_cases: int = 0) -> dict:
    if not code or not code.strip():
        return {"n_pass": 0, "n_total": len(tests), "pass_fraction": 0.0,
                "error": "empty_code", "format": fmt}
    if fmt == "assert":
        return score_asserts(code, tests, timeout)
    if fmt == "stdio":
        return score_stdio(code, tests, timeout, max_cases)
    return {"n_pass": 0, "n_total": len(tests), "pass_fraction": 0.0,
            "error": f"unsupported_format:{fmt}", "format": fmt}


# ----------------------------- sanity check --------------------------------
def check_sanity(rec, scored_completions):
    passrate = rec.get("passrate")
    if passrate is None:
        return None
    scored = [c for c in scored_completions
              if isinstance(c.get("score"), dict)
              and c["score"].get("error") is None]
    if not scored:
        return None  
    fracs = [c["score"]["pass_fraction"] for c in scored]
    worst, best = min(fracs), max(fracs)

    violation = None
    if passrate == 1.0 and worst < 1.0:
        violation = {"id": rec.get("id"), "passrate": passrate,
                     "expected": "all==1.0", "worst": worst, "best": best,
                     "format": scored[0]["score"].get("format")}
    elif passrate == 0.0 and best == 1.0:
        violation = {"id": rec.get("id"), "passrate": passrate,
                     "expected": "all==0.0", "worst": worst, "best": best,
                     "format": scored[0]["score"].get("format")}
    return violation


def load_done_ids(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="out/dolci_clean.jsonl")
    ap.add_argument("--out", default="out/dolci_scored.jsonl")
    ap.add_argument("--violations", default="out/dolci_sanity_violations.jsonl")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    clean_path = Path(args.clean)
    out_path = Path(args.out)
    viol_path = Path(args.violations)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(clean_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    done = load_done_ids(out_path)
    todo = [r for r in records if r.get("id") not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Total: {len(records)} | done: {len(done)} | to do: {len(todo)}")

    n_violations = 0
    fmt_counts = {"assert": 0, "stdio": 0, "empty": 0, "unknown": 0}

    with open(out_path, "a") as out_f, open(viol_path, "a") as viol_f:
        for i, rec in enumerate(todo):
            tests = rec.get("tests", [])
            fmt = detect_format(tests)
            fmt_counts[fmt] = fmt_counts.get(fmt, 0) + 1

            scored_completions = []
            for idx, comp in enumerate(rec.get("outputs", [])):
                code = extract_code(comp)
                ok = syntax_ok(code)
                entry = {
                    "output_index": idx,
                    "code": code,
                    "syntax_ok": ok,
                    "truncated": (code == ""),
                }
                if not ok:
                    entry["score"] = None
                else:
                    entry["score"] = score_completion(code, tests, fmt,
                                                      args.timeout, args.max_cases)
                scored_completions.append(entry)

            rec["test_format"] = fmt
            rec["scored_completions"] = scored_completions

            violation = check_sanity(rec, scored_completions)
            if violation:
                n_violations += 1
                viol_f.write(json.dumps(violation) + "\n")
                viol_f.flush()

            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            scored = [c for c in scored_completions
                      if isinstance(c.get("score"), dict)
                      and c["score"].get("error") is None]
            fracs = sorted(c["score"]["pass_fraction"] for c in scored)
            best = fracs[-1] if fracs else 0.0
            worst = fracs[0] if fracs else 0.0
            flag = "  <-- SANITY VIOLATION" if violation else ""
            print(f"[{i+1}/{len(todo)}] {str(rec['id'])[:28]:28s} fmt={fmt:6s} "
                  f"scored={len(scored)}/{len(scored_completions)} "
                  f"best={best:.2f} worst={worst:.2f} "
                  f"dolci={rec.get('passrate')}{flag}")

    print("\nScoring done.")
    print(f"Format counts: {fmt_counts}")
    print(f"Sanity violations: {n_violations}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize JSONL benchmark results (benchmarks/results/<dir>/*.jsonl) as markdown.

Every row of the output is computed from the raw `run` events; nothing is typed in.
usage: summarize_candle_bench.py RESULTS_DIR > table.md
"""
import json, statistics, sys, collections, pathlib

def load(path):
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out

def main(root):
    root = pathlib.Path(root)
    rows = []
    for f in sorted(root.glob("*.jsonl")):
        if "parity" in f.name:
            continue
        ev = load(f)
        loaded = next((e for e in ev if e.get("event") == "loaded"), {})
        done = next((e for e in ev if e.get("event") == "done"), {})
        runs = [e for e in ev if e.get("event") == "run"]
        if not runs:
            continue
        backend = runs[0].get("backend", f.stem)
        cases = collections.OrderedDict()
        for r in runs:
            cases.setdefault(r["case"], []).append(r)
        for case, rs in cases.items():
            rtf = sorted(r["rtf"] for r in rs)
            stage = collections.defaultdict(list)
            for r in rs:
                for k, v in (r.get("stage_secs") or {}).items():
                    if isinstance(v, (int, float)):
                        stage[k].append(v)
            ttfa = [r["ttfa_secs"] for r in rs if r.get("ttfa_secs") is not None]
            rows.append(dict(
                file=f.name, backend=backend, case=case, n=len(rs),
                audio=statistics.median(r["audio_secs"] for r in rs),
                rtf_min=rtf[0], rtf_med=statistics.median(rtf), rtf_max=rtf[-1],
                ttfa=statistics.median(ttfa) if ttfa else None,
                load=loaded.get("load_secs"), rss=done.get("peak_rss_kib"),
                finite=all(r.get("finite", True) for r in rs),
                peak=statistics.median(r.get("peak", 0) for r in rs),
                stage={k: statistics.median(v) for k, v in stage.items()},
            ))
    print("| Result file | Backend | Case | Audio (s) | n | RTF min | RTF median | RTF max | TTFA (s) | Load (s) | Peak RSS (GiB) | Finite | Peak amp |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for r in rows:
        f = lambda x, p=2: "—" if x is None else f"{x:.{p}f}"
        rss = "—" if not r["rss"] else f"{r['rss']/1048576:.2f}"
        print(f"| {r['file']} | {r['backend']} | {r['case']} | {f(r['audio'],1)} | {r['n']} | {f(r['rtf_min'])} | {f(r['rtf_med'])} | {f(r['rtf_max'])} | {f(r['ttfa'])} | {f(r['load'],1)} | {rss} | {r['finite']} | {f(r['peak'])} |")
    print()
    print("Per-stage wall time (median seconds per run):")
    print()
    for r in rows:
        if r["stage"]:
            st = ", ".join(f"{k}={v:.3f}" for k, v in r["stage"].items())
            print(f"- {r['backend']} / {r['case']}: {st}")

if __name__ == "__main__":
    main(sys.argv[1])

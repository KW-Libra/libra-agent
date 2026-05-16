"""L4 행동 채점 하네스.

각 시나리오를 agent API(:8010 POST /v1/judge-runs)로 돌리거나(cached_result 있으면 재사용)
3 리트머스 + 구조 기준으로 자동 채점하고 통과율 리포트를 만든다.

기준(평가 철학과 일치):
  http_ok          안전 종료(HTTP 200 + decision 존재, 크래시 없음)
  schema           agent_responses 비어있지 않고 각자 agent_id/verdict, decision 에 decision/summary
  core_signal      코어(news·disclosure) 평균 direction 방향이 기대(negative/positive/flat)와 일치
  no_hallucination flat 기대 시 코어가 confident-directional 아님(QUIET/UNAVAILABLE/PARTIAL 또는 conf<=0.35)
  decision_in      최종 decision 이 허용 집합 안
  policy_gate      정책 위반 시나리오에서 USER_DECISION_REQUIRED 발화

사용:
  python scenario_harness/harness.py                 # cached 우선, 없으면 live POST
  python scenario_harness/harness.py --only positive --run --depth medium
  python scenario_harness/harness.py --no-run        # cached 만 채점(비용 0)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
API = "http://127.0.0.1:8010/v1/judge-runs"
HEALTH = "http://127.0.0.1:8010/health"
CORE = ("news", "disclosure")
HONEST_VERDICTS = {"QUIET", "DIRECT_ANSWER_UNAVAILABLE", "PARTIAL_ANSWER"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _post(payload: dict, timeout: float) -> tuple[int, dict]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(API, data=data,
                                 headers={"Content-Type": "application/json; charset=utf-8"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"detail": f"HTTP {e.code}"}
        return e.code, body


def _portfolio(sc: dict) -> dict:
    if "portfolio" in sc:
        return sc["portfolio"]
    pf = json.loads(Path(sc["portfolio_file"]).read_text(encoding="utf-8"))
    return pf["portfolio"] if "portfolio" in pf else pf


def _core_dirs(result: dict) -> list[tuple[str, float, str, float]]:
    out = []
    for a in result.get("agent_responses", []) or []:
        if a.get("agent_id") in CORE:
            out.append((a["agent_id"], float(a.get("direction", 0.0) or 0.0),
                        str(a.get("verdict", "")), float(a.get("confidence", 0.0) or 0.0)))
    return out


def score(sc: dict, result: dict, http: int | None) -> list[tuple[str, bool, str]]:
    exp = sc["expect"]
    dec = result.get("decision", {}) if isinstance(result, dict) else {}
    checks: list[tuple[str, bool, str]] = []

    if exp.get("http_ok"):
        ok = (http in (None, 200)) and isinstance(dec, dict) and bool(dec.get("decision"))
        checks.append(("http_ok", ok,
                       result.get("detail", "") if not ok else f"http={http} decision={dec.get('decision')}"))

    if exp.get("schema"):
        ars = result.get("agent_responses", []) or []
        ok = bool(ars) and all(a.get("agent_id") and a.get("verdict") for a in ars) \
            and bool(dec.get("decision")) and bool(dec.get("summary"))
        checks.append(("schema", ok, f"{len(ars)} agent_responses"))

    if "core_signal" in exp:
        cd = _core_dirs(result)
        want = exp["core_signal"]
        if not cd:
            checks.append((f"core_signal:{want}", False, "no core agent responses"))
        else:
            mean = sum(d for _, d, _, _ in cd) / len(cd)
            ok = (want == "negative" and mean <= -0.2) or \
                 (want == "positive" and mean >= 0.2) or \
                 (want == "flat" and abs(mean) < 0.25)
            detail = " ".join(f"{n}={d:+.2f}({v})" for n, d, v, _ in cd) + f" mean={mean:+.2f}"
            checks.append((f"core_signal:{want}", ok, detail))

    if exp.get("no_hallucination"):
        cd = _core_dirs(result)
        bad = [f"{n}:{v}/conf{c:.2f}/dir{d:+.2f}" for n, d, v, c in cd
               if v not in HONEST_VERDICTS and c > 0.35 and abs(d) >= 0.3]
        checks.append(("no_hallucination", not bad,
                       "honest" if not bad else "confident w/o signal: " + ", ".join(bad)))

    if "decision_in" in exp:
        d = dec.get("decision")
        checks.append((f"decision_in", d in exp["decision_in"], f"decision={d}"))

    if exp.get("policy_gate"):
        d = dec.get("decision")
        checks.append(("policy_gate", d == "USER_DECISION_REQUIRED", f"decision={d}"))

    return checks


def run_scenario(sc: dict, args) -> dict:
    name = sc["name"]
    result: dict = {}
    http: int | None = None
    cached = sc.get("cached_result")
    use_cache = cached and Path(cached).exists() and not args.run

    if use_cache:
        result = json.loads(Path(cached).read_text(encoding="utf-8"))
        src = f"cached:{Path(cached).name}"
    else:
        payload = {
            "query": "이번 뉴스·공시 반영해서 포트폴리오 점검하고 필요하면 리밸런싱 제안해줘",
            "portfolio": _portfolio(sc),
            "knowledge_sources": sc["knowledge"],
            "depth": args.depth, "trigger": "pull", "enable_human_interrupts": False,
        }
        t0 = time.time()
        http, result = _post(payload, timeout=args.timeout)
        src = f"live:{http}:{time.time()-t0:.0f}s"
        outp = HERE / "results" / f"{name}.json"
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    checks = score(sc, result, http)
    passed = all(ok for _, ok, _ in checks)
    return {"name": name, "src": src, "passed": passed, "checks": checks, "desc": sc["desc"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="LIBRA L4 behavioral scoring harness")
    ap.add_argument("--only", help="run only this scenario name")
    ap.add_argument("--run", action="store_true", help="force live POST even if cached_result exists")
    ap.add_argument("--no-run", action="store_true", help="score cached only; skip scenarios without cache")
    ap.add_argument("--depth", default="medium", choices=["medium", "deep"])
    ap.add_argument("--timeout", type=float, default=560.0)
    args = ap.parse_args()

    spec = json.loads((HERE / "scenarios.json").read_text(encoding="utf-8"))
    scenarios = spec["scenarios"]
    if args.only:
        scenarios = [s for s in scenarios if s["name"] == args.only]

    if not args.no_run:
        try:
            with urllib.request.urlopen(HEALTH, timeout=5) as r:
                if r.status != 200:
                    raise RuntimeError
        except Exception:
            print(f"[ERROR] agent API not healthy at {HEALTH}. Start it or use --no-run.")
            return 2

    results = []
    for sc in scenarios:
        if args.no_run and not (sc.get("cached_result") and Path(sc["cached_result"]).exists()):
            print(f"[SKIP] {sc['name']} (no cache, --no-run)")
            continue
        print(f"[RUN ] {sc['name']} ...", flush=True)
        results.append(run_scenario(sc, args))

    lines = ["# LIBRA L4 행동 채점 리포트", ""]
    sp = sum(1 for r in results if r["passed"])
    cp = sum(1 for r in results for _, ok, _ in r["checks"] if ok)
    ct = sum(len(r["checks"]) for r in results)
    lines += [f"- 시나리오 통과: **{sp}/{len(results)}**",
              f"- 기준 통과: **{cp}/{ct}**", ""]
    print("\n" + "=" * 74)
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"[{tag}] {r['name']:18} ({r['src']}) — {r['desc']}")
        lines += [f"## {'✅' if r['passed'] else '❌'} {r['name']} ({r['src']})",
                  f"_{r['desc']}_", ""]
        for cname, ok, detail in r["checks"]:
            mark = "✓" if ok else "✗"
            print(f"   {mark} {cname:22} {detail}")
            lines.append(f"- {mark} `{cname}` — {detail}")
        lines.append("")
    print("=" * 74)
    print(f"시나리오 {sp}/{len(results)} | 기준 {cp}/{ct}")

    rep = HERE / "report.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n리포트: {rep}")
    return 0 if sp == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

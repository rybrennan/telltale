"""Run one FEDn session with the telltale monitor and save what it logged.

    python run_experiment.py <run name> [--fault kind:site:onset:ramp:severity] [--rounds N]

The monitor runs inside FEDn's hook container. Its per-round records come back
through that container's log, the same route a Scaleout user would read them.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from fedn import APIClient

HERE = Path(__file__).parent
N_CLIENTS = 6


def build_class(run: str, fault: dict | None, seed: int):
    """Write this run's copy of the class and import it.

    The platform reuses the running instance when the code string is unchanged,
    so state would carry over between sessions. A unique RUN_TAG gives each
    run a fresh instance.
    """
    src = (HERE / "server_functions.py").read_text()
    assert (
        src.count('RUN_TAG = ""') == 1
        and src.count("EXPERIMENT_SETTINGS = {}") == 1
    )
    src = src.replace('RUN_TAG = ""', f'RUN_TAG = "{run}"')
    settings = {"telltale_seed": seed}
    if fault:
        settings["telltale_fault"] = fault
    src = src.replace(
        "EXPERIMENT_SETTINGS = {}", f"EXPERIMENT_SETTINGS = {settings!r}"
    )
    out = HERE / "_generated" / f"sf_{run.replace('-', '_')}.py"
    out.parent.mkdir(exist_ok=True)
    out.write_text(src)
    spec = importlib.util.spec_from_file_location(out.stem, out)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[out.stem] = mod
    spec.loader.exec_module(mod)
    return mod.ServerFunctions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--fault", default=None)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--seed-model", required=True, help="model id of the seed model"
    )
    args = ap.parse_args()

    fault = None
    if args.fault:
        kind, site, onset, ramp, severity = args.fault.split(":")
        fault = {
            "kind": kind,
            "site": int(site),
            "onset": int(onset),
            "ramp": int(ramp),
            "severity": float(severity),
        }

    out_path = HERE / "results" / f"{args.run}.json"
    if out_path.exists():
        sys.exit(f"{out_path} exists; pick a new run name")

    c = APIClient(host="localhost", port=8092)
    online = [
        x for x in c.get_clients()["result"] if x.get("status") == "online"
    ]
    if len(online) != N_CLIENTS:
        sys.exit(f"need {N_CLIENTS} online clients, found {len(online)}")

    cls = build_class(args.run, fault, args.seed)
    since = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    r = c.start_session(
        name=args.run,
        model_id=args.seed_model,
        server_functions=cls,
        helper="numpyhelper",
        rounds=args.rounds,
        round_timeout=300,
        validate=False,
        min_clients=N_CLIENTS,
        requested_clients=N_CLIENTS,
    )
    sid = r.get("session_id")
    if not sid:
        sys.exit(f"session did not start: {r}")
    while True:
        status = c.get_session_status(sid)
        if str(status).lower() in ("finished", "terminated", "failed"):
            break
        if time.time() - t0 > 3600:
            sys.exit("timed out after an hour")
        time.sleep(5)
    elapsed = int(time.time() - t0)

    log = subprocess.run(
        ["docker", "logs", "--since", since, "hook"],
        capture_output=True,
        text=True,
    )
    text = log.stdout + log.stderr
    records, errors = [], []
    for line in text.splitlines():
        if (
            "TELLTALE_ERROR" in line
            or "crashed" in line
            or "Exec failed" in line
        ):
            errors.append(line)
        i = line.find("TELLTALE {")
        if i >= 0:
            rec = json.loads(line[i + len("TELLTALE ") :])
            if rec["tag"] == args.run:
                records.append(rec)

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "run": args.run,
                "session_id": sid,
                "status": status,
                "seconds": elapsed,
                "rounds_requested": args.rounds,
                "fault": fault,
                "seed": args.seed,
                "n_clients": N_CLIENTS,
                "errors": errors,
                "records": records,
            },
            indent=1,
        )
    )
    last = records[-1] if records else {}
    site = last.get("site", {})
    final = {
        site.get(cid, cid): round(v, 2)
        for cid, v in last.get("cusum", {}).items()
    }
    print(
        f"{args.run}: {status} in {elapsed}s, {len(records)} monitor records, {len(errors)} errors"
    )
    print(
        f"  final CUSUM by site: {dict(sorted(final.items(), key=lambda kv: str(kv[0])))}"
    )
    if errors:
        print("  first error:", errors[0][:300])


if __name__ == "__main__":
    main()

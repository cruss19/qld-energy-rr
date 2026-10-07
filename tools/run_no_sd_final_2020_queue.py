"""Execute the user-approved no-SD final refits sequentially, then ensemble them."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 142, 242)
MODEL = "TCN_starNLL_causal_feature_estimates_no_sd"
MANIFEST = ROOT / "training_output" / "private_queue" / "no_sd_final_2020_queue.json"
RUNNER = ROOT / "tools" / "run_tcn_star_nll_causal_feature_estimates_no_sd_final_2020.py"
BUILDER = ROOT / "tools" / "build_no_sd_extension_2020_ensemble.py"


def write(payload: dict) -> None:
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def main() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("authorization") != "user approved exact three-run list on 2026-10-07":
        raise RuntimeError("Queue authorization record is missing or incorrect")
    if [job["seed"] for job in payload["jobs"]] != list(SEEDS):
        raise RuntimeError("Queue seed order differs from authorization")
    for job in payload["jobs"]:
        seed = int(job["seed"])
        output = ROOT / job["output_directory"]
        completion = output / "completed_run.json"
        if completion.exists() and json.loads(completion.read_text(encoding="utf-8")).get("status") == "complete":
            job["status"] = "COMPLETE"
            write(payload)
            continue
        command = [sys.executable, "-u", str(RUNNER), "--seed", str(seed), "--authorise-full-run"]
        if (output / "latest_complete_epoch.pt").exists():
            command.append("--resume")
        job["status"] = "RUNNING"
        job["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(payload)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            job["status"] = "FAILED"
            job["returncode"] = result.returncode
            write(payload)
            return result.returncode
        job["status"] = "COMPLETE"
        job["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(payload)
    payload["ensemble_status"] = "RUNNING"
    write(payload)
    result = subprocess.run([sys.executable, "-u", str(BUILDER)], cwd=ROOT)
    payload["ensemble_status"] = "COMPLETE" if result.returncode == 0 else "FAILED"
    payload["status"] = "COMPLETE" if result.returncode == 0 else "FAILED"
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(payload)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

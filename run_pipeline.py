from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Sequence

from dataset.dataloader import discover_dataset_files

try:
    from dataset.dataloader import SKIPPED_FILES
except ImportError:
    SKIPPED_FILES = {}


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
PIPELINE_LOG_DIR = MODELS_DIR / "pipeline_logs"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def result_path(dataset: str) -> Path:
    return MODELS_DIR / f"{dataset}_results.json"


def discover_entries():
    """Use the same dynamic discovery logic as dataset/dataloader.py."""
    mapping = discover_dataset_files()
    return [
        {"name": name, "source": source}
        for name, source in sorted(
            mapping.items(),
            key=lambda item: item[0].lower(),
        )
    ]


def extract_multi_metrics(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]["multi"]
    return {
        name: float(metrics[name])
        for name in ("ACC", "NMI", "ARI", "PUR")
    }


def build_train_command(dataset: str, forwarded_args: Sequence[str]) -> List[str]:
    """
    IMPORTANT:
    This version is compatible with the older/improved-2 train.py.

    That train.py does NOT define --models_dir, so we must NOT pass it.
    train.py itself writes to ./models relative to the repository root.
    """
    if "--dataset" in forwarded_args or any(
        item.startswith("--dataset=") for item in forwarded_args
    ):
        raise ValueError(
            "run_pipeline.py controls --dataset automatically; "
            "remove --dataset from extra arguments."
        )

    # improved-2 train.py has no --models_dir option.
    if "--models_dir" in forwarded_args or any(
        item.startswith("--models_dir=") for item in forwarded_args
    ):
        raise ValueError(
            "This improved-2 train.py does not support --models_dir. "
            "Remove that argument."
        )

    return [
        sys.executable,
        "-u",
        str(ROOT / "train.py"),
        "--dataset",
        dataset,
        *forwarded_args,
    ]


def run_one_dataset(
    entry,
    index: int,
    total: int,
    completed: int,
    forwarded_args: Sequence[str],
):
    dataset = entry["name"]
    source = entry["source"]

    print("\n" + "=" * 92, flush=True)
    print(
        f"[PIPELINE] START {index}/{total}: {dataset} | "
        f"successfully completed: {completed}/{total}",
        flush=True,
    )
    print(f"[PIPELINE] Source: {source}", flush=True)
    print("=" * 92, flush=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = PIPELINE_LOG_DIR / f"{index:02d}_{dataset}.log"
    command = build_train_command(dataset, forwarded_args)

    print("[PIPELINE] Command:", shlex.join(command), flush=True)
    print("[PIPELINE] Live log:", log_path, flush=True)

    started = time.time()
    last_lines: List[str] = []

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"# dataset: {dataset}\n")
        log_file.write(f"# source: {source}\n")
        log_file.write(f"# started_at: {now_iso()}\n")
        log_file.write(f"# command: {shlex.join(command)}\n\n")

        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )

        try:
            assert proc.stdout is not None

            for line in proc.stdout:
                print(line, end="", flush=True)
                log_file.write(line)

                last_lines.append(line.rstrip("\n"))
                if len(last_lines) > 100:
                    last_lines.pop(0)

            code = proc.wait()

        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise

    elapsed = time.time() - started
    expected = result_path(dataset)

    if code != 0 or not expected.exists():
        failure = {
            "status": "failed",
            "failed_dataset": dataset,
            "source": source,
            "dataset_index": index,
            "total_datasets": total,
            "successfully_completed": completed,
            "exit_code": code,
            "elapsed_seconds": elapsed,
            "timestamp": now_iso(),
            "command": command,
            "log_path": str(log_path),
            "last_output_lines": last_lines,
        }

        if code == 0 and not expected.exists():
            failure["reason"] = (
                "train.py exited successfully but did not generate "
                f"{expected}. Check whether your improved-2 train.py "
                "contains the JSON-saving code."
            )

        atomic_write_json(
            MODELS_DIR / "pipeline_failure.json",
            failure,
        )
        return False, failure

    metrics = extract_multi_metrics(expected)

    return True, {
        "status": "success",
        "dataset": dataset,
        "source": source,
        "elapsed_seconds": elapsed,
        "result_json": str(expected),
        "log_path": str(log_path),
        "metrics": metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run every dataset dynamically discovered by "
            "dataset/dataloader.py using the GFSAF improved-2 train.py."
        )
    )

    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip datasets whose result JSON already exists.",
    )

    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only these dynamically discovered dataset names.",
    )

    parser.add_argument(
        "--list_only",
        action="store_true",
        help="Only list datasets; do not train.",
    )

    return parser.parse_known_args()


def main() -> int:
    args, forwarded = parse_args()

    try:
        entries = discover_entries()
    except Exception as exc:
        print(
            f"[PIPELINE][FATAL] Dataset discovery failed: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    if args.only:
        names = {entry["name"] for entry in entries}
        missing = [name for name in args.only if name not in names]

        if missing:
            print(
                "[PIPELINE][FATAL] Dataset(s) not discovered: "
                + ", ".join(missing),
                file=sys.stderr,
                flush=True,
            )
            return 2

        wanted = set(args.only)
        entries = [
            entry for entry in entries
            if entry["name"] in wanted
        ]

    print(
        f"[PIPELINE] Dynamically discovered "
        f"{len(entries)} runnable dataset(s):",
        flush=True,
    )

    for index, entry in enumerate(entries, 1):
        print(
            f"  {index:02d}. {entry['name']:<24} <- {entry['source']}",
            flush=True,
        )

    if SKIPPED_FILES:
        print(
            "\n[PIPELINE] dataloader.py explicitly skips:",
            flush=True,
        )
        for filename, reason in SKIPPED_FILES.items():
            print(f"  - {filename}: {reason}", flush=True)

    if not entries:
        print(
            "[PIPELINE][FATAL] No runnable dataset discovered.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    if args.list_only:
        return 0

    if forwarded:
        print(
            "[PIPELINE] Extra train.py args:",
            shlex.join(forwarded),
            flush=True,
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    failure_path = MODELS_DIR / "pipeline_failure.json"
    if failure_path.exists():
        failure_path.unlink()

    started = time.time()
    total = len(entries)
    completed = 0
    records = []

    for index, entry in enumerate(entries, 1):
        dataset = entry["name"]
        existing = result_path(dataset)

        if args.skip_completed and existing.exists():
            metrics = extract_multi_metrics(existing)
            completed += 1

            record = {
                "status": "skipped_completed",
                "dataset": dataset,
                "source": entry["source"],
                "result_json": str(existing),
                "metrics": metrics,
            }
            records.append(record)

            print(
                f"[PIPELINE] SKIP {dataset} | "
                f"completed: {completed}/{total} | "
                f"ACC={metrics['ACC']:.4f} "
                f"NMI={metrics['NMI']:.4f} "
                f"ARI={metrics['ARI']:.4f} "
                f"PUR={metrics['PUR']:.4f}",
                flush=True,
            )
            continue

        ok, record = run_one_dataset(
            entry,
            index,
            total,
            completed,
            forwarded,
        )
        records.append(record)

        atomic_write_json(
            MODELS_DIR / "pipeline_status.json",
            {
                "status": "running" if ok else "failed",
                "updated_at": now_iso(),
                "total_datasets": total,
                "successfully_completed": (
                    completed + (1 if ok else 0)
                ),
                "records": records,
            },
        )

        if not ok:
            print("\n" + "!" * 92, flush=True)
            print(
                f"[PIPELINE][BUG] {dataset} failed. PIPELINE STOPPED.",
                flush=True,
            )
            print(
                f"[PIPELINE][BUG] Successfully completed: "
                f"{completed}/{total}",
                flush=True,
            )
            print(
                f"[PIPELINE][BUG] Log: {record['log_path']}",
                flush=True,
            )
            print(
                f"[PIPELINE][BUG] Report: "
                f"{MODELS_DIR / 'pipeline_failure.json'}",
                flush=True,
            )
            print("!" * 92, flush=True)
            return int(record.get("exit_code") or 1)

        completed += 1
        metrics = record["metrics"]

        print(
            f"\n[PIPELINE] SUCCESS {dataset} | "
            f"completed: {completed}/{total} | "
            f"ACC={metrics['ACC']:.4f} "
            f"NMI={metrics['NMI']:.4f} "
            f"ARI={metrics['ARI']:.4f} "
            f"PUR={metrics['PUR']:.4f}",
            flush=True,
        )

    summary = {
        "status": "success",
        "discovery_mode": (
            "dataset.dataloader.discover_dataset_files"
        ),
        "total_datasets": total,
        "successfully_completed": completed,
        "finished_at": now_iso(),
        "elapsed_seconds": time.time() - started,
        "records": records,
    }

    atomic_write_json(
        MODELS_DIR / "pipeline_summary.json",
        summary,
    )

    atomic_write_json(
        MODELS_DIR / "pipeline_status.json",
        {
            **summary,
            "updated_at": now_iso(),
        },
    )

    print("\n" + "=" * 92, flush=True)
    print(
        f"[PIPELINE] ALL DONE: "
        f"{completed}/{total} dataset(s).",
        flush=True,
    )
    print(
        f"[PIPELINE] Summary JSON: "
        f"{MODELS_DIR / 'pipeline_summary.json'}",
        flush=True,
    )
    print("=" * 92, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

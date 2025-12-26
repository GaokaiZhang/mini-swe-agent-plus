#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run mini-SWE-agent on SWE-bench instances with hint-augmented problem statements.

This is a modified version of swebench_pool_way.py that supports injecting hints
into the problem statement via a --modified-problems JSON file.
"""

import json
import random
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Optional
import os

import typer
import yaml
from datasets import load_dataset
from rich.live import Live

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import subprocess

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.environments.singularity import SingularityEnvironment
from minisweagent.models import get_model
from minisweagent.models.litellm_model import ContextLengthExceeded

from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj

MAX_CONTEXT_RETRIES = 3

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances with hint support."""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


def _safe_text(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, bytes):
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return x.decode(enc, errors="replace")
            except Exception:
                continue
        return x.decode(errors="replace")
    return str(x)


def _write_error_artifacts(instance_dir: Path, payload: dict):
    try:
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / "error.log").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
    except Exception:
        pass


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    output_data = {}
    if output_path.exists():
        output_data = json.loads(output_path.read_text())
    output_data[instance_id] = {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": result,
    }
    output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    if not output_path.exists():
        return
    output_data = json.loads(output_path.read_text())
    if instance_id in output_data:
        del output_data[instance_id]
        output_path.write_text(json.dumps(output_data, indent=2))


def get_swebench_docker_image_name(instance: dict) -> str:
    image_name = instance.get("image_name", None)
    if image_name is None:
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


class MPProgressAgent(DefaultAgent):
    def __init__(self, *args, progress_queue, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_queue = progress_queue
        self.instance_id = instance_id

    def step(self) -> dict:
        try:
            self.progress_queue.put({
                "t": "status",
                "id": self.instance_id,
                "msg": f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
            })
        except Exception:
            pass
        return super().step()


def process_instance_proc(
    instance: dict,
    output_dir: str,
    model_name: Optional[str],
    config_path: str | Path,
    progress_queue,
    docker_start_sem=None,
    environment_class: str = "singularity",
    modified_problems: Optional[dict] = None,  # NEW: modified problems with hints
):
    """Single instance processor with hint support."""
    instance_id = instance["instance_id"]
    instance_dir = Path(output_dir) / instance_id

    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    try:
        progress_queue.put({"t": "start", "id": instance_id})
    except Exception:
        pass

    image_name = get_swebench_docker_image_name(instance)
    config = yaml.safe_load(get_config_path(config_path).read_text())
    model = get_model(model_name, config=config.get("model", {}))

    # NEW: Use modified problem statement if available
    if modified_problems and instance_id in modified_problems:
        mod_data = modified_problems[instance_id]
        if mod_data.get("problem_statement"):
            task = mod_data["problem_statement"]
            try:
                progress_queue.put({"t": "status", "id": instance_id, "msg": "Using hint-augmented problem"})
            except Exception:
                pass
        else:
            task = instance["problem_statement"]
    else:
        task = instance["problem_statement"]

    agent = None
    extra_info = None

    runtime_name = "apptainer" if environment_class == "singularity" else "docker"

    try:
        progress_queue.put({"t": "status", "id": instance_id, "msg": f"Pulling/starting {runtime_name} (queued)"})
    except Exception:
        pass

    if docker_start_sem is not None:
        docker_start_sem.acquire()

    try:
        try:
            try:
                progress_queue.put({"t": "status", "id": instance_id, "msg": f"Pulling/starting {runtime_name}"})
            except Exception:
                pass

            env_config = config.get("environment", {}).copy()
            if environment_class == "singularity":
                sif_cache_dir = os.getenv("SWEBENCH_SIF_CACHE",
                                         "/ocean/projects/cis250260p/gzhang15/cache/apptainer/swebench")
                iid = instance["instance_id"]
                id_safe = iid.replace("__", "_1776_").replace("/", "_")
                sif_filename = f"sweb.eval.x86_64.{id_safe}.sif"
                local_sif = Path(sif_cache_dir) / sif_filename

                if local_sif.exists():
                    env_config["image"] = str(local_sif)
                    try:
                        progress_queue.put({"t": "status", "id": instance_id,
                                           "msg": f"Using cached SIF: {sif_filename}"})
                    except Exception:
                        pass
                else:
                    env_config["image"] = "docker://" + image_name
                    try:
                        progress_queue.put({"t": "status", "id": instance_id,
                                           "msg": f"No cached SIF, pulling: {image_name}"})
                    except Exception:
                        pass
                env = SingularityEnvironment(**env_config)
            else:
                env_config["image"] = image_name
                env = DockerEnvironment(**env_config)
        finally:
            if docker_start_sem is not None:
                docker_start_sem.release()

        exit_status = None
        result = None

        for attempt in range(1, MAX_CONTEXT_RETRIES + 1):
            try:
                agent = MPProgressAgent(
                    model,
                    env,
                    progress_queue=progress_queue,
                    instance_id=instance_id,
                    **config.get("agent", {}),
                )

                if attempt > 1:
                    try:
                        progress_queue.put({"t": "status", "id": instance_id,
                                           "msg": f"Retry {attempt}/{MAX_CONTEXT_RETRIES}"})
                    except Exception:
                        pass

                exit_status, result = agent.run(task)
                break

            except ContextLengthExceeded as e:
                if attempt < MAX_CONTEXT_RETRIES:
                    try:
                        progress_queue.put({"t": "status", "id": instance_id,
                                           "msg": f"Context exceeded, will retry ({attempt}/{MAX_CONTEXT_RETRIES})"})
                    except Exception:
                        pass
                    time.sleep(1)
                else:
                    exit_status = "ContextLengthExceeded"
                    result = str(e)
                    extra_info = {"error": str(e), "attempts": attempt}
                    break

        if agent is not None:
            extra_info = extra_info or {}
            extra_info["cost"] = agent.model.cost
            extra_info["n_calls"] = agent.model.n_calls

            # NEW: Record which hints were used
            if modified_problems and instance_id in modified_problems:
                mod_data = modified_problems[instance_id]
                extra_info["hint_flags"] = mod_data.get("hint_flags", {})
                extra_info["hints_applied"] = mod_data.get("problem_statement") is not None

        return {
            "instance_id": instance_id,
            "exit_status": exit_status,
            "result": result,
            "messages": agent.messages if agent else [],
            "extra_info": extra_info,
        }

    except subprocess.CalledProcessError as e:
        error_payload = {
            "exception": type(e).__name__,
            "returncode": e.returncode,
            "cmd": e.cmd if hasattr(e, "cmd") else None,
            "stdout": _safe_text(e.stdout) if hasattr(e, "stdout") else None,
            "stderr": _safe_text(e.stderr) if hasattr(e, "stderr") else None,
        }
        _write_error_artifacts(instance_dir, error_payload)
        return {
            "instance_id": instance_id,
            "exit_status": "CalledProcessError",
            "result": f"returncode={e.returncode}",
            "messages": [],
            "extra_info": error_payload,
        }

    except Exception as e:
        error_payload = {
            "exception": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        _write_error_artifacts(instance_dir, error_payload)
        return {
            "instance_id": instance_id,
            "exit_status": type(e).__name__,
            "result": str(e),
            "messages": [],
            "extra_info": error_payload,
        }

    finally:
        try:
            progress_queue.put({"t": "done", "id": instance_id})
        except Exception:
            pass


def _progress_listener(progress_queue, progress_manager, stop_event):
    """Background thread that reads progress events from queue."""
    while not stop_event.is_set():
        try:
            msg = progress_queue.get(timeout=0.5)
            if msg["t"] == "start":
                progress_manager.on_instance_start(msg["id"])
            elif msg["t"] == "status":
                progress_manager.update_instance_status(msg["id"], msg["msg"])
            elif msg["t"] == "done":
                pass
        except Exception:
            pass


@app.command(help=_HELP_TEXT)
def main(
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    subset: str = typer.Option("lite", "--subset", "-s"),
    split: str = typer.Option("test", "--split"),
    filter: Optional[str] = typer.Option(None, "--filter", "-f"),
    output: str = typer.Option("swebench_output", "--output", "-o"),
    config: str = typer.Option(
        str(builtin_config_dir / "extra" / "swebench.yaml"), "--config", "-c"
    ),
    redo_existing: bool = typer.Option(False, "--redo-existing", "-r"),
    workers: int = typer.Option(1, "--workers", "-w"),
    environment_class: str = typer.Option("singularity", "--environment-class", "-e"),
    modified_problems: Optional[str] = typer.Option(None, "--modified-problems",
        help="JSON file with modified problem statements (hints)"),
):
    """Main entry point with hint support."""

    # Load modified problems if provided
    modified_problems_data = None
    if modified_problems:
        modified_problems_path = Path(modified_problems)
        if modified_problems_path.exists():
            with open(modified_problems_path) as f:
                modified_problems_data = json.load(f)
            print(f"Loaded {len(modified_problems_data)} modified problems from {modified_problems}")
        else:
            print(f"Warning: Modified problems file not found: {modified_problems}")

    dataset_name = DATASET_MAPPING.get(subset, subset)
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} instances from {dataset_name}")

    instances = list(dataset)
    if filter:
        pattern = re.compile(filter)
        instances = [i for i in instances if pattern.match(i["instance_id"])]
        print(f"Filtered to {len(instances)} instances")

    random.shuffle(instances)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not redo_existing:
        done_ids = {
            d.name for d in output_dir.iterdir()
            if d.is_dir() and (d / f"{d.name}.traj.json").exists()
        }
        instances = [i for i in instances if i["instance_id"] not in done_ids]
        print(f"Skipping {len(done_ids)} already completed instances")

    if not instances:
        print("No instances to process")
        return

    print(f"Processing {len(instances)} instances with {workers} workers")

    manager = mp.Manager()
    progress_queue = manager.Queue()
    docker_start_sem = manager.Semaphore(2)

    progress_manager = RunBatchProgressManager(len(instances), output_dir / f"exit_statuses_{time.time()}.yaml")
    stop_event = threading.Event()
    listener_thread = threading.Thread(
        target=_progress_listener,
        args=(progress_queue, progress_manager, stop_event),
        daemon=True,
    )
    listener_thread.start()

    preds_path = output_dir / "preds.json"

    with Live(progress_manager.get_renderable(), refresh_per_second=2) as live:
        progress_manager.live = live

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance_proc,
                    inst,
                    str(output_dir),
                    model,
                    config,
                    progress_queue,
                    docker_start_sem,
                    environment_class,
                    modified_problems_data,  # Pass modified problems
                ): inst["instance_id"]
                for inst in instances
            }

            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    res = future.result()
                    progress_manager.on_instance_end(
                        res["instance_id"],
                        res["exit_status"],
                        res.get("extra_info", {}),
                    )

                    # Save trajectory
                    instance_dir = output_dir / res["instance_id"]
                    instance_dir.mkdir(parents=True, exist_ok=True)
                    save_traj(
                        instance_dir,
                        res["messages"],
                        res["extra_info"],
                        res["instance_id"],
                    )

                    # Update predictions
                    update_preds_file(
                        preds_path,
                        res["instance_id"],
                        model or "unknown",
                        res["result"] or "",
                    )

                except Exception as e:
                    progress_manager.on_instance_end(instance_id, "Error", {"error": str(e)})
                    traceback.print_exc()

    stop_event.set()
    listener_thread.join(timeout=2)

    print(f"\nResults saved to {output_dir}")
    print(f"Predictions saved to {preds_path}")


if __name__ == "__main__":
    app()

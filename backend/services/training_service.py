from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class TrainingService:
    """Safe dashboard controller for the fixed AMCShield training runners."""

    ACTIVE_STATES = {"starting", "running", "stop_requested"}

    def __init__(self, checkpoints_dir: Path, src_dir: Path):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.src_dir = Path(src_dir)
        self.project_root = self.src_dir.parent
        self.config_path = self.project_root / "configs" / "config.yaml"
        self.dashboard_dir = self.project_root / "logs" / "training_dashboard"
        self.state_path = self.dashboard_dir / "state.json"
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._state = self._read_state()

        # A persisted "running" state cannot prove a process is still alive
        # after the backend restarts. Never launch a duplicate automatically.
        if self._state.get("status") in self.ACTIVE_STATES:
            self._state["status"] = "unknown"
            self._state["message"] = (
                "Backend restarted; process identity must be checked manually "
                "before another run can be launched."
            )
            self._write_state()

    def _read_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"status": "idle", "run": None}

    def _write_state(self) -> None:
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._state, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def _active_run_dir(self) -> Path | None:
        run = self._state.get("run")
        if not run or not run.get("id"):
            return None
        return self.dashboard_dir / run["id"]

    def _refresh_process(self) -> None:
        if self._process is None:
            return
        return_code = self._process.poll()
        if return_code is None:
            self._state["status"] = (
                "stop_requested"
                if self._state.get("status") == "stop_requested"
                else "running"
            )
            return

        run_dir = self._active_run_dir()
        log_path = run_dir / "training.log" if run_dir else None
        log_text = ""
        if log_path and log_path.exists():
            try:
                with log_path.open("r", encoding="utf-8", errors="replace") as stream:
                    stream.seek(max(0, log_path.stat().st_size - 20000))
                    log_text = stream.read()
            except OSError:
                pass

        if "TRAINING PAUSED (manual stop)" in log_text:
            final_status = "paused"
            message = "Training paused; resume checkpoint should be available."
        elif "TRAINING COMPLETED" in log_text:
            final_status = "completed"
            message = "Training runner completed."
        elif return_code == 0:
            final_status = "completed"
            message = "Training runner exited successfully."
        else:
            final_status = "failed"
            message = f"Training runner exited with code {return_code}."

        self._state["status"] = final_status
        self._state["message"] = message
        self._state["return_code"] = return_code
        self._state["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._write_state()
        self._process = None

    def start(self, model: str, resume_only: bool = False) -> dict[str, Any]:
        if model not in {"baseline", "robust"}:
            raise ValueError("Model must be 'baseline' or 'robust'.")

        with self._lock:
            self._refresh_process()
            if self._state.get("status") in self.ACTIVE_STATES:
                raise RuntimeError("A training process is already active.")
            if self._state.get("status") == "unknown":
                raise RuntimeError(
                    "Previous process state is unknown after backend restart. "
                    "Verify it has stopped before launching another run."
                )

            checkpoint_name = (
                "baseline_resume.pt" if model == "baseline"
                else "robust_resume.pt"
            )
            resume_path = self.checkpoints_dir / checkpoint_name
            if resume_only and not resume_path.is_file():
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {checkpoint_name}"
                )

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            run_dir = self.dashboard_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            log_path = run_dir / "training.log"
            stop_path = run_dir / "stop.request"

            runner = (
                "src.train_baseline"
                if model == "baseline"
                else "src.train_robust"
            )
            command = [sys.executable, "-m", runner]
            env = os.environ.copy()
            env["AMC_TRAINING_CONTROL_DIR"] = str(run_dir)

            self._state = {
                "status": "starting",
                "run": {
                    "id": run_id,
                    "name": f"{model.title()} CNN",
                    "model": model,
                    "runner": runner,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "log_path": str(log_path),
                    "stop_path": str(stop_path),
                    "resume_requested": resume_only,
                },
                "message": "Training process is starting.",
            }
            self._write_state()

            try:
                with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
                    creation_flags = (
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        if os.name == "nt" else 0
                    )
                    self._process = subprocess.Popen(
                        command,
                        cwd=str(self.project_root),
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        shell=False,
                        creationflags=creation_flags,
                    )
            except Exception as error:
                self._state["status"] = "failed"
                self._state["message"] = f"Could not start runner: {error}"
                self._write_state()
                raise

            self._state["status"] = "running"
            self._state["pid"] = self._process.pid
            self._state["message"] = "Training process launched."
            self._write_state()
            return self.overview()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_process()
            if self._process is None or self._process.poll() is not None:
                raise RuntimeError("There is no active dashboard training process.")

            run_dir = self._active_run_dir()
            if run_dir is None:
                raise RuntimeError("Active run metadata is unavailable.")

            (run_dir / "stop.request").touch(exist_ok=True)
            self._state["status"] = "stop_requested"
            self._state["message"] = (
                "Graceful stop requested. The current batch will finish "
                "before training saves its resume state."
            )
            self._write_state()
            return self.overview()

    def config(self) -> dict[str, Any]:
        if not self.config_path.is_file():
            raise FileNotFoundError("Training configuration file is missing.")
        with self.config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        return {"config": config, "source": "configs/config.yaml"}

    def checkpoints(self) -> list[dict[str, Any]]:
        items = []
        if self.checkpoints_dir.exists():
            for path in sorted(self.checkpoints_dir.rglob("*")):
                if path.is_file():
                    stat = path.stat()
                    items.append({
                        "name": str(path.relative_to(self.checkpoints_dir)),
                        "path": str(path),
                        "size_bytes": stat.st_size,
                        "modified": stat.st_mtime,
                        "status": "AVAILABLE",
                    })
        return items

    def history(self) -> list[dict[str, Any]]:
        run_dir = self._active_run_dir()
        if not run_dir:
            return []
        log_path = run_dir / "training.log"
        if not log_path.is_file():
            return []

        pattern = re.compile(
            r"Epoch\s*\[(\d+)/(\d+)\].*?"
            r"Train Loss:\s*([0-9.]+).*?"
            r"Train Acc:\s*([0-9.]+)%.*?"
            r"Val Loss:\s*([0-9.]+).*?"
            r"Val Acc:\s*([0-9.]+)%"
        )
        rows = []
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = pattern.search(line)
                if match:
                    epoch, total, train_loss, train_acc, val_loss, val_acc = match.groups()
                    rows.append({
                        "epoch": int(epoch),
                        "total_epochs": int(total),
                        "train_loss": float(train_loss),
                        "train_accuracy": float(train_acc),
                        "val_loss": float(val_loss),
                        "val_accuracy": float(val_acc),
                    })
        return rows

    def logs(self, limit: int = 250) -> list[dict[str, Any]]:
        run_dir = self._active_run_dir()
        if not run_dir:
            return []
        log_path = run_dir / "training.log"
        if not log_path.is_file():
            return []
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            lines = stream.readlines()[-max(1, min(limit, 1000)):]
        return [
            {"time": "", "level": "info", "message": line.rstrip("\r\n")}
            for line in lines
        ]

    def overview(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_process()
            run = dict(self._state.get("run") or {})
            history = self.history()
            latest = history[-1] if history else {}
            config = {}
            try:
                config = self.config()["config"]
            except (OSError, ValueError, yaml.YAMLError):
                pass

            checkpoint_names = {
                item["name"] for item in self.checkpoints()
            }
            model = run.get("model") or "baseline"
            best_name = f"{model}_best.pt" if model else None
            run.update({
                "status": self._state.get("status", "idle"),
                "message": self._state.get("message", ""),
                "epoch": latest.get("epoch"),
                "total_epochs": latest.get("total_epochs", config.get("epochs")),
                "train_accuracy": latest.get("train_accuracy"),
                "train_loss": latest.get("train_loss"),
                "val_accuracy": latest.get("val_accuracy"),
                "val_loss": latest.get("val_loss"),
                "checkpoint_status": (
                    "AVAILABLE" if best_name and best_name in checkpoint_names
                    else "NOT FOUND"
                ),
                "checkpoint_name": best_name,
                "checkpoint_path": (
                    str(self.checkpoints_dir / best_name) if best_name else None
                ),
            })
            return {
                "run": run,
                "training_control": "dashboard",
                "src_exists": self.src_dir.exists(),
                "checkpoint_count": len(checkpoint_names),
            }

    def status(self) -> dict[str, Any]:
        return {
            **self.overview(),
            "checkpoints": self.checkpoints(),
        }

    def progress(self) -> list[dict[str, Any]]:
        # Batch-level progress is not emitted by the current engine.
        # Do not invent percentages or ETA.
        return []

    def models(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "Baseline CNN",
                "architecture": "Standard supervised CNN",
                "status": "AVAILABLE",
                "best": False,
            },
            {
                "name": "Robust CNN",
                "architecture": "PGD adversarially trained CNN",
                "status": "AVAILABLE",
                "best": False,
            },
        ]

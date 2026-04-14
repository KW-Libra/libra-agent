from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .utils import stable_hash


class LibraDecisionStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.queues_dir = self.base_dir / "queues"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.queues_dir.mkdir(parents=True, exist_ok=True)

    def record_result(self, result: Mapping[str, Any]) -> dict[str, str | None]:
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        run_hash = stable_hash(
            {
                "query": result.get("query"),
                "decision": result.get("decision"),
                "trigger": result.get("decision", {}).get("trigger") if isinstance(result.get("decision"), Mapping) else None,
            }
        )[:12]
        run_path = self.runs_dir / f"{timestamp}_{run_hash}.json"
        run_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        decision = result.get("decision", {})
        follow_up_path: Path | None = None
        checkpoint_path: Path | None = None
        if isinstance(decision, Mapping):
            follow_up_at = decision.get("follow_up_at")
            if isinstance(follow_up_at, str) and follow_up_at.strip():
                follow_up_path = self.queues_dir / "follow_ups.jsonl"
                self._append_jsonl(
                    follow_up_path,
                    {
                        "run_path": str(run_path),
                        "query": result.get("query"),
                        "decision": decision.get("decision"),
                        "follow_up_at": follow_up_at,
                        "trigger": decision.get("trigger"),
                    },
                )
            feedback_checkpoint = decision.get("feedback_checkpoint")
            if isinstance(feedback_checkpoint, str) and feedback_checkpoint.strip():
                checkpoint_path = self.queues_dir / "feedback_checkpoints.jsonl"
                self._append_jsonl(
                    checkpoint_path,
                    {
                        "run_path": str(run_path),
                        "query": result.get("query"),
                        "decision": decision.get("decision"),
                        "feedback_checkpoint": feedback_checkpoint,
                        "trigger": decision.get("trigger"),
                    },
                )
        return {
            "run_path": str(run_path),
            "follow_up_queue": str(follow_up_path) if follow_up_path else None,
            "feedback_queue": str(checkpoint_path) if checkpoint_path else None,
        }

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False))
            handle.write("\n")

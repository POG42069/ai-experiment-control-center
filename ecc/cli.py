"""CLI for ECC read-only queries, two-step audited launches, and private dataset uploads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecc.config import ConfigError
from ecc.launch.builder import WheelBuildError
from ecc.launch.datasets import (
    DatasetUploadRefused,
    DatasetUploadService,
    KaggleDatasetError,
    build_monitor_wheel_folder,
)
from ecc.launch.github import GitHubReader
from ecc.launch.service import LaunchRefused, LaunchRequest, LaunchService
from ecc.models import to_jsonable
from ecc.redact import redact
from ecc.secrets import SecretStore
from ecc.service import RunNotFoundError, Service, build_service


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _downsample_df(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    indices = np.linspace(0, len(df) - 1, num=max_points, dtype=int)
    indices = np.unique(indices)
    return df.iloc[indices]


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    records = df.to_dict(orient="records")
    return [to_jsonable(rec) for rec in records]


def _print_json(payload: Any) -> None:
    safe = to_jsonable(payload)
    sys.stdout.write(json.dumps(safe, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def build_launch_service(read_service: Service) -> LaunchService:
    """Build a ``LaunchService`` wired to ``read_service``'s store, settings, and secrets."""
    secrets = getattr(read_service, "_secrets", None)
    if secrets is None:
        secrets = SecretStore()
    return LaunchService(
        store=read_service.store,
        secrets=secrets,
        settings=read_service.settings,
        read_service=read_service,
        github=GitHubReader(),
    )


def build_dataset_upload_service(
    read_service: Service,
    launch_service: LaunchService,
) -> DatasetUploadService:
    """Build a ``DatasetUploadService`` sharing store, settings, secrets, and owner resolver."""
    secrets = getattr(read_service, "_secrets", None)
    if secrets is None:
        secrets = SecretStore()
    return DatasetUploadService(
        store=read_service.store,
        secrets=secrets,
        settings=read_service.settings,
        launch_service=launch_service,
    )


def _add_common_flags(p: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    cfg_default: Any = argparse.SUPPRESS if suppress_defaults else None
    json_default: Any = argparse.SUPPRESS if suppress_defaults else False
    p.add_argument(
        "--config",
        dest="config",
        default=cfg_default,
        help="Path to ecc_config.toml",
    )
    p.add_argument(
        "--json",
        dest="json_flag",
        action="store_true",
        default=json_default,
        help="Output JSON (always enabled; accepted for compatibility)",
    )


def _add_read_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", dest="project", default=None, help="Filter by project slug")
    p.add_argument("--status", dest="status", default=None, help="Filter by run status")
    p.add_argument(
        "--run",
        dest="runs",
        action="extend",
        nargs="+",
        default=[],
        help="One or more run IDs",
    )
    p.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=None,
        help="Metric name(s) (e.g. eval/macro_f1)",
    )
    p.add_argument(
        "--axis",
        dest="axis",
        choices=["step", "epoch"],
        default="step",
        help="X-axis for compare curves",
    )
    p.add_argument(
        "--mode",
        dest="mode",
        choices=["max", "min"],
        default="max",
        help="Optimization direction for best command",
    )
    p.add_argument("--tail", dest="tail", type=int, default=500, help="Log tail line count")
    p.add_argument("--query", dest="query", default=None, help="Log substring filter")
    p.add_argument("--level", dest="level", default=None, help="Log level filter")
    p.add_argument(
        "--source",
        dest="source",
        default=None,
        help="Optional log source override (e.g. 'kaggle' for native Kaggle logs)",
    )
    p.add_argument(
        "--account",
        dest="account",
        default=None,
        help="Filter by account nickname (for kaggle-kernels)",
    )
    p.add_argument(
        "--since-hours",
        dest="since_hours",
        type=float,
        default=72.0,
        help="Lookback window in hours for failed runs",
    )
    p.add_argument(
        "--max-points",
        dest="max_points",
        type=int,
        default=500,
        help="Maximum metric points per run (downsampled if exceeded)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ecc.cli",
        description="AI Experiment Control Center JSON CLI",
    )
    _add_common_flags(parser, suppress_defaults=False)

    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parent = argparse.ArgumentParser(add_help=False)
    _add_common_flags(common_parent, suppress_defaults=True)

    read_parent = argparse.ArgumentParser(add_help=False, parents=[common_parent])
    _add_read_flags(read_parent)

    for read_cmd in (
        "list-projects",
        "list-runs",
        "list-active-runs",
        "get-run",
        "get-metrics",
        "get-logs",
        "compare",
        "failed",
        "best",
        "health",
        "accounts",
        "kaggle-kernels",
    ):
        subparsers.add_parser(read_cmd, parents=[read_parent])

    # ------------------------------------------------------------------
    # launch subcommands
    # ------------------------------------------------------------------
    launch_parser = subparsers.add_parser(
        "launch",
        parents=[common_parent],
        help="Two-step audited Kaggle launch commands",
    )
    launch_sub = launch_parser.add_subparsers(dest="launch_action", required=True)

    lp_plan = launch_sub.add_parser("plan", parents=[common_parent], help="Plan a Kaggle launch")
    lp_plan.add_argument("--account", required=True, help="Kaggle account nickname")
    lp_plan.add_argument("--repo", required=True, help="GitHub repository (owner/repo)")
    lp_plan.add_argument("--branch", required=True, help="Git branch name")
    lp_plan.add_argument("--commit", dest="commit", default=None, help="Optional commit SHA")
    lp_plan.add_argument("--command", dest="launch_command", default="", help="Training command")
    lp_plan.add_argument(
        "--accelerator",
        dest="accelerator",
        default=None,
        help="Accelerator (e.g. NvidiaTeslaT4, TpuV5E8, none)",
    )
    lp_plan.add_argument(
        "--timeout-hours",
        dest="timeout_hours",
        type=float,
        default=1.0,
        help="Session timeout in hours",
    )
    lp_plan.add_argument(
        "--monitor-mode",
        dest="monitor_mode",
        default="runner",
        help="Monitoring mode ('runner' or 'script')",
    )
    lp_plan.add_argument("--project", dest="project", default="", help="W&B project slug")
    lp_plan.add_argument("--run-name", dest="run_name", default="", help="Optional W&B run name")
    lp_plan.add_argument(
        "--requirements",
        dest="requirements",
        default="",
        help="Optional requirements file path relative to repo root",
    )
    lp_plan.add_argument(
        "--preflight",
        dest="preflight",
        action="store_true",
        help="CPU-only preflight check (no GPU quota)",
    )

    lp_exec = launch_sub.add_parser(
        "execute", parents=[common_parent], help="Execute a persisted launch plan"
    )
    lp_exec.add_argument("--plan-id", dest="plan_id", required=True, help="Persisted plan_id")

    lp_status = launch_sub.add_parser(
        "status", parents=[common_parent], help="Check runner kernel status for an account"
    )
    lp_status.add_argument("--account", required=True, help="Kaggle account nickname")

    lp_history = launch_sub.add_parser(
        "history", parents=[common_parent], help="Show audited launch history"
    )
    lp_history.add_argument("--limit", type=int, default=20, help="Maximum history entries")

    # ------------------------------------------------------------------
    # dataset subcommands
    # ------------------------------------------------------------------
    ds_parser = subparsers.add_parser(
        "dataset",
        parents=[common_parent],
        help="Private Kaggle dataset upload and wheel helpers",
    )
    ds_sub = ds_parser.add_subparsers(dest="dataset_action", required=True)

    ds_upload = ds_sub.add_parser(
        "upload", parents=[common_parent], help="Upload or version a private Kaggle dataset"
    )
    ds_upload.add_argument("--account", required=True, help="Kaggle account nickname")
    ds_upload.add_argument("--folder", required=True, help="Local folder to upload")
    ds_upload.add_argument("--slug", required=True, help="Dataset slug (3-50 chars, [a-z0-9-])")
    ds_upload.add_argument("--title", required=True, help="Dataset title (6-50 chars)")
    ds_upload.add_argument(
        "--message",
        dest="message",
        default=None,
        help="Version notes message (used when updating an existing dataset)",
    )

    ds_wheel = ds_sub.add_parser(
        "build-monitor-wheel",
        parents=[common_parent],
        help="Build the experiment_monitor wheel into an empty folder",
    )
    ds_wheel.add_argument("--out", required=True, help="Empty output directory path")

    return parser


def _handle_launch(args: argparse.Namespace, read_service: Service) -> int:
    launch_service = build_launch_service(read_service)
    if not launch_service.is_enabled():
        raise LaunchRefused(
            "Launch control is disabled. Set [launch] enabled = true in ecc_config.toml."
        )

    action = args.launch_action

    if action == "plan":
        accelerator = (
            args.accelerator
            if args.accelerator is not None
            else read_service.settings.launch.default_accelerator
        )
        req = LaunchRequest(
            account=args.account,
            repo=args.repo,
            branch=args.branch,
            commit_sha=args.commit,
            command=args.launch_command,
            accelerator=accelerator,
            timeout_hours=args.timeout_hours,
            monitor_mode=args.monitor_mode,
            project=args.project,
            run_name=args.run_name,
            requirements_file=args.requirements,
            preflight=bool(args.preflight),
            source="claude",
        )
        plan = launch_service.plan_launch(req, source="claude")
        saved = launch_service.save_plan(plan)
        _print_json(
            {
                "plan_id": plan.launch_id,
                "summary": plan.summary,
                "consumes_gpu": plan.consumes_gpu_quota,
                "expires_at": saved["expires_at"],
            }
        )
        return 0

    if action == "execute":
        record = launch_service.execute_saved_plan(
            args.plan_id,
            confirmed_by_user=True,
            source="claude",
        )
        payload = record.to_dict()
        owner = record.kernel_ref.split("/", 1)[0] if "/" in record.kernel_ref else ""
        payload["owner"] = owner
        _print_json(payload)
        return 0 if record.ok else 4

    if action == "status":
        kernel_ref = launch_service.runner_ref(args.account)
        status = launch_service.runner_status(args.account)
        owner = kernel_ref.split("/", 1)[0] if "/" in kernel_ref else ""
        _print_json(
            {
                "account": args.account,
                "owner": owner,
                "kernel_ref": kernel_ref,
                "url": f"https://www.kaggle.com/code/{kernel_ref}",
                "status": status,
            }
        )
        return 0

    if action == "history":
        records = launch_service.list_launches(limit=args.limit)
        _print_json([r.to_dict() for r in records])
        return 0

    raise ValueError(f"Unsupported launch action: {action}")


def _handle_dataset(args: argparse.Namespace, read_service: Service) -> int:
    action = args.dataset_action

    if action == "build-monitor-wheel":
        wheel_file = build_monitor_wheel_folder(args.out)
        _print_json(
            {
                "ok": True,
                "out": str(Path(args.out).resolve()),
                "wheel_file": str(wheel_file),
                "size_bytes": wheel_file.stat().st_size,
            }
        )
        return 0

    if action == "upload":
        launch_service = build_launch_service(read_service)
        ds_service = build_dataset_upload_service(read_service, launch_service)
        result = ds_service.upload(
            account=args.account,
            folder=args.folder,
            slug=args.slug,
            title=args.title,
            message=args.message,
            source="claude",
        )
        _print_json(result.to_dict())
        return 0 if result.ok else 4

    raise ValueError(f"Unsupported dataset action: {action}")


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        service = build_service(config_path=args.config)
        cmd = args.command

        if cmd == "launch":
            return _handle_launch(args, service)

        if cmd == "dataset":
            return _handle_dataset(args, service)

        if cmd == "list-projects":
            projects = service.list_projects()
            _print_json([p.to_dict() for p in projects])
            return 0

        if cmd == "list-runs":
            runs = service.list_runs(project=args.project, status=args.status)
            _print_json([r.to_dict() for r in runs])
            return 0

        if cmd == "list-active-runs":
            runs = service.list_active_runs(project=args.project)
            _print_json([r.to_dict() for r in runs])
            return 0

        if cmd == "get-run":
            if len(args.runs) != 1:
                raise ValueError("get-run requires exactly one --run RUN_ID")
            run = service.get_run_details(args.runs[0])
            if run is None:
                raise RunNotFoundError(f"Unknown run_id: {args.runs[0]}")
            _print_json(run.to_dict())
            return 0

        if cmd == "get-metrics":
            if len(args.runs) != 1:
                raise ValueError("get-metrics requires exactly one --run RUN_ID")
            keys: list[str] | None = None
            if args.metrics:
                keys = []
                for m in args.metrics:
                    keys.extend([part.strip() for part in m.split(",") if part.strip()])
            df = service.get_run_metrics(args.runs[0], keys=keys)
            df = _downsample_df(df, args.max_points)
            _print_json(_df_to_records(df))
            return 0

        if cmd == "get-logs":
            if len(args.runs) != 1:
                raise ValueError("get-logs requires exactly one --run RUN_ID")
            logs = service.get_run_logs(
                args.runs[0],
                tail=args.tail,
                query=args.query,
                level=args.level,
                source=args.source,
            )
            _print_json([line.to_dict() for line in logs])
            return 0

        if cmd == "compare":
            primary_metric = args.metrics[0] if args.metrics else "eval/macro_f1"
            res = service.compare_runs(
                run_ids=args.runs,
                metric=primary_metric,
                axis=args.axis,
            )
            curves_json: dict[str, list[dict[str, Any]]] = {}
            for m_key, cdf in res["curves"].items():
                if cdf.empty:
                    curves_json[m_key] = []
                else:
                    sampled_parts = [
                        _downsample_df(grp, args.max_points)
                        for _, grp in cdf.groupby("run_id", sort=False)
                    ]
                    sampled = pd.concat(sampled_parts, ignore_index=True)
                    curves_json[m_key] = _df_to_records(sampled)

            payload = {
                "runs": res["runs"],
                "table": _df_to_records(res["table"]),
                "curves": curves_json,
                "warnings": res["warnings"],
            }
            _print_json(payload)
            return 0

        if cmd == "failed":
            runs = service.get_failed_runs(project=args.project, since_hours=args.since_hours)
            _print_json([r.to_dict() for r in runs])
            return 0

        if cmd == "best":
            if not args.project:
                raise ValueError("best requires --project PROJECT")
            primary_metric = args.metrics[0] if args.metrics else "eval/macro_f1"
            res = service.get_best_experiment(
                project=args.project,
                metric=primary_metric,
                mode=args.mode,
            )
            _print_json(res)
            return 0

        if cmd == "health":
            _print_json(service.health())
            return 0

        if cmd == "accounts":
            accounts = service.list_accounts()
            _print_json([a.to_dict() for a in accounts])
            return 0

        if cmd == "kaggle-kernels":
            kernels = service.list_kaggle_kernels(nickname=args.account)
            _print_json([k.to_dict() for k in kernels])
            return 0

        raise ValueError(f"Unsupported command: {cmd}")

    except LaunchRefused as exc:
        _print_json({"error": "launch_refused", "reason": redact(str(exc))})
        return 3
    except (DatasetUploadRefused, PermissionError) as exc:
        _print_json({"error": "upload_refused", "reason": redact(str(exc))})
        return 3
    except (KaggleDatasetError, WheelBuildError) as exc:
        _print_json({"error": "operation_failed", "reason": redact(str(exc))})
        return 4
    except RunNotFoundError as exc:
        _print_json({"error": "run_not_found", "message": redact(str(exc))})
        return 2
    except (ValueError, ConfigError) as exc:
        _print_json({"error": "invalid_argument", "message": redact(str(exc))})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Local SQLite metadata store (ARCHITECTURE §6).

Stores only user-authored metadata (projects, notes, account nicknames/status,
kernel links). Never stores secrets or training metrics.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ecc.redact import redact

SCHEMA_VERSION = 3

_SCHEMA_V1_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    wandb_entity TEXT,
    wandb_project TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '',
    baseline_run_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_notes (
    run_id TEXT PRIMARY KEY,
    notes TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    pinned INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
    nickname TEXT PRIMARY KEY,
    platform TEXT NOT NULL CHECK(platform IN ('kaggle', 'colab', 'wandb')),
    username TEXT NOT NULL DEFAULT '',
    secret_ref TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    last_check_at TEXT,
    last_check_ok INTEGER,
    last_check_msg TEXT
);

CREATE TABLE IF NOT EXISTS kernel_links (
    kernel_ref TEXT PRIMARY KEY,
    account TEXT NOT NULL,
    project TEXT NOT NULL,
    run_id TEXT
);
"""

_SCHEMA_V2_SQL = """
CREATE TABLE IF NOT EXISTS launches (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    account TEXT NOT NULL,
    kernel_ref TEXT NOT NULL,
    repo TEXT NOT NULL,
    branch TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    command TEXT NOT NULL,
    accelerator TEXT NOT NULL,
    timeout_s INTEGER NOT NULL,
    monitor_mode TEXT NOT NULL,
    project TEXT NOT NULL DEFAULT '',
    run_name TEXT NOT NULL DEFAULT '',
    preflight INTEGER NOT NULL DEFAULT 0,
    ok INTEGER NOT NULL DEFAULT 0,
    version INTEGER,
    url TEXT,
    message TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_launches_created_at ON launches (created_at DESC);

PRAGMA user_version = 2;
"""

_SCHEMA_V3_SQL = """
CREATE TABLE IF NOT EXISTS pending_plans (
    plan_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    executed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_pending_plans_created_at ON pending_plans (created_at DESC);

CREATE TABLE IF NOT EXISTS dataset_uploads (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    account TEXT NOT NULL,
    ref TEXT NOT NULL,
    folder TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    ok INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'claude'
);

CREATE INDEX IF NOT EXISTS idx_dataset_uploads_created_at ON dataset_uploads (created_at DESC);

PRAGMA user_version = 3;
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return []


class Store:
    """SQLite-backed metadata store with idempotent schema initialization."""

    def __init__(self, db_path: str | Path) -> None:
        self._is_memory = str(db_path) == ":memory:"
        self.db_path = ":memory:" if self._is_memory else str(Path(db_path))
        self._mem_conn: sqlite3.Connection | None = None

        if not self._is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self._mem_conn = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row

        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _close(self, conn: sqlite3.Connection) -> None:
        if conn is not self._mem_conn:
            conn.close()

    def close(self) -> None:
        if self._mem_conn is not None:
            self._mem_conn.close()
            self._mem_conn = None

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.executescript(_SCHEMA_V1_SQL)
                row = conn.execute("PRAGMA user_version").fetchone()
                current_version = int(row[0]) if row else 0
                if current_version < 2:
                    conn.executescript(_SCHEMA_V2_SQL)
                    current_version = 2
                if current_version < 3:
                    cols = {
                        str(r[1])
                        for r in conn.execute("PRAGMA table_info(launches)").fetchall()
                    }
                    if "source" not in cols:
                        conn.execute(
                            "ALTER TABLE launches ADD COLUMN source TEXT NOT NULL DEFAULT 'ui'"
                        )
                    conn.executescript(_SCHEMA_V3_SQL)
        finally:
            self._close(conn)

    def user_version(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("PRAGMA user_version").fetchone()
            return int(row[0]) if row else 0
        finally:
            self._close(conn)

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "slug": row["slug"],
            "name": row["name"],
            "description": row["description"],
            "wandb_entity": row["wandb_entity"],
            "wandb_project": row["wandb_project"],
            "tags": _load_json_list(row["tags"]),
            "notes": row["notes"],
            "baseline_run_id": row["baseline_run_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def upsert_project(
        self,
        slug: str,
        name: str,
        description: str = "",
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
        tags: list[str] | None = None,
        notes: str = "",
        baseline_run_id: str | None = None,
    ) -> dict[str, Any]:
        now = _now_iso()
        tags_json = json.dumps(list(tags) if tags is not None else [])
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO projects (
                        slug, name, description, wandb_entity, wandb_project,
                        tags, notes, baseline_run_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(slug) DO UPDATE SET
                        name = excluded.name,
                        description = excluded.description,
                        wandb_entity = excluded.wandb_entity,
                        wandb_project = excluded.wandb_project,
                        tags = excluded.tags,
                        notes = excluded.notes,
                        baseline_run_id = COALESCE(excluded.baseline_run_id, projects.baseline_run_id),
                        updated_at = excluded.updated_at
                    """,
                    (
                        slug,
                        name,
                        description,
                        wandb_entity,
                        wandb_project,
                        tags_json,
                        notes,
                        baseline_run_id,
                        now,
                        now,
                    ),
                )
            row = conn.execute("SELECT * FROM projects WHERE slug = ?", (slug,)).fetchone()
            assert row is not None
            return self._row_to_project(row)
        finally:
            self._close(conn)

    def get_project(self, slug: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM projects WHERE slug = ?", (slug,)).fetchone()
            return self._row_to_project(row) if row is not None else None
        finally:
            self._close(conn)

    def list_projects(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM projects ORDER BY slug ASC").fetchall()
            return [self._row_to_project(r) for r in rows]
        finally:
            self._close(conn)

    def delete_project(self, slug: str) -> bool:
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute("DELETE FROM projects WHERE slug = ?", (slug,))
                return cur.rowcount > 0
        finally:
            self._close(conn)

    def set_baseline(self, project: str, run_id: str | None) -> None:
        now = _now_iso()
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    "UPDATE projects SET baseline_run_id = ?, updated_at = ? WHERE slug = ?",
                    (run_id, now, project),
                )
                if cur.rowcount == 0:
                    conn.execute(
                        """
                        INSERT INTO projects (
                            slug, name, description, wandb_entity, wandb_project,
                            tags, notes, baseline_run_id, created_at, updated_at
                        ) VALUES (?, ?, '', NULL, NULL, '[]', '', ?, ?, ?)
                        """,
                        (project, project, run_id, now, now),
                    )
        finally:
            self._close(conn)

    # ------------------------------------------------------------------
    # Run notes
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_run_note(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "notes": row["notes"],
            "tags": _load_json_list(row["tags"]),
            "pinned": bool(row["pinned"]),
            "updated_at": row["updated_at"],
        }

    def set_run_note(
        self,
        run_id: str,
        notes: str = "",
        tags: list[str] | None = None,
        pinned: bool = False,
    ) -> dict[str, Any]:
        now = _now_iso()
        tags_json = json.dumps(list(tags) if tags is not None else [])
        pinned_int = 1 if pinned else 0
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO run_notes (run_id, notes, tags, pinned, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        notes = excluded.notes,
                        tags = excluded.tags,
                        pinned = excluded.pinned,
                        updated_at = excluded.updated_at
                    """,
                    (run_id, notes, tags_json, pinned_int, now),
                )
            row = conn.execute("SELECT * FROM run_notes WHERE run_id = ?", (run_id,)).fetchone()
            assert row is not None
            return self._row_to_run_note(row)
        finally:
            self._close(conn)

    def get_run_note(self, run_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM run_notes WHERE run_id = ?", (run_id,)).fetchone()
            return self._row_to_run_note(row) if row is not None else None
        finally:
            self._close(conn)

    # ------------------------------------------------------------------
    # Accounts (no secrets!)
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_account(row: sqlite3.Row) -> dict[str, Any]:
        ok_val = row["last_check_ok"]
        return {
            "nickname": row["nickname"],
            "platform": row["platform"],
            "username": row["username"],
            "secret_ref": row["secret_ref"],
            "enabled": bool(row["enabled"]),
            "last_check_at": row["last_check_at"],
            "last_check_ok": None if ok_val is None else bool(ok_val),
            "last_check_msg": row["last_check_msg"],
        }

    def upsert_account(
        self,
        nickname: str,
        platform: str,
        username: str = "",
        secret_ref: str | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        ref = secret_ref if secret_ref is not None else f"{platform}:{nickname}"
        enabled_int = 1 if enabled else 0
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO accounts (nickname, platform, username, secret_ref, enabled)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(nickname) DO UPDATE SET
                        platform = excluded.platform,
                        username = excluded.username,
                        secret_ref = excluded.secret_ref,
                        enabled = excluded.enabled
                    """,
                    (nickname, platform, username, ref, enabled_int),
                )
            row = conn.execute(
                "SELECT * FROM accounts WHERE nickname = ?", (nickname,)
            ).fetchone()
            assert row is not None
            return self._row_to_account(row)
        finally:
            self._close(conn)

    def list_accounts(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM accounts ORDER BY nickname ASC").fetchall()
            return [self._row_to_account(r) for r in rows]
        finally:
            self._close(conn)

    def delete_account(self, nickname: str) -> bool:
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute("DELETE FROM accounts WHERE nickname = ?", (nickname,))
                return cur.rowcount > 0
        finally:
            self._close(conn)

    def record_account_check(self, nickname: str, ok: bool, msg: str = "") -> None:
        now = _now_iso()
        safe_msg = redact(msg)[:300]
        ok_int = 1 if ok else 0
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE accounts
                    SET last_check_at = ?, last_check_ok = ?, last_check_msg = ?
                    WHERE nickname = ?
                    """,
                    (now, ok_int, safe_msg, nickname),
                )
        finally:
            self._close(conn)

    # ------------------------------------------------------------------
    # Kernel links
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_kernel_link(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "kernel_ref": row["kernel_ref"],
            "account": row["account"],
            "project": row["project"],
            "run_id": row["run_id"],
        }

    def link_kernel(
        self,
        kernel_ref: str,
        account: str,
        project: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO kernel_links (kernel_ref, account, project, run_id)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(kernel_ref) DO UPDATE SET
                        account = excluded.account,
                        project = excluded.project,
                        run_id = excluded.run_id
                    """,
                    (kernel_ref, account, project, run_id),
                )
            row = conn.execute(
                "SELECT * FROM kernel_links WHERE kernel_ref = ?", (kernel_ref,)
            ).fetchone()
            assert row is not None
            return self._row_to_kernel_link(row)
        finally:
            self._close(conn)

    def list_kernel_links(
        self,
        account: str | None = None,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if account is not None:
            clauses.append("account = ?")
            params.append(account)
        if project is not None:
            clauses.append("project = ?")
            params.append(project)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM kernel_links{where} ORDER BY kernel_ref ASC",
                tuple(params),
            ).fetchall()
            return [self._row_to_kernel_link(r) for r in rows]
        finally:
            self._close(conn)

    # ------------------------------------------------------------------
    # Launch audit log
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_launch(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "account": row["account"],
            "kernel_ref": row["kernel_ref"],
            "repo": row["repo"],
            "branch": row["branch"],
            "commit_sha": row["commit_sha"],
            "command": row["command"],
            "accelerator": row["accelerator"],
            "timeout_s": int(row["timeout_s"]),
            "monitor_mode": row["monitor_mode"],
            "project": row["project"],
            "run_name": row["run_name"],
            "preflight": bool(row["preflight"]),
            "ok": bool(row["ok"]),
            "version": None if row["version"] is None else int(row["version"]),
            "url": row["url"],
            "message": row["message"],
            "source": row["source"] if "source" in row.keys() and row["source"] else "ui",
        }

    def record_launch(
        self,
        *,
        account: str,
        kernel_ref: str,
        repo: str,
        branch: str,
        commit_sha: str,
        command: str,
        accelerator: str,
        timeout_s: int,
        monitor_mode: str,
        project: str = "",
        run_name: str = "",
        preflight: bool = False,
        ok: bool = False,
        version: int | None = None,
        url: str | None = None,
        message: str = "",
        source: str = "ui",
        launch_id: str | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        lid = launch_id or uuid.uuid4().hex
        ts = created_at or _now_iso()
        safe_msg = redact(message)
        src = (source or "ui").strip() or "ui"
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO launches (
                        id, created_at, account, kernel_ref, repo, branch, commit_sha,
                        command, accelerator, timeout_s, monitor_mode, project, run_name,
                        preflight, ok, version, url, message, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lid,
                        ts,
                        account,
                        kernel_ref,
                        repo,
                        branch,
                        commit_sha,
                        command,
                        accelerator,
                        int(timeout_s),
                        monitor_mode,
                        project,
                        run_name,
                        1 if preflight else 0,
                        1 if ok else 0,
                        version,
                        url,
                        safe_msg,
                        src,
                    ),
                )
            row = conn.execute("SELECT * FROM launches WHERE id = ?", (lid,)).fetchone()
            assert row is not None
            return self._row_to_launch(row)
        finally:
            self._close(conn)

    def get_launch(self, launch_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM launches WHERE id = ?", (launch_id,)).fetchone()
            return self._row_to_launch(row) if row is not None else None
        finally:
            self._close(conn)

    def list_launches(
        self,
        limit: int = 50,
        *,
        account: str | None = None,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if account is not None:
            clauses.append("account = ?")
            params.append(account)
        if project is not None:
            clauses.append("project = ?")
            params.append(project)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        lim = max(0, int(limit))
        params.append(lim)
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM launches{where} ORDER BY created_at DESC, rowid DESC LIMIT ?",
                tuple(params),
            ).fetchall()
            return [self._row_to_launch(r) for r in rows]
        finally:
            self._close(conn)

    # ------------------------------------------------------------------
    # Pending launch plans (two-step CLI plan -> execute)
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_pending_plan(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "plan_id": row["plan_id"],
            "created_at": row["created_at"],
            "payload_json": row["payload_json"],
            "executed_at": row["executed_at"],
        }

    def save_pending_plan(
        self,
        plan_id: str,
        created_at: str,
        payload_json: str,
    ) -> dict[str, Any]:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO pending_plans (plan_id, created_at, payload_json, executed_at)
                    VALUES (?, ?, ?, NULL)
                    """,
                    (plan_id, created_at, payload_json),
                )
            row = conn.execute(
                "SELECT * FROM pending_plans WHERE plan_id = ?", (plan_id,)
            ).fetchone()
            assert row is not None
            return self._row_to_pending_plan(row)
        finally:
            self._close(conn)

    def get_pending_plan(self, plan_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM pending_plans WHERE plan_id = ?", (plan_id,)
            ).fetchone()
            return self._row_to_pending_plan(row) if row is not None else None
        finally:
            self._close(conn)

    def mark_pending_plan_executed(
        self,
        plan_id: str,
        executed_at: str | None = None,
    ) -> bool:
        ts = executed_at or _now_iso()
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    """
                    UPDATE pending_plans
                    SET executed_at = ?
                    WHERE plan_id = ? AND executed_at IS NULL
                    """,
                    (ts, plan_id),
                )
                return cur.rowcount > 0
        finally:
            self._close(conn)

    # ------------------------------------------------------------------
    # Dataset upload audit log
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_dataset_upload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "account": row["account"],
            "ref": row["ref"],
            "folder": row["folder"],
            "size_bytes": int(row["size_bytes"]),
            "ok": bool(row["ok"]),
            "message": row["message"],
            "source": row["source"],
        }

    def record_dataset_upload(
        self,
        *,
        account: str,
        ref: str,
        folder: str,
        size_bytes: int,
        ok: bool,
        message: str = "",
        source: str = "claude",
        upload_id: str | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        uid = upload_id or uuid.uuid4().hex
        ts = created_at or _now_iso()
        safe_msg = redact(message)
        src = (source or "claude").strip() or "claude"
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO dataset_uploads (
                        id, created_at, account, ref, folder, size_bytes, ok, message, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uid,
                        ts,
                        account,
                        ref,
                        folder,
                        int(size_bytes),
                        1 if ok else 0,
                        safe_msg,
                        src,
                    ),
                )
            row = conn.execute(
                "SELECT * FROM dataset_uploads WHERE id = ?", (uid,)
            ).fetchone()
            assert row is not None
            return self._row_to_dataset_upload(row)
        finally:
            self._close(conn)

    def list_dataset_uploads(
        self,
        limit: int = 50,
        *,
        account: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if account is not None:
            clauses.append("account = ?")
            params.append(account)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        lim = max(0, int(limit))
        params.append(lim)
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM dataset_uploads{where} ORDER BY created_at DESC, rowid DESC LIMIT ?",
                tuple(params),
            ).fetchall()
            return [self._row_to_dataset_upload(r) for r in rows]
        finally:
            self._close(conn)

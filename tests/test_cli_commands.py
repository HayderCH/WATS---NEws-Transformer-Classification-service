"""Smoke tests for the Typer CLI commands."""

import json
import os
from pathlib import Path
import zipfile

import sqlalchemy as sa
from typer.testing import CliRunner

from app.db.models import Feedback, ReviewItem


runner = CliRunner()


def test_seed_db_cli_command(tmp_path, monkeypatch):
    db_path = tmp_path / "cli_seed.db"
    db_url = f"sqlite:///{db_path.as_posix()}"

    monkeypatch.setenv("DB_URL", db_url)

    from app.core.config import get_settings

    get_settings.cache_clear()

    from scripts.manage import app

    env = {**os.environ, "DB_URL": db_url}
    result = runner.invoke(app, ["seed-db", "--overwrite"], env=env)
    assert result.exit_code == 0, result.output

    assert db_path.exists()

    engine = sa.create_engine(db_url, future=True)
    try:
        insp = sa.inspect(engine)
        assert set(insp.get_table_names()) >= {"feedback", "review_items"}

        with engine.connect() as conn:
            feedback_count = conn.execute(
                sa.select(sa.func.count()).select_from(Feedback)
            ).scalar_one()
            review_count = conn.execute(
                sa.select(sa.func.count()).select_from(ReviewItem)
            ).scalar_one()

        assert feedback_count > 0
        assert review_count > 0
    finally:
        engine.dispose()

    # Second invocation without overwrite should report already populated
    result_again = runner.invoke(app, ["seed-db"], env=env)
    assert result_again.exit_code == 0, result_again.output
    assert "Already populated" in result_again.output


def test_bundle_artifacts_cli(tmp_path):
    source_dir = tmp_path / "bundle_src"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "artifact.txt").write_text("hello", encoding="utf-8")

    output_dir = tmp_path / "archives"

    from scripts.manage import app

    result = runner.invoke(
        app,
        [
            "bundle-artifacts",
            "--sources",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--label",
            "cli-test",
        ],
    )
    assert result.exit_code == 0, result.output

    archives = list(output_dir.glob("bundle_*.zip"))
    assert len(archives) == 1
    archive_path = archives[0]

    with zipfile.ZipFile(archive_path) as zf:
        assert "bundle_src/nested/artifact.txt" in zf.namelist()
        manifest = json.loads(zf.read("artifact-manifest.json"))
        assert manifest["label"] == "cli-test"
        assert manifest["push_requested"] is False
        assert any("bundle_src" in entry for entry in manifest["sources"])


def test_bundle_artifacts_push_cli(monkeypatch, tmp_path):
    source_dir = tmp_path / "bundle_src"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "artifact.txt").write_text("hello", encoding="utf-8")

    output_dir = tmp_path / "archives"

    class StubPublisher:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, str | None]] = []

        def upload_file(
            self,
            file_path,
            *,
            destination_name=None,
        ):
            path_obj = Path(file_path)
            self.calls.append((path_obj, destination_name))
            return f"stub://{destination_name or path_obj.name}"

    stub = StubPublisher()

    monkeypatch.setenv("ARTIFACT_STORE_TYPE", "s3")
    monkeypatch.setenv("ARTIFACT_S3_BUCKET", "stub-bucket")
    monkeypatch.setenv("ARTIFACT_PUSH_DEFAULT", "0")

    from app.core.config import get_settings

    get_settings.cache_clear()

    monkeypatch.setattr(
        "scripts.manage.create_artifact_publisher",
        lambda settings: stub,
    )

    from scripts.manage import app

    result = runner.invoke(
        app,
        [
            "bundle-artifacts",
            "--sources",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--push",
            "--remote-name",
            "custom.zip",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Uploaded archive to: stub://custom.zip" in result.output
    assert len(stub.calls) == 1
    uploaded_path, dest_name = stub.calls[0]
    assert dest_name == "custom.zip"
    assert uploaded_path.exists()

    archives = list(output_dir.glob("bundle_*.zip"))
    assert archives, "archive should exist for manifest validation"
    with zipfile.ZipFile(archives[0]) as zf:
        manifest = json.loads(zf.read("artifact-manifest.json"))
        assert manifest["push_requested"] is True
        assert manifest["remote_name"] == "custom.zip"
        assert manifest["settings"]["artifact_store_type"] == "s3"

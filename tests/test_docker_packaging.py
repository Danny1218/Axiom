"""Dockerfile, compose, and .dockerignore contract (Phase 53)."""

from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_dockerfile_uses_slim_base_and_serve_lock():
    df = (_root() / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:" in df and "slim" in df
    assert "[serve,lock]" in df or '"[serve,lock]"' in df
    assert "CMD" in df and "serve" in df
    assert "portfolio_trained" not in df.lower()  # no example bundle baked into image
    assert "0.0.0.0" in df or "HOST=" in df


def test_docker_compose_exposes_and_mounts_bundles():
    yml = (_root() / "docker-compose.yml").read_text(encoding="utf-8")
    assert "8000:8000" in yml
    assert "bundles:/bundles" in yml or "./bundles:/bundles" in yml
    assert "AXIOM_BUNDLE_PATH" in yml
    assert "AXIOM_API_KEY" in yml
    assert "AXIOM_REQUIRE_API_KEY" in yml
    assert "AXIOM_BUNDLE_SECRET" in yml
    assert "HOST" in yml and "PORT" in yml


def test_dockerfile_requires_api_key_in_production():
    df = (_root() / "Dockerfile").read_text(encoding="utf-8")
    assert "AXIOM_REQUIRE_API_KEY" in df
    ign = (_root() / ".dockerignore").read_text(encoding="utf-8")
    assert ".venv" in ign or "venv" in ign
    assert ".pytest_cache" in ign or "**/.pytest_cache" in ign
    assert "tests" in ign


def test_readme_documents_docker_build_run_compose_curl():
    text = (_root() / "readme.md").read_text(encoding="utf-8")
    assert "## Docker deployment" in text
    assert "docker build" in text
    assert "docker run" in text
    assert "docker compose" in text
    assert "curl" in text.lower()
    assert "AXIOM_BUNDLE_PATH" in text
    assert "change-me-in-production" not in text or "AXIOM_API_KEY" in text
    assert "Building only" in text or "start the server" in text.lower()

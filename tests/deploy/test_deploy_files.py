from pathlib import Path


def test_dockerfile_runs_uvicorn_on_hf_space_port():
    dockerfile = Path("Dockerfile").read_text()

    assert "EXPOSE 7860" in dockerfile
    assert "GENACADEMY_DATA_DIR=/data" in dockerfile
    assert "HF_HOME=/app/.cache/huggingface" in dockerfile
    assert "SentenceTransformer('all-MiniLM-L6-v2')" in dockerfile
    assert "scripts/start_hf_space.sh" in dockerfile


def test_space_readme_declares_docker_sdk_and_port():
    readme = Path("README.md").read_text()

    assert "sdk: docker" in readme
    assert "app_port: 7860" in readme
    assert "GenAcademy RAG" in readme


def test_dockerignore_excludes_local_state():
    dockerignore = Path(".dockerignore").read_text()

    assert ".env" in dockerignore
    assert ".venv" in dockerignore
    assert ".github/" in dockerignore
    assert "data/" in dockerignore
    assert "docs/" in dockerignore
    assert "eval/runs/" in dockerignore
    assert "tests/" in dockerignore


def test_start_script_bootstraps_then_runs_web_app():
    start_script = Path("scripts/start_hf_space.sh").read_text()

    assert "python -m genacademy_rag.deploy.bootstrap" in start_script
    assert "uvicorn genacademy_rag.web.main:app" in start_script


def test_env_example_is_docker_env_file_compatible():
    lines = Path(".env.example").read_text().splitlines()
    assert "# GENACADEMY_DATA_DIR=/absolute/path/to/data" in lines
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        value = stripped.split("=", 1)[1]
        assert "#" not in value, f"inline comment is not Docker --env-file compatible: {line}"

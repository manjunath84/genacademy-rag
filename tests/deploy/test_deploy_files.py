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

import pytest


def test_smoke_http_checks_login_page(monkeypatch, capsys):
    import scripts.smoke_http as smoke_http

    class _Response:
        status_code = 200
        text = '<form action="/login"><input name="csrf_token">member@genacademy.local</form>'

        def raise_for_status(self):
            pass

    monkeypatch.setattr(smoke_http.requests, "get", lambda url, timeout: _Response())

    smoke_http.main(["--base-url", "http://127.0.0.1:7860"])

    assert "HTTP SMOKE OK" in capsys.readouterr().out


def test_smoke_http_fails_when_login_marker_missing(monkeypatch):
    import scripts.smoke_http as smoke_http

    class _Response:
        status_code = 200
        text = "not the login page"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(smoke_http.requests, "get", lambda url, timeout: _Response())

    with pytest.raises(SystemExit) as exc:
        smoke_http.main(["--base-url", "http://127.0.0.1:7860"])

    assert "login marker not found" in str(exc.value)

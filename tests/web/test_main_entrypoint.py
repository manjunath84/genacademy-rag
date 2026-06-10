def test_main_entrypoint_builds_default_app(monkeypatch):
    import importlib
    import sys

    import genacademy_rag.web.app as app_module

    built = object()
    monkeypatch.setattr(app_module, "build_default_app", lambda: built)

    main_module = importlib.import_module("genacademy_rag.web.main")
    main_module = importlib.reload(main_module)

    try:
        assert main_module.app is built
    finally:
        sys.modules.pop("genacademy_rag.web.main", None)

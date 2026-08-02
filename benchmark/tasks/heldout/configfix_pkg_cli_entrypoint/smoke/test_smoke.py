def test_smoke_import():
    import importlib
    assert importlib.import_module("demo.commands")

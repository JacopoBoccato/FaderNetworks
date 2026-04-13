from pathlib import Path
import runpy


TESTS_DIR = Path(__file__).resolve().parent


def _run(script_name: str) -> None:
    runpy.run_path(str(TESTS_DIR / script_name), run_name="__main__")


def test_model_smoke() -> None:
    _run("model_test.py")


def test_training_smoke() -> None:
    _run("training_test.py")


def test_loader_smoke() -> None:
    _run("test_loader.py")

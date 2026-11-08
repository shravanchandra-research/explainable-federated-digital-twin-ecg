"""Lightweight syntax test for deposited experiment scripts; does not run data-dependent experiments."""
from pathlib import Path
import py_compile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"

class ExperimentScriptSyntaxTests(unittest.TestCase):
    def test_all_deposited_experiment_scripts_compile(self):
        scripts = sorted(SCRIPT_ROOT.rglob("experiment_*.py"))
        self.assertEqual(len(scripts), 17, "Update this expected count when adding/removing experiment scripts.")
        for script in scripts:
            with self.subTest(script=script.relative_to(REPO_ROOT).as_posix()):
                py_compile.compile(str(script), doraise=True)

if __name__ == "__main__":
    unittest.main()

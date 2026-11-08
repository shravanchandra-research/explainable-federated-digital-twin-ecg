"""Syntax checks for all experiment scripts and reference package modules."""
from pathlib import Path
import py_compile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT_ROOT=ROOT/"scripts"
SOURCE_ROOT=ROOT/"src"/"efdt"

class SourceSyntaxTests(unittest.TestCase):
    def test_all_17_experiment_scripts_compile(self):
        files=sorted(SCRIPT_ROOT.rglob("experiment_*.py"))
        self.assertEqual(len(files),17)
        for path in files:
            with self.subTest(file=path.relative_to(ROOT).as_posix()):
                py_compile.compile(str(path),doraise=True)

    def test_all_reference_package_modules_compile(self):
        files=sorted(SOURCE_ROOT.rglob("*.py"))
        self.assertGreaterEqual(len(files),10)
        for path in files:
            with self.subTest(file=path.relative_to(ROOT).as_posix()):
                py_compile.compile(str(path),doraise=True)

if __name__=="__main__": unittest.main()

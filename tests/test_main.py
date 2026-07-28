"""
Tests for BEES main module (parser + BEES init)

Run: pytest -v tests/test_main.py
"""

import os
import sys
import yaml
import pytest
from unittest import mock

import bees.main as main
from bees.common import InputError  # raised by read_yaml_file on missing file


def _get_parse_and_load_input():
    """Lazy import to avoid BEES.py env check during test collection."""
    if "bees_env" not in sys.executable:
        sys.executable = sys.executable + "bees_env"
    from BEES import parse_and_load_input
    return parse_and_load_input

# --------------------------
# Shared fixtures/utilities
# --------------------------

@pytest.fixture(autouse=True)
def patch_paths_and_deps(tmp_path, monkeypatch):
    """
    - Redirect BEES_PATH to temp dir.
    - Create projects/minimal/ for the parser's default path.
    - Stub git helpers for deterministic logs.
    """
    if "bees_env" not in sys.executable:
        monkeypatch.setattr(sys, "executable", sys.executable + "bees_env")

    bees_path = tmp_path / "bees_path"

    (bees_path / 'projects' / 'minimal').mkdir(parents=True)

    # Patch globals used by main.py and its imported common
    monkeypatch.setattr(main, 'BEES_PATH', str(bees_path))
    monkeypatch.setattr(main.common, 'BEES_PATH', str(bees_path))

    # Patch BEES.py (the parser file)
    import BEES as bees_cli
    monkeypatch.setattr(bees_cli, 'BEES_PATH', str(bees_path))

    # Stable git info
    monkeypatch.setattr(main.common, 'get_git_branch', lambda: 'test-branch')
    monkeypatch.setattr(main.common, 'get_git_commit', lambda: ('abcdef', '2025-07-27'))

    yield


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)

# --------------------------
# Tests: parse_and_load_input (from BEES.py)
# --------------------------

def test_parser_uses_default_input_when_none_provided(tmp_path, monkeypatch):
    default_dir = os.path.join(main.BEES_PATH, 'projects', 'minimal')
    os.makedirs(default_dir, exist_ok=True)
    default_yaml = os.path.join(default_dir, 'input.yml')
    _write_yaml(default_yaml, {
        'project': 'DefaultProj',
        'species': [], 'enzymes': [],
        'environment': {'temperature': 300},
        'database': {'name': 'db'},
        'settings': {'end_time': 10, 'verbose': 20},
    })

    old_argv = sys.argv
    sys.argv = ['BEES.py']
    try:
        parse_and_load_input = _get_parse_and_load_input()
        out = parse_and_load_input()
    finally:
        sys.argv = old_argv

    assert out['project'] == 'DefaultProj'
    assert out['settings']['verbose'] == 20
    assert out['settings']['end_time'] == 10


def test_parser_cli_overrides_yaml(tmp_path):
    yml = tmp_path / "custom.yml"
    _write_yaml(yml, {
        'project': 'YAMLProj',
        'species': [], 'enzymes': [],
        'environment': {'temperature': 310},
        'database': {'name': 'db'},
        'settings': {'end_time': 100, 'time_step': 0.5, 'verbose': 30}
    })

    old_argv = sys.argv
    sys.argv = [
        'BEES.py',
        '--input_file', str(yml),
        '--project', 'CLIProj',
        '--verbose', '10',
        '--output_directory', 'results/run_01',
    ]
    try:
        parse_and_load_input = _get_parse_and_load_input()
        out = parse_and_load_input()
    finally:
        sys.argv = old_argv

    assert out['project'] == 'CLIProj'
    assert out['settings']['verbose'] == 10
    assert out['settings']['output_directory'] == 'results/run_01'
    assert out['settings']['end_time'] == 100
    assert out['settings']['time_step'] == 0.5


def test_parser_missing_yaml_raises_inputerror(tmp_path):
    old_argv = sys.argv
    sys.argv = ['BEES.py', '--input_file', str(tmp_path / 'does_not_exist.yml')]
    try:
        parse_and_load_input = _get_parse_and_load_input()
        with pytest.raises(InputError):
            parse_and_load_input()
    finally:
        sys.argv = old_argv


def test_parser_requires_project_when_yaml_omits_it(tmp_path):
    yml = tmp_path / "no_project.yml"
    _write_yaml(yml, {
        'species': [], 'enzymes': [],
        'environment': {'temperature': 300},
        'database': {'name': 'db'},
        'settings': {'end_time': 10, 'verbose': 20},
    })

    old_argv = sys.argv
    sys.argv = ['BEES.py', '--input_file', str(yml)]
    try:
        parse_and_load_input = _get_parse_and_load_input()
        with pytest.raises(ValueError, match="Project name is required"):
            parse_and_load_input()
    finally:
        sys.argv = old_argv


def test_parser_injects_settings_when_missing(tmp_path):
    yml = tmp_path / "no_settings.yml"
    _write_yaml(yml, {
        'project': 'P',
        'species': [], 'enzymes': [],
        'environment': {'temperature': 300},
        'database': {'name': 'db'},
    })

    old_argv = sys.argv
    sys.argv = ['BEES.py', '--input_file', str(yml), '--verbose', '40', '--output_directory', 'out']
    try:
        parse_and_load_input = _get_parse_and_load_input()
        out = parse_and_load_input()
    finally:
        sys.argv = old_argv

    assert out['project'] == 'P'
    assert out['settings']['verbose'] == 40
    assert out['settings']['output_directory'] == 'out'


# --------------------------
# Tests: BEES.__init__ 
# --------------------------

class DummySchema:
    def __init__(self, **kwargs):
        self.project = kwargs.get('project')
        self.species = kwargs.get('species', [])
        self.enzymes = kwargs.get('enzymes', [])
        class E: pass
        self.environment = E()
        env = kwargs.get('environment', {})
        self.environment.temperature = env.get('temperature')
        self.environment.pH = env.get('pH', 7.0)
        class D: pass
        self.database = D()
        db = kwargs.get('database', {})
        self.database.name = db.get('name')
        class S: pass
        self.settings = S()
        st = kwargs.get('settings', {})
        self.settings.end_time = st.get('end_time')
        self.settings.verbose = st.get('verbose')
        self.settings.time_step = st.get('time_step', 1.0)
    def model_dump(self, exclude_unset):
        return {}

@mock.patch('bees.main.Logger')
@mock.patch('bees.main.InputBase', new=DummySchema)
def test_BEES_init_success(MockLogger):
    data = {
        'project': 'testproj',
        'species': [{'label': 'A'}],
        'enzymes': [{'name': 'E'}],
        'environment': {'temperature': 310, 'pH': 7.4},
        'database': {'name': 'db'},
        'settings': {'end_time': 15, 'verbose': 10, 'time_step': 0.5},
    }
    bees = main.BEES(data)

    assert os.path.isdir(bees.project_directory)
    assert bees.bees_object.project == 'testproj'
    MockLogger.assert_called_once()


@mock.patch('bees.main.Logger')
def test_BEES_init_dir_fail(MockLogger, monkeypatch):
    data = {'project': 'proj'}
    monkeypatch.setattr(os, 'makedirs', lambda *a, **k: (_ for _ in ()).throw(OSError("deny")))
    with pytest.raises(FileNotFoundError):
        main.BEES(data)


@mock.patch('bees.main.Logger')
@mock.patch('bees.main.InputBase', side_effect=Exception("schema error"))
def test_BEES_init_schema_fail(MockInputBase, MockLogger):
    data = {
        'project': 'x',
        'species': [{'label': 'A'}],
        'enzymes': [{'name': 'E'}],
        'environment': {'temperature': 300, 'pH': 7.0},
        'database': {'name': 'db'},
        'settings': {'end_time': 5, 'verbose': 20, 'time_step': 1.0},
    }
    with pytest.raises(ValueError):
        main.BEES(data)

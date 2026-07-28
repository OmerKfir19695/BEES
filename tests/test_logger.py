""""
Test BEES logger module


This module Test the logger module .
To run the tests, use pytest and the command line: pytest -v tests/test_logger.py

"""


import os
from pathlib import Path
import time
import logging
import pytest

from bees.logger import Logger, logger as global_logger
import bees.common as common

@ pytest.fixture(autouse=True)
def reset_logger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Reset the Logger singleton and the global 'BEES' logger before each test.
    """
    # 1) remove any existing handlers from the module-level logger
    for handler in list(global_logger.handlers):
        global_logger.removeHandler(handler)
    global_logger.setLevel(logging.NOTSET)

    # 3) stub out any external dependencies
    monkeypatch.setattr(common, "get_git_branch", lambda path=None: "testbranch")
    monkeypatch.setattr(common, "get_git_commit", lambda path=None: ("fakehash", "fakedate"))
    monkeypatch.setattr(common, "time_lapse", lambda t0: "00:00:00")
    monkeypatch.setattr(common, "dict_to_str", lambda d: "\n".join(f"{k}: {v}" for k, v in d.items()))

    yield

@ pytest.fixture
def tmp_dir(tmp_path: Path):
    """
    A clean temporary directory for our logs.
    """
    d = tmp_path / "myproject"
    d.mkdir()
    return str(d)


def test_reinit_does_not_accumulate_handlers_and_sets_t0(tmp_path: Path):
    # First init: should create exactly 3 handlers (console + main file + error file)
    d1 = tmp_path / "p1"; d1.mkdir()
    d2 = tmp_path / "p2"; d2.mkdir()
    logger1 = Logger(project_directory=str(d1), verbose=logging.INFO, t0=time.time())
    assert len(global_logger.handlers) == 3
    assert hasattr(logger1, "t0")

    # Second init in the same process: must NOT accumulate handlers (_setup_handlers
    # clears them first), and MUST fully initialize. Regression: a previous
    # class-level guard early-returned here, leaving the instance without self.t0
    # so log_footer() crashed with "'Logger' object has no attribute 't0'" whenever
    # BEES().execute() ran more than once per process.
    logger2 = Logger(project_directory=str(d2), verbose=logging.DEBUG, t0=time.time())
    assert len(global_logger.handlers) == 3   # no duplicate accumulation
    assert hasattr(logger2, "t0")             # fully initialized on re-init

    assert isinstance(logger1, Logger)
    assert isinstance(logger2, Logger)


def test_info_and_debug_levels(tmp_dir: str, capsys: pytest.CaptureFixture[str]):
    """
    INFO goes to console, DEBUG does not when verbose=INFO.
    Both go to the main log file.
    """
    logger = Logger(project_directory=tmp_dir, verbose=logging.INFO, t0=time.time())
    # clear initial header output
    capsys.readouterr()

    logger.info("hello-info")
    logger.debug("hello-debug")

    out = capsys.readouterr().out
    assert "hello-info" in out
    assert "hello-debug" not in out

    project_name = os.path.basename(tmp_dir)
    main_log = os.path.join(tmp_dir, f"{project_name}.log")
    with open(main_log) as f:
        content = f.read()
    assert "hello-info" in content
    assert "hello-debug" in content


def test_warning_error_critical_to_console_and_error_file(tmp_dir: str, capsys: pytest.CaptureFixture[str]):
    """
    WARNING+ should appear on console at WARNING level, and ERROR+
    should go to the error log.
    """
    logger = Logger(project_directory=tmp_dir, verbose=logging.WARNING, t0=time.time())
    capsys.readouterr()

    logger.warning("warn-this")
    logger.error("err-this")
    logger.critical("crit-this")

    out = capsys.readouterr().out
    assert "warn-this" in out
    assert "err-this" in out
    assert "crit-this" in out

    project_name = os.path.basename(tmp_dir)
    err_log = os.path.join(tmp_dir, f"{project_name}_errors.log")
    with open(err_log) as f:
        err_content = f.read()
    assert "err-this" in err_content
    assert "crit-this" in err_content
    assert "warn-this" not in err_content


def test_helper_methods_and_args(tmp_dir: str, capsys: pytest.CaptureFixture[str]):
    """
    Test .always(), .log_max_time_reached(), .log_footer(), .log_args().
    """
    logger = Logger(project_directory=tmp_dir, verbose=10, t0=time.time())
    capsys.readouterr()

    logger.always("go-always")
    out = capsys.readouterr().out
    assert "ALWAYS: go-always" in out

    capsys.readouterr()
    logger.log_max_time_reached("01:23:45")
    out = capsys.readouterr().out
    assert "execution terminated because the maximum run time was reached" in out.lower()
    

    capsys.readouterr()
    logger.log_footer(success=True)
    out = capsys.readouterr().out
    assert "Total BEES execution time:" in out
    assert "BEES execution completed successfully." in out

    capsys.readouterr()
    logger.log_footer(success=False)
    out = capsys.readouterr().out
    assert "BEES execution terminated with errors." in out

    capsys.readouterr()
    schema = {"x": 1, "verbose": 20, "y": "yes"}
    logger.log_args(schema)
    out = capsys.readouterr().out
    assert "Full input arguments" in out
    assert "verbose: info" in out


def test_file_backup(tmp_dir: str):
    """
    Ensure old logs get backed up into log_archive on re-init.
    """
    project_name = os.path.basename(tmp_dir)
    main_log = os.path.join(tmp_dir, f"{project_name}.log")
    err_log  = os.path.join(tmp_dir, f"{project_name}_errors.log")
    archive  = os.path.join(tmp_dir, "log_archive")

    # create old logs
    with open(main_log, "w") as f:
        f.write("OLD_MAIN")
    with open(err_log, "w") as f:
        f.write("OLD_ERR")

    # reset handlers
    for handler in list(global_logger.handlers):
        global_logger.removeHandler(handler)
    global_logger.setLevel(logging.NOTSET)

    # re-init logger to trigger backup
    Logger(project_directory=tmp_dir, verbose=logging.DEBUG, t0=time.time())

    archived = sorted(os.listdir(archive))
    assert any(x.startswith(f"{project_name}_main.old") for x in archived)
    assert any(x.startswith(f"{project_name}_errors.old") for x in archived)

    # originals cleared
    with open(main_log) as f:
        assert f.read() != "OLD_MAIN"
    with open(err_log) as f:
        assert f.read() != "OLD_ERR"

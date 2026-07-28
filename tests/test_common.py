""""
Test BEES common module


This module Test the common module which contains functions which are shared across multiple  modules.
To run the tests, use pytest and the command line: pytest -v tests/test_common.py


"""


import pytest
import os

import time
import yaml
from unittest.mock import patch
import re


# Import functions and constants from the common module
from bees.common import (
    get_git_branch,
    get_git_commit,
    InputError,
    read_yaml_file,
    save_yaml_file,
    from_yaml,
    to_yaml,
    globalize_paths,
    globalize_path,
    time_lapse,
    dict_to_str,
    heavy_atom_count,
    get_ontology_equivalents,
)

# Mock BEES_PATH for isolated testing
@pytest.fixture(autouse=True)
def mock_bees_paths(tmp_path):
    """
    Fixture to mock BEES_PATH to temporary directory
    for isolated testing of file operations.
    Also creates a dummy .git directory for git-related tests.
    """
    mock_bees_root = tmp_path / 'BEES_ROOT'
    mock_git_dir = mock_bees_root / '.git'

    os.makedirs(mock_bees_root, exist_ok=True)
    mock_git_dir.mkdir(exist_ok=True) # Create dummy .git directory for git tests

    # Patch the constants in the common module
    with patch('bees.common.BEES_PATH', str(mock_bees_root)):
        yield
        # Cleanup after tests

def test_get_git_branch():
    """Test get_git_branch function."""
    # Mock subprocess.check_output for a controlled test environment
    with patch('subprocess.check_output') as mock_subproc:
        # Mocking the command to simulate git branch output
        mock_subproc.return_value = b'  main\n* develop\n'
        branch = get_git_branch()
        assert branch == 'develop'

        mock_subproc.return_value = b'* master\n'
        branch = get_git_branch()
        assert branch == 'master'

    # Test when .git directory does not exist (mock os.path.exists)
    with patch('os.path.exists', return_value=False):
        branch = get_git_branch()
        assert branch == ''

def test_get_git_commit():
    """Test get_git_commit function."""
    with patch('subprocess.check_output') as mock_subproc:
        # Mocking the command to simulate git log output
        mock_subproc.return_value = b'abcdef1234567890\nThu Jan 1 00:00:00 1970 +0000\n'
        commit, date = get_git_commit()
        assert commit == 'abcdef1234567890'
        assert date == 'Thu Jan 1 00:00:00 1970 +0000'

    # Test when .git directory does not exist (mock os.path.exists)
    with patch('os.path.exists', return_value=False):
        commit, date = get_git_commit()
        assert commit == ''
        assert date == ''

def test_InputError():
    """Test InputError exception."""
    with pytest.raises(InputError, match="This is a test error"):
        raise InputError("This is a test error")

def test_read_yaml_file(tmp_path):
    """Test read_yaml_file function."""
    # Test reading a valid YAML file
    test_yaml_content = {"key": "value", "number": 123}
    test_yaml_path = tmp_path / "test.yml"
    with open(test_yaml_path, "w") as f:
        yaml.dump(test_yaml_content, f)

    content = read_yaml_file(str(test_yaml_path))
    assert content == test_yaml_content

    # Test with project_directory for relative path resolution (file path itself)
    nested_dir = tmp_path / "project_dir"
    nested_dir.mkdir()
    nested_yaml_path = nested_dir / "nested.yml"
    with open(nested_yaml_path, "w") as f:
        yaml.dump({"nested_key": "nested_value"}, f)
    
    content = read_yaml_file("nested.yml", project_directory=str(nested_dir))
    assert content == {"nested_key": "nested_value"}

    # Test non-existent file
    with pytest.raises(InputError, match="Could not find the YAML file"):
        read_yaml_file(str(tmp_path / "non_existent.yml"))

    # Test invalid YAML content
    invalid_yaml_path = tmp_path / "invalid.yml"
    with open(invalid_yaml_path, "w") as f:
        f.write("key: - value") # Invalid YAML syntax (sequence entry where mapping key is expected)

    # The error message can vary slightly between PyYAML versions,
    # so a more general regex for YAMLError is appropriate.
    with pytest.raises(yaml.YAMLError):
        read_yaml_file(str(invalid_yaml_path))

    # Test invalid path type
    with pytest.raises(InputError, match="path must be a string"):
        read_yaml_file(123)


def test_get_ontology_equivalents_acp_thioester_permutations():
    eq = get_ontology_equivalents("3-oxo-(5Z)-dodecenoyl-[ACP]")
    assert "(5z)-3-oxododecenoyl-[acp]" in eq

    eq2 = get_ontology_equivalents("(5Z)-3-oxododecenoyl-[ACP]")
    assert "3-oxo-(5z)-dodecenoyl-[acp]" in eq2

def test_save_yaml_file(tmp_path):
    """Test save_yaml_file function."""
    output_path = tmp_path / "output.yml"
    data_to_save = {"data_key": "data_value", "list_data": [1, 2, 3]}
    save_yaml_file(str(output_path), data_to_save)

    assert output_path.exists()
    with open(output_path, "r") as f:
        loaded_data = yaml.safe_load(f)
    assert loaded_data == data_to_save

    # Test saving to a non-existent directory
    nested_output_dir = tmp_path / "new_dir" / "sub_dir"
    nested_output_path = nested_output_dir / "nested_output.yml"
    save_yaml_file(str(nested_output_path), {"nested": True})
    assert nested_output_path.exists()

    # Test invalid path type
    with pytest.raises(InputError, match="path must be a string"):
        save_yaml_file(123, {})

def test_from_yaml():
    """Test from_yaml function."""
    yaml_string = """
    name: Test
    value: 42
    """
    data = from_yaml(yaml_string)
    assert data == {"name": "Test", "value": 42}

def test_to_yaml():
    """Test to_yaml function."""
    data = {"name": "Test", "value": 42}
    yaml_string = to_yaml(data)
    # Use safe_load to verify the output YAML string
    loaded_data = yaml.safe_load(yaml_string)
    assert loaded_data == data

    # Test with multiline string representation
    data_multiline = {"description": "This is a\nmultiline\nstring."}
    yaml_string_multiline = to_yaml(data_multiline) # Use to_yaml to apply the custom representer
    # The string_representer uses style='|'. PyYAML often adds '|- ' for block literals.
    # The regex should be flexible for leading spaces and the exact chomping indicator.
    # It should match "description: " followed by optional " |-" and then the multiline content.
    assert re.search(r"description:\s*\|-?\s*\n\s*This is a\n\s*multiline\n\s*string\.?", yaml_string_multiline) is not None


def test_globalize_paths(tmp_path):
    """Test globalize_paths function."""
    original_content = """
path_to_calc: /old/path/calcs/Species/mol1.xyz
project_directory: /old/path/
another_path: /some/other/file.txt
    """
    original_file = tmp_path / "config.yml"
    original_file.write_text(original_content)

    new_project_dir_path = tmp_path / "new_project_root" # Keep as Path object
    os.makedirs(new_project_dir_path, exist_ok=True)

    # Simulate the structure expected by globalize_path
    (new_project_dir_path / "calcs" / "Species").mkdir(parents=True, exist_ok=True)

    # Pass new_project_dir as string to globalize_paths as it expects a string
    globalized_file_path = globalize_paths(str(original_file), str(new_project_dir_path))
    
    assert "globalized" in globalized_file_path
    
    with open(globalized_file_path, 'r') as f:
        content = f.read()

    # Check that paths were correctly rebased
    # Use os.path.join for robust path construction in assertions
    expected_calc_path = os.path.join(os.path.normpath(str(new_project_dir_path)), 'alcs', 'Species', 'mol1.xyz')
    expected_project_dir_line = f"project_directory: {os.path.normpath(str(new_project_dir_path)).rstrip(os.sep) + os.sep}"

    # Assert that the rebased paths are present in the content
    assert expected_calc_path.replace(os.sep, '/') in content.replace('\\', '/')
    assert expected_project_dir_line in content
    assert "another_path: /some/other/file.txt" in content # Should remain unchanged

    # Test case where no changes are needed
    current_project_root_path = tmp_path / "current_project_root"
    os.makedirs(current_project_root_path, exist_ok=True)
    (current_project_root_path / "calcs" / "Species").mkdir(parents=True, exist_ok=True)

    # Construct the content for the "no change" test case to ensure exact match
    # It's crucial that this matches what globalize_path would return if no change is needed.
    # Ensure consistent path normalization and trailing slash for the content.
    normalized_current_project_root_str = os.path.normpath(str(current_project_root_path)).rstrip(os.sep) + os.sep
    # The content must exactly match what globalize_path would return for no change.
    # This means preserving leading spaces and ensuring correct newlines. 
    no_change_content = f"""
path_to_calc: {normalized_current_project_root_str}calcs/Species/mol1.xyz
project_directory: {normalized_current_project_root_str}
"""
    # Split, strip each line, and join with os.linesep to ensure consistent newlines
    # and remove any leading/trailing blank lines from the f-string itself.
    lines_for_no_change_file = [line.strip() + os.linesep for line in no_change_content.strip().splitlines()]
    # Add a final newline if the original template had one after the last line,
    # or ensure the file ends consistently with how globalize_path would output it.
    final_no_change_content = "".join(lines_for_no_change_file)
    if not final_no_change_content.endswith(os.linesep):
        final_no_change_content += os.linesep
    
    no_change_file = tmp_path / "no_change.yml"
    no_change_file.write_text(final_no_change_content)
    
    globalized_no_change_path = globalize_paths(str(no_change_file), str(current_project_root_path))
    assert globalized_no_change_path == str(no_change_file) # No _globalized file should be created

def test_globalize_path():
    """Test globalize_path function."""
    project_dir = os.path.normpath("/new/project/").rstrip(os.sep) + os.sep # Ensure project_dir is normalized for the test
    
    # Test with /calcs/Species/
    path_str = "path_to_calc: /old/path/calcs/Species/mol.xyz\n" # Input string has newline
    expected = "path_to_calc: /new/project/alcs/Species/mol.xyz\n"
    assert globalize_path(path_str, project_dir).replace('\\', '/') == expected.replace('\\', '/')

    # Test with /calcs/TSs/
    path_str = "another_calc: /old/path/calcs/TSs/ts1.log\n"
    expected = "another_calc: /new/project/calcs/TSs/ts1.log\n"
    assert globalize_path(path_str, project_dir).replace('\\', '/') == expected.replace('\\', '/')

    # Test with project_directory field
    path_str = "  project_directory: /old/project/root\n"
    expected = "  project_directory: /new/project/\n"
    assert globalize_path(path_str, project_dir).replace('\\', '/') == expected.replace('\\', '/')

    # Test with no relevant path pattern
    path_str = "just some text\n"
    assert globalize_path(path_str, project_dir) == path_str

    # Test with project_directory already in path (should not modify)
    # This string must exactly match what globalize_path would output if no change is made.
    path_str_already_in_path = f"path_to_calc: {project_dir}calcs/Species/mol.xyz\n"
    assert globalize_path(path_str_already_in_path, project_dir) == path_str_already_in_path

    # Test with a path that is just the path, no key
    path_str_only = "/old/path/calcs/Species/mol.xyz\n"
    expected_only = "/new/project/alcs/Species/mol.xyz\n"
    assert globalize_path(path_str_only, project_dir) == expected_only

    # Test with a path that is just the project directory, no key
    path_str_project_dir_only = "/old/project/root/\n"
    expected_project_dir_only = "/new/project/\n"
    assert globalize_path(path_str_project_dir_only, project_dir) == expected_project_dir_only

    # Test with a path that is already the project directory, no key, no change expected
    path_str_already_globalized = "/new/project/\n"
    assert globalize_path(path_str_already_globalized, project_dir) == path_str_already_globalized

    # Test with no trailing newline in input string
    path_str_no_newline = "path_to_calc: /old/path/calcs/Species/mol.xyz"
    expected_no_newline = "path_to_calc: /new/project/alcs/Species/mol.xyz"
    assert globalize_path(path_str_no_newline, project_dir) == expected_no_newline

    # Test with no trailing newline, already globalized
    path_str_no_newline_already_globalized = f"path_to_calc: {project_dir}calcs/Species/mol.xyz"
    assert globalize_path(path_str_no_newline_already_globalized, project_dir) == path_str_no_newline_already_globalized

    # Test with leading spaces and no key prefix
    path_str_leading_ws = "  /old/path/calcs/Species/mol.xyz\n"
    expected_leading_ws = "  /new/project/alcs/Species/mol.xyz\n"
    assert globalize_path(path_str_leading_ws, project_dir) == expected_leading_ws

    # Test with leading spaces and no key prefix, no newline
    path_str_leading_ws_no_newline = "  /old/path/calcs/Species/mol.xyz"
    expected_leading_ws_no_newline = "  /new/project/alcs/Species/mol.xyz"
    assert globalize_path(path_str_leading_ws_no_newline, project_dir) == expected_leading_ws_no_newline


def test_string_representer():
    """Test string_representer for YAML multiline strings."""
    # This function is typically used by yaml.dump, so we'll test it indirectly
    # by dumping a dictionary with a multiline string.
    data = {"key": "single line string"}
    yaml_str = yaml.dump(data, Dumper=yaml.Dumper, default_flow_style=False)
    assert "single line string" in yaml_str
    assert "|" not in yaml_str # Should not use block literal for single line

    data_multiline = {"description": "This is a\nmultiline\nstring."}
    yaml_str_multiline = to_yaml(data_multiline) # Use to_yaml to apply the custom representer
    # The regex checks for the key, optional chomping indicator and the multiline content
    assert re.search(r"description:\s*\|-?\s*\n\s*This is a\n\s*multiline\n\s*string\.?", yaml_str_multiline) is not None


"""
Bond-length and distance-matrix utilities were removed from `bees.common` since they are not
used by BEES' core pipeline. The associated tests were intentionally removed.
"""

def test_time_lapse():
    """Test time_lapse function."""
    t0 = time.time()
    time.sleep(1.1) # Sleep for a bit to ensure non-zero time
    elapsed_time_str = time_lapse(t0)
    # Check format, can't check exact time due to execution variations
    assert re.match(r'(\d+ days, )?\d{2}:\d{2}:\d{2}', elapsed_time_str) is not None

    # Test with a known time difference for specific formatting
    mock_start_time = time.time() - (2 * 24 * 3600 + 3 * 3600 + 4 * 60 + 5) # 2 days, 03:04:05
    with patch('time.time', return_value=mock_start_time + (2 * 24 * 3600 + 3 * 3600 + 4 * 60 + 5)):
        assert time_lapse(mock_start_time) == "2 days, 03:04:05"

    mock_start_time_no_days = time.time() - (3 * 3600 + 4 * 60 + 5) # 03:04:05
    with patch('time.time', return_value=mock_start_time_no_days + (3 * 3600 + 4 * 60 + 5)):
        assert time_lapse(mock_start_time_no_days) == "03:04:05"


def test_dict_to_str():
    """Test dict_to_str function."""
    test_dict = {
        "key1": "value1",
        "key2": {
            "nested_key1": 123,
            "nested_key2": "abc"
        },
        "key3": True
    }
    expected_str = (
        "key1: value1\n"
        "key2:\n"
        "  nested_key1: 123\n"
        "  nested_key2: abc\n"
        "key3: True\n"
    )
    assert dict_to_str(test_dict) == expected_str

    # Test with empty dict
    assert dict_to_str({}) == ""

    # Test with different level
    test_dict_level = {"outer": {"inner": "val"}}
    expected_str_level = (
        "outer:\n"
        "  inner: val\n"
    )
    assert dict_to_str(test_dict_level, level=0) == expected_str_level

def test_heavy_atom_count_valid_smiles():
    # ethanol: C-C-O = 3 heavy atoms
    assert heavy_atom_count("CCO") == 3


def test_heavy_atom_count_empty_or_invalid_returns_none():
    assert heavy_atom_count("") is None
    assert heavy_atom_count(None) is None
    assert heavy_atom_count("not_a_smiles") is None


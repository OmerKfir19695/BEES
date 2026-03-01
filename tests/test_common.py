""""
Test BEES common module


This module Test the common module which contains functions which are shared across multiple  modules.
To run the tests, use pytest and the command line: pytest -v tests/test_common.py

"""


import pytest
import os

import time
import datetime
import yaml
import numpy as np
from unittest.mock import patch
import re
import math


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
    get_ordinal_indicator,
    get_number_with_ordinal_indicator,
    extremum_list,
    get_extremum_index,
    sum_list_entries,
    sort_two_lists_by_the_first,
    check_that_all_entries_are_in_list,
    key_by_val,
    is_str_float,
    is_str_int,
    clean_text,
    time_lapse,
    dict_to_str,
    timedelta_from_str,
    convert_list_index_0_to_1,
    calculate_arrhenius_rate_coefficient,
    R, # Gas constant
    EA_UNIT_CONVERSION, # Energy unit conversion dictionary
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


def test_get_ordinal_indicator():
    """Test get_ordinal_indicator function."""
    assert get_ordinal_indicator(1) == 'st'
    assert get_ordinal_indicator(2) == 'nd'
    assert get_ordinal_indicator(3) == 'rd'
    assert get_ordinal_indicator(4) == 'th'
    assert get_ordinal_indicator(11) == 'th'
    assert get_ordinal_indicator(13) == 'th'
    assert get_ordinal_indicator(21) == 'st'
    assert get_ordinal_indicator(22) == 'nd'
    assert get_ordinal_indicator(100) == 'th'

def test_get_number_with_ordinal_indicator():
    """Test get_number_with_ordinal_indicator function."""
    assert get_number_with_ordinal_indicator(1) == '1st'
    assert get_number_with_ordinal_indicator(2) == '2nd'
    assert get_number_with_ordinal_indicator(10) == '10th'
    assert get_number_with_ordinal_indicator(12) == '12th'
    assert get_number_with_ordinal_indicator(23) == '23rd'

"""
Bond-length and distance-matrix utilities were removed from `bees.common` since they are not
used by BEES' core pipeline. The associated tests were intentionally removed.
"""

def test_extremum_list():
    """Test extremum_list function."""
    assert extremum_list([1, 2, 3]) == 1
    assert extremum_list([1, 2, 3], return_min=False) == 3
    assert extremum_list([5]) == 5
    assert extremum_list([None, 1, 2, None]) == 1
    assert extremum_list([None, 1, 2, None], return_min=False) == 2
    assert extremum_list([]) is None
    assert extremum_list([None, None]) is None

def test_get_extremum_index():
    """Test get_extremum_index function."""
    assert get_extremum_index([1, 2, 0, 3]) == 2
    assert get_extremum_index([1, 2, 0, 3], return_min=False) == 3
    assert get_extremum_index([5]) == 0
    
    # Test with None values and ensure correct index is returned
    assert get_extremum_index([None, 1, 0, None, 2]) == 2
    assert get_extremum_index([None, 1, 0, None, 2], return_min=False) == 4
    
    assert get_extremum_index([]) is None
    assert get_extremum_index([None, None]) is None
    assert get_extremum_index([10, 20, 0], skip_values=[0]) == 0 # Should skip 0 and find min of [10, 20]
    assert get_extremum_index([None, None, 5, 2, 8], skip_values=[None]) == 3 # Test with initial None values

def test_sum_list_entries():
    """Test sum_list_entries function."""
    assert sum_list_entries([1, 2, 3]) == 6
    assert sum_list_entries([1.0, 2.5, 3.5]) == 7.0
    assert sum_list_entries([1, 2, 3], multipliers=[10, 1, 0.1]) == pytest.approx(12.3)
    assert sum_list_entries([1, None, 3]) is None
    assert sum_list_entries([]) == 0.0 # Sum of empty list is 0.0

def test_sort_two_lists_by_the_first():
    """Test sort_two_lists_by_the_first function."""
    list1 = [3, 1, 2]
    list2 = ['c', 'a', 'b']
    sorted1, sorted2 = sort_two_lists_by_the_first(list1, list2)
    assert sorted1 == [1, 2, 3]
    assert sorted2 == ['a', 'b', 'c']

    list1_with_none = [3, None, 1, 2]
    list2_with_none = ['c', 'x', 'a', 'b']
    sorted1_none, sorted2_none = sort_two_lists_by_the_first(list1_with_none, list2_with_none)
    assert sorted1_none == [1, 2, 3]
    assert sorted2_none == ['a', 'b', 'c'] # 'x' should be removed

    with pytest.raises(InputError, match="Arguments must be lists"):
        sort_two_lists_by_the_first(1, [2])

    with pytest.raises(InputError, match="Entries of list1 must be either floats or integers"):
        sort_two_lists_by_the_first(['a', 1], [1, 2])

    with pytest.raises(InputError, match="Both lists must be the same length"):
        sort_two_lists_by_the_first([1, 2], [3])

def test_check_that_all_entries_are_in_list():
    """Test check_that_all_entries_are_in_list function."""
    assert check_that_all_entries_are_in_list([1, 2, 3], [3, 1, 2]) is True
    assert check_that_all_entries_are_in_list([1, 2, 3], [1, 2]) is False # Different length
    assert check_that_all_entries_are_in_list([1, 2, 3], [1, 2, 4]) is False # Missing entry
    assert check_that_all_entries_are_in_list([], []) is True
    assert check_that_all_entries_are_in_list([1], []) is False # Different length

def test_key_by_val():
    """Test key_by_val function."""
    test_dict = {"a": 1, "b": 2, "c": "value"}
    assert key_by_val(test_dict, 1) == "a"
    assert key_by_val(test_dict, "value") == "c"
    assert key_by_val(test_dict, 2) == "b" 

    with pytest.raises(ValueError, match="Could not find value"):
        key_by_val(test_dict, 99)

def test_is_str_float():
    """Test is_str_float function."""
    assert is_str_float("1.23") is True
    assert is_str_float("1") is True
    assert is_str_float("-0.5") is True
    assert is_str_float("abc") is False
    assert is_str_float("1.2.3") is False
    assert is_str_float(None) is False

def test_is_str_int():
    """Test is_str_int function."""
    assert is_str_int("123") is True
    assert is_str_int("-45") is True
    assert is_str_int("1.0") is False
    assert is_str_int("abc") is False
    assert is_str_int(None) is False

def test_clean_text():
    """Test clean_text function."""
    assert clean_text("  hello world  ") == "hello world"
    assert clean_text("\nhello\nworld\n") == "hello\nworld"
    assert clean_text('"text"') == "text"
    assert clean_text('text,') == "text"
    # This assertion should now pass with the improved clean_text logic
    assert clean_text('  "\nhello world,\n"  ') == "hello world"
    assert clean_text("") == ""
    assert clean_text(' " text " ') == "text" # Test with spaces and quotes

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

def test_timedelta_from_str():
    """Test timedelta_from_str function."""
    assert timedelta_from_str("1hr") == datetime.timedelta(hours=1)
    assert timedelta_from_str("30m") == datetime.timedelta(minutes=30)
    assert timedelta_from_str("15s") == datetime.timedelta(seconds=15)
    assert timedelta_from_str("1hr30m") == datetime.timedelta(hours=1, minutes=30)
    assert timedelta_from_str("1hr30m15s") == datetime.timedelta(hours=1, minutes=30, seconds=15)
    assert timedelta_from_str("24hr") == datetime.timedelta(hours=24)
    assert timedelta_from_str("") == datetime.timedelta(0)
    assert timedelta_from_str("invalid") is None

def test_convert_list_index_0_to_1():
    """Test convert_list_index_0_to_1 function."""
    assert convert_list_index_0_to_1([0, 1, 2]) == [1, 2, 3]
    assert convert_list_index_0_to_1([1, 2, 3], direction=-1) == [0, 1, 2]
    assert convert_list_index_0_to_1((0, 1), direction=1) == (1, 2)

    with pytest.raises(ValueError, match="The resulting list from converting"):
        convert_list_index_0_to_1([0], direction=-1) # Should result in [-1]

def test_calculate_arrhenius_rate_coefficient():
    """Test calculate_arrhenius_rate_coefficient function."""
    # Test with typical values (example from RMG docs or similar)
    A = 1e8 # cm^3/(mol*s)
    n = 0.5
    Ea = 10000 # J/mol
    T = 300 # K
    
    # k = A * T^n * exp(-Ea / (R * T))
    expected_k = A * (T ** n) * math.exp(-1 * (Ea / (R * T)))
    assert calculate_arrhenius_rate_coefficient(A, n, Ea, T, Ea_units='J/mol') == pytest.approx(expected_k)

    # Test with kJ/mol
    Ea_kJ = 10 # kJ/mol
    Ea_J = Ea_kJ * 1e3 # Convert to J/mol
    expected_k_kJ = A * (T ** n) * math.exp(-1 * (Ea_J / (R * T)))
    assert calculate_arrhenius_rate_coefficient(A, n, Ea_kJ, T, Ea_units='kJ/mol') == pytest.approx(expected_k_kJ)

    # Test with kcal/mol
    Ea_kcal = 2.39 # kcal/mol (approx 10 kJ/mol)
    Ea_J_kcal = Ea_kcal * EA_UNIT_CONVERSION['kcal/mol']
    expected_k_kcal = A * (T ** n) * math.exp(-1 * (Ea_J_kcal / (R * T)))
    assert calculate_arrhenius_rate_coefficient(A, n, Ea_kcal, T, Ea_units='kcal/mol') == pytest.approx(expected_k_kcal)

    # Test unsupported units
    with pytest.raises(ValueError, match="Unsupported Ea units"):
        calculate_arrhenius_rate_coefficient(A, n, Ea, T, Ea_units='invalid_unit')


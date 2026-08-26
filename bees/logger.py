
"""
BEES logger module

"""
 
import datetime
import os
import sys
import shutil
import time
from typing import Optional
import logging

from bees.common import VERSION, BEES_PATH, get_git_commit, get_git_branch, time_lapse, dict_to_str

# Initialize a global logger for BEES module.
# This logger can be used directly throughout BEES code once configured.
logger = logging.getLogger('BEES')

# Disable propagation to the root logger to avoid duplicate messages and allow this logger to handle its messages completely.
logger.propagate = False

# Suppress warnings from specific external libraries if they are known to be noisy
logging.getLogger('matplotlib.font_manager').disabled = True


class Logger(object):
    """
    The Bees Logger class.

    This class is responsible for setting up and configuring the global 'bees' logger.
    It should be instantiated once at the very beginning of the BEES application's
    execution (typically in main.py) to configure the logging system.

    It manages console output, a comprehensive main log file, and a dedicated error log file.
  

    Args:
        project_directory (str): The project directory path.
        verbose (Optional[int]): The logging level, optional. 10 - debug, 20 - info, 30 - warning.
                                 ``None`` to avoid logging to file.
        t0 (float): Initial time when the project was spawned, stored as a float (from time.time()).

    Attributes:
        project (str): The project name (derived from project_directory).
        project_directory (str): The project directory path.
        verbose (Optional[int]): The logging level, optional. 10 - debug, 20 - info, 30 - warning.
                                 ``None`` to avoid logging to file.
        t0 (float): Initial time when the project was spawned, stored as a float.
        log_file (str): The path to the log file.
    """

    def __init__(self,
                 project_directory: str,
                 verbose: Optional[int],
                 t0: float, # Changed type hint to float
                 ):

        # NOTE: no class-level "initialized once" guard. A previous guard
        # early-returned on the 2nd Logger() in a process, leaving the instance
        # without self.t0 (and other attrs) -> log_footer() crashed with
        # "'Logger' object has no attribute 't0'" whenever BEES().execute() ran
        # more than once per process. Duplicate handlers are already prevented
        # by _setup_handlers() (it clears existing handlers before adding), so
        # re-initializing per run is safe and correctly re-points handlers at
        # the new project_directory.

        # Derive project name from project_directory for logging purposes
        self.project = os.path.basename(project_directory) 
        self.project_directory = project_directory
        self.t0 = t0
        self.log_file = os.path.join(self.project_directory, 'bees.log') # This seems redundant with main_log_file_path

        # Map the custom verbose level to Python's standard logging levels.
        # If verbose is None, default to INFO for console output.
        self.console_level = verbose if verbose != None else logging.INFO


        # Define full paths for log files, based on project name and directory.
        self.main_log_file_path = os.path.join(self.project_directory, f'{self.project}.log')
        self.error_log_file_path = os.path.join(self.project_directory, f'{self.project}_errors.log')

        # File logging levels
        self.main_file_level = logging.DEBUG  # Main log file logs everything from DEBUG up
        self.error_file_level = logging.ERROR # Error log file logs only ERROR and CRITICAL

        self._setup_handlers()

        # Log the header and initial project info using the configured logger
        self.log_header()
        self.info(f"Logger initialized successfully in: {self.project_directory}")
        self.info(f"Console output level: {logging.getLevelName(self.console_level)}")
        self.info(f"Main log file: {self.main_log_file_path} (level: {logging.getLevelName(self.main_file_level)})")
        self.info(f"Error log file: {self.error_log_file_path} (level: {logging.getLevelName(self.error_file_level)})")
        self.info(f"Project '{self.project}' started at: {datetime.datetime.fromtimestamp(self.t0).strftime('%Y-%m-%d %H:%M:%S')}") # Convert timestamp to datetime object for logging


    def _setup_handlers(self):
        """
        Configures and attaches handlers (console, main file, error file) to the global 'BEES' logger.
        Also handles backing up existing log files before creating new ones.
        """
        os.makedirs(self.project_directory, exist_ok=True)
        while logger.handlers:
            logger.removeHandler(logger.handlers[0])

        # Formatters
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')  
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # --- Console Handler ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # --- Main Log File Handler ---
        if os.path.isfile(self.main_log_file_path):
            self._backup_log_file(self.main_log_file_path, f'{self.project}_main.old')

        main_file_handler = logging.FileHandler(filename=self.main_log_file_path, mode='w')
        main_file_handler.setLevel(self.main_file_level)
        main_file_handler.setFormatter(file_formatter)
        logger.addHandler(main_file_handler)

        # --- Error Log File Handler ---
        if os.path.isfile(self.error_log_file_path):
            self._backup_log_file(self.error_log_file_path, f'{self.project}_errors.old')

        error_file_handler = logging.FileHandler(filename = self.error_log_file_path, mode='w')
        error_file_handler.setLevel(self.error_file_level)
        error_file_handler.setFormatter(file_formatter)
        logger.addHandler(error_file_handler)

        logger.setLevel(min(self.console_level, self.main_file_level, self.error_file_level))
    def _backup_log_file(self, file_path: str, base_name: str):
        """
        Helper method to backup an existing log file before it's overwritten.
        Moves the existing file to a 'log_archive' subdirectory with a timestamp.

        Args:
            file_path (str): The full path to the log file to be backed up.
            base_name (str): A base name for the archived log file (e.g., 'bees_main.old').
        """
        
        log_archive_dir = os.path.join(self.project_directory, 'log_archive')
        os.makedirs(log_archive_dir, exist_ok=True) # Ensure archive directory exists

        local_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_file_name = f'{base_name}.{local_time}.log'
        archived_file_path = os.path.join(log_archive_dir, archived_file_name)

        try:
            shutil.copy(file_path, archived_file_path)
            os.remove(file_path)
            logger.info(f"Backed up old log file '{os.path.basename(file_path)}' to: {archived_file_path}")
        except Exception as e:
            logger.error(f"Failed to backup or remove log file '{file_path}': {e}")


     # --- Wrapper methods for common logging levels, now using standard logging ---

    def debug(self, message: str):
        """Logs a message with DEBUG severity."""
        logger.debug(message)

    def info(self, message: str):
        """Logs a message with INFO severity."""
        logger.info(message)

    def warning(self, message: str):
        """Logs a message with WARNING severity."""
        logger.warning(message)

    def error(self, message: str):
        """Logs a message with ERROR severity."""
        logger.error(message)

    def critical(self, message: str):
        """Logs a message with CRITICAL severity."""
        logger.critical(message)

    def always(self, message: str):
        """
        Logs a message that is intended to be seen almost always,
        by using the INFO level with a distinguishing prefix.
        """
        logger.info(f"ALWAYS: {message}")

    # --- Specialized logging functions from your original code ---

    def log_header(self):
        """
        Output a header to the log.
        This function relies on `VERSION`, `bees_path`, `get_git_commit`, `get_git_branch`
        being available from `bees.common`.
        """
        self.always(f'BEES execution initiated on {time.asctime()}\n\n'
                    f'################################################################\n'
                    f'#                                                              #\n'
                    f'#       Biochemical Engine for Enzymatic modelS (BEES)         #\n'
                    f'#                                                              #\n'
                    f'#                                                              #\n'
                    f'#       Version:{VERSION}{" " * (10 - len(str(VERSION)))}                                     #\n'
                    f'#                                                              #\n'
                    f'################################################################\n\n')

        head, date = get_git_commit(path=BEES_PATH)
        branch_name = get_git_branch(path=BEES_PATH)
        if head and date:
            self.always(
                'The current git HEAD for BEES is:\n'
                f'    {head}\n'
                f'    {date}'
            )
        if branch_name and branch_name != 'main':
            self.always(f'    (running on the {branch_name} branch)\n')
        else:
            self.always('\n')
        self.always(f'Starting project {self.project}')


    def log_max_time_reached(self, max_time: str):
        """
        Log that the maximum run time was reached.

        Args:
            max_time (str): The maximum BEES walltime.
        """
        execution_time = time_lapse(self.t0)
        self.always(f'\n\n\nTotal BEES execution time: {execution_time}')
        self.always('BEES execution terminated because the maximum run time was reached.')
    

    def log_footer(self, success: bool = True): # Added success parameter
        """
        Output a footer to the log.

        Args:
            success (bool): True if the execution was successful, False otherwise.
        """
        execution_time = time_lapse(self.t0)
        try:
            log_size_bytes = os.path.getsize(self.main_log_file_path)
            units = ["B", "KiB", "MiB", "GiB", "TiB"]
            size = float(log_size_bytes)
            unit = units[0]
            for u in units[1:]:
                if size < 1024.0:
                    break
                size /= 1024.0
                unit = u
            self.always(
                f"Main log file size: {size:.2f} {unit} "
                f"({log_size_bytes} bytes)  |  {self.main_log_file_path}"
            )
        except Exception:
            # Avoid hiding real errors during shutdown due to log-size reporting.
            pass
        self.always(f'\n\n\nTotal BEES execution time: {execution_time}')
        if success:
            self.always('BEES execution completed successfully.')
        else:
            self.always('BEES execution terminated with errors.')
        self.always(f'BEES execution terminated on {time.asctime()}\n')

    def log_args(self, schema: dict):
        """Log the full input arguments (debug level)."""
        verbose_map = {10: 'debug', 20: 'info', 30: 'warning', None: 'info'}
        schema_copy = schema.copy()
        schema_copy['verbose'] = verbose_map[schema_copy.get('verbose', 20)]

        self.debug(f'\n\nFull input arguments:\n\n'
                   f'{dict_to_str(schema_copy)}')

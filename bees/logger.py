
"""
BEES logger module

"""
 
import datetime
import os
import sys
import shutil
import time
from typing import Dict, List, Optional, Tuple
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

    def log_species_to_calculate(self,
                                 species_keys: List[int],
                                 species_dict: Dict[int, dict]):
        """
        Report the species to be calculated in the next iteration.
        The 'label' is used for reporting, it is principally the
        BEES species label as entered by the user, or the label determined
        by BEES if it does not contain forbidden characters and is
        descriptive (i.e., NOT "S(1056)").
    
        Args:
            species_keys (List[int]): Entries are bees species indices.
            species_dict (dict): The bees species dictionary.

     
        """
        if len(species_keys):
            self.info('\n\nSpecies to calculate thermodynamic data for:')
            # Handle cases where species_dict might be empty or missing 'object'
            try:
                max_label_length = max([len(spc_dict['label']) for key, spc_dict in species_dict.items() if key in species_keys] + [6])
                # Ensure 'object' and 'molecule' exist before trying to call to_smiles()
                max_smiles_length = max([len(spc_dict['object'].molecule[0].to_smiles()) 
                                        for key, spc_dict in species_dict.items() 
                                        if key in species_keys and 'object' in spc_dict and spc_dict['object'] and hasattr(spc_dict['object'], 'molecule') and spc_dict['object'].molecule
                                        ] + [6])
            except (KeyError, AttributeError):
                self.warning("Could not determine max label/SMILES length for species logging. Missing keys or attributes in species_dict.")
                max_label_length = 20 # Default
                max_smiles_length = 30 # Default

            space1 = ' ' * (max_label_length - len('Label') + 1)
            space2 = ' ' * (max_smiles_length - len('SMILES') + 1)
            self.info(f'Label{space1} SMILES{space2}   Reason for calculating thermo')
            self.info(f'-----{space1} ------{space2}   -----------------------------')
            for key in species_keys:
                spc_dict = species_dict[key]
                try:
                    # Check if 'object' and 'molecule' exist before accessing
                    if 'object' in spc_dict and spc_dict['object'] and hasattr(spc_dict['object'], 'molecule') and spc_dict['object'].molecule:
                        smiles = spc_dict['object'].molecule[0].to_smiles()
                    else:
                        smiles = "N/A" # Fallback if SMILES object is not available
                except (KeyError, AttributeError):
                    smiles = "N/A" # Fallback if SMILES object is not available
                space1 = ' ' * (max_label_length - len(spc_dict.get('label', 'N/A')) + 1)
                space2 = ' ' * (max_smiles_length - len(smiles) + 1)
                reasons = spc_dict.get('reasons', ['No reason provided'])
                one = '1. ' if len(reasons) > 1 else '   '
                self.info(f"{spc_dict.get('label', 'N/A')}{space1} {smiles}{space2} {one}{reasons[0]}")
                for j, reason in enumerate(reasons):
                    if j > 0:
                        self.info(f"{' ' * (max_label_length + max_smiles_length + 4)}{j + 1}. {reasons[j]}")

    

    def log_reactions_to_calculate(self,
                                   reaction_keys: List[int],
                                   reaction_dict: Dict[int, dict]):
        """
        Report reaction rate coefficients to be calculated in the next iteration.
        The reaction 'label' is used for reporting.

        Args:
            reaction_keys (List[int]): Entries are bees reaction indices.
            reaction_dict (dict): The bees reaction dictionary.
        """
        if len(reaction_keys):
            self.info('\n\nReactions to calculate high-pressure limit rate coefficients for:')
            try:
                max_label_length = max([len(rxn_dict['label']) for key, rxn_dict in reaction_dict.items() if key in reaction_keys] + [6])
                max_smiles_length = max([len(rxn_dict['SMILES label']) for key, rxn_dict in reaction_dict.items() if key in reaction_keys] + [6])
            except (KeyError, AttributeError):
                self.warning("Could not determine max label/SMILES length for reactions logging. Missing keys or attributes in reaction_dict.")
                max_label_length = 20
                max_smiles_length = 30
            max_label_length = max(max_label_length, max_smiles_length) # Take the max of both for consistent column width
            space1 = ' ' * (max_label_length - len('Label') + 1)
            self.info(f'Label{space1} Reason for calculating rate coefficient')
            self.info(f'-----{space1} ---------------------------------------')
            for key in reaction_keys:
                rxn_dict = reaction_dict[key]
                space1 = ' ' * (max_label_length - len(rxn_dict.get('label', 'N/A')) + 1)
                self.info(f"\n{rxn_dict.get('label', 'N/A')}{space1} {rxn_dict.get('reasons', 'No reason provided')}")
                self.info(f"{rxn_dict.get('SMILES label', 'N/A')}\n")
                if hasattr(rxn_dict.get('object'), 'family') and rxn_dict['object'].family is not None:
                    label = rxn_dict['object'].family if isinstance(rxn_dict['object'].family, str) \
                        else rxn_dict['object'].family.label
                    self.info(f'RMG family: {label}\n')


    def log_species_summary(self, species_dict: Dict[int, dict]):
        """
        Report species summary.
        Assumes species_dict contains 'label', 'object' (with molecule attribute) and 'reasons' and 'converged'.
    
        Args:
            species_dict (dict): The bees species dictionary.
        """
        if species_dict:
            self.info('\n\n\nSPECIES SUMMARY')
            converged_keys, unconverged_keys = _get_converged_and_unconverged_keys(species_dict)

            try:
                max_label_length = max([len(spc_dict['label']) for spc_dict in species_dict.values()] + [6])
                max_smiles_length = max([len(spc_dict['object'].molecule[0].to_smiles()) 
                                        for spc_dict in species_dict.values() 
                                        if 'object' in spc_dict and spc_dict['object'] and hasattr(spc_dict['object'], 'molecule') and spc_dict['object'].molecule
                                        ] + [6])
            except (KeyError, AttributeError):
                self.warning("Could not determine max label/SMILES length for species summary. Missing keys or attributes in species_dict.")
                max_label_length = 20
                max_smiles_length = 30

            if len(converged_keys):
                self.info('\nSpecies for which thermodynamic data was calculated:\n')
                space1 = ' ' * (max_label_length - len('label') + 1)
                space2 = ' ' * (max_smiles_length - len('SMILES') + 1)
                self.info(f'Label{space1} SMILES{space2} Reason for calculating thermo for this species')
                self.info(f'-----{space1} ------{space2} ----------------------------------------------')
                for key in converged_keys:
                    spc_dict = species_dict[key]
                    try:
                        if 'object' in spc_dict and spc_dict['object'] and hasattr(spc_dict['object'], 'molecule') and spc_dict['object'].molecule:
                            smiles = spc_dict['object'].molecule[0].to_smiles()
                        else:
                            smiles = "N/A"
                    except (KeyError, AttributeError):
                        smiles = "N/A"
                    space1 = ' ' * (max_label_length - len(spc_dict.get('label', 'N/A')) + 1)
                    space2 = ' ' * (max_smiles_length - len(smiles) + 1)
                    self.info(f"{spc_dict.get('label', 'N/A')}{space1} {smiles}{space2} {spc_dict.get('reasons', 'No reason provided')}")
            else:
                self.info('\nNo species thermodynamic calculation converged!')

            if len(unconverged_keys):
                self.info('\nSpecies for which thermodynamic data did not converge:')
                space1 = ' ' * (max_label_length - len('label') + 1)
                space2 = ' ' * (max_smiles_length - len('SMILES') + 1)
                self.info(f'        Label{space1} SMILES{space2} Reason for calculating thermo for this species')
                self.info(f'        -----{space1} ------{space2} ----------------------------------------------')
                for key in unconverged_keys:
                    spc_dict = species_dict[key]
                    try:
                        if 'object' in spc_dict and spc_dict['object'] and hasattr(spc_dict['object'], 'molecule') and spc_dict['object'].molecule:
                            smiles = spc_dict['object'].molecule[0].to_smiles()
                        else:
                            smiles = "N/A"
                    except (KeyError, AttributeError):
                        smiles = "N/A"
                    space1 = ' ' * (max_label_length - len(spc_dict.get('label', 'N/A')) + 1)
                    space2 = ' ' * (max_smiles_length - len(smiles) + 1)
                    self.info(f"(FAILED) {spc_dict.get('label', 'N/A')}{space1} "
                              f"{smiles}{space2} {spc_dict.get('reasons', 'No reason provided')}")
            else:
                self.info('\nAll species calculated by BEES successfully converged')

    def log_reactions_summary(self, reactions_dict: Dict[int, dict]):
        """   
        Report rate coefficient summary.

        Args:
            reactions_dict (dict): The bees reactions dictionary.
        """
        # Check if reactions_dict is empty before proceeding
        
        if reactions_dict:
            self.info('\n\n\nRATE COEFFICIENTS SUMMARY')
            converged_keys, unconverged_keys = _get_converged_and_unconverged_keys(reactions_dict)

            try:
                max_label_length = max([len(rxn_dict['label']) for rxn_dict in reactions_dict.values()] + [6])
            except KeyError:
                self.warning("Could not determine max label length for reactions summary. Missing 'label' in reaction_dict.")
                max_label_length = 20

            if len(converged_keys):
                self.info('\nReactions for which rate coefficients were calculated:\n')
                space1 = ' ' * (max_label_length - len('label') + 1)
                self.info(f'Label{space1} Reason for calculating rate coefficient for this reaction')
                self.info(f'-----{space1} ---------------------------------------------------------')
                for key in converged_keys:
                    rxn_dict = reactions_dict[key]
                    space1 = ' ' * (max_label_length - len(rxn_dict.get('label', 'N/A')) + 1)
                    self.info(f"{rxn_dict.get('label', 'N/A')}{space1} {rxn_dict.get('reasons', 'No reason provided')}")
            else:
                self.info('\nNo reaction rate coefficient calculation converged!')

            if len(unconverged_keys):
                self.info('\nReactions for which rate coefficient calculations were unsuccessful:')
                space1 = ' ' * (max_label_length - len('label') + 1)
                self.info(f'        Label{space1} Reason for calculating rate coefficient for this reaction')
                self.info(f'        -----{space1} ---------------------------------------------------------')
                for key in unconverged_keys:
                    rxn_dict = reactions_dict[key]
                    space1 = ' ' * (max_label_length - len(rxn_dict.get('label', 'N/A')) + 1)
                    self.info(f"(FAILED) {rxn_dict.get('label', 'N/A')}{space1} {rxn_dict.get('reasons', 'No reason provided')}")
            else:
                self.info('\nAll reaction rate coefficients calculations in this iteration successfully converged.')

    def log_unconverged_species_and_reactions(self,
                                              species_keys: List[int],
                                              species_dict: Dict[int, dict],
                                              reaction_keys: List[int],
                                              reaction_dict: Dict[int, dict],
                                              ):
        """
        Report unconverged species and reactions.
    
        Args:
            species_keys (List[int]): Entries are bees species indices.
            species_dict (dict): The bees species dictionary.
            reaction_keys (List[int]): Entries are bees reaction indices.
            reaction_dict (dict): The bees reaction dictionary.
        """
        if len(species_keys):
            self.info('\nThermodynamic calculations for the following species did NOT converge:')
            try:
                max_label_length = max([len(spc_dict['label'])
                                        for key, spc_dict in species_dict.items() if key in species_keys] + [6])
            except KeyError:
                self.warning("Could not determine max label length for species logging. Missing 'label' in species_dict.")
                max_label_length = 20
            space1 = ' ' * (max_label_length - len('label') + 1)
            self.info(f'Label{space1} SMILES')
            self.info(f'-----{space1} ------')
            for key in species_keys:
                spc_dict = species_dict[key]
                label = spc_dict.get('label', 'N/A')
                space1 = ' ' * (max_label_length - len(label))
                try:
                    if 'object' in spc_dict and spc_dict['object'] and hasattr(spc_dict['object'], 'molecule') and spc_dict['object'].molecule:
                        smiles = spc_dict['object'].molecule[0].to_smiles()
                    else:
                        smiles = "N/A"
                except (KeyError, AttributeError):
                    smiles = "N/A"
                self.info(f"{label}{space1} {smiles}")
            self.info('\n')
        elif len(species_dict.keys()):
            self.info('\nAll species thermodynamic calculations in this iteration successfully converged.\n')

        if len(reaction_keys):
            self.info('\nRate coefficient calculations for the following reactions did NOT converge:')
            for key in reaction_keys:
                self.info(reaction_dict[key].get('label', 'N/A'))
            self.info('\n')
        elif len(reaction_dict.keys()):
            self.info('\nAll reaction rate coefficients calculations in this iteration successfully converged.\n')


    def log_args(self, schema: dict):
        """
        Log the full arguments used in BEES. debug level only.
        At INFO level, only a concise summary is shown via log_input_summary.
        """
        verbose_map = {10: 'debug', 20: 'info', 30: 'warning', None: 'info'}
        schema_copy = schema.copy()
        schema_copy['verbose'] = verbose_map[schema_copy.get('verbose', 20)]

        self.debug(f'\n\nFull input arguments:\n\n'
                   f'{dict_to_str(schema_copy)}')

    def log_input_summary(self, schema: dict):
        """
        Log a concise input summary at INFO level.
        Species/enzyme names are listed; long fields (sequences, SMILES) are omitted.
        Full details are available at DEBUG level via log_args.
        """
        species = schema.get("species", [])
        enzymes = schema.get("enzymes", [])

        species_labels = [s.get("label", "?") for s in species]
        enzyme_labels = [
            f"{e.get('label', '?')} ({e.get('ecnumber', '?')})" for e in enzymes
        ]

        self.info(f"Species: {', '.join(species_labels)}")
        self.info(f"Enzymes: {', '.join(enzyme_labels)}")

        # Full dump at debug level only
        self.log_args(schema)



def _get_converged_and_unconverged_keys(object_dict: Dict[int, dict]) -> Tuple[List[int], List[int]]:
    """
    Get converged keys and unconverged keys from a dictionary representing species or reactions.

    Args:
        object_dict (dict): A dictionary representing ``species_dict`` or ``reactions_dict``.

    Returns:
        Tuple[List[int], List[int]]: ``converged_keys`` and ``unconverged_keys``.
    """

    converged_keys, unconverged_keys = list(), list()
    for key, sub_dict in object_dict.items():
        if 'converged' in sub_dict.keys() and sub_dict['converged']:
            converged_keys.append(key)
        else:
            unconverged_keys.append(key)
    return converged_keys, unconverged_keys

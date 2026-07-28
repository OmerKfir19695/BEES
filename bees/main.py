"""
Main BEES application module for kinetic model generation and refinement.

This module provides the BEES class, which orchestrates the kinetic model generation pipeline.
The module processes the YAML input files containing, validates inputs against Pydantic schemas, initializes project
directories and logging, and executes the model generation workflow.

This is probably the most important module in the code.
 run the code by executing this script directly python bees.py --input_file + directory
(decribed in BEES.py as well)
"""


import os
import time
from typing import Any, Dict

import bees.common as common
from bees.logger import Logger
from bees.schema import InputBase
from bees.reaction_generator import ReactionGenerator
from bees.enlarger import IterativeEnlarger
from bees.exporter import EnlargerExporter

# Base paths
BEES_PATH = common.BEES_PATH


class BEES():
    """
    The main BEES application class.
    Orchestrates: setup, schema validation, logging, execution.
    Also, here is where the model generation will be started. right now it is just a placeholder.
    """

    def __init__(self, input_data: Dict[str, Any]):
        self.t0 = time.time()
        self.input_data = input_data
        self.project: str = input_data.get("project", "default_project")

        # 1. Setup Directories
        self._setup_directories()

        # 2. Initialize Logger
        self._setup_logging()

        # 3. Log System Info
        self._log_system_info()

        # 4. Load Chemical Ontology
        self._load_ontology()

        # 5. Validate Schema
        self._validate_input()

    def _setup_directories(self):
        """Determine and create project/output directories."""
        # 1. Determine the base directory for the project
        if "project_directory" in self.input_data:
            self.base_directory = self.input_data["project_directory"]
            if not os.path.isabs(self.base_directory):
                # If relative, it's relative to the installation root
                self.base_directory = os.path.normpath(os.path.join(common.BEES_PATH, self.base_directory))
        else:
            # Default to projects/<project_name>
            self.base_directory = os.path.join(common.BEES_PATH, "projects", self.project)

        # 2. Set project_directory to 'output' inside base_directory
        # This is where logs and output files will go by default
        self.project_directory = os.path.join(self.base_directory, "output")

        specified_output_dir = self.input_data.get("settings", {}).get("output_directory")
        if specified_output_dir:
            if os.path.isabs(specified_output_dir):
                if not specified_output_dir.startswith(self.base_directory):
                    
                    self.output_directory = specified_output_dir
                else:
                    self.output_directory = specified_output_dir
            else:
                self.output_directory = os.path.join(self.base_directory, specified_output_dir)
        else:
            self.output_directory = self.project_directory

        try:
            os.makedirs(self.project_directory, exist_ok=True)
            if self.output_directory != self.project_directory:
                os.makedirs(self.output_directory, exist_ok=True)
        except OSError as e:
            raise FileNotFoundError(f"Failed to create output directory: {e}")

    def _setup_logging(self):
        """Initialize the BEES logger."""
        self.logger = Logger(
            project_directory=self.project_directory,
            verbose=self.input_data.get("settings", {}).get("verbose"),
            t0=self.t0,
        )

    def _log_system_info(self):
        """Log BEES version and Git information."""
        self.logger.info(f"BEES version: {common.VERSION}")
        git_branch = common.get_git_branch()
        commit_hash, commit_date = common.get_git_commit()
        self.logger.info(f"Git branch: {git_branch}")
        self.logger.info(f"Git commit: {commit_hash} ({commit_date})")
        self.logger.log_args(self.input_data)

    def _load_ontology(self):
        """Load and invert the chemical ontology for alias matching."""
        ontology_path = os.path.join(BEES_PATH, "db", "ontology.yaml")
        self.ontology = common.load_and_invert_ontology(ontology_path)
        if self.ontology:
            self.logger.info(f"Loaded chemical ontology from {ontology_path} ({len(self.ontology)} aliases)")

    def _validate_input(self):
        """Validate input data against BEES schema."""
        try:
            self.bees_object = InputBase(**self.input_data)
            self.logger.info("Input validated successfully against the schema.")
            verified_input_path = os.path.join(self.project_directory, "input.yml")
            common.save_yaml_file(
                verified_input_path, self.bees_object.model_dump(exclude_unset=True)
            )
            self.logger.info(f"Saving validated input to {verified_input_path}")
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            self.logger.log_footer(success=False)
            raise ValueError(f"Invalid input parameters provided: {e}")

        self.logger.info(
            f"BEES project {self.project} initialized successfully in {common.time_lapse(self.t0)}."
        )

    def execute(self):
        """
        Execute the BEES kinetic model generation pipeline.
        There are two modes: iterative ( rate-based iterative model enlargement) 
        and batch (reaction network generation only, without kinetic parameters and pruning ).

        Current functionality:
        - Logs project initialization and input summary (species count, enzymes count, temperature, database info)
        - Validates solver configuration and simulation end time settings
       
        
        Returns:
            dict: Execution results (depending on the mode)
        """

        self.logger.info(f"Starting BEES execution for project '{self.project}'...")

        # Log input summary
        reactive_species = [s for s in self.bees_object.species if s.reactive]
        cofactor_species = [s for s in self.bees_object.species if not s.reactive]
        reactive_enzymes = [e for e in self.bees_object.enzymes if e.reactive]
        self.logger.info(
            f"Input: {len(reactive_species)} reactive species, "
            f"{len(cofactor_species)} cofactors, "
            f"{len(reactive_enzymes)} enzymes"
        )
        self.logger.info(f"Environment: T={self.bees_object.environment.temperature} K, "
                        f"pH={self.bees_object.environment.pH}")
        self.logger.info(f"Database: '{self.bees_object.database.name}'")

        # Log estimation settings
        est_enabled = getattr(self.bees_object.settings, "estimate_kinetics", False)
        est_backend = getattr(self.bees_object.settings, "kinetics_estimator", None)
        smiles_mode = getattr(self.bees_object.settings, "smiles_mode", "auto")
        if est_enabled and est_backend:
            self.logger.info(f"Kinetics estimation: {est_backend} (smiles_mode={smiles_mode})")
        elif est_enabled:
            self.logger.info("Kinetics estimation: enabled (no backend specified)")
        else:
            self.logger.debug("Kinetics estimation: disabled")
        if hasattr(self.bees_object, "settings") and hasattr(self.bees_object.settings, "end_time") and self.bees_object.settings.end_time is not None:
            self.logger.info(f"The simulation will run until {self.bees_object.settings.end_time} time units.")
        else:
            self.logger.info("No simulation end time specified (reaction network generation only).")
    
        
        # Resolve DB path from database.name (e.g. db -> db/db.csv)
        db_name = self.bees_object.database.name.strip()
        db_path_from_name = os.path.join(BEES_PATH, "db", f"{db_name}.csv")
        default_db = os.path.join(BEES_PATH, "db", "db.csv")
        db_path = db_path_from_name if os.path.exists(db_path_from_name) else default_db
        if getattr(self.bees_object, "seed_model", None):
            self.logger.info(f"Using seed model: {self.bees_object.seed_model}")
        reaction_generator = ReactionGenerator(
            bees_object=self.bees_object,
            logger=self.logger,
            output_directory=self.output_directory
        )
        
        # Load kinetic database with ontology
        reaction_generator.load_kinetic_database(db_path, ontology=self.ontology)

        # Decide execution mode: iterative (rate-based enlargement) vs batch (discovery only)
        settings = self.bees_object.settings
        _mode = getattr(settings, "simulation_mode", None)
        if _mode == "iterative":
            use_iterative = True
        elif _mode == "batch":
            use_iterative = False
        else:
            # backward-compatible auto-detect: iterative when ANY termination
            # criterion is set (end_time, conversion target, or rate-ratio).
            has_termination = (
                settings.end_time is not None
                or bool(getattr(settings, "termination_conversion", None))
                or getattr(settings, "termination_rate_ratio", None) is not None
            )
            use_iterative = (
                has_termination
                and getattr(settings, "toleranceMoveToCore", 0) > 0
            )

        if use_iterative:
            # ----------------------------------------------------------
            # Rate-based iterative model enlargement (simulator, rates, flux)
            # ----------------------------------------------------------
            self.logger.info("Mode: Rate-based iterative enlargement")
            enlarger = IterativeEnlarger(
                bees_object=self.bees_object,
                reaction_generator=reaction_generator,
                logger=self.logger,
                output_directory=self.output_directory,
            )
            enlarger_result = enlarger.run()

            # Export reactions summary 
            reaction_generator.reactions = (
                list(enlarger_result.model.core_reactions)
                + list(enlarger_result.model.edge_reactions)
            )
            summary_path = None
            if reaction_generator.reactions:
                core_ids = {id(r) for r in enlarger_result.model.core_reactions}
                summary_path = reaction_generator.export_reactions_summary(core_rxn_ids=core_ids)
                self.logger.info(f"Exported reaction summary to: {summary_path}")

            # Export simulation profiles if requested
            profiles_path = None
            if getattr(settings, "save_simulation_profiles", False):
                exporter = EnlargerExporter(
                    model=enlarger_result.model,
                    profiles=enlarger_result.simulation_profiles,
                    output_directory=self.output_directory,
                    logger=self.logger,
                    reaction_id_by_sig=enlarger._reaction_id_by_sig,
                    reaction_first_seen_iter=enlarger._reaction_first_seen_iter,
                    reaction_core_enter_iter=enlarger._reaction_core_enter_iter,
                    reaction_obj_by_sig=enlarger._reaction_obj_by_sig,
                    iteration_summaries=enlarger._iteration_summaries,
                    save_reaction_tree_plots=getattr(settings, "save_reaction_tree_plots", False),
                    save_simulation_plots=getattr(settings, "save_simulation_plots", False),
                    plot_max_species=getattr(settings, "plot_max_species", None),
                    plot_exclude_enzymes=getattr(settings, "plot_exclude_enzymes", True),
                    plot_exclude_cofactors=getattr(settings, "plot_exclude_cofactors", True),
                    reaction_tree_layout=getattr(settings, "reaction_tree_layout", "graphviz"),
                    reaction_tree_rankdir=getattr(settings, "reaction_tree_rankdir", "TB"),
                    reaction_tree_fontsize=int(getattr(settings, "reaction_tree_fontsize", 8) or 8),
                    core_seen_labels=getattr(enlarger, "_core_seen_labels", set()),
                    bees_object=self.bees_object,
                )
                profiles_path = exporter.export_simulation_profiles()

            # Export simulation plots (concentration vs time per iteration) if requested
            if getattr(settings, "save_simulation_plots", False):
                if 'exporter' not in locals():
                    exporter = EnlargerExporter(
                        model=enlarger_result.model,
                        profiles=enlarger_result.simulation_profiles,
                        output_directory=self.output_directory,
                        logger=self.logger,
                        reaction_id_by_sig=enlarger._reaction_id_by_sig,
                        reaction_first_seen_iter=enlarger._reaction_first_seen_iter,
                        reaction_core_enter_iter=enlarger._reaction_core_enter_iter,
                        reaction_obj_by_sig=enlarger._reaction_obj_by_sig,
                        iteration_summaries=enlarger._iteration_summaries,
                        save_reaction_tree_plots=getattr(settings, "save_reaction_tree_plots", False),
                        save_simulation_plots=getattr(settings, "save_simulation_plots", False),
                        plot_max_species=getattr(settings, "plot_max_species", None),
                        plot_exclude_enzymes=getattr(settings, "plot_exclude_enzymes", True),
                        plot_exclude_cofactors=getattr(settings, "plot_exclude_cofactors", True),
                        reaction_tree_layout=getattr(settings, "reaction_tree_layout", "graphviz"),
                        reaction_tree_rankdir=getattr(settings, "reaction_tree_rankdir", "TB"),
                        reaction_tree_fontsize=int(getattr(settings, "reaction_tree_fontsize", 8) or 8),
                        core_seen_labels=getattr(enlarger, "_core_seen_labels", set()),
                        bees_object=self.bees_object,
                    )
                exporter.export_simulation_plots()

            # Export flux analysis
            if 'exporter' not in locals():
                exporter = EnlargerExporter(
                    model=enlarger_result.model,
                    profiles=enlarger_result.simulation_profiles,
                    output_directory=self.output_directory,
                    logger=self.logger,
                    reaction_id_by_sig=enlarger._reaction_id_by_sig,
                    reaction_first_seen_iter=enlarger._reaction_first_seen_iter,
                    reaction_core_enter_iter=enlarger._reaction_core_enter_iter,
                    reaction_obj_by_sig=enlarger._reaction_obj_by_sig,
                    iteration_summaries=enlarger._iteration_summaries,
                    save_reaction_tree_plots=getattr(settings, "save_reaction_tree_plots", False),
                    save_simulation_plots=getattr(settings, "save_simulation_plots", False),
                    plot_max_species=getattr(settings, "plot_max_species", None),
                    plot_exclude_enzymes=getattr(settings, "plot_exclude_enzymes", True),
                    plot_exclude_cofactors=getattr(settings, "plot_exclude_cofactors", True),
                    reaction_tree_layout=getattr(settings, "reaction_tree_layout", "graphviz"),
                    reaction_tree_rankdir=getattr(settings, "reaction_tree_rankdir", "TB"),
                    reaction_tree_fontsize=int(getattr(settings, "reaction_tree_fontsize", 8) or 8),
                    core_seen_labels=getattr(enlarger, "_core_seen_labels", set()),
                    bees_object=self.bees_object,
                )
            flux_path = exporter.export_flux_analysis()
            core_edge_csv_paths = exporter.export_core_edge_reaction_species_csvs()
            sbml_path = exporter.export_sbml()

            execution_time = common.time_lapse(self.t0)
            self.logger.info(
                f"BEES execution for project '{self.project}' completed in {execution_time}."
            )
            self.logger.log_footer(success=True)

            return {
                'success': True,
                'project': self.project,
                'mode': 'iterative',
                'n_reactions': enlarger_result.final_core_reactions,
                'n_core_species': enlarger_result.final_core_species,
                'n_edge_species': enlarger_result.final_edge_species,
                'iterations': enlarger_result.iterations,
                'converged': enlarger_result.converged,
                'convergence_reason': enlarger_result.convergence_reason,
                'execution_time': execution_time,
                'summary_path': summary_path,
                'profiles_path': profiles_path,
                'flux_path': flux_path,
                'core_edge_csv_paths': core_edge_csv_paths,
                'sbml_path': sbml_path,
                'message': (
                    f'Iterative enlargement completed in {enlarger_result.iterations} '
                    f'iteration(s). Core: {enlarger_result.final_core_species} species, '
                    f'{enlarger_result.final_core_reactions} reactions.'
                ),
            }

        else:
            # ----------------------------------------------------------
            # Batch mode (reaction generation only)
            # ----------------------------------------------------------
            self.logger.info("Mode: Batch reaction network generation")

            reactions = reaction_generator.generate_reactions()

            summary_path = None
            if reactions:
                summary_path = reaction_generator.export_reactions_summary()
                self.logger.info(f"Exported reaction summary to: {summary_path}")

            execution_time = common.time_lapse(self.t0)
            self.logger.info(
                f"BEES execution for project '{self.project}' completed in {execution_time}."
            )
            self.logger.log_footer(success=True)

            return {
                'success': True,
                'project': self.project,
                'mode': 'batch',
                'n_reactions': len(reactions),
                'execution_time': execution_time,
                'summary_path': summary_path,
                'sbml_path': None,
                'message': f'Execution completed successfully. Generated {len(reactions)} reaction(s).'
            }
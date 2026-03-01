#!/usr/bin/env python3

"""
Reaction Template Module
------------------------
Determines reaction types and templates based on enzyme EC numbers.
Provides simple product inference for enzymatic reactions.

EC Number Classification:
- EC 1.x.x.x: Oxidoreductases (electron transfer)
- EC 2.x.x.x: Transferases (group transfer)
- EC 3.x.x.x: Hydrolases (hydrolysis)
- EC 4.x.x.x: Lyases (addition/removal without hydrolysis)
- EC 5.x.x.x: Isomerases (intramolecular rearrangement)
- EC 6.x.x.x: Ligases (bond formation with ATP)
- EC 7.x.x.x: Translocases (movement across membranes)

"""

import logging
import re
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

# Logger
logger = logging.getLogger('BEES')


class ECClass(Enum):
    """EC Number main classes."""
    OXIDOREDUCTASE = 1  # Electron transfer reactions
    TRANSFERASE = 2      # Group transfer reactions
    HYDROLASE = 3        # Hydrolysis reactions
    LYASE = 4            # Addition/elimination reactions
    ISOMERASE = 5        # Isomerization reactions
    LIGASE = 6           # Bond formation with ATP cleavage
    TRANSLOCASE = 7      # Translocation reactions
    UNKNOWN = 0


@dataclass
class ReactionTemplate:
    """
    Reaction template describing the type and stoichiometry of a biochemical reaction.
    
    Attributes:
        template_type (str): Type of reaction (e.g., "phosphorylation", "hydrolysis")
        ec_class (ECClass): Main EC class
        reactants (List[str]): List of reactant labels
        products (List[str]): List of product labels (may be inferred)
        stoichiometry (Dict[str, int]): Stoichiometric coefficients (negative=consumed, positive=produced)
        cofactors (List[str]): Required cofactors
        description (str): Human-readable description
        reversible (bool): Whether reaction is reversible
    """
    template_type: str
    ec_class: ECClass
    reactants: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)
    stoichiometry: Dict[str, int] = field(default_factory=dict)
    cofactors: List[str] = field(default_factory=list)
    description: str = ""
    reversible: bool = False
    
    def __repr__(self):
        return (f"ReactionTemplate(type={self.template_type}, class={self.ec_class.name}, "
                f"reactants={self.reactants} → products={self.products})")


def parse_ec_number(ec_number: str) -> Tuple[int, int, int, int]:
    """
    Parse EC number string into components.
    
    Args:
        ec_number (str): EC number in format "EC 1.2.3.4" or "1.2.3.4"
        
    Returns:
        Tuple[int, int, int, int]: (class, subclass, sub-subclass, serial)
        
    Raises:
        ValueError: If EC number format is invalid
    """
    # Remove "EC " prefix if present
    ec_str = ec_number.strip().upper()
    if ec_str.startswith("EC "):
        ec_str = ec_str[3:].strip()
    
    # Parse components
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)\.(\d+)$', ec_str)
    if not match:
        raise ValueError(f"Invalid EC number format: {ec_number}. Expected 'EC X.X.X.X' or 'X.X.X.X'")
    
    return tuple(int(x) for x in match.groups())


def get_ec_class(ec_number: str) -> ECClass:
    """
    Get the main EC class from an EC number.
    
    Args:
        ec_number (str): EC number
        
    Returns:
        ECClass: Main enzyme class
    """
    try:
        main_class, _, _, _ = parse_ec_number(ec_number)
        return ECClass(main_class)
    except (ValueError, KeyError):
        logger.warning(f"Could not determine EC class for {ec_number}")
        return ECClass.UNKNOWN


# Base templates for each EC class
EC_BASE_TEMPLATES: Dict[ECClass, Dict] = {
    ECClass.OXIDOREDUCTASE: {
        "template_type": "oxidoreduction",
        "description": "Oxidoreduction reaction",
        "reversible": True,
    },
    ECClass.TRANSFERASE: {
        "template_type": "group_transfer",
        "description": "Generic group transfer",
        "cofactors": [],
        "reversible": False,
    },
    ECClass.HYDROLASE: {
        "template_type": "hydrolysis",
        "description": "Hydrolysis reaction",
        "reversible": False,
    },
    ECClass.LYASE: {
        "template_type": "elimination",
        "description": "Addition or elimination reaction",
        "reversible": True,
    },
    ECClass.ISOMERASE: {
        "template_type": "isomerization",
        "description": "Intramolecular rearrangement",
        "reversible": True,
    },
    ECClass.LIGASE: {
        "template_type": "ligation",
        "description": "Bond formation coupled to ATP hydrolysis",
        "cofactors": ["ATP", "Mg2+"],
        "reversible": False,
    },
    ECClass.TRANSLOCASE: {
        "template_type": "translocation",
        "description": "Movement across membrane",
        "reversible": False,
    },
}

# Subclass-specific overrides
EC_SUBCLASS_OVERRIDES: Dict[ECClass, Dict[int, Dict]] = {
    ECClass.OXIDOREDUCTASE: {
        1: {
            "template_type": "oxidation_CH-OH",
            "description": "Oxidation of CH-OH group",
        },
        2: {
            "template_type": "oxidation_C=O",
            "description": "Oxidation of aldehyde or ketone",
        },
        3: {
            "template_type": "oxidation_CH-CH",
            "description": "Oxidation of CH-CH group",
        },
    },
    ECClass.TRANSFERASE: {
        7: {
            "template_type": "phosphorylation",
            "description": "Phosphate group transfer from ATP",
            "cofactors": ["ATP", "Mg2+"],
        },
        3: {
            "template_type": "acyl_transfer",
            "description": "Acyl group transfer",
        },
        4: {
            "template_type": "glycosyl_transfer",
            "description": "Glycosyl group transfer",
        },
    },
    ECClass.HYDROLASE: {
        1: {
            "template_type": "ester_hydrolysis",
            "description": "Hydrolysis of ester bonds",
        },
        2: {
            "template_type": "glycoside_hydrolysis",
            "description": "Hydrolysis of glycosidic bonds",
        },
        4: {
            "template_type": "peptide_hydrolysis",
            "description": "Hydrolysis of peptide bonds",
        },
    },
}


def determine_template_from_ec(ec_number: str, substrate: Optional[str] = None) -> ReactionTemplate:
    """
    Determine reaction template based on EC number.
    
    Uses EC classification to infer reaction type. More specific templates
    can be determined by analyzing EC sub-classes.
    
    Args:
        ec_number (str): Enzyme EC number
        substrate (str, optional): Substrate name for more specific inference
        
    Returns:
        ReactionTemplate: Inferred reaction template
    """
    try:
        main_class, sub_class, _, _ = parse_ec_number(ec_number)
        ec_class = ECClass(main_class)
    except (ValueError, KeyError):
        logger.warning(f"Invalid EC number {ec_number}, using generic template")
        return ReactionTemplate(
            template_type="generic",
            ec_class=ECClass.UNKNOWN,
            description="Generic enzymatic reaction"
        )
    
    # Get base template for this EC class, or fall back to generic
    base = EC_BASE_TEMPLATES.get(
        ec_class,
        {
            "template_type": "generic",
            "description": "Generic enzymatic reaction",
            "reversible": False,
        },
    )
    
    # Get subclass override if present
    subclass_overrides = EC_SUBCLASS_OVERRIDES.get(ec_class, {})
    override = subclass_overrides.get(sub_class, {})
    
    # Merge, with subclass override taking precedence
    template_kwargs = {**base, **override, "ec_class": ec_class}
    
    return ReactionTemplate(**template_kwargs)


def _infer_hydrolysis_products(substrate: str, template: ReactionTemplate, cofactor: Optional[str]) -> List[str]:
    """Helper function for hydrolysis product inference."""
    if "glycos" in template.template_type.lower() or "sacchar" in substrate.lower():
        return [f"{substrate} fragments"]
    return [f"{substrate} (hydrolyzed)"]


def _infer_ester_hydrolysis_products(substrate: str, template: ReactionTemplate, cofactor: Optional[str]) -> List[str]:
    """Helper function for ester hydrolysis product inference."""
    if "ATP" in substrate or "ADP" in substrate:
        return ["ADP" if "ATP" in substrate else "AMP", "Pi"]
    return [f"{substrate} (hydrolyzed)"]


def _infer_oxidation_products(substrate: str, template: ReactionTemplate, cofactor: Optional[str]) -> List[str]:
    """Helper function for oxidation product inference."""
    products = [f"{substrate} (oxidized)"]
    if cofactor and "NAD" in cofactor.upper():
        if "NAD+" in cofactor.upper():
            products.extend(["NADH", "H+"])
        elif "NADP+" in cofactor.upper():
            products.extend(["NADPH", "H+"])
        elif "NADH" in cofactor.upper():
            products.extend(["NAD+", "H+"])
        elif "NADPH" in cofactor.upper():
            products.extend(["NADP+", "H+"])
    return products


# Product inference rules by template type
PRODUCT_INFERENCE_RULES: Dict[str, Callable[[str, ReactionTemplate, Optional[str]], List[str]]] = {
    "phosphorylation": lambda s, t, c: [f"{s}-phosphate", "ADP"],
    "hydrolysis": _infer_hydrolysis_products,
    "ester_hydrolysis": _infer_ester_hydrolysis_products,
    "isomerization": lambda s, t, c: [f"{s} (isomer)"],
    "ligation": lambda s, t, c: [f"{s} (ligated)", "ADP", "Pi"],
    "group_transfer": lambda s, t, c: [f"{s} (modified)"],
    "generic": lambda s, t, c: [f"{s} (product)"],
}


def _add_cofactor_products(products: List[str], cofactor: Optional[str], template_type: Optional[str] = None) -> List[str]:
    """
    Add cofactor-derived products to the product list.
    
    Rules:
    - ATP consumed → ADP + Pi produced (except for phosphorylation)
    - NAD+ consumed → NADH + H+ produced
    - NADP+ consumed → NADPH + H+ produced
    - NADH consumed → NAD+ + H+ produced
    - NADPH consumed → NADP+ + H+ produced
    
    Args:
        products: Existing product list
        cofactor: Cofactor name if present
        template_type: Optional template type to refine rules
        
    Returns:
        Updated product list with cofactor products
    """
    if not cofactor or cofactor.lower() in ['none', 'null', '']:
        return products
    
    cofactor_upper = cofactor.upper()
    products_set = {p.upper() for p in products}  # Case-insensitive check
    
    # Handle ATP
    if "ATP" in cofactor_upper:
        if "ADP" not in products_set:
            products.append("ADP")
        # Only add Pi if it's not a phosphorylation reaction
        if template_type != "phosphorylation" and "PI" not in products_set:
            products.append("Pi")
    
    # Handle NAD+/NADH
    if "NAD+" in cofactor_upper:
        if "NADH" not in products_set:
            products.append("NADH")
        if "H+" not in products_set:
            products.append("H+")
    elif "NADH" in cofactor_upper:
        if "NAD+" not in products_set:
            products.append("NAD+")
        if "H+" not in products_set:
            products.append("H+")
    
    # Handle NADP+/NADPH
    if "NADP+" in cofactor_upper:
        if "NADPH" not in products_set:
            products.append("NADPH")
        if "H+" not in products_set:
            products.append("H+")
    elif "NADPH" in cofactor_upper:
        if "NADP+" not in products_set:
            products.append("NADP+")
        if "H+" not in products_set:
            products.append("H+")
    
    return products


def infer_products(
    substrate: str,
    enzyme_label: str,
    template: ReactionTemplate,
    cofactor: Optional[str] = None,
    database_products: Optional[str] = None
) -> List[str]:
    """
    Infer reaction products based on template and substrate.
    
    This is a simple rule-based approach. Future versions will use:
    - SMARTS pattern matching on substrate structure
    - Chemical transformation rules
    - Product prediction ML models
    
    Args:
        substrate (str): Substrate name
        enzyme_label (str): Enzyme name
        template (ReactionTemplate): Reaction template
        cofactor (str, optional): Cofactor if present
        database_products (str, optional): Products from database (if available)
        
    Returns:
        List[str]: Predicted product names
    """
    # If database has products, use them but ensure cofactor products are added
    if database_products:
        products = [p.strip() for p in re.split(r'[+,;]', database_products)]
        products = [p for p in products if p]  # Remove empty strings
        # Add cofactor products if not already present
        products = _add_cofactor_products(products, cofactor, template.template_type)
        return products
    
    # Get inference function for template type
    template_type = template.template_type
    
    # Handle oxidation templates (can start with "oxidation")
    if template_type.startswith("oxidation"):
        products = _infer_oxidation_products(substrate, template, cofactor)
    else:
        # Look up inference function in dictionary
        inference_func = PRODUCT_INFERENCE_RULES.get(
            template_type,
            PRODUCT_INFERENCE_RULES["generic"]
        )
        products = inference_func(substrate, template, cofactor)
    
    # Ensure cofactor products are added (in case they weren't in the inference)
    products = _add_cofactor_products(products, cofactor, template.template_type)
    
    logger.debug(f"Inferred products for {substrate} + {enzyme_label}: {products}")
    return products


def create_reaction_from_database(
    substrate: str,
    enzyme_label: str,
    ec_number: str,
    cofactor: Optional[str] = None,
    database_products: Optional[str] = None
) -> ReactionTemplate:
    """
    Create a complete reaction template with products from database information.
    
    Args:
        substrate (str): Substrate name
        enzyme_label (str): Enzyme name
        ec_number (str): EC number
        cofactor (str, optional): Cofactor name
        database_products (str, optional): Products from database
        
    Returns:
        ReactionTemplate: Complete reaction template with inferred/known products
    """
    # Determine template from EC number
    template = determine_template_from_ec(ec_number, substrate)
    
    # Set reactants
    template.reactants = [substrate]
    
    # Add cofactors that are consumed in the reaction
    # For ligase reactions, ATP is consumed (ATP → ADP + Pi)
    # For phosphorylation reactions, ATP is consumed (ATP → ADP)
    if template.cofactors:
        # Add ATP to reactants if it's in the cofactors list and the reaction consumes it
        if "ATP" in template.cofactors:
            # ATP is consumed in ligation and phosphorylation reactions
            if template.template_type in ["ligation", "phosphorylation"]:
                if "ATP" not in template.reactants:
                    template.reactants.append("ATP")
    
    # Also add cofactor from parameter if provided
    if cofactor and cofactor.lower() not in ['none', 'null', '']:
        if cofactor not in template.reactants:
            template.reactants.append(cofactor)
    
    # Infer products
    template.products = infer_products(
        substrate=substrate,
        enzyme_label=enzyme_label,
        template=template,
        cofactor=cofactor,
        database_products=database_products
    )
    
    # Ensure cofactor products are added based on reactants (not just cofactor parameter)
    # This handles cases where ATP/NAD+/NADPH are in reactants directly
    for reactant in template.reactants:
        if reactant.upper() == "ATP":
            if "ADP" not in [p.upper() for p in template.products]:
                template.products.append("ADP")
            # Only add Pi if it's not a phosphorylation reaction
            if template.template_type != "phosphorylation" and "PI" not in [p.upper() for p in template.products]:
                template.products.append("Pi")
        elif reactant.upper() == "NAD+":
            if "NADH" not in [p.upper() for p in template.products]:
                template.products.append("NADH")
            if "H+" not in [p.upper() for p in template.products]:
                template.products.append("H+")
        elif reactant.upper() == "NADH":
            if "NAD+" not in [p.upper() for p in template.products]:
                template.products.append("NAD+")
            if "H+" not in [p.upper() for p in template.products]:
                template.products.append("H+")
        elif reactant.upper() == "NADP+":
            if "NADPH" not in [p.upper() for p in template.products]:
                template.products.append("NADPH")
            if "H+" not in [p.upper() for p in template.products]:
                template.products.append("H+")
        elif reactant.upper() == "NADPH":
            if "NADP+" not in [p.upper() for p in template.products]:
                template.products.append("NADP+")
            if "H+" not in [p.upper() for p in template.products]:
                template.products.append("H+")
    
    # Create stoichiometry
    for reactant in template.reactants:
        template.stoichiometry[reactant] = -1
    for product in template.products:
        template.stoichiometry[product] = 1
    
    logger.debug(f"Created reaction template: {template}")
    return template





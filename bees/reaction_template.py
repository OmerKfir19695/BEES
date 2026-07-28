#!/usr/bin/env python3

"""Reaction template construction and product inference from EC numbers."""

import logging
import re
from typing import List, Optional, Dict, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger('BEES')


class ECClass(Enum):
    """EC main classification (1=Oxidoreductase … 7=Translocase)."""
    OXIDOREDUCTASE = 1  
    TRANSFERASE = 2      
    HYDROLASE = 3        
    LYASE = 4            
    ISOMERASE = 5        
    LIGASE = 6           
    TRANSLOCASE = 7      
    UNKNOWN = 0


@dataclass
class ReactionTemplate:
    template_type: str
    ec_class: ECClass
    reactants: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)
    stoichiometry: Dict[str, int] = field(default_factory=dict)
    cofactors: List[str] = field(default_factory=list)
    description: str = ""
    reversible: bool = True
    
    def __repr__(self):
        return (f"ReactionTemplate(type={self.template_type}, class={self.ec_class.name}, "
                f"reactants={self.reactants} → products={self.products})")


def parse_ec_number(ec_number: str) -> Tuple[int, int, int, int]:
    """Parse EC number string ("EC 1.2.3.4" or "1.2.3.4") into (class, subclass, sub-subclass, serial)."""
    ec_str = ec_number.strip().upper()
    if ec_str.startswith("EC "):
        ec_str = ec_str[3:].strip()
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)\.(\d+)$', ec_str)
    if not match:
        raise ValueError(f"Invalid EC number format: {ec_number}. Expected 'EC X.X.X.X' or 'X.X.X.X'")
    
    return tuple(int(x) for x in match.groups())


def get_ec_class(ec_number: str) -> ECClass:
    try:
        main_class, _, _, _ = parse_ec_number(ec_number)
        if main_class not in (1, 2, 3, 4, 5, 6, 7):
            logger.warning(f"EC class {main_class} out of range for {ec_number}")
            return ECClass.UNKNOWN
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
    },
    ECClass.HYDROLASE: {
        "template_type": "hydrolysis",
        "description": "Hydrolysis reaction",
    },
    ECClass.LYASE: {
        "template_type": "elimination",
        "description": "Addition or elimination reaction",
    },
    ECClass.ISOMERASE: {
        "template_type": "isomerization",
        "description": "Intramolecular rearrangement",
    },
    ECClass.LIGASE: {
        "template_type": "ligation",
        "description": "Bond formation coupled to ATP hydrolysis",
        "cofactors": ["ATP", "Mg2+"],
    },
    ECClass.TRANSLOCASE: {
        "template_type": "translocation",
        "description": "Movement across membrane",
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


def determine_template_from_ec(ec_number: str) -> ReactionTemplate:
    try:
        main_class, sub_class, _, _ = parse_ec_number(ec_number)
        if main_class not in (1, 2, 3, 4, 5, 6, 7):
            ec_class = ECClass.UNKNOWN
        else:
            ec_class = ECClass(main_class)
    except (ValueError, KeyError):
        logger.warning(f"Invalid EC number {ec_number}, using generic template")
        return ReactionTemplate(
            template_type="generic",
            ec_class=ECClass.UNKNOWN,
            description="Generic enzymatic reaction"
        )
    
    base = EC_BASE_TEMPLATES.get(
        ec_class,
        {
            "template_type": "generic",
            "description": "Generic enzymatic reaction",
        },
    )
    
    subclass_overrides = EC_SUBCLASS_OVERRIDES.get(ec_class, {})
    override = subclass_overrides.get(sub_class, {})
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


# Product inference rules by template type.
# Oxidation types use startswith("oxidation") in infer_products. Others fall back to "generic".
PRODUCT_INFERENCE_RULES: Dict[str, Callable[[str, ReactionTemplate, Optional[str]], List[str]]] = {
    "phosphorylation": lambda s, t, c: [f"{s}-phosphate", "ADP"],
    "hydrolysis": _infer_hydrolysis_products,
    "ester_hydrolysis": _infer_ester_hydrolysis_products,
    "glycoside_hydrolysis": _infer_hydrolysis_products,
    "peptide_hydrolysis": _infer_hydrolysis_products,
    "isomerization": lambda s, t, c: [f"{s} (isomer)"],
    "ligation": lambda s, t, c: [f"{s} (ligated)", "ADP", "Pi"],
    "group_transfer": lambda s, t, c: [f"{s} (modified)"],
    "acyl_transfer": lambda s, t, c: [f"{s} (modified)"],
    "glycosyl_transfer": lambda s, t, c: [f"{s} (modified)"],
    "elimination": lambda s, t, c: [f"{s} (product)"],
    "oxidoreduction": _infer_oxidation_products,
    "translocation": lambda s, t, c: [f"{s} (translocated)"],
    "generic": lambda s, t, c: [f"{s} (product)"],
}


def _add_cofactor_products(
    products: List[str],
    cofactor_source: Optional[Union[str, List[str]]] = None,
    template_type: Optional[str] = None,
) -> List[str]:
    """Add cofactor-derived products: ATP→ADP+Pi, NAD+→NADH+H+, NADP+→NADPH+H+, etc."""
    if not cofactor_source:
        return products

    if isinstance(cofactor_source, str):
        if cofactor_source.lower() in ('none', 'null', ''):
            return products
        sources = [cofactor_source]
        use_substring_match = True  # "NAD+" in "NADP+" for cofactor param
    else:
        sources = [s for s in cofactor_source if s and str(s).lower() not in ('none', 'null', '')]
        use_substring_match = False  # exact match for reactant labels

    products_set = {p.upper() for p in products}

    for src in sources:
        su = str(src).upper()
        if use_substring_match:
            has_atp = "ATP" in su
            # Check NADP before NAD to avoid "NAD+" matching "NADP+"
            has_nadp = "NADP+" in su or "NADPH" in su
            has_nad = (("NAD+" in su or "NADH" in su) and not has_nadp)
        else:
            has_atp = su == "ATP"
            has_nadp = su in ("NADP+", "NADPH")
            has_nad = su in ("NAD+", "NADH")

        if has_atp:
            if "ADP" not in products_set:
                products.append("ADP")
                products_set.add("ADP")
            if template_type != "phosphorylation" and "PI" not in products_set:
                products.append("Pi")
                products_set.add("PI")

        if has_nad and not has_nadp:
            is_oxidized = "NAD+" in su or su == "NAD+"
            if is_oxidized:
                if "NADH" not in products_set:
                    products.append("NADH")
                    products_set.add("NADH")
            else:
                if "NAD+" not in products_set:
                    products.append("NAD+")
                    products_set.add("NAD+")
            if "H+" not in products_set:
                products.append("H+")
                products_set.add("H+")

        if has_nadp:
            is_oxidized = "NADP+" in su or su == "NADP+"
            if is_oxidized:
                if "NADPH" not in products_set:
                    products.append("NADPH")
                    products_set.add("NADPH")
            else:
                if "NADP+" not in products_set:
                    products.append("NADP+")
                    products_set.add("NADP+")
            if "H+" not in products_set:
                products.append("H+")
                products_set.add("H+")

    return products


def infer_products(
    substrate: str,
    enzyme_label: str,
    template: ReactionTemplate,
    cofactor: Optional[str] = None,
    database_products: Optional[str] = None
) -> List[str]:
    """Infer reaction products from template and substrate using rule-based lookup."""
    if database_products:
        products = [p.strip() for p in re.split(r'[+,;]', database_products)]
        products = [p for p in products if p]
        products = _add_cofactor_products(products, cofactor, template.template_type)
        return products

    template_type = template.template_type
    # Handle oxidation templates (can start with "oxidation")
    if template_type.startswith("oxidation"):
        products = _infer_oxidation_products(substrate, template, cofactor)
    else:
        inference_func = PRODUCT_INFERENCE_RULES.get(
            template_type,
            PRODUCT_INFERENCE_RULES["generic"]
        )
        products = inference_func(substrate, template, cofactor)

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
    """Create a reaction template with products from database info or rule-based inference."""
    template = determine_template_from_ec(ec_number)
    template.reactants = [substrate]
    if template.cofactors:
        if "ATP" in template.cofactors and template.template_type in ["ligation", "phosphorylation"]:
            if "ATP" not in template.reactants:
                template.reactants.append("ATP")
    if cofactor and cofactor.lower() not in ['none', 'null', '']:
        if cofactor not in template.reactants:
            template.reactants.append(cofactor)
    template.products = infer_products(
        substrate=substrate,
        enzyme_label=enzyme_label,
        template=template,
        cofactor=cofactor,
        database_products=database_products
    )

    template.products = _add_cofactor_products(
        template.products, template.reactants, template.template_type
    )
    for reactant in template.reactants:
        template.stoichiometry[reactant] = -1
    for product in template.products:
        template.stoichiometry[product] = 1
    
    logger.debug(f"Created reaction template: {template}")
    return template


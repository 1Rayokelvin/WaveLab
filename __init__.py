"""
WaveLab - Optical Field Simulations and Topological Singularity Analysis
==========================================================================

A Python library for electromagnetic field simulations using plane wave expansion.

Public API
----------
Main workflow:
    >>> from wavelab import get_config, setup_experiment
    >>> config = get_config()
    >>> config.source.num_modes = 10000
    >>> expt = setup_experiment(config)
    >>> E, derivs, B = expt.compute_on_op(z=0.0)

Author: Mayank Soni
Year: 2025
"""

__version__ = "0.0.0"

# Core configuration
from .config import (
    Config,
    OpConfig,
    SourceConfig,
    RandomConfig,
    SpectrumConfig,
    get_config,
    load_config
)

# Core Pipeline
from .setup import ExperimentSetup, Beam
from .engine import FieldEngine

# Utilities & Physics Math
from .utils import (
    get_stokes_parameters,
    get_ellipse_parameters,
    decompose_in_polarization_basis,
)

# Singularity Finders
from .singularities import SingularityFinder

def setup_experiment(config: Config) -> FieldEngine:
    """
    High-level wrapper to generate the beam and initialize the computation engine.
    """
    setup = ExperimentSetup(config)
    beam = setup.generate_beam()
    return FieldEngine(beam, config)


__all__ = [
    # Config
    "Config",
    "OpConfig", 
    "SourceConfig",
    "RandomConfig",
    "SpectrumConfig",
    "get_config",
    "load_config",
    
    # Engine Pipeline
    "setup_experiment",
    "ExperimentSetup",
    "Beam",
    "FieldEngine",
    
    # Utilities
    "get_stokes_parameters",
    "get_ellipse_parameters", 
    "decompose_in_polarization_basis",

    # Topologies
    "SingularityFinder",
]
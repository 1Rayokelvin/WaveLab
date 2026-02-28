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
from .version import current_version, current_version_str
__version_info__ = current_version
__version__ = current_version_str

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
from .beam_maker import BeamMaker, Beam
from .engine_maker import FieldEngine

# Utilities & Physics Math
from .utils import (
    get_stokes_params,
    get_pol_ellipse_params,
    decompose_in_basis,
)

# Singularity Finders
from .singularities import SingularityFinder

def setup_experiment(config: Config) -> FieldEngine:
    """
    High-level wrapper to generate the beam and initialize the computation engine
    from config object.
    """
    beam = BeamMaker(config).generate_beam()
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
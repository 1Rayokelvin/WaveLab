"""
config.py
========================

This module defines the configuration data structures for WaveLab experiments.
It employs strict type checking and validation to ensure physical parameters
are valid before computation begins.

It includes serialization support (JSON) and extensive documentation for 
angular spectrum generation.
"""

import json
import warnings
import numpy as np
from dataclasses import dataclass, field, is_dataclass, asdict, fields
from typing import Tuple, List, Dict, Any, Optional, Literal, Callable, Union, Type
from .spectras import spatial_spectras,spectral_spectras

CURRENT_VERSION = 1
# =========================================================================
#                       VALIDATION HELPERS
# =========================================================================

def _check_scalar(val: Any, name: str, dtype: Type, allow_complex: bool = False) -> Any:
    """
    Strictly validates a scalar value against a specific type.
    """
    # 1. Reject complex numbers for real-valued fields
    if not allow_complex and isinstance(val, (complex, np.complex128, np.complex64)):
        raise TypeError(f"Config Error ['{name}']: Expected {dtype.__name__}, got complex '{val}'")
    
    # 2. Strict Integer check (prevent floats being passed as ints)
    if dtype is int:
        if not isinstance(val, (int, np.integer)):
             raise TypeError(f"Config Error ['{name}']: \
                             Expected strict integer, got {type(val).__name__} '{val}'")
    
    # 3. Final conversion/check
    try:
        return dtype(val)
    except (ValueError, TypeError):
        raise TypeError(f"Config Error ['{name}']: \
                        Cannot convert {type(val).__name__} to {dtype.__name__}")


def _coerce_tuple(val: Any, length: int, name: str, dtype: Type = float) -> Tuple:
    """
    Validates sequences, allows list->tuple, but enforces dtype strictly.
    """
    if not hasattr(val, "__iter__") or isinstance(val, (str, bytes)):
        raise TypeError(f"Config Error ['{name}']: Expected a sequence, got {type(val).__name__}")
    
    val_list = list(val)
    if len(val_list) != length:
        raise ValueError(f"Config Error ['{name}']: Expected {length} elements, got {len(val_list)}")

    return tuple(_check_scalar(x, f"{name}[{i}]", dtype, allow_complex=(dtype is complex)) 
                 for i, x in enumerate(val_list))


# =========================================================================
#                      SAVING/LOADING CONFIGS
# =========================================================================
class SerializableConfig:
    """
    Base class providing JSON serialization.
    Can serialize/deserialize Python function references (module + name).
    """
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        out = {}
        if self.__class__.__name__ == "Config":
            out["__version__"] = CURRENT_VERSION

        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = self._serialize_value(value)
        return out

    def _serialize_value(self, val):
        """Recursive helper for serialization."""
        if val is None:
            return None
        elif is_dataclass(val):
            return val.to_dict()
        elif isinstance(val, (complex, np.complex64, np.complex128)):
            return {"__complex__": True, "real": val.real, "imag": val.imag}
        elif callable(val):
            return self._serialize_callable(val)
        elif isinstance(val, (list, tuple, np.ndarray)):
            if isinstance(val, np.ndarray): val = val.tolist()
            return [self._serialize_value(item) for item in val]
        elif isinstance(val, dict):
            return {k: self._serialize_value(v) for k, v in val.items()}
        elif isinstance(val, np.generic): 
            return val.item()
        else:
            return val

    def _serialize_callable(self, func):
        if func is None: return None 

        for cls in [spatial_spectras, spectral_spectras]:
            for attr_name in dir(cls):
                # Avoid unnecessary introspection on non-callables
                if attr_name.startswith("__"): continue
                attr = getattr(cls, attr_name)
                if attr is func:
                    return {
                        "__callable__": True,
                        "type": "builtin",
                        "name": attr_name,
                    }

        return {"__callable__": True, "type": "custom", "name": func.__name__}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Reconstruct configuration from a dictionary."""       
        # 1. Version Check
        if "__version__" in data:
            version = data.pop("__version__")
            if version != CURRENT_VERSION:
                data = cls._migrate(data, version)

        # 2. Strict Typo Checking
        valid_fields = {f.name for f in fields(cls)}
        incoming_fields = set(data.keys())
        unknown = incoming_fields - valid_fields
        if unknown:
            raise ValueError(f"Unknown configuration fields for {cls.__name__}: {unknown}")

        # 3. Explicit Registry
        sub_classes = {
            'op': OpConfig,
            'source': SourceConfig,
            'random': RandomConfig,
            'spectrum': SpectrumConfig,
            'spatial': SpatialConfig,
            'spectral': SpectralConfig
        }

        init_args = {}
        for k, v in data.items():
            if v is None:
                init_args[k] = None
            
            elif k in sub_classes and isinstance(v, dict):
                init_args[k] = sub_classes[k].from_dict(v)
            
            else:
                init_args[k] = cls._deserialize_value(v)
        
        return cls(**init_args)

    @staticmethod
    def _migrate(data: Dict[str, Any], version: int) -> Dict[str, Any]:
        if version > CURRENT_VERSION:
            raise ValueError("Config version is newer than supported.")
        # future migrations go here
        raise ValueError(f"Unsupported config version: {version}")

    @classmethod
    def _deserialize_value(cls, val):
        """Recursive helper for deserialization."""
        if val is None:
            return None
            
        if isinstance(val, dict):
            if val.get("__complex__"):
                return complex(val["real"], val["imag"])
            
            if val.get("__callable__"):
                return cls._deserialize_callable(val)
            
            return {k: cls._deserialize_value(v) for k, v in val.items()}
            
        elif isinstance(val, list):
            return [cls._deserialize_value(item) for item in val]
        
        return val

    @staticmethod
    def custom_callable_fail(*args, **kwargs):
        """Dummy callable as a placeholder for custom functions while deserializing from json."""
        raise RuntimeError(
            "Custom callable(s) not initialized properly! You must redefine it before execution."
            )

    @staticmethod
    def _deserialize_callable(data):
        """Restores functions from spectras.py."""        
        if data.get("type") == "custom":
            warnings.warn(
                f"Custom callable '{data.get('name')}' found in json. "
                "It must be redefined before execution.",
                UserWarning
            )
            # Return the dummy failure function instead of None
            return SerializableConfig.custom_callable_fail
        
        target_name = data.get("name")
        for cls in [spatial_spectras, spectral_spectras]:
            if hasattr(cls, target_name):
                return getattr(cls, target_name)
        
        raise RuntimeError("Deserializing callable failed. Callable must be from spectras.py or have type 'custom'. ")
    
    def save(self, filename: str):
            """Saves the config dataclass as json at filename"""
            with open(filename, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filename: str):
        """Loads json at filename into the config data calss : cls"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
# =========================================================================
#                       CONFIGURATION CLASSES
# =========================================================================

@dataclass(slots=True)
class OpConfig(SerializableConfig):
    """
    Observation Plane Configuration.
    
    Defines the spatial grid where the fields are computed.

    Attributes
    ----------
    spacing : float
        Grid pixel spacing. Must be > 0.
    size : tuple[float, float]
        Physical size of the plane (width, height).
    center : tuple[float, float]
        Center position (x, y) of the plane.
    """
    spacing: float = 0.05
    size: Tuple[float, float] = (10.0, 10.0)
    center: Tuple[float, float] = (0.0, 0.0)

    def validate(self):
        """Performs strict type and value checking."""
        self.spacing = _check_scalar(self.spacing, "op.spacing", float)
        if self.spacing <= 0: 
            raise ValueError("op.spacing must be > 0")
        
        self.size = _coerce_tuple(self.size, 2, "op.size", float)
        if any(d <= 0 for d in self.size): 
            raise ValueError("op.size must be positive")
        
        self.center = _coerce_tuple(self.center, 2, "op.center", float)
    
    def __post_init__(self):
        self.validate()

@dataclass(slots=True)
class SourceConfig(SerializableConfig):
    """
    Light Source Configuration.
    
    Defines the physical properties of the beam before spatial shaping.

    Attributes
    ----------
    intensity_scale : float
        Scalar multiplier for field intensity.
    wavelength : float | List[float]
        Wavelength(s).
    pol_vect : Tuple[complex, complex]
        Jones vector (Ex, Ey) defining base polarization. Will be normalized.
    beam_axis : Tuple[float, float, float]
        Average propagation direction vector (kx, ky, kz).
        Will be normalized, pol_vect is defined in this frame.
    num_modes : int
        Number of plane wave modes used for the Monte Carlo integration.
    angular_support : 'hemisphere' or 'sphere'
        Domain of the angular spectrum representation.
    """
    intensity_scale: float = 1e3
    wavelength: Union[float, List[float]] = 1.0
    pol_vect: Tuple[complex, complex] = (1+0j, 0+0j)
    beam_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    num_modes: int = 6000
    angular_support: Literal["hemisphere", "sphere"] = "hemisphere"

    def validate(self):
        """Performs strict type and value checking."""
        self.intensity_scale = _check_scalar(self.intensity_scale, "source.intensity_scale", float)
        if self.intensity_scale <= 0: 
            raise ValueError("source.intensity_scale must be > 0")
        
        self.num_modes = _check_scalar(self.num_modes, "source.num_modes", int)
        if self.num_modes <= 0: 
            raise ValueError("source.num_modes must be > 0")

        if np.isscalar(self.wavelength):
            self.wavelength = _check_scalar(self.wavelength, "source.wavelength", float)
            if self.wavelength <= 0: 
                raise ValueError("Wavelength must be > 0")
        else:
            if isinstance(self.wavelength, (np.ndarray, list, tuple)):
                self.wavelength = [_check_scalar(w, "source.wavelength", float) for w in self.wavelength]
                if any(w <= 0 for w in self.wavelength): 
                    raise ValueError("All wavelengths must be > 0")
            else:
                 raise TypeError(f"Expected float or list of floats for wavelength, got {type(self.wavelength)}")

        self.pol_vect = _coerce_tuple(self.pol_vect, 2, "source.pol_vect", dtype=complex)
        norm = np.linalg.norm(self.pol_vect)
        if norm < 1e-13:
            raise ValueError('pol_vect must have reasonable norm.')
        self.pol_vect = (self.pol_vect[0]/norm, self.pol_vect[1]/norm)

        # Beam Axis
        axis = _coerce_tuple(self.beam_axis, 3, "source.beam_axis", dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-13: 
            raise ValueError("beam_axis must have resonable norm")
        self.beam_axis = (axis[0]/norm, axis[1]/norm, axis[2]/norm)

        self.angular_support = str(self.angular_support).lower()
        if self.angular_support not in ["hemisphere", "sphere"]:
            raise ValueError("angular_support must be 'hemisphere' or 'sphere'.")

    def __post_init__(self):
        self.validate()

@dataclass(slots=True)
class RandomConfig(SerializableConfig):
    """
    Stochastic Process Configuration.
    
    Controls which aspects of the light field are randomized during generation.

    Attributes
    ----------
    seed : int
        Seed for the random number generator.
    phase : bool
        If True, applies random phase [0, 2pi] to modes.
    pol_rot : bool
        If True, applies random rotation to polarization vector per mode.
    pol_state : bool
        If True, samples fully random polarization states (Poincaré sphere).
        Overrides `pol_rot`.
    complex_amplitude : bool
        If True, applies a complex random normal factor to amplitude per mode.
    """
    seed: int = 24459
    phase: bool = True
    pol_rot: bool = True
    pol_state: bool = False
    complex_amplitude: bool = True

    def validate(self):
        """Performs strict type checking."""
        for f in ["phase", "pol_rot", "pol_state", "complex_amplitude"]:
            val = getattr(self, f)
            if not isinstance(val, (bool, np.bool_)):
                raise TypeError(f"Config Error ['random.{f}']: Expected bool, got {type(val).__name__}")
        
        self.seed = _check_scalar(self.seed, "random.seed", int)

        if self.pol_rot and self.pol_state:
            warnings.warn(
                "RandomConfig: Both 'pol_rot' and 'pol_state' are True. "
                "'pol_state' (Random Poincaré) overrides 'pol_rot'.", 
                UserWarning
            )

    def off(self):
        """
        Turn off all randomness (make deterministic/coherent).
        
        Returns
        -------
        self
        """
        self.phase = False
        self.pol_rot = False
        self.pol_state = False
        self.complex_amplitude = False
        return self

    def __post_init__(self):
        self.validate()

@dataclass(slots=True)
class SpatialConfig(SerializableConfig):
    """
    Configuration for the spatial (k-space) profile of the beam.
    """
    profile: Callable = field(default=spatial_spectras.gaussian)
    params: Dict[str, Any] = field(default_factory=lambda: {"sigma_k_perp": 1.5})
    vectorised: bool = True

    def uniform(self):
        """Set spectral envelope to Uniform (no scaling applied)."""
        self.profile = spatial_spectras.uniform
        self.params = {}
        return self

    def gaussian(self, sigma_k_perp: float = 1.5):
        """Set spatial profile to a Gaussian beam."""
        self.profile = spatial_spectras.gaussian
        self.params = {'sigma_k_perp': sigma_k_perp}
        self.vectorised = True
        return self

    def tophat(self, k_perp_max: float = 1.0):
        """Set spatial profile to a Top-Hat beam."""
        self.profile = spatial_spectras.tophat
        self.params = {'k_perp_max': k_perp_max}
        self.vectorised = True
        return self
        
    def laguerre_gauss(self, *, p: int = 0, l: int = 0, sigma_k_perp: float = 0.5):
        """Set spatial profile to a Laguerre-Gauss mode."""
        if p < 0: raise ValueError("LG index p must be >= 0")
        self.profile = spatial_spectras.laguerre_gauss
        self.params = {'p': p, 'l': l, 'sigma_k_perp': sigma_k_perp}
        self.vectorised = True
        return self
    
    def hermite_gauss(self, *, l: int = 0, m: int = 0, sigma_k_perp: float = 0.5):
        """Set spatial profile to a Hermite-Gauss mode."""
        if l < 0 or m < 0: raise ValueError("HG indices l, m must be >= 0")
        self.profile = spatial_spectras.hermite_gauss
        self.params = {'l': l, 'm': m, 'sigma_k_perp': sigma_k_perp}
        self.vectorised = True
        return self

    def bessel_gauss(self, *, theta_deg: float = 10.0, sigma_theta: float = 0.05, l: int = 0):
        """Set spatial profile to a Bessel-Gauss mode."""
        if theta_deg > 70:
            warnings.warn("Large Bessel cone angles interact strangely with hemisphere clipping.", UserWarning)
        self.profile = spatial_spectras.bessel_gauss
        self.params = {'theta_0': np.radians(theta_deg), 'sigma_theta': sigma_theta, 'l': l}
        self.vectorised = True
        return self

    def custom(self, fn: Callable, vectorised: bool = False, **params):
        """Set a user-defined spatial profile."""
        self.profile = fn
        self.params = params
        self.vectorised = vectorised
        self.validate()
        return self

    def validate(self):
        """Validates the spatial spectrum against dummy k-vectors."""
        if getattr(self.profile, '__name__', '') == 'custom_callable_fail': 
            return
        
        try:
            test_k = np.random.randn(3, 2) if self.vectorised else np.random.randn(3)
            res = self.profile(test_k, **self.params)
            
            if self.vectorised:
                if not hasattr(res, "__len__") or len(res) != 2: 
                    raise ValueError("Vectorised function must return an array matching input shape.")
            else:
                if not np.isscalar(res) and not isinstance(res, complex): 
                    raise ValueError("Non-vectorised function must return a scalar.")
        except Exception as e:
            raise RuntimeError(f"Spatial profile failed validation: {e}")

    def __post_init__(self):
        self.validate()

@dataclass(slots=True)
class SpectralConfig(SerializableConfig):
    """
    Configuration for the spectral (wavelength) envelope of polychromatic beams.
    """
    profile: Callable = field(default=spectral_spectras.uniform)
    params: Dict[str, Any] = field(default_factory=dict)

    def uniform(self):
        """Set spectral envelope to Uniform (no scaling applied)."""
        self.profile = spectral_spectras.uniform
        self.params = {}
        return self

    def gaussian(self, center: float = 8.0, sigma: float = 0.5):
        """Set spectral envelope to a Gaussian distribution."""
        self.profile = spectral_spectras.gaussian
        self.params = {'center': center, 'sigma': sigma}
        return self
    
    def lorentzian(self, center: float = 8.0, gamma: float = 0.5):
        """Set spectral envelope to a Lorentzian distribution."""
        self.profile = spectral_spectras.lorentzian
        self.params = {'center': center, 'gamma': gamma}
        return self

    def tophat(self, center: float = 8.0, width: float = 1.0):
        """Set spectral envelope to a Top-Hat (Rectangular) distribution."""
        self.profile = spectral_spectras.tophat
        self.params = {'center': center, 'width': width}
        return self

    def custom(self, fn: Callable, **params):
        """Set a user-defined spectral envelope."""
        self.profile = fn
        self.params = params
        self.validate()
        return self

    def validate(self):
        """Validates the polychromatic envelope with a test wavelength."""
        if getattr(self.profile, '__name__', '').endswith('fail'): 
            return
        
        try:
            test_wl = 8.0
            result = self.profile(test_wl, **self.params)
            if not np.isscalar(result):
                raise ValueError(f"Spectral envelope must return a scalar. Got {type(result)}.")
            if not np.isfinite(result):
                raise ValueError(f"Spectral envelope returned non-finite value: {result}")
        except Exception as e:
            raise RuntimeError(
                f"Spectral profile failed validation: {e}\n"
                f"Ensure signature is: fn(wavelength: float, **params) -> float"
            ) from e

    def __post_init__(self):
        self.validate()

@dataclass(slots=True)
class SpectrumConfig(SerializableConfig):
    """
    Master Configuration for the beam's Spatial and Spectral characteristics.
    """
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)

    def validate(self):
        self.spatial.validate()
        self.spectral.validate()

@dataclass(slots=True)
class Config(SerializableConfig):
    """
    Main WaveLab Configuration.
    
    Aggregates all sub-configurations and computational settings.

    Attributes
    ----------
    backend : str
        Computational backend. Options: "auto", "numpy", "numba".
    op : OpConfig
        Observation plane settings.
    source : SourceConfig
        Light source settings.
    random : RandomConfig
        Stochastic settings.
    spectrum : SpectrumConfig
        Angular spectrum settings.
    verbose : bool
        If True, prints progress details.
    """
    backend: Literal["auto", "numpy", "numba"] = "auto"
    op: OpConfig = field(default_factory=OpConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    random: RandomConfig = field(default_factory=RandomConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    verbose: bool = False

    def validate(self):
        """
        Cascades validation through all children and checks global consistency.
        """
        # Cascade
        self.op.validate()
        self.source.validate()
        self.random.validate()
        self.spectrum.validate()

        # Backend Check
        self.backend = str(self.backend).lower()
        allowed = ["numba", "numpy", "auto"]
        if self.backend not in allowed:
            raise ValueError(f"Invalid backend: {self.backend}. Must be one of {allowed}")

        # Aliasing Check (Physics check)
        wls = np.atleast_1d(self.source.wavelength)
        min_wl = np.min(wls)
        if self.op.spacing > min_wl / 2:
            warnings.warn(
                f"Spatial aliasing detected. Grid spacing ({self.op.spacing} um) "
                f"is larger than half the minimum wavelength ({min_wl/2:.4f} um).", 
                UserWarning
            )

    def __post_init__(self):
        self.validate()

def get_config() -> Config:
    """
    Factory function to create a default configuration.
    
    Returns
    -------
    Config
        Initialized with default values.
    """
    return Config()

def load_config(filename: str) -> Config:
    """
    Load a WaveLab configuration from a JSON file.
    
    Parameters
    ----------
    filename : str
        Path to the saved JSON configuration file.
        
    Returns
    -------
    Config
        The configuration object.
    """
    return get_config().load(filename)
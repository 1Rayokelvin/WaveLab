"""
config_stuff.py
========================

This module defines the configuration data structures for WaveLab experiments.
It employs strict type checking and validation to ensure physical parameters
are valid before computation begins.

It includes serialization support (JSON) and extensive documentation for 
angular spectrum generation.
"""
"""
Configuration & Experiment Schema
=================================

The foundation of the WaveLab pipeline. This module defines the strict data 
structures (`Config`) used to describe physical parameters of beam, 
and observation plane geometry.

It ensures physical validity via type checking and provides serialization 
(JSON) for experiment reproducibility.

Pipeline Context:
    1. Config (config_stuff.py) -> **START HERE**. Define physics.
    2. Beam   (beam_stuff.py)   -> Pass Config to BeamMaker.
    3. Engine (engine_stuff.py) -> Use Beam + Config for field evaluation.
"""


import json
import warnings
import numpy as np
from dataclasses import dataclass, field, is_dataclass, fields
from typing import Tuple, List, Dict, Any, Optional, Literal, Callable, Union, Type
from .spectra import KSpaceSpectra, PolychromaticSpectra
from .version import current_version

# =========================================================================
#                       VALIDATION HELPERS
# =========================================================================

def _check_scalar(val: Any, name: str, dtype: Type, allow_complex: bool = False) -> Any:
    """
    Strictly validates a scalar value against a specific type.
    """
    if not allow_complex and isinstance(val, (complex, np.complex128, np.complex64)):
        raise TypeError(f"Config Error['{name}']: Expected {dtype.__name__}, got complex '{val}'")
    if dtype is int and not isinstance(val, (int, np.integer)):
        raise TypeError(f"Config Error ['{name}']: Expected strict integer, got {type(val).__name__} '{val}'")
    try:
        return dtype(val)
    except (ValueError, TypeError):
        raise TypeError(f"Config Error ['{name}']: Cannot convert {type(val).__name__} to {dtype.__name__}")


def _coerce_tuple(val: Any, length: int, name: str, dtype: Type = float) -> Tuple:
    """
    Validates sequences, allows list->tuple, but enforces dtype strictly.
    """
    if not hasattr(val, "__iter__") or isinstance(val, (str, bytes)):
        raise TypeError(f"Config Error ['{name}']: Expected a sequence, got {type(val).__name__}")
    
    val_list = list(val)
    if len(val_list) != length:
        raise ValueError(f"Config Error ['{name}']: Expected {length} elements, got {len(val_list)}")
    return tuple(_check_scalar(x, f"{name}[{i}]", dtype, allow_complex=(dtype is complex)) for i, x in enumerate(val_list))


# =========================================================================
#                      SAVING/LOADING CONFIGS
# =========================================================================

class SerializableConfig:
    """Base class providing JSON serialization and generic config features."""
    def to_dict(self) -> Dict[str, Any]:
        out = {}
        if self.__class__.__name__ == "Config":
            out["__version__"] = current_version
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = self._serialize_value(value)
        return out

    def _serialize_value(self, val):
        if val is None: return None
        elif is_dataclass(val): return val.to_dict()
        elif isinstance(val, (complex, np.complex64, np.complex128)): 
            return {"__complex__": True, "real": val.real, "imag": val.imag}
        elif callable(val): return self._serialize_callable(val)
        elif isinstance(val, (list, tuple, np.ndarray)): 
            return[self._serialize_value(i) for i in (val.tolist() if isinstance(val, np.ndarray) else val)]
        elif isinstance(val, dict): return {k: self._serialize_value(v) for k, v in val.items()}
        elif isinstance(val, np.generic): return val.item()
        return val

    def _serialize_callable(self, func):
        if func is None: return None 
        for cls in [KSpaceSpectra, PolychromaticSpectra]:
            for attr_name in dir(cls):
                if attr_name.startswith("__"): continue
                attr = getattr(cls, attr_name)
                if attr is func: 
                    return {
                        "__callable__": True, 
                        "type": "builtin", 
                        "class": cls.__name__,
                        "name": attr_name
                    }
        return {"__callable__": True, "type": "custom", "name": func.__name__}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if "__version__" in data:
            version = tuple(data.pop("__version__"))
            if version != current_version: data = cls._migrate(data, version)
        valid_fields = {f.name for f in fields(cls)}
        incoming_fields = set(data.keys())
        unknown = incoming_fields - valid_fields
        if unknown: raise ValueError(f"Unknown configuration fields for {cls.__name__}: {unknown}")

        sub_classes = {
            'op': OpConfig, 'source': SourceConfig, 'randomize': RandomizeConfig, 
            'k_space': KSpaceConfig, 'polychromatic': PolychromaticConfig
            }
        init_args = {}
        for k, v in data.items():
            if v is None: init_args[k] = None
            elif k in sub_classes and isinstance(v, dict): 
                init_args[k] = sub_classes[k].from_dict(v)
            else: init_args[k] = cls._deserialize_value(v)
        return cls(**init_args)

    @staticmethod
    def _migrate(data: Dict[str, Any], version: int) -> Dict[str, Any]:
        raise ValueError(f"Unsupported config version: {version}")

    @classmethod
    def _deserialize_value(cls, val):
        if val is None: return None
        if isinstance(val, dict):
            if val.get("__complex__"): return complex(val["real"], val["imag"])
            if val.get("__callable__"): return cls._deserialize_callable(val)
            return {k: cls._deserialize_value(v) for k, v in val.items()}
        elif isinstance(val, list): return [cls._deserialize_value(item) for item in val]
        return val

    @staticmethod
    def custom_callable_fail(*args, **kwargs):
        raise RuntimeError(
            "Custom callable(s) not initialized properly! You must redefine it before execution."
            )

    @staticmethod
    def _deserialize_callable(data):
        if data.get("type") == "custom":
            warnings.warn(
                f"Custom callable '{data.get('name')}' found in json. Redefine before execution.", 
                UserWarning
            )
            return SerializableConfig.custom_callable_fail
            
        target_name = data.get("name")
        target_class = data.get("class")
        
        # Routing based on explicit class name
        if target_class == "KSpaceSpectra" and hasattr(KSpaceSpectra, target_name):
            return getattr(KSpaceSpectra, target_name)
        if target_class == "PolychromaticSpectra" and hasattr(PolychromaticSpectra, target_name):
            return getattr(PolychromaticSpectra, target_name)
            
        raise RuntimeError("Deserializing callable failed.")
    
    def save(self, filename: str):
        if not filename.endswith(".json"): raise ValueError("filename must end with '.json'")
        with open(filename, 'w') as f: json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filename: str):
        with open(filename, 'r') as f: data = json.load(f)
        return cls.from_dict(data)


# =========================================================================
#                       CONFIGURATION CLASSES
# =========================================================================

@dataclass(slots=True)
class OpConfig(SerializableConfig):
    """
    Observation Plane Configuration.
    
    Defines the spatial grid where the optical field is evaluated. Access this
    via `config.op`.

    Example:
        `config.op.size = (10.0, 10.0)`
        `config.op.spacing = 0.05`

    Attributes
    ----------
    spacing : float
        Grid pixel spacing. Must be strictly > 0.
    size : tuple[float, float]
        Physical size of the rectangular plane (width, height).
    center : tuple[float, float]
        Center position (x, y) of the plane relative to the global origin (0,0).
    """
    spacing: float = 0.05
    size: Tuple[float, float] = (10.0, 10.0)
    center: Tuple[float, float] = (0.0, 0.0)

    def validate(self):
        self.spacing = _check_scalar(self.spacing, "op.spacing", float)
        if self.spacing <= 0: raise ValueError("op.spacing must be > 0")
        self.size = _coerce_tuple(self.size, 2, "op.size", float)
        if any(d <= 0 for d in self.size): raise ValueError("op.size must be positive")
        self.center = _coerce_tuple(self.center, 2, "op.center", float)
    
    def __post_init__(self):
        self.validate()


@dataclass(slots=True)
class RandomizeConfig(SerializableConfig):
    """
    Stochastic Process Configuration.
    
    Controls which aspects of the light field are randomized during generation
    to simulate partially coherent beams, diffuse light, or specific speckle stats.
    Access this via `config.source.randomize`.

    Example:
        To simulate fully coherent deterministic light:
        `config.source.randomize.off()`

    Attributes
    ----------
    seed : int
        Seed for the random number generator ensuring reproducible fields.
    phase : bool
        If True, applies a random phase uniform in [0, 2pi] to each mode.
    pol_rot : bool
        If True, applies a random rotation to the polarization vector per mode.
    pol_state : bool
        If True, samples fully random polarization states from the Poincaré sphere.
        Overrides `pol_rot`.
    amplitude : bool
        If True, applies a complex random normal factor to the mode amplitude.
    """
    seed: int = 24459
    phase: bool = True
    pol_rot: bool = True
    pol_state: bool = False
    amplitude: bool = True

    def validate(self):
        for f in["phase", "pol_rot", "pol_state", "amplitude"]:
            val = getattr(self, f)
            if not isinstance(val, (bool, np.bool_)):
                raise TypeError(f"Config Error ['randomize.{f}']: Expected bool, got {type(val).__name__}")
        self.seed = _check_scalar(self.seed, "randomize.seed", int)
        if self.pol_rot and self.pol_state:
            warnings.warn(
                "Both 'pol_rot' and 'pol_state' are True. 'pol_state' overrides 'pol_rot'.", 
                UserWarning
                )

    def off(self) -> "RandomizeConfig":
        """Turns off all randomness, making the beam perfectly coherent and deterministic."""
        self.phase = False
        self.pol_rot = False
        self.pol_state = False
        self.amplitude = False
        return self

    def __post_init__(self):
        self.validate()


@dataclass(slots=True)
class KSpaceConfig(SerializableConfig):
    """
    k-Space Spatial Profile Configuration.
    
    Controls how plane wave modes are weighted depending on their wavevector.
    Access this via `config.source.k_space`.

    Use fluent methods to modify the distribution:
        `config.source.k_space.gaussian(sigma_k_perp=2.0)`
        `config.source.k_space.laguerre_gauss(p=0, l=1)`

    Attributes
    ----------
    profile : Callable
        The active profile function (from `KSpaceSpectra` or custom).
    params : Dict[str, Any]
        Keyword arguments dynamically passed into the `profile` callable.
    vectorised : bool
        If True, indicates the callable supports numpy array vectorization.
    """
    profile: Callable = field(default=KSpaceSpectra.gaussian)
    params: Dict[str, Any] = field(default_factory=lambda: {"sigma_k_perp": 1.5})
    vectorised: bool = True

    def uniform(self) -> "KSpaceConfig":
        """Sets k-space profile to Uniform (no scaling applied)."""
        self.profile = KSpaceSpectra.uniform
        self.params = {}
        return self

    def gaussian(self, sigma_k_perp: float = 1.5) -> "KSpaceConfig":
        """Sets k-space profile to a Gaussian envelope."""
        self.profile = KSpaceSpectra.gaussian
        self.params = {'sigma_k_perp': sigma_k_perp}
        self.vectorised = True
        return self

    def tophat(self, k_perp_max: float = 1.0) -> "KSpaceConfig":
        """Sets k-space profile to a Top-Hat sharp cutoff."""
        self.profile = KSpaceSpectra.tophat
        self.params = {'k_perp_max': k_perp_max}
        self.vectorised = True
        return self
        
    def laguerre_gauss(self, *, p: int = 0, l: int = 0, sigma_k_perp: float = 0.5) -> "KSpaceConfig":
        """Sets k-space profile to a Laguerre-Gauss (OAM Vortex) mode."""
        if p < 0: raise ValueError("LG index p must be >= 0")
        self.profile = KSpaceSpectra.laguerre_gauss
        self.params = {'p': p, 'l': l, 'sigma_k_perp': sigma_k_perp}
        self.vectorised = True
        return self
    
    def hermite_gauss(self, *, l: int = 0, m: int = 0, sigma_k_perp: float = 0.5) -> "KSpaceConfig":
        """Sets k-space profile to a Hermite-Gauss mode."""
        if l < 0 or m < 0: raise ValueError("HG indices l, m must be >= 0")
        self.profile = KSpaceSpectra.hermite_gauss
        self.params = {'l': l, 'm': m, 'sigma_k_perp': sigma_k_perp}
        self.vectorised = True
        return self

    def bessel_gauss(self, *, theta_deg: float = 10.0, sigma_theta: float = 0.05, l: int = 0) -> "KSpaceConfig":
        """Sets k-space profile to a Bessel-Gauss mode."""
        if theta_deg > 70: 
            warnings.warn(
                "Large Bessel cone angles interact strangely with hemisphere clipping.",
                UserWarning
                )
        self.profile = KSpaceSpectra.bessel_gauss
        self.params = {'theta_0': np.radians(theta_deg), 'sigma_theta': sigma_theta, 'l': l}
        self.vectorised = True
        return self

    def custom(self, fn: Callable, vectorised: bool = False, **params) -> "KSpaceConfig":
        """
        Sets a user-defined custom k-space profile function.
        
        Expected signature:
            If vectorised=True:  fn(k: np.ndarray[3, N], **params) -> np.ndarray[N]
            If vectorised=False: fn(k: np.ndarray[3], **params) -> complex
        """
        self.profile = fn
        self.params = params
        self.vectorised = vectorised
        self.validate()
        return self

    def validate(self):
        if getattr(self.profile, '__name__', '') == 'custom_callable_fail': return
        try:
            test_k = np.random.randn(3, 2) if self.vectorised else np.random.randn(3)
            res = self.profile(test_k, **self.params)
            if self.vectorised and (not hasattr(res, "__len__") or len(res) != 2): 
                raise ValueError("Vectorised function must return an array matching input shape.")
            elif not self.vectorised and not np.isscalar(res) and not isinstance(res, complex): 
                raise ValueError("Non-vectorised function must return a scalar.")
        except Exception as e:
            raise RuntimeError(f"K space profile failed validation: {e}")

    def __post_init__(self):
        self.validate()


@dataclass(slots=True)
class PolychromaticConfig(SerializableConfig):
    """
    Polychromatic Envelope Configuration.
    
    Controls the spectral (wavelength) distribution of the beam. Access this
    via `config.source.polychromatic`.
    
    Use fluent methods to modify the distribution:
        `config.source.polychromatic.gaussian(center=8.0, sigma=0.5)`

    Attributes
    ----------
    profile : Callable
        The active profile function (from `PolychromaticSpectra` or custom).
    params : Dict[str, Any]
        Keyword arguments passed into the `profile` callable.
    """
    profile: Callable = field(default=PolychromaticSpectra.uniform)
    params: Dict[str, Any] = field(default_factory=dict)

    def uniform(self) -> "PolychromaticConfig":
        """Sets Uniform polychromatic envelope (no scaling applied)."""
        self.profile = PolychromaticSpectra.uniform
        self.params = {}
        return self

    def gaussian(self, center: float = 8.0, sigma: float = 0.5) -> "PolychromaticConfig":
        """Sets Gaussian polychromatic envelope."""
        self.profile = PolychromaticSpectra.gaussian
        self.params = {'center': center, 'sigma': sigma}
        return self
    
    def lorentzian(self, center: float = 8.0, gamma: float = 0.5) -> "PolychromaticConfig":
        """Sets Lorentzian (natural broadening) polychromatic envelope."""
        self.profile = PolychromaticSpectra.lorentzian
        self.params = {'center': center, 'gamma': gamma}
        return self

    def tophat(self, center: float = 8.0, width: float = 1.0) -> "PolychromaticConfig":
        """Sets Top-Hat bandpass polychromatic envelope."""
        self.profile = PolychromaticSpectra.tophat
        self.params = {'center': center, 'width': width}
        return self

    def custom(self, fn: Callable, **params) -> "PolychromaticConfig":
        """
        Sets a user-defined polychromatic envelope.
            
        Expected signature: fn(wavelength, **params) -> float
        """
        self.profile = fn
        self.params = params
        self.validate()
        return self

    def validate(self):
        if getattr(self.profile, '__name__', '').endswith('fail'): return
        try:
            result = self.profile(8.0, **self.params)
            if not np.isscalar(result): raise ValueError(f"Polychromatic envelope must return a scalar. Got {type(result)}.")
            if not np.isfinite(result): raise ValueError(f"Polychromatic envelope returned non-finite value: {result}")
        except Exception as e:
            raise RuntimeError(f"Polychromatic profile failed validation: {e}\nEnsure signature is: fn(wavelength: float, **params) -> float") from e

    def __post_init__(self):
        self.validate()


@dataclass(slots=True)
class SourceConfig(SerializableConfig):
    """
    Light Source Configuration. 
    
    Aggregates all physical parameters defining the initial state of the light
    field, including nested properties like spatial modes and polarization.
    Access this via `config.source`.

    Example:
        `config.source.intensity_scale = 1e3`
        `config.source.wavelength = 0.532`
        `config.source.k_space.gaussian(sigma_k_perp=2.0)`
        `config.source.randomize.off()`

    Attributes
    ----------
    k_space : KSpaceConfig
        Sub-configuration dictating the spatial distribution of momentum (k-vectors).
    polychromatic : PolychromaticConfig
        Sub-configuration dictating the spectral/wavelength envelope.
    randomize : RandomizeConfig
        Sub-configuration dictating random noise/coherence features.
    intensity_scale : float
        Scalar multiplier for total field intensity. Must be > 0.
    wavelength : Union[float, List[float]]
        The physical wavelength(s) of the simulation.
    pol_vect : Tuple[complex, complex]
        Jones vector (E1, E2) defining base polarization, normalized upon init.
    beam_axis : Tuple[float, float, float]
        Average propagation direction vector (kx, ky, kz), normalized upon init.
    num_modes : int
        Number of plane wave modes used for the angular spectrum sum.
    theta_max : float
        Maximum half-cone angle (radians) restricting the generated wavevectors.
    """
    k_space: KSpaceConfig = field(default_factory=KSpaceConfig)
    polychromatic: PolychromaticConfig = field(default_factory=PolychromaticConfig)
    randomize: RandomizeConfig = field(default_factory=RandomizeConfig)
    
    intensity_scale: float = 1e3
    wavelength: Union[float, List[float]] = 1.0
    pol_vect: Tuple[complex, complex] = (1+0j, 0+0j)
    beam_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    num_modes: int = 6000
    theta_max: float = np.pi/2

    def validate(self):
        self.intensity_scale = _check_scalar(self.intensity_scale, "source.intensity_scale", float)
        if self.intensity_scale <= 0: raise ValueError("source.intensity_scale must be > 0")
        
        self.num_modes = _check_scalar(self.num_modes, "source.num_modes", int)
        if self.num_modes <= 0: raise ValueError("source.num_modes must be > 0")

        if np.isscalar(self.wavelength):
            self.wavelength = _check_scalar(self.wavelength, "source.wavelength", float)
            if self.wavelength <= 0: raise ValueError("Wavelength must be > 0")
        else:
            if isinstance(self.wavelength, (np.ndarray, list, tuple)):
                self.wavelength =[_check_scalar(w, "source.wavelength", float) for w in self.wavelength]
                if any(w <= 0 for w in self.wavelength): raise ValueError("All wavelengths must be > 0")
            else:
                 raise TypeError(f"Expected float or list of floats for wavelength, got {type(self.wavelength)}")

        self.pol_vect = _coerce_tuple(self.pol_vect, 2, "source.pol_vect", dtype=complex)
        norm = np.linalg.norm(self.pol_vect)
        if norm < 1e-13: raise ValueError('pol_vect must have reasonable norm.')
        self.pol_vect = (self.pol_vect[0]/norm, self.pol_vect[1]/norm)

        axis = _coerce_tuple(self.beam_axis, 3, "source.beam_axis", dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-13: raise ValueError("beam_axis must have reasonable norm")
        self.beam_axis = (axis[0]/norm, axis[1]/norm, axis[2]/norm)
        
        self.theta_max = _check_scalar(self.theta_max, "source.theta_max", float)
        if not 0 < self.theta_max < np.pi:
            raise ValueError("For fibonacci sampling, maximum angle must be in (0, pi).")
        
        self.k_space.validate()
        self.polychromatic.validate()
        self.randomize.validate()
        
    def __post_init__(self):
        self.validate()


@dataclass(slots=True)
class Config(SerializableConfig):
    """
    Main WaveLab Configuration.
    
    This is the root tree for initializing an experiment. It manages the global 
    backend, the observation plane settings, and the complex light source properties.

    Example:
        `config = get_config()`
        `config.op.size = (20.0, 20.0)`
        `config.source.k_space.gaussian(sigma_k_perp=1.0)`
        `config.source.randomize.off()`

    Attributes
    ----------
    backend : Literal["auto", "numpy", "numba"]
        Computational backend to execute evaluations. Defaults to "auto".
    op : OpConfig
        Settings for the Observation Plane (grid size, center, resolution).
    source : SourceConfig
        Settings for the Light Source (beam axis, wavelength, spatial/spectral profiles).
    verbose : bool
        If True, prints execution details and deep warnings.
    """
    backend: Literal["auto", "numpy", "numba"] = "auto"
    op: OpConfig = field(default_factory=OpConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    verbose: bool = False

    def validate(self):
        self.op.validate()
        self.source.validate()

        self.backend = str(self.backend).lower()
        allowed = ["numba", "numpy", "auto"]
        if self.backend not in allowed:
            raise ValueError(f"Invalid backend: {self.backend}. Must be one of {allowed}")

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
        Initialized with standard defaults.
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
        The deserialized configuration object.
    """
    return get_config().load(filename)

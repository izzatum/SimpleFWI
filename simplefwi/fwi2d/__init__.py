"""SI PyTorch acoustic FWI; legacy simplefwi exports remain separate."""
from .types import Acquisition, BoundaryConfig, Grid2D, SolveReport, SolverConfig, validate_velocity
from .datasets import load_marmousi
from .api import AcousticFWI
from .objective import FWIObjective
from .parameters import VelocityParameters
from .inversion import InversionRunner, InversionResult, StepResult
from .regularization import Regularizer
from .curvature import Curvature, TransformedCurvature
from .recording import ArrayPayload, Recorder, RecordingPolicy
from .integration import AcceptedStateRecorder
from .checkpoint import capture_checkpoint, save_checkpoint, load_checkpoint, restore_checkpoint
from .planning import StageRequest, StagePlan, PlanningError, plan_stage
from .dispersion import DispersionReport, assess_dispersion
from .experiments import camembert_config, nested_surface_acquisition, run_experiment
from .optimizers import OptimizerRecipe, make_optimizer, validate_optimizer
from .uncertainty import LocalGaussian, PriorPrecision, SmoothingPrecision, CGConfig, PrecisionError

__all__ = [
    'AcousticFWI','Acquisition','BoundaryConfig','Grid2D','SolveReport','SolverConfig',
    'validate_velocity','load_marmousi','FWIObjective','VelocityParameters',
    'InversionRunner','InversionResult','StepResult','Regularizer','Curvature','TransformedCurvature',
    'ArrayPayload','Recorder','RecordingPolicy','AcceptedStateRecorder','capture_checkpoint',
    'save_checkpoint','load_checkpoint','restore_checkpoint','StageRequest','StagePlan',
    'PlanningError','plan_stage','DispersionReport','assess_dispersion','camembert_config','nested_surface_acquisition','run_experiment',
    'OptimizerRecipe','make_optimizer','validate_optimizer','LocalGaussian','PriorPrecision',
    'SmoothingPrecision','CGConfig','PrecisionError',
]

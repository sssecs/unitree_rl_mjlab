from mjlab.envs.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
from .motion_library import *  # noqa: F401, F403

# Backward-compatible runtime extension for EgoDex+PICO kneeling datasets.
# It patches the motion-library class used by SparseWholeBodyCommand before any
# environment instance is built, while leaving the original public API intact.
from .kneel_extensions import activate_kneel_extensions

activate_kneel_extensions()

from .kneel_motion_library import *  # noqa: F401, F403,E402
from .kneel_rewards import *  # noqa: F401, F403,E402

import chex 
import optax
import collections
from acme import types

@chex.dataclass
class Trajectory:
    observations: types.NestedArray  # [T, B, ...]
    actions: types.NestedArray  # [T, B, ...]
    rewards: chex.ArrayNumpy  # [T, B]
    dones: chex.ArrayNumpy  # [T, B]
    discounts: chex.ArrayNumpy  # [T, B]c


@chex.dataclass
class LearnerState:
    params: chex.Array
    opt_state: optax.OptState

Transition = collections.namedtuple("Transition", field_names=["obs_tm1", "action_tm1", "reward_t", "discount_t", "obs_t", "done"])

import dm_env
import sdepy 
from sdepy import *
from typing import *
import multiprocessing as mp


def _validate_timestep(ts: dm_env.TimeStep) -> dm_env.TimeStep:
    """Some timesteps may come with rewards or discounts set to None, we
  replace such values by 0. (resp 1.)."""
    if ts.reward is None or ts.discount is None:
        ts = ts._replace(reward=0., discount=1.)
    return ts


def actor_target(environment, controller_handle: mp.connection.Connection,
                 max_steps: Optional[int] = None) -> None:
    
    ts = environment.reset()

    ts = _validate_timestep(ts)

    controller_handle.send(ts)

    steps = 0
    while True:
        if max_steps is not None and steps >= max_steps:
            break
        action = controller_handle.recv()
        ts = environment.step(action.tolist())
        if ts.last():
            reset_ts = environment.reset()
            ts = ts._replace(observation=reset_ts.observation)

        ts = _validate_timestep(ts)

        controller_handle.send(ts)
        steps += 1

@integrate
def my_process(t, x, theta=0.15, k=-1, sigma=0.2):
    return {'dt': k*(theta - x), 'dw': sigma}  
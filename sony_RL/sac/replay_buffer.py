from collections import deque
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from segment_tree import SumTree, MinTree

class NStepBuffer:
    """
    Buffer for calculating n-step returns.
    """

    def __init__(
        self,
        gamma=0.99,
        nstep=3,
    ):
        self.discount = [gamma ** i for i in range(nstep)]
        self.nstep = nstep
        self.state = deque(maxlen=self.nstep)
        self.action = deque(maxlen=self.nstep)
        self.reward = deque(maxlen=self.nstep)

    def append(self, state, action, reward):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)

    def get(self):
        assert len(self.reward) > 0

        state = self.state.popleft()
        action = self.action.popleft()
        reward = self.nstep_reward()
        return state, action, reward

    def nstep_reward(self):
        reward = np.sum([r * d for r, d in zip(self.reward, self.discount)])
        self.reward.popleft()
        return reward

    def is_empty(self):
        return len(self.reward) == 0

    def is_full(self):
        return len(self.reward) == self.nstep

    def __len__(self):
        return len(self.reward)


class ReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(
        self,
        buffer_size,
        state_space,
        action_space,
        gamma,
        nstep,
    ):
        #assert len(state_space.shape) in (1, 3)

        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.nstep = nstep
        self.state_shape = state_space.shape
        self.use_image = len(self.state_shape) >= 3

        if self.use_image:
            # Store images as a list of LazyFrames, which uses 4 times less memory.
            self.state = [None] * buffer_size
            self.next_state = [None] * buffer_size
        else:
            self.state = np.empty((buffer_size, *state_space.shape), dtype=np.float32)
            self.next_state = np.empty((buffer_size, *state_space.shape), dtype=np.float32)

        '''if type(action_space) == Box:
            self.action = np.empty((buffer_size, *action_space.shape), dtype=np.float32)
        elif type(action_space) == Discrete:
            self.action = np.empty((buffer_size, 1), dtype=np.int32)
        else:
            NotImplementedError'''
        self.action = np.empty((buffer_size, *action_space.shape), dtype=np.float32)
        self.reward = np.empty((buffer_size, 1), dtype=np.float32)
        self.done = np.empty((buffer_size, 1), dtype=np.float32)

        if nstep != 1:
            self.nstep_buffer = NStepBuffer(gamma, nstep)

    def append(self, state, action, reward, done, next_state, episode_done=None):

        if self.nstep != 1:
            self.nstep_buffer.append(state, action, reward)

            if self.nstep_buffer.is_full():
                state, action, reward = self.nstep_buffer.get()
                self._append(state, action, reward, done, next_state)

            if done or episode_done:
                while not self.nstep_buffer.is_empty():
                    state, action, reward = self.nstep_buffer.get()
                    self._append(state, action, reward, done, next_state)

        else:
            self._append(state, action, reward, done, next_state)

    def _append(self, state, action, reward, done, next_state):
        self.state[self._p] = state
        self.action[self._p] = action
        self.reward[self._p] = float(reward)
        self.done[self._p] = float(done)
        self.next_state[self._p] = next_state

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def _sample_idx(self, batch_size):
        return np.random.randint(low=0, high=self._n, size=batch_size)

    def _sample(self, idxes):
        if self.use_image:
            state = np.empty((len(idxes), *self.state_shape))
            next_state = state.copy()
            for i, idx in enumerate(idxes):
                state[i, ...] = self.state[idx]
                next_state[i, ...] = self.next_state[idx]
        else:
            state = self.state[idxes]
            next_state = self.next_state[idxes]

        return (
            state,
            self.action[idxes],
            self.reward[idxes],
            self.done[idxes],
            next_state,
        )

    def sample(self, batch_size):
        idxes = self._sample_idx(batch_size)
        batch = self._sample(idxes)
        # Use fake weight to use the same interface with PER.
        weight = np.ones((), dtype=np.float32)
        return weight, batch


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Replay Buffer.
    """

    def __init__(
        self,
        buffer_size,
        state_space,
        action_space,
        gamma,
        nstep,
        alpha=0.6,
        beta=0.4,
        beta_steps=10 ** 5,
        min_pa=0.0,
        max_pa=1.0,
        eps=0.01,
    ):
        super(PrioritizedReplayBuffer, self).__init__(
            buffer_size=buffer_size,
            state_space=state_space,
            action_space=action_space,
            gamma=gamma,
            nstep=nstep,
        )

        self.alpha = alpha
        self.beta = beta
        self.beta_diff = (1.0 - beta) / beta_steps
        self.min_pa = min_pa
        self.max_pa = max_pa
        self.eps = eps
        self._cached_idxes = None

        tree_size = 1
        while tree_size < buffer_size:
            tree_size *= 2
        self.tree_sum = SumTree(tree_size)
        self.tree_min = MinTree(tree_size)

    def _append(self, state, action, reward, next_state, done):
        # Assign max priority when stored for the first time.
        self.tree_min[self._p] = self.max_pa
        self.tree_sum[self._p] = self.max_pa
        super()._append(state, action, reward, next_state, done)

    def _sample_idx(self, batch_size):
        total_pa = self.tree_sum.reduce(0, self._n)
        rand = np.random.rand(batch_size) * total_pa
        idxes = [self.tree_sum.find_prefixsum_idx(r) for r in rand]
        self.beta = min(1.0, self.beta + self.beta_diff)
        return idxes

    def sample(self, batch_size):
        assert self._cached_idxes is None, "Update priorities before sampling."

        self._cached_idxes = self._sample_idx(batch_size)
        weight = self._calculate_weight(self._cached_idxes)
        batch = self._sample(self._cached_idxes)
        return weight, batch

    def _calculate_weight(self, idxes):
        min_pa = self.tree_min.reduce(0, self._n)
        weight = [(self.tree_sum[i] / min_pa) ** -self.beta for i in idxes]
        weight = np.array(weight, dtype=np.float32)
        return np.expand_dims(weight, axis=1)

    def update_priority(self, abs_td):
        assert self._cached_idxes is not None, "Sample batch before updating priorities."
        assert abs_td.shape[1:] == (1,)
        pa = np.array(self._calculate_pa(abs_td), dtype=np.float32).flatten()
        for i, idx in enumerate(self._cached_idxes):
            self.tree_sum[idx] = pa[i]
            self.tree_min[idx] = pa[i]
        self._cached_idxes = None

    @partial(jax.jit, static_argnums=0)
    def _calculate_pa(self, abs_td: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip((abs_td + self.eps) ** self.alpha, self.min_pa, self.max_pa)

class RolloutBuffer:
    """
    Rollout Buffer.
    """

    def __init__(
        self,
        buffer_size,
        state_space,
        action_space,
    ):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size

        self.state = np.empty((buffer_size, *state_space.shape), dtype=np.float32)
        self.reward = np.empty((buffer_size, 1), dtype=np.float32)
        self.done = np.empty((buffer_size, 1), dtype=np.float32)
        self.log_pi = np.empty((buffer_size, 1), dtype=np.float32)
        self.next_state = np.empty((buffer_size, *state_space.shape), dtype=np.float32)
        self.action = np.empty((buffer_size, *action_space.shape), dtype=np.float32)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.state[self._p] = state
        self.action[self._p] = action
        self.reward[self._p] = float(reward)
        self.done[self._p] = float(done)
        self.log_pi[self._p] = float(log_pi)
        self.next_state[self._p] = next_state

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def get(self):
        return (
            self.state,
            self.action,
            self.reward,
            self.done,
            self.log_pi,
            self.next_state,
        )
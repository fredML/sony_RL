import os
from datetime import timedelta
from time import sleep, time

import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter


class Trainer:
    """
    Trainer.
    """

    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        action_repeat=1,
        num_agent_steps=10 ** 6,
        eval_interval=10 ** 4,
        num_eval_episodes=10,
        save_params=False,
        save_interval=10**4
    ):
        assert num_agent_steps % action_repeat == 0
        assert eval_interval % action_repeat == 0

        # Envs.
        self.env = env
        self.env_test = env_test

        # Algorithm.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": [], "num_steps":[]}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.param_dir = os.path.join(log_dir, "param")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "summary"))

        # Other parameters.
        self.action_repeat = action_repeat
        self.num_agent_steps = num_agent_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.save_params = save_params
        self.save_interval = save_interval

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Initialize the environment.
        _,_,_, state = self.env.reset()

        for step in tqdm(range(1, self.num_agent_steps + 1)):
            state = self.algo.step(self.env, state)

            if self.algo.is_update():
                self.algo.update(self.writer)

            if step % self.eval_interval == 0:
                self.evaluate(self.algo.agent_step)
            
            if step % self.save_interval == 0:
                if self.save_params:
                    self.algo.save_params(os.path.join(self.param_dir, f"step{step}"))

        # Wait for the logging to be finished.
        sleep(2)

    def evaluate(self, step):
        total_return = 0.0
        num_steps = []
        for _ in range(self.num_eval_episodes):
            _,_,_, state = self.env_test.reset()
            for k in range (10):
                action = self.algo.select_action(state)
                #state, reward, done, _ = self.env_test.step(action)
                done, reward, _, state = self.env_test.step(action)
                done = done == 2
                if done:
                    num_steps.append(k)
                    break
            
            total_return += self.env_test.total_reward

        # Log mean return.
        mean_return = total_return / self.num_eval_episodes
        num_steps = np.array(num_steps)
        if len(num_steps) > 0:
            mean_num_steps = np.mean(num_steps)
        
        else:
            mean_num_steps = -1

        # To TensorBoard.
        self.writer.add_scalar("test/return", mean_return, step * self.action_repeat)
        self.writer.add_scalar("test/num_steps", mean_num_steps, step * self.action_repeat)
        # To CSV.
        self.log["step"].append(step * self.action_repeat)
        self.log["return"].append(mean_return)
        self.log["num_steps"].append(mean_num_steps)
        #pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to standard output.
        #print(f"Num steps: {step * self.action_repeat:<6}   Return: {mean_return:<5.1f}   Time: {self.time}")

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))

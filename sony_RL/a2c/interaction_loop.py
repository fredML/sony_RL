import multiprocessing as mp
from copy import deepcopy
import numpy as np
import tree
from a2c_utils import actor_target, my_process
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

def interaction_loop(agent, make_env, max_learner_steps, replay, batch_size, num_actors):
 
    plots = dict(
                actions_mean=[],
                value_loss=[],
                policy_loss=[],
                value_mean=[],
                value_target_mean=[],
                mean_reward=[],
                bootstrapped_q=[],
                )

    environment = make_env()
    parent_pipes, children_pipes = zip(*[mp.Pipe(duplex=True) for _ in range(num_actors)])
    actors = [mp.Process(target=actor_target, args=(environment, pipe, max_learner_steps)) for  pipe in children_pipes]
    for actor in actors:
        actor.start()

    compt = 0
    for learner_step in tqdm(range(max_learner_steps)):
        ts = tree.map_structure(lambda *x: np.stack(x, axis=0), *[pipe.recv() for pipe in parent_pipes])
        if learner_step>0:
            for i in range (len(ts.observation)):
                replay.add(obs_tm1=timestep.observation[i],
                        action_tm1=actions[i],
                        reward_t=ts.reward[i],
                        discount_t=ts.discount[i],
                        obs_t=ts.observation[i],
                        done = (ts.step_type[i] == 2))
            if len(replay._memory) >= batch_size:
                transitions = replay.sample_batch(min(batch_size,len(replay._memory)))
                logs = agent.learner_step(transitions)
                compt +=1
                for name in logs.keys():
                    if name in plots.keys():
                        plots[name].append(logs[name])
                
                if (compt % 50 == 0) & (compt > 0):
                    clear_output(wait  = True)
                
                    for key in plots.keys():
                        plt.figure(figsize=(15,5))
                        if key.startswith('actions'):
                            plt.plot(np.array(plots[key])[:,0],label='Average theta move per epoch')
                            plt.plot(np.array(plots[key])[:,1],label='Average phi move per epoch')
                        else:
                            plt.plot(plots[key],label='Average '+ key + ' per epoch')
                        plt.legend()
                        plt.show()

        actions = agent.actor_step(ts.observation)

        actions = actions + my_process(x0=0,paths=1)(np.linspace(0,1,actions.shape[0]))
        for i, pipe in enumerate(parent_pipes):
            pipe.send(actions[i])
        timestep = deepcopy(ts)

    for actor in actors:
        actor.join()

    return plots
import numpy as np
import pickle
import os

from src import MountainCar, Config


def GenerateTrajectories(initial_seed=0, num_trajectories=100, num_evaluations=1000, verbose=False):

    config = Config()
    config.norm_state = True
    epsilon = 0.1
    gamma = 0.99

    trajectories = []
    seeds = []
    for i in range(num_trajectories):
        print("Currently working on seed {0}...".format(i + initial_seed))
        np.random.seed(i + initial_seed)
        # initialize environment
        env = MountainCar(config)
        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_terminations = []
        # generate trajectory
        terminal = False
        curr_s = env.get_current_state()
        trajectory_states.append(curr_s)
        while not terminal:
            curr_a = pumping_action(curr_s, rprob=epsilon)
            curr_s, reward, terminal = env.step(curr_a)
            trajectory_states.append(curr_s)
            trajectory_actions.append(curr_a)
            trajectory_rewards.append(reward)
            trajectory_terminations.append(terminal)
        env.reset()
        # compute estimated discounted returns for each state in the trajectory using monte carlo rollouts
        sample_discounted_returns = np.zeros(len(trajectory_states), dtype=np.float64)
        for j in range(len(trajectory_states) - 1):
            sample_discounted_returns[j] += estimate_expected_discounted_return(env=env, init_s=trajectory_states[j],
                                                                                init_a=trajectory_actions[j],
                                                                                epsilon=epsilon, gamma=gamma,
                                                                                samples=num_evaluations)
        if verbose:
            for j in range(len(trajectory_states)):
                print("Step: {0}\tState: {1}\tEstimated Return: {2}".format(j+1, trajectory_states[j],
                                                                            sample_discounted_returns[j]))
        trajectories.append(list(zip(trajectory_states, trajectory_actions, trajectory_rewards, trajectory_terminations,
                                     sample_discounted_returns)))
        seeds.append(i+initial_seed)

    trajectories_path = os.path.join(os.getcwd(), 'MountainCar_Trajectories.p')
    if os.path.isfile(trajectories_path):
        with open(trajectories_path, mode='rb') as trajectories_file:
            trajectories_dict = pickle.load(trajectories_file)
        for i in range(len(seeds)):
            if seeds[i] not in trajectories_dict['seeds']:
                trajectories_dict['seeds'].append(seeds[i])
                trajectories_dict['trajectories'].append(trajectories[i])
    else:
        trajectories_dict = {
            'seeds': seeds,
            'trajectories': trajectories
        }

    with open(trajectories_path, mode='wb') as trajectories_file:
        pickle.dump(trajectories_dict, trajectories_file)


def pumping_action(s: np.ndarray, rprob=0.1):
    assert s.size == 2
    p = np.random.rand()
    if p > rprob:
        return 1 + np.sign(s[1])
    else:
        return np.random.randint(low=0, high=3)


def estimate_expected_discounted_return(env: MountainCar, init_s: np.ndarray, init_a: int, epsilon=0.1, gamma=0.99,
                                        samples=50000):
    assert init_s.size == 2
    discounted_returns = np.zeros(samples, dtype=np.float)
    for i in range(samples):
        env.set_state(init_s, normalized=True)
        current_discounted_return = 0
        curr_gamma = 1.0
        terminal = False
        curr_a = init_a
        while not terminal:
            curr_s, curr_reward, terminal = env.step(curr_a)
            curr_a = pumping_action(curr_s, epsilon)
            current_discounted_return += curr_gamma * curr_reward
            curr_gamma *= gamma
        discounted_returns[i] += current_discounted_return
    return np.average(discounted_returns)


if __name__ == '__main__':
    GenerateTrajectories(initial_seed=0, num_trajectories=130, num_evaluations=1000, verbose=False)

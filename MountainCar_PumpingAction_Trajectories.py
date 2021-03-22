import numpy as np
import pickle
import os
import time
import argparse

from src import MountainCar, Config

NUM_TRANSITIONS = 200001


def GenerateTrajectories(seed=0, num_evaluations=1000, verbose=False):

    data_dir = os.path.join(os.getcwd(), 'mountain_car_prediction_data_{0}evaluations'.format(num_evaluations))
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, 'seed'+str(seed)+'.p')

    config = Config()
    config.norm_state = True
    epsilon = 0.1
    gamma = 0.99
    num_states = 2

    print("Currently working on seed {0}...".format(seed))
    np.random.seed(seed)
    # initialize environment
    env = MountainCar(config)
    states = np.zeros((NUM_TRANSITIONS, num_states), dtype=np.float64)
    actions = np.zeros(NUM_TRANSITIONS, dtype=np.int32)
    rewards = np.zeros(NUM_TRANSITIONS, dtype=np.int8)
    terminations = np.zeros(NUM_TRANSITIONS, dtype=np.int8)

    # generate trajectory
    states[0] = env.get_current_state()
    curr_a = pumping_action(states[0])
    actions[0] = np.int32(curr_a)
    rewards[0] = np.int8(0)
    terminations[0] = np.int8(0)
    i = 1
    while i < NUM_TRANSITIONS:
        next_s, next_reward, next_term = env.step(curr_a)
        next_a = pumping_action(next_s, rprob=epsilon)

        states[i] += next_s
        actions[i] += np.int32(next_a)
        rewards[i] += np.int8(next_reward)
        terminations[i] += np.int8(next_term)

        curr_a = next_a
        i += 1

        if next_term:
            env.reset()
            states[i] = env.get_current_state()
            curr_a = pumping_action(states[i])
            actions[i] = np.int32(curr_a)
            rewards[i] = np.int8(0)
            terminations[i] = np.int8(0)
            i += 1

    # compute estimated discounted returns for each state in the trajectory using monte carlo rollouts
    avg_discounted_returns = np.zeros(NUM_TRANSITIONS, dtype=np.float64)
    ste_discounted_returns = np.zeros(NUM_TRANSITIONS, dtype=np.float64)
    for j in range(NUM_TRANSITIONS):
        if terminations[j] != 1:
            avg, ste = estimate_expected_discounted_return(env=env, init_s=states[j], init_a=actions[j],
                                                           epsilon=epsilon, gamma=gamma, samples=num_evaluations)
            avg_discounted_returns[j] += avg
            ste_discounted_returns[j] += ste

    if verbose:
        for j in range(NUM_TRANSITIONS):
            print("Step: {0}\tState: {1}\tEstimated Return: {2}\tStandard Error {3}".format(j+1,
                                    states[j], avg_discounted_returns[j], ste_discounted_returns[j]))

    trajectory_dict = {
        'states': states, 'actions': actions, 'rewards': rewards, 'terminations': terminations,
        'avg_discounted_return': avg_discounted_returns, 'ste_discounted_returns': ste_discounted_returns
    }

    with open(data_path, mode='wb') as trajectories_file:
        pickle.dump(trajectory_dict, trajectories_file)


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
    avg = np.average(discounted_returns)
    ste = np.std(discounted_returns, ddof=1) / np.sqrt(samples)
    return avg, ste


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', action='store', type=int, default=0)
    parser.add_argument('-ne', '--num_evaluations', action='store', type=int, default=30)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parameters = parser.parse_args()

    start_time = time.time()
    GenerateTrajectories(seed=parameters.seed, num_evaluations=parameters.num_evaluations, verbose=parameters.verbose)
    end_time = time.time()
    total_running_time = (end_time - start_time) / 60
    print("The total running time was: {0}".format(total_running_time))

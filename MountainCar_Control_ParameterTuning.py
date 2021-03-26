import numpy as np
import pickle
import os
import time
import argparse

from src import MountainCar, Config, LinearFunctionApproximator, RadialBasisFunction
from src import IDBD, Adam, AutoStep, SGD, SIDBD

DEBUG = False


class Experiment:

    def __init__(self, exp_arguments, results_path, tunable_parameter_values):

        self.results_path = results_path
        self.verbose = exp_arguments.verbose
        self.tunable_parameter_values = tunable_parameter_values
        self.stepsize_method = exp_arguments.stepsize_method
        self.config = Config()

        """ Feature Function Setup """
        self.config.state_dims = 2          # number of dimension in mountain car
        self.config.state_lims = np.array(((-1,1), (-1,1)), dtype=np.float64)   # state bounds in mountain car
        self.config.initial_centers = np.array(((0,0),(.25,.25),(.25,-.25),(-.25,-.25),(-.25,.25)), dtype=np.float64)
        self.config.sigma = 0.5
        self.config.init_noise_mean = 0.0
        self.config.init_noise_var = 0.01
        self.feature_function = RadialBasisFunction(self.config)    # stays constant regardless of the parameter value

        """ Environment and Policy Setup """
        self.num_actions = 3                # number of actions in mountain car
        self.config.norm_state = True
        self.config.num_obs_features = self.config.initial_centers.shape[0]        # number of initial centers
        self.config.max_num_features = self.config.initial_centers.shape[0] + 1    # arbitrary since features are fixed
        self.training_data = exp_arguments.training_data_size
        self.epsilon = 0.1                  # reasonable choice in mountain car environment
        self.gamma = 0.99                   # discount factor
        self.checkpoint = 5000
        assert self.training_data % self.checkpoint == 0

        """ Experiment Setup """
        self.sample_size = exp_arguments.sample_size

        """ Stepsize adaptation settings"""
        self.config.parameter_size = self.config.num_obs_features
        if self.stepsize_method == 'idbd':
            # non-tunable parameters
            self.config.init_beta = np.log(0.001)
            self.parameter_name = 'meta_stepsize'
            self.stepsize_method_class = IDBD
        elif self.stepsize_method == 'sidbd':
            # non-tunable parameters
            self.config.init_beta = -np.log((1/0.001) - 1)  # equivalent to starting with a stepsize of 0.001
            self.parameter_name = 'meta_stepsize'
            self.stepsize_method_class = SIDBD
        elif self.stepsize_method in ['adam', 'slow_adam']:
            # non-tunable parameters
            self.config.beta1 = 0.9 if self.stepsize_method == 'adam' else 0.0
            self.config.beta2 = 0.99
            self.config.eps = 1e-08
            self.parameter_name = 'initial_stepsize'
            self.stepsize_method_class = Adam
            self.config.restart_ma = False
        elif self.stepsize_method == 'autostep':
            # non-tunable parameters
            self.config.tau = 10000.0
            self.config.init_stepsize = 0.001
            self.parameter_name = 'meta_stepsize'
            self.stepsize_method_class = AutoStep
        elif self.stepsize_method in ['sgd', 'rescaled_sgd']:
            # non-tunable parameters
            self.parameter_name = 'stepsize'
            self.stepsize_method_class = SGD
            self.config.rescale = (self.stepsize_method == 'rescaled_sgd')
        else:
            raise ValueError("Unrecognized stepsize adaptation method.")

    def _print(self, astring):
        if self.verbose:
            print(astring)

    def set_tunable_parameter_value(self, val):
        if self.stepsize_method in ['idbd', 'sidbd']:
            self.config.theta = val
        elif self.stepsize_method in ['adam', 'slow_adam']:
            self.config.init_alpha = val
        elif self.stepsize_method == 'autostep':
            self.config.mu = val
        elif self.stepsize_method in ['sgd', 'rescaled_sgd']:
            self.config.alpha = val
        else:
            raise ValueError("Unrecognized stepsize adaptation method.")

    def run(self):
        np.random.seed(0)
        results = {'parameter_name': self.parameter_name,
                   'sample_size': self.sample_size,
                   'parameter_values': self.tunable_parameter_values,
                   'avg_return': np.zeros(len(self.tunable_parameter_values)),
                   'avg_episodes_per_run': np.zeros(len(self.tunable_parameter_values))}

        for j, pv in enumerate(self.tunable_parameter_values):
            self._print("Parameter value: {0}".format(pv))

            self.set_tunable_parameter_value(pv)
            avg_return_per_run = np.zeros(self.sample_size, dtype=np.float64)
            episodes_per_run = np.zeros(self.sample_size, dtype=np.int32)

            for i in range(self.sample_size):
                self._print("\tRun number: {0}".format(i+1))
                env = MountainCar(self.config)
                approximators = []
                stepsize_method = []
                for _ in range(self.num_actions):
                    approximators.append(LinearFunctionApproximator(self.config))
                    stepsize_method.append(self.stepsize_method_class(self.config))

                avg_return_per_checkpoint = np.zeros(self.training_data // self.checkpoint, dtype=np.float64)

                return_per_episode = []
                curr_checkpoint = 0
                current_return = 0.0

                # initial features and action
                curr_s = env.get_current_state()
                curr_obs_feats = self.feature_function.get_observable_features(curr_s)  # current observable features
                for k in range(self.training_data):
                    # get current action values
                    curr_avs = self.get_action_values(curr_obs_feats, approximators)    # current action values
                    curr_a = self.epsilon_greedy_policy(curr_avs)                       # current action
                    # execute action
                    next_s, r, term = env.step(curr_a)                     # r = reward
                    # get next observable features
                    next_obs_feats = self.feature_function.get_observable_features(next_s)
                    # get next action values and action
                    next_avs = self.get_action_values(next_obs_feats, approximators)
                    next_a = self.epsilon_greedy_policy(next_avs)
                    # compute the TD error for Sarsa(0)
                    td_error = r + self.gamma * (1 - term) * next_avs[next_a] - curr_avs[curr_a]
                    # update weight vector
                    _, _, new_weights = stepsize_method[curr_a].update_weight_vector(td_error, curr_obs_feats,
                                                                             approximators[curr_a].get_weight_vector())
                    # store new weights
                    approximators[curr_a].update_weight_vector(new_weights)
                    # update feature and action information, and keep track of progress
                    curr_obs_feats = next_obs_feats
                    current_return += r
                    # handle cases where weights diverge
                    if np.sum(np.isnan(new_weights)) > 0 or np.sum(np.isinf(new_weights)) > 0:
                        print("The weights diverged!")
                        avg_return_per_checkpoint[curr_checkpoint:] -= self.checkpoint
                        break
                    # check if terminal state
                    if term:
                        # store summaries
                        episodes_per_run[i] += 1
                        return_per_episode.append(current_return)
                        current_return *= 0.0
                        # reset environment
                        env.reset()
                        curr_s = env.get_current_state()
                        curr_obs_feats = self.feature_function.get_observable_features(curr_s)

                    if (k + 1) % self.checkpoint == 0:
                        if len(return_per_episode) == 0:
                            avg_return_per_checkpoint[curr_checkpoint] -= self.checkpoint
                        else:
                            avg_return_per_checkpoint[curr_checkpoint] += np.average(return_per_episode)
                            return_per_episode = []
                        curr_checkpoint += 1

                if DEBUG:
                    import matplotlib.pyplot as plt
                    x = np.arange(self.training_data // self.checkpoint)
                    plt.plot(x, avg_return_per_checkpoint)
                    plt.show()
                    plt.close()

                avg_return_per_run[i] = np.average(avg_return_per_checkpoint)
                self._print("\t\tAverage Return per Run: {0:.4f}".format(avg_return_per_run[i]))
                self._print("\t\tEpisodes Completed: {0}".format(episodes_per_run[i]))

            results['avg_return'][j] += np.average(avg_return_per_run)
            results['avg_episodes_per_run'][j] += np.average(episodes_per_run)
            self._print("Average Return: {0:.4f}".format(results['avg_return'][j]))
            self._print("Average Episodes Completed: {0:.4f}".format(results['avg_episodes_per_run'][j]))

        self.store_results(results)

    def get_action_values(self, features, approximators):
        action_values = np.zeros(self.num_actions, dtype=np.float64)
        for k in range(self.num_actions):
            action_values[k] += approximators[k].get_prediction(features)
        return action_values

    def epsilon_greedy_policy(self, action_values:np.ndarray):
        p = np.random.rand()
        if p > self.epsilon:
            argmax_av = np.random.choice(np.flatnonzero(action_values == action_values.max()))
            return argmax_av
        else:
            return np.random.randint(self.num_actions)

    def store_results(self, results):
        file_path = os.path.join(self.results_path, 'parameter_tuning_results.p')
        with open(file_path, mode='wb') as results_file:
            pickle.dump(results, results_file)
        print("Results successfully stored.")


def main():
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ss', '--sample_size', action='store', default=1, type=int)
    parser.add_argument('-tds', '--training_data_size', action='store', default=200000, type=int)
    parser.add_argument('-ssm', '--stepsize_method', action='store', default='sgd', type=str,
                        choices=['sgd', 'adam', 'slow_adam', 'idbd', 'autostep', 'rescaled_sgd', 'sidbd'])
    parser.add_argument('-tpv', '--tunable_parameter_values', action='store', nargs='+', type=float, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_parameters = parser.parse_args()

    task_name = 'mountain_car_task'
    results_path = os.path.join(os.getcwd(), 'results', 'parameter_tuning', task_name, exp_parameters.stepsize_method)
    os.makedirs(results_path, exist_ok=True)

    init_time = time.time()
    exp = Experiment(exp_parameters, results_path, exp_parameters.tunable_parameter_values)
    exp.run()
    finish_time = time.time()
    elapsed_time = (finish_time - init_time) / 60
    print("Running time in minutes: {0:.4f}".format(elapsed_time))


if __name__ == '__main__':
    main()

    # for centers: np.array(((0,0),(.25,.25),(.25,-.25),(-.25,-.25),(-.25,.25)), dtype=np.float64) with N(0,0.01) noise
    # SGD parameter values: stepsize in {0.25 0.2 0.1 0.05 0.03 0.01 0.0075 0.005 0.003 0.001}
    #   - 0.0075 completed the most episodes during the whole training period
    # Adam parameter values: initial stepsize in {0.06 0.05 0.04 0.03 0.02 0.01 0.009 0.008 0.007 0.005}
    #   - 0.02 completed the most episodes during the whole training period
    # Adam without momentum (called slow_adam in the code) values: initial stepsize in
    #   {0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01}
    #   - 0.03 completed the most episodes during the whole training period
    # TIDBD parameter values: theta in {0.045 0.04 0.035 0.03 0.025 0.02 0.015 0.01 0.005 0.001}
    #   - 0.02 completed the most episodes during the whole training period, but diverged when adding new features
    #     we still use it for the add features experiment
    # AutoStep parameter values: mu in {0.05 0.04 0.03 0.02 0.01 0.009 0.008 0.007 0.006 0.005}
    #   - 0.009 completed the most episodes during the whole training period
    # Rescaled SGD parameter values: stepsize in {0.1 0.05 0.03 0.02 0.01 0.009 0.008 0.007 0.005 0.001}
    #   - 0.01 completed the most episodes during the whole training period
    # STIDBD parameter values: theta in {0.05 0.04 0.035 0.03 0.025 0.02 0.015 0.01 0.007 0.005}
    #   - 0.02 completed the most episodes during the whole training period

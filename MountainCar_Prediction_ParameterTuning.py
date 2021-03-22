import numpy as np
import pickle
import os
import time
import argparse

from src import Config, LinearFunctionApproximator, RadialBasisFunction
from src import IDBD, Adam, AutoStep, SGD, SIDBD

DEBUG = False


class Experiment:

    def __init__(self, exp_arguments, results_path, tunable_parameter_values):

        self.results_path = results_path
        self.verbose = exp_arguments.verbose
        self.tunable_parameter_values = tunable_parameter_values
        self.stepsize_method = exp_arguments.stepsize_method
        self.sample_size = exp_arguments.sample_size
        self.num_transitions = 200000
        self.checkpoint = 1000
        self.config = Config()

        self.data_path = os.path.join(os.getcwd(), 'mountain_car_prediction_data_30evaluations')
        assert len(os.listdir(self.data_path)) >= self.sample_size

        """ Feature Function Setup """
        self.config.state_dims = 2          # number of dimension in mountain car
        self.config.state_lims = np.array(((-1,1), (-1,1)), dtype=np.float64)   # state bounds in mountain car
        self.config.initial_centers = np.array(((0,0),(.25,.25),(.25,-.25),(-.25,-.25),(-.25,.25)), dtype=np.float64)
        # self.config.initial_centers = np.array(((0,0),(.25,.25),(.25,-.25),(-.25,-.25),(-.25,.25),
        #                                         (0.6,0), (-0.6,0), (0,0.6), (0,-0.6),
        #                                         (0.8,0.8), (0.8,-0.8), (-0.8, -0.8), (-0.8, 0.8),
        #                                         (1,0), (-1,0), (0,1), (0,-1)), dtype=np.float64)
        self.config.sigma = 0.5
        self.config.init_noise_mean = 0.0
        self.config.init_noise_var = 0.01
        self.feature_function = RadialBasisFunction(self.config)    # stays constant regardless of the parameter value

        """ Environment and Policy Setup """
        self.num_actions = 3                # number of actions in mountain car
        self.config.num_obs_features = self.config.initial_centers.shape[0]        # number of initial centers
        self.config.max_num_features = self.config.initial_centers.shape[0] + 1    # arbitrary since features are fixed
        self.gamma = 0.99                   # discount factor

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
        results = {'parameter_name': self.parameter_name,
                   'sample_size': self.sample_size,
                   'parameter_values': self.tunable_parameter_values,
                   'avg_mse': np.zeros(len(self.tunable_parameter_values))}

        for j, pv in enumerate(self.tunable_parameter_values):
            self._print("Parameter value: {0}".format(pv))

            self.set_tunable_parameter_value(pv)
            avg_mse_per_run = np.zeros(self.sample_size, dtype=np.int32)

            for i in range(self.sample_size):
                self._print("\tRun number: {0}".format(i+1))

                approximators = []
                stepsize_method = []
                for _ in range(self.num_actions):
                    approximators.append(LinearFunctionApproximator(self.config))
                    stepsize_method.append(self.stepsize_method_class(self.config))

                mse_per_checkpoint = np.zeros(self.num_transitions // self.checkpoint, dtype=np.float64)
                curr_checkpoint = 0

                # load data
                states, actions, rewards, terminations, avg_disc_return = self.load_data(seed_number=i)
                """ Start of Training"""
                curr_obs_feats = self.feature_function.get_observable_features(states[0])  # current observable features
                k = 0
                while k < self.num_transitions:
                    curr_a = actions[k]                                                 # current action
                    curr_av = approximators[curr_a].get_prediction(curr_obs_feats)      # current value
                    next_s = states[k+1]                # next state
                    next_r = rewards[k+1]               # next reward
                    next_term = terminations[k+1]       # next termination
                    next_a = actions[k+1]               # next action

                    # get next observable features
                    next_obs_feats = self.feature_function.get_observable_features(next_s)
                    # get next action values and action
                    next_av = approximators[next_a].get_prediction(next_obs_feats)
                    # compute the TD error for Sarsa(0)
                    td_error = next_r + self.gamma * (1 - next_term) * next_av - curr_av
                    # update weight vector
                    _, _, new_weights = stepsize_method[curr_a].update_weight_vector(td_error, curr_obs_feats,
                                                                             approximators[curr_a].get_weight_vector())
                    approximators[curr_a].update_weight_vector(new_weights)
                    # handle cases where weights diverge
                    if np.sum(np.isnan(new_weights)) > 0 or np.sum(np.isinf(new_weights)) > 0:
                        mse_per_checkpoint[curr_checkpoint:] += 1000000
                        print("The weights diverged!")
                        break

                    # update feature information, checkpoint, and k, and keep track of progress
                    curr_obs_feats = next_obs_feats
                    mse_per_checkpoint[curr_checkpoint] += np.square(curr_av - avg_disc_return[k]) / self.checkpoint
                    k += 1
                    if k % self.checkpoint == 0:
                        curr_checkpoint += 1

                    # handle terminal states
                    if next_term:
                        k += 1
                        curr_obs_feats = self.feature_function.get_observable_features(states[k])

                if DEBUG:
                    import matplotlib.pyplot as plt
                    x = np.arange(self.num_transitions // self.checkpoint)
                    plt.plot(x, mse_per_checkpoint)
                    plt.show()
                    plt.close()

                avg_mse_per_run[i] = np.average(mse_per_checkpoint)
                self._print("\t\tAverage Mean Squared Error: {0:.4f}".format(avg_mse_per_run[i]))

            results['avg_mse'][j] += np.average(avg_mse_per_run)
            self._print("Average MSE: {0:.4f}".format(results['avg_mse'][j]))

        self.store_results(results)

    def store_results(self, results):
        file_path = os.path.join(self.results_path, 'parameter_tuning_results.p')
        with open(file_path, mode='wb') as results_file:
            pickle.dump(results, results_file)
        print("Results successfully stored.")

    def load_data(self, seed_number):
        with open(os.path.join(self.data_path, 'seed' + str(seed_number) + '.p'), mode='rb') as data_file:
            data_dict = pickle.load(data_file)
        states = data_dict['states']
        actions = data_dict['actions']
        rewards = data_dict['rewards']
        terminations = data_dict['terminations']
        avg_discounted_return = data_dict['avg_discounted_return']
        return states, actions, rewards, terminations, avg_discounted_return


def main():
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ss', '--sample_size', action='store', default=1, type=int)
    parser.add_argument('-ssm', '--stepsize_method', action='store', default='sgd', type=str,
                        choices=['sgd', 'adam', 'slow_adam', 'idbd', 'autostep', 'rescaled_sgd', 'sidbd'])
    parser.add_argument('-tpv', '--tunable_parameter_values', action='store', nargs='+', type=float, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_parameters = parser.parse_args()

    task_name = 'mountain_car_prediction_task'
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
    # SGD parameter values: stepsize in {0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01}
    #   - 0.05 completed the most episodes during the whole training period
    # Adam parameter values: initial stepsize in {0.5 0.4 0.3 0.2 0.1 0.09 0.08 0.07 0.06 0.05}
    #   - 0.2 completed the most episodes during the whole training period
    # Adam without momentum (called slow_adam in the code) values: initial stepsize in
    #   {0.5 0.3 0.2 0.1 0.09 0.08 0.07 0.06 0.05 0.01}
    #   - 0.1 completed the most episodes during the whole training period
    # TIDBD parameter values: theta in {0.009 0.008 0.007 0.006 0.005 0.004 0.003 0.002 0.001 0.0009}
    #   - 0.005 completed the most episodes during the whole training period
    # AutoStep parameter values: mu in {0.04 0.03 0.02 0.01 0.009 0.008 0.007 0.006 0.005 0.001}
    #   - 0.02 completed the most episodes during the whole training period
    # Rescaled SGD parameter values: stepsize in {0.5 0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.01}
    #   - 0.02 completed the most episodes during the whole training period
    # STIDBD parameter values: theta in {0.01 0.009 0.008 0.007 0.006 0.005 0.004 0.003 0.002 0.001}
    #   - 0.01 completed the most episodes during the whole training period

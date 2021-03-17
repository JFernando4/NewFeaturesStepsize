"""
implementation of autostep according to this paper: https://arxiv.org/pdf/1903.03252.pdf (Algorithm 6)
"""
import numpy as np
from src.util import Config, check_attribute


class AutoTIDBD:

    def __init__(self, config: Config):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        parameter_size          int             10                  size of the weight vector
        theta                   float           0.01                meta-stepsize
        tau                     float           10000               eta decay
        init_stepsize           float           0.001               initial stepsize value
        """
        self.parameter_size = check_attribute(config, attr_name='parameter_size', default_value=10, data_type=int)
        self.theta = check_attribute(config, attr_name='theta', default_value=0.01, data_type=float)
        self.tau = check_attribute(config, attr_name='tau', default_value=10000.0, data_type=float)
        self.init_stepsize = check_attribute(config, attr_name='init_stepsize', default_value=0.001, data_type=float)

        self.beta = np.ones(self.parameter_size, dtype=np.float64) * np.log(self.init_stepsize)
        self.alpha = np.ones(self.parameter_size) * self.init_stepsize
        self.eta = np.zeros(self.parameter_size, dtype=np.float64)
        self.h = np.zeros(self.parameter_size, dtype=np.float64)
        self.z = np.zeros(self.parameter_size, dtype=np.float64)

    def update_weight_vector(self, td_error, features, weights, discounted_next_features):
        gradient = td_error * features
        feat_diff = discounted_next_features - features

        d_feat_diff_h = np.abs(td_error * feat_diff * self.h)
        eta_trace = self.eta - (1/self.tau) * self.alpha * feat_diff * self.z * (np.abs(gradient * self.h) - self.eta)
        self.eta = np.max((d_feat_diff_h, eta_trace), axis=0)

        pos_eta = self.eta > 0
        self.beta[pos_eta] -= self.theta * (1/self.eta[pos_eta]) * td_error * feat_diff[pos_eta] * self.h[pos_eta]
        M = -np.exp(self.beta) * np.dot(feat_diff, self.z)
        M[M < 1] = 1
        self.beta -= np.log(M)
        self.alpha = np.exp(self.beta)
        self.z = features               # this is the case when lambda = 0

        new_weights = weights + self.alpha * td_error * self.z
        self.h = np.clip(1 + self.alpha * feat_diff * self.z, a_min=0, a_max=None) + self.alpha * td_error * self.z
        return gradient, self.alpha, new_weights

    def increase_size(self, k: int):
        new_parameter_size = self.parameter_size + k

        # increase size of beta
        new_beta = np.zeros(new_parameter_size, dtype=np.float64)
        new_beta[:self.parameter_size] += self.beta
        new_beta[self.parameter_size:] += np.log(self.init_stepsize)
        # increasing size of stepsizes
        new_stepsizes = np.zeros(new_parameter_size)
        new_stepsizes[:self.parameter_size] += self.alpha
        new_stepsizes[self.parameter_size:] += self.init_stepsize
        # increasing the size of eta
        new_eta = np.zeros(new_parameter_size)
        new_eta[:self.parameter_size] += self.eta
        # increasing the size of h
        new_h = np.zeros(new_parameter_size)
        new_h[:self.parameter_size] += self.h
        # increasing size of z
        new_z = np.zeros(new_parameter_size)
        new_z[:self.parameter_size] += self.z

        self.parameter_size += k
        self.beta = new_beta
        self.alpha = new_stepsizes
        self.eta = new_eta
        self.h = new_h
        self.z = new_z


def boyan_chain_test(steps=50000):
    from src.env.BoyanChain import BoyanChain
    from src.env.RandomFeatures_task import LinearFunctionApproximator
    from src.util import Config
    import matplotlib.pyplot as plt

    config = Config()
    checkpoint = 100

    """ Environment Setup """
    config.init_noise_var = 0.1
    config.num_obs_features = 4
    config.max_num_features = 9
    """ AutoTIDBD Setup """
    config.parameter_size = 4
    config.theta = 0.001
    config.tau = 10000
    config.init_stepsize = 0.001
    # to keep track of learning progress
    run_avg_msve = np.zeros(steps // checkpoint, dtype=np.float64)
    current_checkpoint = 0
    avg_msve = 0

    env = BoyanChain(config)
    approximator = LinearFunctionApproximator(config)
    optimizer = AutoTIDBD(config)
    """ Start of Learning"""
    curr_obs_feats = env.get_observable_features()
    for s in range(steps):
        state_value = approximator.get_prediction(curr_obs_feats)
        optimal_value = env.compute_true_value()
        # step in the environment
        _, r, next_obs_feats, term = env.step()
        next_state_value = approximator.get_prediction(next_obs_feats)
        # compute td error
        td_error = r + (1 - term) * next_state_value - state_value
        # update weights
        _, _, new_weights = optimizer.update_weight_vector(td_error, features=curr_obs_feats,
                                                           weights=approximator.get_weight_vector(),
                                                           discounted_next_features=next_obs_feats)
        approximator.update_weight_vector(new_weights)
        # update features
        curr_obs_feats = next_obs_feats
        # keep track of progress
        avg_msve += np.square(state_value - optimal_value) / checkpoint
        # check if terminal state
        if term:
            env.reset()
            curr_obs_feats = env.get_observable_features()
        # store learning progress so far
        if (s+1) % checkpoint == 0:
            run_avg_msve[current_checkpoint] += avg_msve
            avg_msve *= 0
            current_checkpoint += 1

        if (s+1) == (steps//2):
            env.add_feature(k=4, noise=0.0, fake_feature=False)
            approximator.increase_num_features(4)
            optimizer.increase_size(4)
            curr_obs_feats = env.get_observable_features()

    print("The average MSVE is: {0:0.4f}".format(np.average(run_avg_msve)))

    xaxis = np.arange(run_avg_msve.size) + 1
    plt.plot(xaxis, run_avg_msve)
    plt.show()
    plt.close()


if __name__ == '__main__':
    boyan_chain_test(steps=200000)

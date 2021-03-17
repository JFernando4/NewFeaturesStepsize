"""
implementation of autostep according to this paper:
http://incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/Tuning-free%20step-size%20adapt.pdf
"""
import numpy as np
from src.util import Config, check_attribute


class AutoStep:

    def __init__(self, config: Config):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        parameter_size          int             10                  size of the weight vector
        mu                      float           0.01                value of mu in table 1
        tau                     float           10000               value of tau in table 1
        init_stepsize           float           0.001               initial stepsize value
        """
        self.parameter_size = check_attribute(config, attr_name='parameter_size', default_value=10, data_type=int)
        self.mu = check_attribute(config, attr_name='mu', default_value=0.01, data_type=float)
        self.tau = check_attribute(config, attr_name='tau', default_value=10000.0, data_type=float)
        self.init_stepsize = check_attribute(config, attr_name='init_stepsize', default_value=0.001, data_type=float)

        self.stepsizes = np.ones(self.parameter_size) * self.init_stepsize
        self.v = np.zeros(self.parameter_size)
        self.h = np.zeros(self.parameter_size)

    def update_weight_vector(self, error, features: np.ndarray, weights: np.ndarray):
        """
        Implementation of the update function in Table 1 of the paper
        :param error: np.float64 corresponding to target - prediction (delta in Table 1 of the paper)
        :param features: np array of shape (num_weights, ) corresponding to the features used to compute the prediction
        :param weights: np array of shape (num_weights, ) corresponding to the current weights of the approximator
        :return: gradient, stepsizes, updated weights
        """
        gradient = error * features     # gradient for a linear approximator with squared error loss
        dxh = gradient * self.h
        abs_dxh = np.abs(dxh)           # first term inside the max in line 4 of the paper
        temp_v = self.v + (1/self.tau) * self.stepsizes * np.square(features) * (abs_dxh - self.v)  # second term
        v = np.max((abs_dxh, temp_v), axis=0)   # max(first term, second term) corresponding to line 4 in Table 1
        if np.sum(v>0) > 0:                     # checks at least one term in v is positive, v is strictly positive
            self.stepsizes[v > 0] *= np.exp((self.mu * dxh)[v > 0] / v[v > 0])  # line 5 in Table 1

        M = np.max((np.dot(self.stepsizes, np.square(features)), 1))            # line 6 in Table 1
        self.stepsizes /= M                                                     # lin 7  in Table 1
        new_weights = weights + self.stepsizes * gradient
        self.h = self.h * (1 - self.stepsizes * np.square(features)) + self.stepsizes * gradient
        return gradient, self.stepsizes, new_weights

    def increase_size(self, k: int):
        new_parameter_size = self.parameter_size + k

        # increasing size of stepsizes
        new_stepsizes = np.zeros(new_parameter_size)
        new_stepsizes[:self.parameter_size] += self.stepsizes
        new_stepsizes[self.parameter_size:] += self.init_stepsize
        # increasing the size of v
        new_v = np.zeros(new_parameter_size)
        new_v[:self.parameter_size] += self.v
        # increasing the size of h
        new_h = np.zeros(new_parameter_size)
        new_h[:self.parameter_size] += self.h

        self.parameter_size += k
        self.stepsizes = new_stepsizes
        self.v = new_v
        self.h = new_h


def perfect_features_test():
    from src.env.Amatrix_task import Amatrix

    n = 50
    m = 3
    env = Amatrix(n, m)

    features = env.Amatrix  # perfect features
    weights = np.zeros(n)

    config = Config()
    config.parameter_size = n
    config.ini_stepsize = 0.01
    autostep = AutoStep(config)

    sample_size = 100000
    for i in range(sample_size):
        rand_row = np.random.randint(n)
        target = env.sample_target(rand_row, noisy=True)

        pred_features = features[rand_row, :]
        prediction = np.dot(pred_features, weights)
        error = target - prediction
        gradient, new_stepsize, new_weight_vector = autostep.update_weight_vector(error, pred_features, weights)
        weights = new_weight_vector
        if (i+1) % 10000 ==0:
            print("Sample number: {0}".format(i+1))
            print("\tPrediction error:{0}".format(error))

    print("Theta star:\n{0}".format(env.theta_star))
    print("Estimated theta:\n{0}".format(weights))
    difference = np.sqrt(np.sum(np.square(env.theta_star - weights)))
    print("L2 norm of difference:\n{0}".format(difference))


def imperfect_features_test():
    from src.env.Amatrix_task import Amatrix

    n = 10
    m = 2
    env = Amatrix(n, m)

    features = env.get_approx_A()   # first m features
    weights = np.zeros(m)

    config = Config()
    config.parameter_size = m
    config.ini_stepsize = 0.1
    autostep = AutoStep(config)

    sample_size = 50000
    for i in range(sample_size):
        rand_row = np.random.randint(n)
        target = env.sample_target(rand_row, noisy=True)

        pred_features = features[rand_row, :]
        prediction = np.dot(pred_features, weights)
        error = target - prediction
        gradient, new_stepsize, new_weight_vector = autostep.update_weight_vector(error, pred_features, weights)
        weights = new_weight_vector
        print("Sample number: {0}".format(i + 1))
        print("\tPrediction error:{0}".format(error))

    print("Theta star:\n{0}".format(env.theta_star))
    print("Estimated theta:\n{0}".format(weights))


def adding_good_features_test():
    from src.env.Amatrix_task import Amatrix

    n = 10
    m = 2
    env = Amatrix(n, m)

    features = env.get_approx_A()   # first m features
    weights = np.zeros(m)

    config = Config()
    config.parameter_size = m
    config.ini_stepsize = 0.1
    autostep = AutoStep(config)

    sample_size = 10000
    additional_features = 8
    for k in range(additional_features + 1):
        print("Number of features in the representation: {0}".format(autostep.parameter_size))
        for i in range(sample_size):
            rand_row = np.random.randint(n)
            target = env.sample_target(rand_row, noisy=True)

            pred_features = features[rand_row, :]
            prediction = np.dot(pred_features, weights)
            error = target - prediction
            gradient, new_stepsize, new_weight_vector = autostep.update_weight_vector(error, pred_features, weights)
            weights = new_weight_vector
            if ((i+1) % 100) == 0:
                print("\tSample number: {0}".format(i + 1))
                print("\t\tPrediction error:{0}".format(error))

        print("Theta star:\n{0}".format(env.theta_star))
        print("Estimated theta:\n{0}".format(weights))

        if autostep.parameter_size < n:
            print("Adding new feature...")
            new_feature = env.get_new_good_features(1)
            features = np.hstack((features, new_feature))
            autostep.increase_size(1)

            new_weights = np.zeros(m+1)
            new_weights[:m] = weights
            m += 1
            weights = new_weights


def adding_bad_features_test():
    from src.env.Amatrix_task import Amatrix

    n = 10
    m = 5
    env = Amatrix(n, m)

    features = env.get_approx_A()   # first m features
    weights = np.zeros(m)

    config = Config()
    config.parameter_size = m
    config.init_stepsize = 0.01
    autostep = AutoStep(config)

    sample_size = 50000
    additional_features = 30
    for k in range(additional_features + 1):
        print("Number of features in the representation: {0}".format(autostep.parameter_size))
        for i in range(sample_size):
            rand_row = np.random.randint(n)
            target = env.sample_target(rand_row, noisy=True)

            pred_features = features[rand_row, :]
            prediction = np.dot(pred_features, weights)
            error = target - prediction
            gradient, new_stepsize, new_weight_vector = autostep.update_weight_vector(error, pred_features, weights)
            weights = new_weight_vector
            if ((i+1) % 50000) == 0:
                print("\tSample number: {0}".format(i + 1))
                print("\t\tPrediction error: {0}".format(error))

        print("Theta star:\n{0}".format(env.theta_star))
        print("Estimated theta:\n{0}".format(weights))

        if k < additional_features:
            print("Adding new feature...")
            new_feature = env.get_new_bad_features(1)
            features = np.hstack((features, new_feature))
            autostep.increase_size(1)

            new_weights = np.zeros(m+1)
            new_weights[:m] = weights
            m += 1
            weights = new_weights


if __name__ == '__main__':
    adding_bad_features_test()

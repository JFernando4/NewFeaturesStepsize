"""
Implementation of SGD using numpy
"""

import numpy as np
from src.util import Config, check_attribute


class SGD:

    def __init__(self, config: Config):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        parameter_size          int             10                  size of the weight vector
        alpha                   float           0.001               step size parameter
        rescale                 bool            False               if True, keeps a running average  of the square sum
                                                                    of the features, which resets every time a new
                                                                    feature is added
        """
        self.parameter_size = check_attribute(config, attr_name='parameter_size', default_value=10, data_type=int)
        self.alpha = check_attribute(config, attr_name='alpha', default_value=0.001, data_type=float)
        self.rescale = check_attribute(config, attr_name='rescale', default_value=False, data_type=bool)

        self.stepsizes = np.ones(self.parameter_size) * self.alpha
        self.count = 0
        self.running_sum = 0

    def update_weight_vector(self, error, features, weights):
        stepsizes = np.zeros(self.parameter_size, dtype=np.float64) + self.stepsizes
        if self.rescale:
            self.count += 1
            self.running_sum += np.dot(features, features)
            stepsizes /= (self.running_sum / self.count)
        # assumes that error = target - prediction
        gradient = error * features
        new_weight_vector = weights + stepsizes * gradient

        return gradient, stepsizes, new_weight_vector

    def increase_size(self, k: int, init_stepsize=None):
        new_parameter_size = self.parameter_size + k
        new_stepsizes = np.zeros(new_parameter_size)
        new_stepsizes[:self.parameter_size] += self.stepsizes
        if init_stepsize is not None:
            new_stepsizes[self.parameter_size:] += init_stepsize
        else:
            new_stepsizes[self.parameter_size:] += self.alpha

        if self.rescale:
            self.count *= 0
            self.running_sum *= 0

        self.parameter_size += k
        self.stepsizes = new_stepsizes


def perfect_features_test():
    from src.env.Amatrix_task import Amatrix

    n = 3
    m = 2
    env = Amatrix(n, m)

    features = env.Amatrix  # perfect features
    weights = np.zeros(n)

    config = Config()
    config.parameter_size = n
    config.alpha = 0.001
    sgd = SGD(config)

    sample_size = 100000
    for i in range(sample_size):
        rand_row = np.random.randint(n)
        target = env.sample_target(rand_row, noisy=True)

        pred_features = features[rand_row, :]
        prediction = np.dot(pred_features, weights)
        error = target - prediction
        gradient, new_stepsize, new_weight_vector = sgd.update_weight_vector(error, pred_features, weights)
        weights = new_weight_vector
        if (i+1) % 10000 == 0:
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
    config.alpha = 0.001
    sgd = SGD(config)

    sample_size = 50000
    for i in range(sample_size):
        rand_row = np.random.randint(n)
        target = env.sample_target(rand_row, noisy=True)

        pred_features = features[rand_row, :]
        prediction = np.dot(pred_features, weights)
        error = target - prediction
        gradient, new_stepsize, new_weight_vector = sgd.update_weight_vector(error, pred_features, weights)
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
    config.alpha = 0.001
    sgd = SGD(config)

    sample_size = 10000
    additional_features = 8
    for k in range(additional_features + 1):
        print("Number of features in the representation: {0}".format(sgd.parameter_size))
        for i in range(sample_size):
            rand_row = np.random.randint(n)
            target = env.sample_target(rand_row, noisy=True)

            pred_features = features[rand_row, :]
            prediction = np.dot(pred_features, weights)
            error = target - prediction
            gradient, new_stepsize, new_weight_vector = sgd.update_weight_vector(error, pred_features, weights)
            weights = new_weight_vector
            if ((i+1) % 100) == 0:
                print("\tSample number: {0}".format(i + 1))
                print("\t\tPrediction error:{0}".format(error))

        print("Theta star:\n{0}".format(env.theta_star))
        print("Estimated theta:\n{0}".format(weights))

        if sgd.parameter_size < n:
            print("Adding new feature...")
            new_feature = env.get_new_good_features(1)
            features = np.hstack((features, new_feature))
            sgd.increase_size(1)

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
    config.alpha = 0.001
    sgd = SGD(config)

    sample_size = 50000
    additional_features = 30
    for k in range(additional_features + 1):
        print("Number of features in the representation: {0}".format(sgd.parameter_size))
        for i in range(sample_size):
            rand_row = np.random.randint(n)
            target = env.sample_target(rand_row, noisy=True)

            pred_features = features[rand_row, :]
            prediction = np.dot(pred_features, weights)
            error = target - prediction
            gradient, new_stepsize, new_weight_vector = sgd.update_weight_vector(error, pred_features, weights)
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
            sgd.increase_size(1)

            new_weights = np.zeros(m+1)
            new_weights[:m] = weights
            m += 1
            weights = new_weights


if __name__ == '__main__':
    adding_bad_features_test()

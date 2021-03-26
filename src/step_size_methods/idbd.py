"""
Implementation of IDBD as of this paper: https://www.aaai.org/Papers/AAAI/1992/AAAI92-027.pdf
"""

import numpy as np
from src.util import Config, check_attribute


class IDBD:

    def __init__(self, config: Config):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        parameter_size          int             10                  size of the weight vector
        init_beta               float           log(0.001)          initial value of beta
        theta                   float           0.1                 meta-stepsize parameter
        """
        self.parameter_size = check_attribute(config, attr_name='parameter_size', default_value=10, data_type=int)
        self.init_beta = check_attribute(config, attr_name='init_beta', default_value=np.log(0.001), data_type=float)
        self.theta = check_attribute(config, attr_name='theta', default_value=0.1, data_type=float)

        self.beta = np.ones(self.parameter_size) * self.init_beta
        self.beta_min = -100
        self.h = np.zeros(self.parameter_size)

    def update_weight_vector(self, error, features, weights):
        gradient = error * features
        change_in_beta = self.theta * gradient * self.h
        self.beta += change_in_beta
        np.clip(self.beta, a_min=self.beta_min, a_max=None, out=self.beta)
        new_stepsize = np.exp(self.beta)
        new_weight_vector = weights + new_stepsize * gradient

        alpha_diff = 1 - new_stepsize * (features ** 2)
        alpha_diff[alpha_diff < 0] *= 0
        self.h = self.h * alpha_diff + new_stepsize * gradient
        return gradient, new_stepsize, new_weight_vector

    def increase_size(self, k: int):
        new_parameter_size = self.parameter_size + k
        new_betas = np.zeros(new_parameter_size)
        new_betas[:self.parameter_size] += self.beta
        new_betas[self.parameter_size:] += self.init_beta

        new_h = np.zeros(new_parameter_size)
        new_h[:self.parameter_size] += self.h

        self.parameter_size += k
        self.beta = new_betas
        self.h = new_h


def perfect_features_test():
    from src.env.Amatrix_task import Amatrix

    n = 3
    m = 2
    env = Amatrix(n, m)

    features = env.Amatrix  # perfect features
    weights = np.zeros(n)

    config = Config()
    config.parameter_size = n
    config.theta = 100.0
    config.init_beta = np.log(0.001)
    idbd = IDBD(config)

    sample_size = 100000
    for i in range(sample_size):
        rand_row = np.random.randint(n)
        target = env.sample_target(rand_row, noisy=True)

        pred_features = features[rand_row, :]
        prediction = np.dot(pred_features, weights)
        error = target - prediction
        gradient, new_stepsize, new_weight_vector = idbd.update_weight_vector(error, pred_features, weights)
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
    config.theta = 0.001
    config.init_beta = 0
    idbd = IDBD(config)

    sample_size = 50000
    for i in range(sample_size):
        rand_row = np.random.randint(n)
        target = env.sample_target(rand_row, noisy=True)

        pred_features = features[rand_row, :]
        prediction = np.dot(pred_features, weights)
        error = target - prediction
        gradient, new_stepsize, new_weight_vector = idbd.update_weight_vector(error, pred_features, weights)
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
    config.theta = 0.001
    config.init_beta = -2
    idbd = IDBD(config)

    sample_size = 10000
    additional_features = 8
    for k in range(additional_features + 1):
        print("Number of features in the representation: {0}".format(idbd.parameter_size))
        for i in range(sample_size):
            rand_row = np.random.randint(n)
            target = env.sample_target(rand_row, noisy=True)

            pred_features = features[rand_row, :]
            prediction = np.dot(pred_features, weights)
            error = target - prediction
            gradient, new_stepsize, new_weight_vector = idbd.update_weight_vector(error, pred_features, weights)
            weights = new_weight_vector
            if ((i+1) % 100) == 0:
                print("\tSample number: {0}".format(i + 1))
                print("\t\tPrediction error:{0}".format(error))

        print("Theta star:\n{0}".format(env.theta_star))
        print("Estimated theta:\n{0}".format(weights))

        if idbd.parameter_size < n:
            print("Adding new feature...")
            new_feature = env.get_new_good_features(1)
            features = np.hstack((features, new_feature))
            idbd.increase_size(1)

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
    config.theta = 0.01
    config.init_beta = np.log(0.0001)
    idbd = IDBD(config)

    sample_size = 50000
    additional_features = 30
    for k in range(additional_features + 1):
        print("Number of features in the representation: {0}".format(idbd.parameter_size))
        for i in range(sample_size):
            rand_row = np.random.randint(n)
            target = env.sample_target(rand_row, noisy=True)

            pred_features = features[rand_row, :]
            prediction = np.dot(pred_features, weights)
            error = target - prediction
            gradient, new_stepsize, new_weight_vector = idbd.update_weight_vector(error, pred_features, weights)
            weights = new_weight_vector
            if ((i+1) % 25000) == 0:
                print("\tSample number: {0}".format(i + 1))
                print("\t\tPrediction error: {0}".format(error))

        print("Theta star:\n{0}".format(env.theta_star))
        print("Estimated theta:\n{0}".format(weights))

        if k < additional_features:
            print("Adding new feature...")
            new_feature = env.get_new_bad_features(1)
            features = np.hstack((features, new_feature))
            idbd.increase_size(1)

            new_weights = np.zeros(m+1)
            new_weights[:m] = weights
            m += 1
            weights = new_weights


if __name__ == '__main__':
    perfect_features_test()


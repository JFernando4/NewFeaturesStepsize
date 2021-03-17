"""
numpy implementation of adam according to this paper: https://arxiv.org/pdf/1412.6980.pdf
"""
import numpy as np
from src.util import Config, check_attribute


class Adam:

    def __init__(self, config: Config):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        parameter_size          int             10                  size of the weight vector
        init_alpha              float           0.001               initial value of the stepsize
        beta1                   float           0.9                 initial value of beta1
        beta2                   float           0.99                initial value of beta2
        eps                     float           1e-08               epsilon value to prevent division by zero
        restart_ma              bool            False               whether to restart the moving averages every time
                                                                    a new feature is added
        """
        self.parameter_size = check_attribute(config, attr_name='parameter_size', default_value=10, data_type=int)
        self.init_alpha = check_attribute(config, attr_name='init_alpha', default_value=0.001, data_type=float)
        self.beta1 = check_attribute(config, attr_name='beta1', default_value=0.9, data_type=float)
        self.beta2 = check_attribute(config, attr_name='beta2', default_value=0.99, data_type=float)
        self.eps = check_attribute(config, attr_name='eps', default_value=1e-08, data_type=float)
        self.restart_ma = check_attribute(config, attr_name='restart_ma', default_value=False, data_type=bool)

        self.stepsize = np.ones(self.parameter_size, dtype=np.float64) * self.init_alpha
        self.m = np.zeros(self.parameter_size)
        self.v = np.zeros(self.parameter_size)
        self.t = np.zeros(self.parameter_size)

    def update_weight_vector(self, error, features, weights):
        gradient = error * features
        self.t += 1.0
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradient)
        mhat = self.m / (1 - (self.beta1 ** self.t))
        vhat = self.v / (1 - (self.beta2 ** self.t))
        new_stepsize = self.stepsize / (np.sqrt(vhat) + self.eps)
        new_weights = weights + new_stepsize * mhat
        return gradient, new_stepsize, new_weights

    def increase_size(self, k: int):
        new_parameter_size = self.parameter_size + k
        # update steptsize vector
        new_stepsize = np.ones(new_parameter_size) * self.init_alpha
        new_m = np.zeros(new_parameter_size)
        new_v = np.zeros(new_parameter_size)
        new_t = np.zeros(new_parameter_size)
        if not self.restart_ma:
            # update first moment vector
            new_m[:self.parameter_size] += self.m
            # update second moment vector
            new_v[:self.parameter_size] += self.v
            # update time vector
            new_t[:self.parameter_size] += self.t

        self.parameter_size += k
        self.stepsize = new_stepsize
        self.m = new_m
        self.v = new_v
        self.t = new_t


def perfect_features_test():
    from src.env.Amatrix_task import Amatrix

    n = 20
    m = 3
    env = Amatrix(n, m)

    features = env.Amatrix  # perfect features
    weights = np.random.rand(n)

    config = Config()
    config.parameter_size = n
    config.init_alpha = 0.001
    adam = Adam(config)

    sample_size = 100000
    for i in range(sample_size):
        rand_row = np.random.randint(n)
        target = env.sample_target(rand_row, noisy=True)

        pred_features = features[rand_row, :]
        prediction = np.dot(pred_features, weights)
        error = target - prediction
        gradient, new_stepsize, new_weight_vector = adam.update_weight_vector(error, pred_features, weights)
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

    n = 4
    m = 2
    env = Amatrix(n, m)

    features = env.get_approx_A()   # first m features
    weights = np.random.rand(m)

    config = Config()
    config.parameter_size = m
    config.init_alpha = 0.001
    adam = Adam(config)

    sample_size = 50000
    for i in range(sample_size):
        rand_row = np.random.randint(n)
        target = env.sample_target(rand_row, noisy=True)

        pred_features = features[rand_row, :]
        prediction = np.dot(pred_features, weights)
        error = target - prediction
        gradient, new_stepsize, new_weight_vector = adam.update_weight_vector(error, pred_features, weights)
        weights = new_weight_vector
        print("Sample number: {0}".format(i+1))
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
    config.init_alpha = 0.0001
    adam = Adam(config)

    sample_size = 10000
    additional_features = 8
    for k in range(additional_features + 1):
        print("Number of features in the representation: {0}".format(adam.parameter_size))
        for i in range(sample_size):
            rand_row = np.random.randint(4)
            target = env.sample_target(rand_row, noisy=True)

            pred_features = features[rand_row, :]
            prediction = np.dot(pred_features, weights)
            error = target - prediction
            gradient, new_stepsize, new_weight_vector = adam.update_weight_vector(error, pred_features, weights)
            weights = new_weight_vector
            if ((i+1) % 5000) == 0:
                print("\tSample number: {0}".format(i + 1))
                print("\t\tPrediction error:{0}".format(error))

        print("Theta star:\n{0}".format(env.theta_star))
        print("Estimated theta:\n{0}".format(weights))

        if adam.parameter_size < n:
            print("Adding new feature...")
            new_feature = env.get_new_good_features(1)
            features = np.hstack((features, new_feature))
            adam.increase_size(1)

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
    config.init_alpha = 0.001
    adam = Adam(config)

    sample_size = 50000
    additional_features = 5
    for k in range(additional_features + 1):
        print("Number of features in the representation: {0}".format(adam.parameter_size))
        for i in range(sample_size):
            rand_row = np.random.randint(n)
            target = env.sample_target(rand_row, noisy=True)

            pred_features = features[rand_row, :]
            prediction = np.dot(pred_features, weights)
            error = target - prediction
            gradient, new_stepsize, new_weight_vector = adam.update_weight_vector(error, pred_features, weights)
            weights = new_weight_vector
            if ((i+1) % 50000) == 0:
                print("\tSample number: {0}".format(i + 1))
                print("\t\tPrediction error:{0}".format(error))

        print("Theta star:\n{0}".format(env.theta_star))
        print("Estimated theta:\n{0}".format(weights))

        if k < additional_features:
            print("Adding new feature...")
            new_feature = env.get_new_bad_features(1)
            features = np.hstack((features, new_feature))
            adam.increase_size(1)

            new_weights = np.zeros(m+1)
            new_weights[:m] = weights
            m += 1
            weights = new_weights


if __name__ == '__main__':
    adding_good_features_test()


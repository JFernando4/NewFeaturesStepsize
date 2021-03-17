import numpy as np

from src.util import Config, check_attribute


class RandomFeatures:

    def __init__(self, config: Config):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        num_true_features       int             20                  number of features used to compute the target
        num_obs_features        int             5                   number of observable features for the function
                                                                    approximator
        max_num_features        int             20000               maximum number of features that can be added
        """
        self.num_true_features = check_attribute(config, 'num_true_features', 20)
        self.num_obs_features = check_attribute(config, 'num_obs_features', 5)
        self.max_num_features = check_attribute(config, 'max_num_features', 20000)
        assert self.num_obs_features <= self.num_true_features
        assert self.num_true_features <= self.max_num_features

        self.theta = np.random.uniform(0, 1, size=self.num_true_features)
        self.theta /= np.linalg.norm(self.theta)

        self.feature_type = np.zeros(self.max_num_features, dtype=bool)
        self.feature_type[:self.num_obs_features] += True       # True if real feature, otherwise False

    def sample_observation(self, noisy=False):
        """
        Samples a target, observable features, and the best approximation to the target given the observable features.
        :param noisy: whether to add noise to the target the best approximation
        :return: target, observable features, best approximation
        """
        true_features = np.random.standard_normal(self.num_true_features)
        observable_features, num_real_features = self.get_observable_features(true_features)
        target = np.dot(self.theta, true_features)
        best_approximation = np.dot(self.theta[:num_real_features], true_features[:num_real_features])
        if noisy:
            noise = np.random.normal(loc=0, scale=np.sqrt(0.3))     # normal distribution with mu = 0, sigma^2 = 0.3
            target += noise
            best_approximation += noise
        return target, observable_features, best_approximation

    def get_observable_features(self, true_features):
        """
        Creates a vector of observable_features of size self.current_obs_features. The i-th entry in the vector
        corresponds to a true feature if the corresponding entry of self.feature_type is true. Otherwise, the entry
        is sampled from a standard normal distribution.
        :param true_features: vector of real features given by the RandomFeatures class
        :return: vector of true and fake features
        """
        observable_features = np.zeros(self.num_obs_features, dtype=np.float64)

        feature_type = self.feature_type[:self.num_obs_features]
        num_real_features = np.sum(feature_type)
        num_fake_features = self.num_obs_features - num_real_features

        observable_features[feature_type] += true_features[:num_real_features]
        observable_features[np.logical_not(feature_type)] += np.random.standard_normal(num_fake_features)
        return observable_features, num_real_features

    def add_new_feature(self, k: int, true_feature=True):
        """
        Increases the number of observable features given to the function approximator.
        :param k: number of new features
        :param true_feature: whether to add a true feature or a fake feature
        """
        # error handling
        assert k > 0
        if self.num_obs_features + k > self.max_num_features:
            raise ValueError("The maximum number of features has been exceeded.")
        # add new feature
        if true_feature and np.sum(self.feature_type) < self.num_true_features:
            self.feature_type[self.num_obs_features: self.num_obs_features + k] += True
        self.num_obs_features += k


class LinearFunctionApproximator:

    def __init__(self, config: Config):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        num_obs_features        int             5                   number of initial true features
        max_num_features        int             20000               maximum number of features that can be added
        """
        self.num_obs_features = check_attribute(config, 'num_obs_features', 5)
        self.max_num_features = check_attribute(config, 'max_num_features', 20000)
        self.theta_hat = np.zeros(self.max_num_features, dtype=np.float64)

    def get_prediction(self, features):
        assert len(features) == self.num_obs_features
        weights = self.theta_hat[:self.num_obs_features]
        prediction = np.dot(features, weights)
        return prediction

    def get_weight_vector(self):
        return self.theta_hat[:self.num_obs_features]

    def update_weight_vector(self, new_weight_vector):
        assert self.theta_hat[:self.num_obs_features].size == new_weight_vector.size
        self.theta_hat[:self.num_obs_features] = new_weight_vector

    def increase_num_features(self, k: int):
        # error handling
        assert k > 0
        if self.num_obs_features + k > self.max_num_features:
            raise OverflowError("The maximum number of features has been exceeded.")
        # increases number of features
        self.num_obs_features += k


def test_random_features_generator(sample_size=10, compute_sample_statistics=True):

    config = Config()
    config.num_true_features = 2
    config.num_obs_features = 2
    config.max_num_features = 20000
    task = RandomFeatures(config)
    print("The value of theta is:\n{0}".format(task.theta))
    print("The norm of theta is: {0}".format(np.linalg.norm(task.theta)))

    for i in range(sample_size):
        target, observable_features, best_approximation = task.sample_observation(noisy=False)
        print("The features are: {0}\tThe target is:{1}".format(observable_features, target))

    if compute_sample_statistics:
        num_samples = 100000
        samples = np.zeros(num_samples)
        for i in range(num_samples):
            target, _, _ = task.sample_observation(noisy=False)
            samples[i] += target
        # The sample average and sample variance of the target should be 0 and 1, respectively.
        print("The sample average of the target is: {:.2f}".format(np.average(samples)))
        print("The sample variance of the target is: {:.2f}".format(np.var(samples)))


def test_function_approximator(num_features=20, initial_features=20, num_iterations=10000, chkpt=100, plot_mse=True,
                               noisy=True, add_features=False,  add_true_features=True, feature_add_interval=100,
                               mixed_features=False):

    from src.step_size_methods import SGD
    config = Config()
    # task setup
    config.num_true_features = num_features
    config.num_obs_features = initial_features      # same as function approximator
    config.max_num_features = 20000                 # same as function approximator
    task = RandomFeatures(config)

    # function approximator setup
    approximator = LinearFunctionApproximator(config)

    # optimizer setup
    config.parameter_size = initial_features
    config.alpha = 0.001
    optimizer = SGD(config)

    # for plotting
    mse_per_chpt = np.zeros(num_iterations // chkpt, dtype=np.float64)
    mse = 0
    current_chpt = 0

    # training loop
    for i in range(num_iterations):
        target, observable_features, best_approximation = task.sample_observation(noisy=noisy)
        prediction = approximator.get_prediction(observable_features)
        error = target - prediction
        _, _, new_weights = optimizer.update_weight_vector(error, observable_features,
                                                           approximator.get_weight_vector())
        approximator.update_weight_vector(new_weights)

        squared_loss = np.square(error)
        mse += squared_loss / chkpt
        if (i + 1) % chkpt == 0:
            # reporting and saving
            print("Iteration number: {0}".format(i + 1))
            print("\tTarget: {0:.4f}".format(target))
            print("\tPrediction: {0:.4f}".format(prediction))
            print("\tMean Squared Error: {0:.4f}".format(mse))
            mse_per_chpt[current_chpt] += mse
            mse *= 0
            current_chpt += 1

        if add_features and (i + 1) % feature_add_interval == 0:
            task.add_new_feature(k=1, true_feature=add_true_features)
            approximator.increase_num_features(k=1)
            optimizer.increase_size(k=1)
            if mixed_features:
                add_true_features = not add_true_features

    if plot_mse:
        # plots
        import matplotlib.pyplot as plt
        x_axis = np.arange(num_iterations // chkpt)
        plt.plot(x_axis, mse_per_chpt)
        plt.show()
        plt.close()


if __name__ == "__main__":
    import argparse

    test_random_features_generator(sample_size=10, compute_sample_statistics=True)
    # test_function_approximator(num_features=4, initial_features=3, num_iterations=100000, chkpt=100, plot_mse=True,
    #                            noisy=True, add_features=True,  add_true_features=True, feature_add_interval=50000,
    #                            mixed_features=False)





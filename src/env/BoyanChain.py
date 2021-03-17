import numpy as np

from src.util import Config, check_attribute


class BoyanChain:

    def __init__(self, config: Config):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        init_noise_var          float           0.1                 initial variance of the feature noise
        num_obs_features        int             4                   number of observable features for the function
                                                                    approximator
        """
        self.init_noise_var = check_attribute(config, 'init_noise_var', 0.1)
        self.num_obs_features = check_attribute(config, 'num_obs_features', 4)

        self.noise_var = np.zeros(self.num_obs_features, dtype=np.float64) + self.init_noise_var
        self.noise_mean = 0

        self.num_true_features = 4

        self.num_states = 13
        self.optimal_weights_norm = np.sqrt(24.0 ** 2 + 16 ** 2 + 8 ** 2)
        self.optimal_weights = np.array((0, -8, - 16, -24), dtype=np.float64) / self.optimal_weights_norm
        self.state_features, self.observable_features = self._initialize_feature_matrices()
        # steady state distribution computed using the partial sum of the Cesaro limit with 10 million iterations
        self.steady_state_distribution = np.array(
            [0.1084, 0.0723, 0.0723, 0.0722, 0.0724, 0.072,  0.0729, 0.0712, 0.0745, 0.0678, 0.0813, 0.0542, 0.1084])

        self.current_state = 12
        self.current_features = self.state_features[self.current_state]

    def _initialize_feature_matrices(self):
        state_features = np.zeros((self.num_states, self.num_true_features), dtype=np.float64)
        for i in range(self.num_states):
            main_feature_index = int(np.ceil(i / self.num_true_features))
            diff = (self.num_true_features - ((i-1) % self.num_true_features + 1)) / self.num_true_features
            state_features[i, main_feature_index] = 1 - diff
            if main_feature_index > 0:
                state_features[i, main_feature_index - 1] = diff

        observable_features = np.zeros((self.num_states, self.num_obs_features), dtype=np.float64)
        for i in range(self.num_obs_features):
            observable_features[:, i] = state_features[:, i % self.num_true_features]
        return state_features, observable_features

    def step(self):
        if self.current_state == 1:
            decrease = 1
            r = -2.0
        else:
            r = -3.0
            decrease = np.random.choice([1,2])
        sp = self.current_state - decrease
        next_features = self.state_features[sp]

        self.current_state = sp
        self.current_features = next_features
        terminal = self.current_state == 0

        obs_features = self.get_observable_features()
        r /= self.optimal_weights_norm
        return self.current_state, r, obs_features, terminal

    def get_observable_features(self, add_noise=True):
        # obs_features = np.zeros(self.num_obs_features)
        # for i in range(self.num_true_features):
        #     obs_features[np.arange(i, self.num_obs_features, self.num_true_features)] += self.current_features[i]
        obs_features = np.zeros(self.num_obs_features, dtype=np.float64) + self.observable_features[self.current_state]

        if add_noise:
            obs_features += np.random.normal(loc=self.noise_mean, scale=np.sqrt(self.noise_var))
        return obs_features

    def add_feature(self, k=1, noise=None, fake_feature=False):
        new_noise_var = np.zeros(self.num_obs_features + k)
        new_noise_var[:self.num_obs_features] += self.noise_var
        if noise is not None:
            new_noise_var[self.num_obs_features:] += noise
        else:
            new_noise_var[self.num_obs_features:] += self.init_noise_var

        obs_features = np.zeros((self.num_states, self.num_obs_features + k), dtype=np.float64)
        obs_features[:, :self.num_obs_features] += self.observable_features
        if not fake_feature:
            obs_features[:, self.num_obs_features:] += self.state_features[:, (np.arange(k)+self.num_obs_features) % 4]

        self.observable_features = obs_features
        self.noise_var = new_noise_var
        self.num_obs_features += k

    def reset(self):
        self.current_state = 12
        self.current_features = np.array([0, 0, 0, 1], dtype=np.float64)

    def compute_true_value(self):
        return np.dot(self.optimal_weights, self.current_features)

    def compute_msve(self, weight_vector):
        current_state = self.current_state
        current_features = self.current_features

        msve = 0
        for i in range(self.num_states):
            self.current_state = i
            self.current_features = self.state_features[i]
            true_value = self.compute_true_value()
            approximate_value = np.dot(weight_vector, self.get_observable_features(add_noise=False))
            msve += self.steady_state_distribution[i] * np.square(true_value - approximate_value)

        self.current_state = current_state
        self.current_features = current_features
        return msve


def test_environment_features():
    np.random.seed(0)
    config = Config()
    config.init_noise_var = 0.0
    config.num_obs_features = 4

    env = BoyanChain(config)

    def run_env(e: BoyanChain, s=20):
        for i in range(s):
            print("Step number: {0}".format(i + 1))
            current_state = e.current_state
            next_state, _, observed_features, terminal = e.step()
            print("\tMoved: {0} --> {1}".format(current_state, next_state))
            print("\tObserved Features: {0}".format(observed_features))

            if terminal:
                e.reset()

    run_env(env, 20)

    print("\nAdding 4 features without noise...")
    env.reset()
    env.add_feature(4)
    run_env(env, 20)

    print("\nAdding 4 features with noise...")
    env.reset()
    env.add_feature(4, noise=1)
    run_env(env, 20)


def learning_value_function(sample_size=100000, checkpoint=1000):
    np.random.seed(0)
    config = Config()
    config.init_noise_var = 0.1
    config.num_obs_features = 4

    env = BoyanChain(config)

    theta = np.zeros(config.num_obs_features, dtype=np.float64)
    theta_star = env.optimal_weights
    alpha = 0.005

    def train(th, th_star, e: BoyanChain, ss, ckpt):
        e.reset()
        current_features = e.get_observable_features()
        mean_square_value_diff = 0.0
        for i in range(ss):
            current_value = np.dot(current_features, th)
            optimal_value = np.dot(e.current_features, th_star)
            current_state, reward, next_features, terminal = e.step()

            next_value = np.dot(next_features, th)
            temporal_diff = reward + (1 - int(terminal)) * next_value - current_value
            th += alpha * temporal_diff * current_features

            mean_square_value_diff += np.square(current_value - optimal_value) / ckpt
            if (i+1) % ckpt == 0:
                print("Training Step: {0}".format(i+1))
                print("\tEstimated MSVE: {0:.4f}".format(mean_square_value_diff))
                print("\tTrue MSVE: {0:.4f}".format(e.compute_msve(th)))
                mean_square_value_diff *= 0

            current_features = next_features

            if terminal:
                e.reset()
                current_features = e.get_observable_features()

    print("First phase of training...")
    train(theta, theta_star, env, sample_size, checkpoint)
    env.add_feature(4, 0.0)

    print("\n\nSecond phase of training...")
    new_theta = np.zeros(8, dtype=np.float64)
    new_theta[:4] += theta
    train(new_theta, theta_star, env, sample_size, checkpoint)


if __name__ == '__main__':
    test_environment_features()
    import time
    start = time.time()
    learning_value_function(sample_size=100000)
    end = time.time()
    print("The time in minutes is: {0}".format((end - start) / 60))

    # P = np.array([
    #     #0  1  2  3  4  5  6  7  8  9 10 11  12
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],    # 0
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # 1
    #     [1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # 2
    #     [0, 1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # 3
    #     [0, 0, 1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # 4
    #     [0, 0, 0, 1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0],    # 5
    #     [0, 0, 0, 0, 1/2, 1/2, 0, 0, 0, 0, 0, 0, 0],    # 6
    #     [0, 0, 0, 0, 0, 1/2, 1/2, 0, 0, 0, 0, 0, 0],    # 7
    #     [0, 0, 0, 0, 0, 0, 1/2, 1/2, 0, 0, 0, 0, 0],    # 8
    #     [0, 0, 0, 0, 0, 0, 0, 1/2, 1/2, 0, 0, 0, 0],    # 9
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1/2, 1/2, 0, 0, 0],    # 10
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1/2, 1/2, 0, 0],    # 11
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/2, 1/2, 0],    # 12
    # ]
    # )
    # n = 10000000
    # Pn = P
    # avg_Pn = np.zeros((13, 13)) + P * (1/n)
    # for i in range(n-1):
    #     Pn = np.matmul(P, Pn)
    #     avg_Pn += Pn * (1/n)
    # print(np.round(avg_Pn[0,:], 4))











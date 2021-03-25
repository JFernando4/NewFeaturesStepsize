import numpy as np
from src.util import check_attribute, Config


class RadialBasisFunction:

    def __init__(self, config=None):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        state_dims              int             2                   number of state dimensions
        state_lims              np.array        ((-1,1),(-1,1))     upper an lower limits of the value of each
                                                                    state dimension, the first dimension must be equal
                                                                    to state_dims
        init_centers            np.ndarray      ((0.1,0),           the centers for the radial basis functions
                                                (0,0.1),
                                                (-0.1,0),
                                                (0,-0.1))
        sigma                   float           0.5                 the width of each feature
        init_noise_mean         float           0.0                 mean of the noise of each feature
        init_noise_var          float           0.1                 variance of the noise noise of each feature
        """
        self.state_dims = check_attribute(config, 'state_dims', 2)
        self.state_lims = check_attribute(config, 'state_lims', np.array(((-1,1), (-1,1)), dtype=np.float64))
        assert self.state_lims.shape[0] == self.state_dims
        self.centers = check_attribute(config, 'initial_centers', np.array(((0.1,0), (0,0.1), (-0.1,0), (0,-0.1))))
        assert self.centers.shape[1] == self.state_dims
        self.sigma = check_attribute(config, 'sigma', 0.5)
        self.init_noise_mean = check_attribute(config, 'init_noise_mean', default_value=0.0)
        self.init_noise_var = check_attribute(config, 'init_noise_var', default_value=0.1)

        self.num_features = self.centers.shape[0]
        self.relevant_features = np.ones(self.num_features, dtype=np.float64)
        self.noise_mean = np.zeros(self.num_features, dtype=np.float64) + self.init_noise_mean
        self.noise_var = np.zeros(self.num_features, dtype=np.float64) + self.init_noise_var
        # flag to indicate if irrelevant or noisy features have been added
        self.added_noisy_or_irrelevant_features = False

    def get_features(self, state:np.ndarray):
        assert state.size == self.state_dims
        assert np.sum(state >= self.state_lims[:, 0]) + np.sum(state <= self.state_lims[:, 1]) == 4

        centered_state = state - self.centers
        norm_centered_states = np.sum(centered_state * centered_state, axis=1)
        features = np.exp(- norm_centered_states / (2 * self.sigma**2))
        return features

    def get_observable_features(self, state: np.ndarray):
        num_copies = int(np.ceil(self.num_features / self.centers.shape[0]))
        features = self.get_features(state)
        obs_features = np.tile(features, num_copies)[:self.num_features]
        obs_features *= self.relevant_features
        obs_features += np.random.normal(self.noise_mean, np.sqrt(self.noise_var))
        return obs_features

    def add_feature(self, k=1, noise_mean=None, noise_var=None, fake_feature=False):
        self.added_noisy_or_irrelevant_features = True
        new_relevant_features = np.ones(self.num_features + k, dtype=np.float64)
        new_relevant_features[:self.num_features] = self.relevant_features
        if fake_feature:
            new_relevant_features[self.num_features:] *= 0.0

        new_noise_mean = np.zeros(self.num_features + k, dtype=np.float64)
        new_noise_mean[:self.num_features] += self.noise_mean
        new_noise_mean[self.num_features:] += self.init_noise_mean if noise_mean is None else noise_mean

        new_noise_var = np.zeros(self.num_features + k, dtype=np.float64)
        new_noise_var[:self.num_features] += self.noise_var
        new_noise_var[self.num_features:] += self.init_noise_var if noise_var is None else noise_var

        self.num_features += k
        self.relevant_features = new_relevant_features
        self.noise_mean = new_noise_mean
        self.noise_var = new_noise_var

    def add_centers(self, centers: np.ndarray, noise_mean=None, noise_var=None):
        assert len(centers.shape) == 2 and centers.shape[1] == 2
        if self.added_noisy_or_irrelevant_features:
            raise NotImplementedError("Cannot add more centers after adding irrelevant or noisy features!")
        num_new_centers = self.num_features + centers.shape[0]
        new_centers = np.zeros((num_new_centers, 2), dtype=np.float64)
        new_centers[:self.num_features, :] += self.centers
        new_centers[self.num_features:, :] += centers

        new_noise_mean = np.zeros(num_new_centers, dtype=np.float64)
        new_noise_mean[:self.num_features] += self.noise_mean
        new_noise_mean[self.num_features:] += self.init_noise_mean if noise_mean is None else noise_mean

        new_noise_var = np.zeros(num_new_centers, dtype=np.float64)
        new_noise_var[:self.num_features] += self.noise_var
        new_noise_var[self.num_features:] += self.init_noise_var if noise_var is None else noise_var

        new_relevant_features = np.ones(num_new_centers, dtype=np.float64)
        new_relevant_features[:self.num_features] *= self.relevant_features

        self.centers = new_centers
        self.num_features += centers.shape[0]
        self.noise_mean = new_noise_mean
        self.noise_var = new_noise_var
        self.relevant_features = new_relevant_features

    def add_random_center(self):
        new_center = np.random.uniform(low=self.state_lims[:,0], high=self.state_lims[:,1]).reshape(1,2)
        self.add_centers(new_center, noise_var=0, noise_mean=0)


class MountainCar:
    """
    Environment Specifications:
    Number of Actions = 3
    Observation Dimension = 2 (position, velocity)
    Observation Dtype = np.float64
    Reward = -1 at every step
    """

    def __init__(self, config):
        assert isinstance(config, Config)
        """ Parameters:
        Name:                       Type            Default:        Description(omitted when self-explanatory):
        norm_state                  bool            True            whether to normalize the state between -1 and 1 
        """
        self.config = config

        # environment related variables
        self.norm_state = check_attribute(config, 'norm_state', True)

        # internal state of the environment
        position = -0.6 + np.random.random() * 0.2
        velocity = 0.0
        self.current_state = np.array((position, velocity), dtype=np.float64)
        self.actions = np.array([0, 1, 2], dtype=int)  # 0 = backward, 1 = coast, 2 = forward
        self.high = np.array([0.5, 0.07], dtype=np.float64)
        self.low = np.array([-1.2, -0.07], dtype=np.float64)
        self.action_dictionary = {0: -1,   # accelerate backwards
                                  1: 0,    # coast
                                  2: 1}    # accelerate forwards

    def reset(self):
        # random() returns a random float in the half open interval [0,1)
        position = -0.6 + np.random.random() * 0.2
        velocity = 0.0
        self.current_state = np.array((position, velocity), dtype=np.float64)
        if self.norm_state:
            return self.normalize(self.current_state)
        else:
            return self.current_state

    " Update environment "
    def step(self, A):

        if A not in self.actions:
            raise ValueError("The action should be one of the following integers: {0, 1, 2}.")
        action = self.action_dictionary[A]
        terminate = False

        current_position = self.current_state[0]
        current_velocity = self.current_state[1]

        velocity = current_velocity + (0.001 * action) - (0.0025 * np.cos(3 * current_position))
        position = current_position + velocity

        if velocity > 0.07:
            velocity = 0.07
        elif velocity < -0.07:
            velocity = -0.07

        if position < -1.2:
            position = -1.2
            velocity = 0.0
        elif position > 0.5:
            position = 0.5
            terminate = True

        reward = -1 if not terminate else 0

        self.current_state = np.array((position, velocity), dtype=np.float64)
        return self.get_current_state(), reward, terminate

    @staticmethod
    def normalize(state):
        """ normalize to [-1, 1] """
        temp_state = np.zeros(shape=2, dtype=np.float64)
        temp_state[0] = (state[0] + 0.35) / 0.85

        temp_state[1] = (state[1]) / 0.07
        return temp_state

    def get_current_state(self):
        if self.norm_state:
            return self.normalize(self.current_state)
        else:
            return self.current_state

    def set_state(self, state: np.ndarray, normalized=True):
        """
        Set the current state of the environment
        :param state: new environment state
        :param normalized: indicates whether the new state is normalized between -1 and 1
        """
        if not normalized:
            self.current_state = state
        else:
            temp_state = np.zeros(state.shape) + state
            temp_state[0] = state[0] * 0.85 - 0.35
            temp_state[1] = state[1] * 0.07
            self.current_state = temp_state


def random_policy_test(steps=100, verbose=False):
    print("==== Results with Random Policy ====")
    config = Config()
    actions = 3

    config.current_step = 0
    env = MountainCar(config)

    cumulative_reward = 0
    terminations = 0
    steps_per_episode = []

    episode_steps = 0

    for i in range(steps):
        A = np.random.randint(actions)
        old_state = env.get_current_state()
        next_S, R, terminate = env.step(A)
        if verbose:
            print("Old state:", np.round(old_state, 3), "-->",
                  "Action:", A, "-->",
                  "New state:", np.round(next_S, 3))
        cumulative_reward += R
        episode_steps += 1
        if terminate:
            if verbose:
                print("\n## Reset ##\n")
            if terminate:
                terminations += 1
                steps_per_episode.append(episode_steps)
                episode_steps *= 0
            env.reset()

    if not terminate:
        steps_per_episode.append(episode_steps)

    print("Number of steps per episode:", steps_per_episode)
    print("Number of episodes that reached the end:", terminations)
    average_length = np.average(episode_steps)
    print("The average number of steps per episode was:", average_length)
    print("Cumulative reward:", cumulative_reward)
    print("\n\n")


def pumping_action_test(steps=100, verbose=False):
    print("==== Results with Pumping Action Policy ====")
    config = Config()

    config.current_step = 0
    env = MountainCar(config)

    steps_per_episode = []
    return_per_episode = []

    episode_steps = 0
    episode_return = 0
    terminations = 0
    for i in range(steps):
        current_state = env.get_current_state()
        A = 1 + np.sign(current_state[1])
        old_state = env.get_current_state()
        next_S, R, terminate = env.step(A)
        if verbose:
            print("Old state:", np.round(old_state, 3), "-->",
                  "Action:", A, "-->",
                  "New state:", np.round(next_S, 3))

        episode_steps += 1
        episode_return += R
        if terminate:
            terminations += 1
            if verbose:
                print("\n## Reset ##\n")
            env.reset()
            steps_per_episode.append(episode_steps)
            return_per_episode.append(episode_return)
            episode_steps *= 0
            episode_return *= 0

    print("Number of steps per episode:", steps_per_episode)
    print("Number of successful episodes:", terminations)
    print("Return per episode:", return_per_episode)
    print("The average return per episode is:", np.mean(return_per_episode))


def sarsa_zero_test(steps=10000, add_new_centers=False, number_of_irrelevant_features=0):
    import matplotlib.pyplot as plt
    from src.env.RandomFeatures_task import LinearFunctionApproximator
    from src.step_size_methods.sgd import SGD

    # epsilon greedy policy
    def choose_action(av_array: np.ndarray, epsilon):
        p = np.random.rand()
        if p > epsilon:
            argmax_av = np.random.choice(np.flatnonzero(av_array == av_array.max()))
            return argmax_av
        else:
            return np.random.randint(av_array.size)

    # for computing action values
    def get_action_values(n, features, approximator_list):
        action_values = np.zeros(n, dtype=np.float64)
        for k in range(n):
            action_values[k] += approximator_list[k].get_prediction(features)
        return action_values

    completed_episodes_per_run = []
    for _ in range(1):
        print("==== Results for Sarsa(0) with Epsilon Greedy Policy ====")
        config = Config()

        # setting up feature function
        config.state_dims = 2
        config.state_lims = np.array(((-1,1), (-1,1)), dtype=np.float64)
        # config.initial_centers = np.array(((0.0,0.0), (-1.8,0), (1.8,0), (0.0,-1.8), (0.0,1.8)), dtype=np.float64)
        config.initial_centers = np.array(((0.0, 0.0), (0.25,0.25), (0.25,-0.25), (-0.25,-0.25), (-0.25,0.25)), dtype=np.float64)
        config.sigma = 0.5
        config.init_noise_mean = 0.0
        config.init_noise_var = 0.01
        feature_function = RadialBasisFunction(config)

        # setting up environment
        config.norm_state = True
        env = MountainCar(config)

        # function approximator and optimizer parameters
        num_actions = 3
        random_action_prob = 0.1
        gamma = 0.99
        config.num_obs_features = feature_function.num_features
        config.max_num_features = 200  # as long as this is more than 12
        config.num_actions = num_actions
        config.alpha = 0.005
        config.rescale = False
        config.parameter_size = feature_function.num_features
        function_approximator = []
        optimizer = []
        # one instance for each action
        for i in range(num_actions):
            function_approximator.append(LinearFunctionApproximator(config))
            optimizer.append(SGD(config))

        # setting up summaries
        all_episodes_return = []
        episode_return = 0

        # setting up initial state, action, features, and action values
        curr_s = env.get_current_state()
        curr_features = feature_function.get_observable_features(curr_s)
        curr_avs = get_action_values(num_actions, curr_features, function_approximator)
        curr_a = choose_action(curr_avs, random_action_prob)
        midpoint_episode = 0
        for i in range(steps):
            # get current action values
            curr_avs = get_action_values(num_actions, curr_features, function_approximator)
            # execute current action
            next_s, r, terminal = env.step(curr_a)
            next_features = feature_function.get_observable_features(next_s)
            # get next action values and action
            next_action_values = get_action_values(num_actions, next_features, function_approximator)
            next_action = choose_action(next_action_values, random_action_prob)
            # compute TD error for Sarsa(0)
            td_error = r + gamma * (1 - terminal) * next_action_values[next_action] - curr_avs[curr_a]
            # update weight vector
            _, ss, new_weights = optimizer[curr_a].update_weight_vector(td_error, curr_features,
                                                                function_approximator[curr_a].get_weight_vector())
            function_approximator[curr_a].update_weight_vector(new_weights)
            # set current features and action
            curr_features = next_features
            curr_a = next_action
            # keep track of sum of rewards
            episode_return += r
            # if terminal state
            if terminal:
                env.reset()
                all_episodes_return.append(episode_return)
                episode_return *= 0
                curr_s = env.get_current_state()
                curr_features = feature_function.get_observable_features(curr_s)
                curr_avs = get_action_values(num_actions, curr_features, function_approximator)
                curr_a = choose_action(curr_avs, random_action_prob)
            # if midpoint of training
            if (i+1) == (steps//2):
                if add_new_centers:
                    new_centers = np.array(((0,0), (0.25,0.25), (0.25,-0.25), (-0.25,-0.25), (-0.25,0.25)),
                                           dtype=np.float64)
                    feature_function.add_centers(new_centers, noise_var=0, noise_mean=0)
                    for k in range(num_actions):
                        function_approximator[k].increase_num_features(new_centers.shape[0])
                        optimizer[k].increase_size(new_centers.shape[0], init_stepsize=0.25)
                if number_of_irrelevant_features > 0:
                    new_feature_mean = 0.0
                    new_feature_var = 0.05
                    fake_features = True
                    feature_function.add_feature(number_of_irrelevant_features, noise_mean=new_feature_mean,
                                                 noise_var=new_feature_var, fake_feature=fake_features)
                    for k in range(num_actions):
                        function_approximator[k].increase_num_features(number_of_irrelevant_features)
                        optimizer[k].increase_size(number_of_irrelevant_features)
                curr_features = feature_function.get_observable_features(curr_s)
                midpoint_episode = len(all_episodes_return)
        completed_episodes_per_run.append(len(all_episodes_return))
        print("Number of episodes completed: {0}".format(len(all_episodes_return)))
    print("Average episodes completed: {0:0.4f}".format(np.average(completed_episodes_per_run)))

    print("Return per episode:\n", all_episodes_return)
    plt.plot(np.arange(len(all_episodes_return)) + 1, all_episodes_return)
    plt.vlines(x=midpoint_episode, ymin=-800, ymax=0)
    plt.ylim((-800,0))
    plt.show()
    plt.close()


if __name__ == "__main__":
    import time

    test_random_policy = False
    test_pumping_action = False
    test_sarsa_zero = True

    if test_random_policy:
        random_policy_test(steps=10000, verbose=False)
    if test_pumping_action:
        pumping_action_test(steps=10000, verbose=False)
    if test_sarsa_zero:
        start_time = time.time()
        sarsa_zero_test(steps=200000, add_new_centers=True, number_of_irrelevant_features=0)
        end_time = time.time()
        print('Total running time in minutes: {0:0.2f}'.format((end_time-start_time)/60))

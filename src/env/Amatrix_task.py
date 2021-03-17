import numpy as np


def projection(u: np.ndarray, v: np.ndarray):
    numerator = np.dot(u, v)
    denominator = np.dot(u, u)
    return (numerator / denominator) * u


def gram_schmidt_process(n:int, normalize=True):
    """
    :param n: number of vectors to produce
    """
    X = np.random.random(size=(n, n))
    U = np.zeros(shape=(n, n), dtype=np.float64)
    for i in range(n):
        current_vec = X[:, i]
        new_vector = current_vec
        for j in range(i):
            new_vector -= projection(U[:, j], new_vector)
        if normalize:
            U[:, i] = new_vector / np.sqrt(np.dot(new_vector, new_vector))
        else:
            U[:, i] = new_vector
    return U


def linearly_independent_basis(n:int):
    """
    Creates a linearly independent basis from random vectors with norm of 1.
    See this link to understand why these type of vectors create a linearly independent basis:
    https://math.stackexchange.com/questions/432447/probability-that-n-vectors-drawn-randomly-from-mathbbrn-are-linearly-ind
    :param n: number of vectors in the basis
    :return: n by n matrix with linearly independent columns
    """
    X = np.random.uniform(low=-1, high=1, size=(n,n))
    X /= np.linalg.norm(X, axis=0)
    return X


class Amatrix:

    def __init__(self, n: int, m: int, orthogonal=True):
        assert m <= n
        self.orthogonal = orthogonal
        if self.orthogonal:
            self.Amatrix = gram_schmidt_process(n, normalize=True)
        else:
            self.Amatrix = linearly_independent_basis(n)
        self.theta_star = np.random.normal(0, 3, n)
        self.theta_star = self.theta_star / np.linalg.norm(self.theta_star)
        self.n = n
        self.m = m
        self.noise_stddev = 0.1

    def sample_target(self, k: int, noisy=False):
        target = np.dot(self.Amatrix[k, :], self.theta_star)
        if noisy:
            target += np.random.normal(0, self.noise_stddev)
        return target

    def get_approx_A(self):
        return self.Amatrix[:, :self.m]

    def get_new_good_features(self, k: int):
        assert self.m + k <= self.n
        new_features = self.Amatrix[:, self.m:(self.m + k)]
        self.m += k
        return new_features

    def get_new_bad_features(self, k: int):
        approx_A = self.get_approx_A()
        new_features = np.matmul(approx_A, np.random.standard_normal((self.m,k)))
        new_features = new_features / np.linalg.norm(new_features, axis=0)  # normalized features
        return new_features

    def get_new_partially_good_feature(self, k: int, principal_feature_weight=1/2):
        assert self.m + k <= self.n
        approx_A = self.get_approx_A()
        linearly_dependent_part = np.sum(approx_A, axis=1, keepdims=True) * (1 - principal_feature_weight)
        linearly_independent_part = self.Amatrix[:, self.m:(self.m + k)] * principal_feature_weight
        new_features = linearly_dependent_part + linearly_independent_part
        new_features = new_features / np.linalg.norm(new_features, axis=0)  # normalized features
        return new_features


def amatrix_test():

    n = 4
    m = 2
    env = Amatrix(n=n, m=m)

    # checking norm of theta star is 1
    print("Theta star:\n{0}".format(env.theta_star))
    print("Norm of theta star: {0}".format(np.linalg.norm(env.theta_star)))

    # printing A matrix
    print("Matrix A:\n{0}".format(env.Amatrix))

    # printing approximate A matrix
    approx_Amatrix = env.get_approx_A()
    print("Approximate A matrix:\n{0}".format(approx_Amatrix))
    bad_features = env.get_new_bad_features(4)

    # testing bad features and checking they're normalized
    print("Bad features:\n{0}".format(bad_features))
    print("(Bad features)^T (Bad features):\n{0}".format(np.matmul(bad_features.T, bad_features)))
    print("The result above should have ones in the diagonal.")

    # testing partially good features
    partially_good_features = env.get_new_partially_good_feature(2, principal_feature_weight=1/4)
    print("Partially good features:\n{0}".format(partially_good_features))
    print("Norm of partially good features:\n{0}".format(np.linalg.norm(partially_good_features, axis=0)))

    # testing good features and adding them to the approximate A matrix
    good_features = env.get_new_good_features(2)
    print("Good features:\n{0}".format(good_features))
    new_approx_Amatrix = np.hstack((approx_Amatrix, good_features))
    print("New approximate A matrix:\n{0}".format(new_approx_Amatrix))

    sample_size = 10
    print("Generating {0} samples...".format(sample_size))
    for i in range(sample_size):
        rand_row = np.random.randint(n)
        print("\tSample {0}: {1}".format(i+1, env.sample_target(rand_row)))
        print("\t\tNoisy Sample: {0}".format(env.sample_target(rand_row, True)))


def gsp_test():
    n = 4
    Amatrix = gram_schmidt_process(n)
    print("Matrix A:\n{0}".format(Amatrix))
    print("A^T A:\n{0}".format(np.matmul(Amatrix.T, Amatrix)))
    print("A^T A should be equal to the identity matrix except for some precision error.")


if __name__ == "__main__":
    amatrix_test()

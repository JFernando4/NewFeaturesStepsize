import pickle
import numpy as np
import os

class Config:
    """
    Used to store the arguments of all the functions in the package src. If a function requires arguments,
    it's definition will be: func(config), where config has all the parameters needed for the function.
    """
    def __init__(self):
        pass

    def print(self):
        for k, v in self.__dict__.items():
            print("Parameter name: {0}, \tParameter Value: {1}".format(k, v))


def check_attribute(object_type, attr_name, default_value, choices=None, data_type=None):
    if not hasattr(object_type, attr_name):
        print("Creating attribute", attr_name)
        setattr(object_type, attr_name, default_value)
    if choices:
        if getattr(object_type, attr_name) not in choices:
            raise ValueError("The possible values for this attribute are: " + str(choices))
    if data_type:
        if not isinstance(default_value, data_type):
            raise ValueError("Wrong data type. The expected data type is: " + str(data_type))
    return getattr(object_type, attr_name)


def check_dict_else_default(dict_type, key_name, default_value):
    assert isinstance(dict_type, dict)
    if key_name not in dict_type.keys():
        dict_type[key_name] = default_value
    return dict_type[key_name]


def check_multiple_attributes(object_type, *attributes):
    """
    checks if an object has the desired arguments
    attributes should be a list of 3-tuples with ("name", default_value, [choices], data_type))
    """
    values = []
    for attr in attributes:
        assert len(attr) == 4
        values.append(
            check_attribute(object_type, attr_name=attr[0], default_value=attr[1], choices=attr[2],
                            data_type=attr[3])
        )
    return values


""" For loading results """
def load_results(results_path):
    with open(results_path, mode='rb') as results_file:
        results = pickle.load(results_file)
    return results


def get_average_and_standard_error(num_samples, data_array: np.ndarray, max_entry=10000, bin_size=1):
    """
    Computes the average and standard error over the rows of a 2D array. It can bin together several columns of the
    array by taking the average over consecutive columns.
    :param num_samples: number of total training steps
    :param data_array: 2D array where the shape is (number of runs, number of checkpoints)
    :param max_entry: used for turning nans and infs to a very large number instead
    :param bin_size: number of columns to  bin together
    :return:
    """
    sample_size = data_array.shape[0]
    # replacing nan and inf for max entry
    data_array[np.isnan(data_array)] = max_entry
    data_array[data_array > max_entry] = max_entry
    # bin columns together
    if bin_size > 1:
        assert num_samples % bin_size == 0
        binned_data_array = np.zeros((sample_size, num_samples // bin_size), dtype=np.float64)
        indices = np.arange(num_samples // bin_size) * bin_size
        for i in range(bin_size):
            binned_data_array += data_array[:, indices + i] / bin_size
    else:
        binned_data_array = data_array
    # compute average and standard error
    data_avg = np.average(binned_data_array, axis=0)
    data_ste = np.std(binned_data_array, axis=0) / np.sqrt(sample_size)
    return data_avg, data_ste


def load_avg_learning_curve(results_dirpath, results_name, learning_curve_name, method_name, num_samples, bin_size=5,
                            secondary_results_name=None, use_moving_sume=False):
    """
    If available, loads the average learning curve for a given method. Otherwise, it computes the average
    learning curve along with standard error and stores it for future use.
    :param results_dirpath: path to directory with the result file or the learning curve file
    :param results_name: filename of raw results
    :param learning_curve_name: filename of the average learning curve
    :param method_name: name of the method
    :param num_samples: number of total training_steps
    :param bin_size: number of columns averaged together in each bin
    :param secondary_results_name: in the case there are multiple results in results_dict, this a string that specifies
                                   which results should be used
    :param use_moving_sume: whether to take the moving sum along the rows of the data array
    :return: (average, standard_error)
    """
    learning_curve_path = os.path.join(results_dirpath, learning_curve_name)
    if os.path.isfile(learning_curve_path):                     # check if average learning curve file exists
        avg_learning_curve = load_results(learning_curve_path)
    else:                                                       # compute learning curve otherwise
        results_dict = load_results(os.path.join(results_dirpath, results_name))
        if secondary_results_name is None:
            if use_moving_sume:
                ms_results = np.apply_along_axis(lambda a: moving_sum(a,bin_size), 1,
                                                 results_dict[method_name]) + bin_size
                avg_learning_curve = get_average_and_standard_error(data_array=ms_results, bin_size=1,
                                                                    num_samples=num_samples)
            else:
                avg_learning_curve = get_average_and_standard_error(data_array=results_dict[method_name],
                                                                    bin_size=bin_size, num_samples=num_samples)
        else:
            if use_moving_sume:
                ms_results = np.apply_along_axis(lambda a: moving_sum(a, bin_size), 1,
                                                 results_dict[method_name][secondary_results_name]) + bin_size
                avg_learning_curve = get_average_and_standard_error(data_array=ms_results, bin_size=1,
                                                                    num_samples=num_samples)
            else:
                avg_learning_curve = get_average_and_standard_error(bin_size=bin_size, num_samples=num_samples,
                                                        data_array=results_dict[method_name][secondary_results_name])
        with open(learning_curve_path, mode='wb') as learning_curve_file:
            pickle.dump(avg_learning_curve, learning_curve_file)
    return avg_learning_curve


def moving_sum(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src import load_results, get_average_and_standard_error, load_avg_learning_curve

NUM_SAMPLES = 2000
Xaxis = (np.arange(NUM_SAMPLES) + 1) * 50
MIDPOINT = 50000
RESCALED_SGD_COLORS = {'solid': '#3B3B3B', 'lighter': '#c4c4c4'}


def get_num_features(exp_type):
    """
    :param exp_type: name of the experiment
    :return: (number of initial features, number of good features added after initialization,
              number of bad features added after initialization)
    """
    if exp_type == 'add_bad_feat':
        return 3, 0, 1      # initial good features, added new features, added bad features
    elif exp_type == 'add_good_feat':
        return 3, 1, 0
    elif exp_type == 'continuously_add_bad':
        return 3, 0, 200
    elif exp_type == 'add_one_good_one_bad':
        return 3, 1, 1
    elif exp_type == 'add_one_good_10_bad':
        return 3, 1, 10
    elif exp_type == 'add_one_good_100_bad':
        return 3, 1, 100
    else:
        raise ValueError("{0} is not a valid method.".format(exp_type))


def sgd_plots(save_plots=False, bin_size=5):
    assert NUM_SAMPLES % bin_size == 0

    main_method = 'sgd'
    results_dir = os.path.join(os.getcwd(), 'results', 'noisy_random_features_task_add_features', 'num_true_features_4')
    ylim = (0.2, 1.2)
    Xaxis = (np.arange(NUM_SAMPLES // bin_size) + 1) * (50 * bin_size)

    colors = [
        "#78b5d2",  # blue
        "#ffdd8e",  # yellow
        "#b696d7",  # purple
    ]

    lighter_colors = [
        "#c6dfec",  # blue
        "#fff0cc",  # yellow
        "#d9c8ea",  # purple
    ]

    experiment_names = ['add_good_feat', 'add_bad_feat', 'continuously_add_bad']
    for exp_name in experiment_names:

        results_dict = load_results(os.path.join(results_dir, main_method, exp_name, 'results.p'))

        names = [k for k in results_dict.keys() if k != 'sample_size']
        for i, n in enumerate(names):
            avg, ste = get_average_and_standard_error(data_array=results_dict[n], bin_size=bin_size, num_samples=NUM_SAMPLES)
            plt.plot(Xaxis, avg, color=colors[i], label=n)
            plt.fill_between(Xaxis, avg + ste, avg - ste, color=lighter_colors[i])

        avg, ste = load_avg_learning_curve(results_dirpath=os.path.join(results_dir, 'rescaled_sgd', exp_name),
                                           bin_size=bin_size, results_name='results.p', method_name='rescaled_sgd',
                                           learning_curve_name='avg_learning_curve_bins' + str(bin_size) + '.p',
                                           num_samples=NUM_SAMPLES)
        plt.plot(Xaxis, avg, color=RESCALED_SGD_COLORS['solid'], linestyle='dotted')
        plt.fill_between(Xaxis, avg + ste, avg - ste, color=RESCALED_SGD_COLORS['lighter'])

        if exp_name != 'continuously_add_bad':
            plt.vlines(x=MIDPOINT, ymin=0, ymax=100, colors='#BD2A4E', linestyles='dashed')
        plt.legend()
        plt.xlabel("Training Examples", fontsize=18)
        plt.ylabel("Mean\nSquared\nError", fontsize=18, rotation=0, labelpad=50)
        plt.title("{0}".format(exp_name))
        plt.ylim(ylim)
        if save_plots:
            plt.savefig(exp_name + '.svg', dpi=200)
        else:
            plt.show()
        plt.close()

    ########################################
    # add_one_good_100_bad data #####
    ########################################
    experiment_names = ['add_one_good_one_bad', 'add_one_good_10_bad', 'add_one_good_100_bad']
    og_names = ['small', 'med', 'large']
    names = []
    for i in range(len(og_names)):  # every combination of (x,y) for x,y in {'small', 'med', 'large'}
        for j in range(len(og_names)):
            names.append('good-' + og_names[i] + '_bad-' + og_names[j])

    for exp_name in experiment_names:
        results_dict = load_results(os.path.join(results_dir, main_method, exp_name, 'results.p'))

        n = 3
        for j in range(n):
            avg, ste = load_avg_learning_curve(results_dirpath=os.path.join(results_dir, 'rescaled_sgd', exp_name),
                                               bin_size=bin_size, results_name='results.p', method_name='rescaled_sgd',
                                               learning_curve_name='avg_learning_curve_bins' + str(bin_size) + '.p',
                                               num_samples=NUM_SAMPLES)
            plt.plot(Xaxis, avg, color=RESCALED_SGD_COLORS['solid'], linestyle='dotted')
            plt.fill_between(Xaxis, avg + ste, avg - ste, color=RESCALED_SGD_COLORS['lighter'])

            i = 0
            for k in range(n):
                name_index = j * n + k
                avg, ste = get_average_and_standard_error(data_array=results_dict[names[name_index]], bin_size=bin_size,
                                                          num_samples=NUM_SAMPLES)
                plt.plot(Xaxis, avg, color=colors[i], label=names[name_index])
                plt.fill_between(Xaxis, avg + ste, avg - ste, color=lighter_colors[i])
                i += 1
            plt.vlines(x=MIDPOINT, ymin=0, ymax=100, colors='#BD2A4E', linestyles='dashed')
            plt.legend()
            plt.title("Good features with {0} stepsize, {1}".format(og_names[j], exp_name))
            plt.xlabel("Training Examples", fontsize=18)
            plt.ylabel("Mean\nSquared\nError", fontsize=18, rotation=0, labelpad=50)
            plt.ylim(ylim)
            if save_plots:
                plt.savefig(exp_name + og_names[j] + '.svg', dpi=200)
            else:
                plt.show()
            plt.close()


def stepsize_methods_plots(save_plots=False, plot_learning_curves=False, plot_stepsizes=False, bin_size=10):

    experiment_names = ['add_good_feat', 'add_bad_feat', 'add_one_good_one_bad',
                        'add_one_good_10_bad', 'add_one_good_100_bad', 'continuously_add_bad']
    methods = ['idbd', 'autostep', 'adam', 'sidbd']
    results_dir = os.path.join(os.getcwd(), 'results', 'noisy_random_features_task_add_features', 'num_true_features_4')
    Xaxis = (np.arange(NUM_SAMPLES // bin_size) + 1) * (50 * bin_size)

    methods_colors = [
        "#332288",  # blue
        "#DDCC77",  # yellow
        "#AA4499",  # purple
        "#117733",  # green-ish
    ]

    methods_lighter_colors = [
        "#b3add3",  # blue
        "#f2ebcc",  # yellow
        "#ebd3e7",  # purple
        "#b8d5c2",  # green-sih
    ]

    stepsize_colors = [
        "#44AA99",   # greenish
        "#DDCC77",   # yellowish
        "#CC6677"    # pinkish
    ]

    """ Plot Learning Curves """
    if plot_learning_curves:
        ylim = (0.2,1.2)
        for exp_name in experiment_names:
            for i, m in enumerate(methods):

                avg, ste = load_avg_learning_curve(results_dirpath=os.path.join(results_dir, m, exp_name),
                                                   bin_size=bin_size, results_name='results.p', method_name=m,
                                                   learning_curve_name='avg_learning_curve_bins' + str(bin_size) + '.p',
                                                   num_samples=NUM_SAMPLES)

                plt.plot(Xaxis, avg, color=methods_colors[i], label=m)
                plt.fill_between(Xaxis, avg + ste, avg - ste, color=methods_lighter_colors[i])

            avg, ste = load_avg_learning_curve(results_dirpath=os.path.join(results_dir, 'rescaled_sgd', exp_name),
                                               bin_size=bin_size, results_name='results.p', method_name='rescaled_sgd',
                                               learning_curve_name='avg_learning_curve_bins' + str(bin_size) + '.p',
                                               num_samples=NUM_SAMPLES)
            plt.plot(Xaxis, avg, color=RESCALED_SGD_COLORS['solid'], label='rescaled_sgd', linestyle='dotted')
            plt.fill_between(Xaxis, avg + ste, avg - ste, color=RESCALED_SGD_COLORS['lighter'])

            plt.legend()
            plt.xlabel('Training Examples', fontsize=18)
            plt.ylabel("Mean\nSquared\nError", fontsize=18, rotation=0, labelpad=50)
            plt.title('{0}'.format(exp_name))
            if exp_name != 'continuously_add_bad':
                plt.vlines(x=MIDPOINT, ymin=0, ymax=100, colors='#BD2A4E', linestyles='dashed')
            plt.ylim(ylim)
            if save_plots:
                plt.savefig('learning_curve_' + exp_name + '.svg', dpi=200)
            else:
                plt.show()
            plt.close()

    """ Plot Stepsizes """
    methods = ['idbd', 'autostep', 'adam', 'rescaled_sgd', 'sidbd']
    if plot_stepsizes:
        ylim = (0.0, 0.025)
        Xaxis = (np.arange(NUM_SAMPLES) + 1) * 50
        for exp_name in experiment_names:
            for i, m in enumerate(methods):

                avg_stepsizes, _ = load_avg_learning_curve(results_dirpath=os.path.join(results_dir, m, exp_name),
                                                           bin_size=1, results_name='results.p', method_name='stepsizes',
                                                           learning_curve_name='avg_stepsize_bins' + str(1) + '.p',
                                                           num_samples=NUM_SAMPLES)

                num_features = get_num_features(exp_name)
                current_features = 0
                for j, nf in enumerate(num_features):
                    for k in range(current_features, current_features + nf):
                        plt.plot(Xaxis, avg_stepsizes[:, k], color=stepsize_colors[j])
                    current_features += nf
                plt.title('{0}, {1}'.format(exp_name, m))
                if exp_name != 'continuously_add_bad':
                    plt.vlines(x=MIDPOINT, ymin=0, ymax=100, colors='#BD2A4E', linestyles='dashed')
                plt.ylim(ylim)
                plt.xlabel('Training Examples', fontsize=18)
                plt.ylabel("Stepsize", fontsize=18, rotation=0, labelpad=50)
                if save_plots:
                    plt.savefig("stepsize_plot_" + exp_name + '_' + m + '.svg', dpi=200)
                else:
                    plt.show()
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sgd_plots', action='store_true', default=False)
    parser.add_argument('--stepsize_methods_plots', action='store_true', default=False)
    parser.add_argument('--plot_learning_curves', action='store_true', default=False)
    parser.add_argument('--plot_stepsizes', action='store_true', default=False)
    parser.add_argument('-sp', '--save_plots', action='store_true', default=False)
    parser.add_argument('-bs', '--bin_size', action='store', default=1, type=int)
    plot_parameters = parser.parse_args()

    if plot_parameters.sgd_plots:
        sgd_plots(bin_size=plot_parameters.bin_size, save_plots=plot_parameters.save_plots)
    if plot_parameters.stepsize_methods_plots:
        stepsize_methods_plots(bin_size=plot_parameters.bin_size, save_plots=plot_parameters.save_plots,
                               plot_learning_curves=plot_parameters.plot_learning_curves,
                               plot_stepsizes=plot_parameters.plot_stepsizes)

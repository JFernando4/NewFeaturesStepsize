import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src import get_average_and_standard_error, load_results, load_avg_learning_curve, moving_sum

NUM_SAMPLES = 200000
Xaxis = (np.arange(NUM_SAMPLES) + 1) * 100
MIDPOINT = 100000
RESCALED_SGD_COLORS = {'solid': '#3B3B3B', 'lighter': '#c4c4c4'}


def get_num_features(exp_type):
    """
    :param exp_type: name of the experiment
    :return: (number of initial features, number of good features added after initialization,
              number of bad features added after initialization)
    """
    if exp_type == 'add_bad_feats':
        return 5, 0, 5      # initial good features, added new features, added bad features
    elif exp_type == 'add_good_feats':
        return 5, 5, 0
    elif exp_type == 'continuously_add_bad':
        return 5, 0, 200
    elif exp_type == 'add_5_good_5_bad':
        return 5, 5, 5
    elif exp_type == 'add_5_good_20_bad':
        return 5, 5, 20
    elif exp_type == 'add_5_good_100_bad':
        return 5, 5, 100
    else:
        raise ValueError("{0} is not a valid method.".format(exp_type))


def sgd_plots(save_plots=False, fake_features=False, bin_size=1000):
    assert NUM_SAMPLES % bin_size == 0
    downsample = 1000
    Xaxis = np.arange(NUM_SAMPLES - bin_size + 1) + 1

    method = 'sgd'
    ms_func = lambda a: moving_sum(a, bin_size)

    if fake_features:
        task_name = 'mountain_car_task_add_fake_features'
        experiment_names = ['add_good_feats', 'add_bad_feats', 'continuously_add_bad']
    else:
        task_name = 'mountain_car_task_add_features'
        experiment_names = ['add_good_feats', 'add_bad_feats', 'continuously_add_bad']

    results_dir = os.path.join(os.getcwd(), 'results', task_name)
    ylim = (0.0, 8 * (bin_size // 1000))

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

    for exp_name in experiment_names:

        results_dict = load_results(os.path.join(results_dir, method, exp_name, 'results.p'))

        names = [k for k in results_dict.keys() if k != 'sample_size']
        for i, n in enumerate(names):
            moving_sum_per_run = np.apply_along_axis(ms_func, 1, results_dict[n]['reward_per_step']) + bin_size
            avg, ste = get_average_and_standard_error(data_array=moving_sum_per_run, bin_size=1,
                                                      num_samples=NUM_SAMPLES)
            plt.plot(Xaxis[::downsample], avg[::downsample], color=colors[i], label=n)
            plt.fill_between(Xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
                             color=lighter_colors[i])

        avg, ste = load_avg_learning_curve(results_dirpath=os.path.join(results_dir, 'rescaled_sgd', exp_name),
                                           bin_size=bin_size, results_name='results.p', method_name='rescaled_sgd',
                                           learning_curve_name='avg_learning_curve_bins' + str(bin_size) + '.p',
                                           num_samples=NUM_SAMPLES, secondary_results_name='reward_per_step',
                                           use_moving_sume=True)
        plt.plot(Xaxis[::downsample], avg[::downsample], color=RESCALED_SGD_COLORS['solid'], linestyle='dotted', label='rescaled_sgd')
        plt.fill_between(Xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
                         color=RESCALED_SGD_COLORS['lighter'])

        if exp_name != 'continuously_add_bad':
            plt.vlines(x=MIDPOINT, ymin=0, ymax=100, colors='#BD2A4E', linestyles='dashed')
        plt.legend()
        plt.xlabel("Training Examples", fontsize=18)
        plt.ylabel("Episodes Completed\nOver {0} Steps".format(bin_size), fontsize=18, rotation=0, labelpad=100)
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
    experiment_names = ['add_5_good_5_bad', 'add_5_good_20_bad', 'add_5_good_100_bad']
    og_names = ['small', 'med', 'large']
    names = []
    for i in range(len(og_names)):  # every combination of (x,y) for x,y in {'small', 'med', 'large'}
        for j in range(len(og_names)):
            names.append('good-' + og_names[i] + '_bad-' + og_names[j])

    for exp_name in experiment_names:
        results_dict = load_results(os.path.join(results_dir, method, exp_name, 'results.p'))

        n = 3
        for j in range(n):
            avg, ste = load_avg_learning_curve(results_dirpath=os.path.join(results_dir, 'rescaled_sgd', exp_name),
                                               bin_size=bin_size, results_name='results.p', method_name='rescaled_sgd',
                                               learning_curve_name='avg_learning_curve_bins' + str(bin_size) + '.p',
                                               num_samples=NUM_SAMPLES, secondary_results_name='reward_per_step',
                                               use_moving_sume=True)
            plt.plot(Xaxis[::downsample], avg[::downsample], color=RESCALED_SGD_COLORS['solid'], linestyle='dotted', label='rescaled_sgd')
            plt.fill_between(Xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample], color=RESCALED_SGD_COLORS['lighter'])

            i = 0
            for k in range(n):
                name_index = j * n + k
                moving_sum_per_run = np.apply_along_axis(ms_func, 1,
                                                         results_dict[names[name_index]]['reward_per_step']) + bin_size
                avg, ste = get_average_and_standard_error(data_array=moving_sum_per_run, bin_size=1,
                                                          num_samples=NUM_SAMPLES)
                plt.plot(Xaxis[::downsample], avg[::downsample], color=colors[i], label=names[name_index])
                plt.fill_between(Xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
                                 color=lighter_colors[i])
                i += 1

            plt.vlines(x=MIDPOINT, ymin=0, ymax=100, colors='#BD2A4E', linestyles='dashed')
            plt.legend()
            plt.title("Good features with {0} stepsize, {1}".format(og_names[j], exp_name))
            plt.xlabel("Training Examples", fontsize=18)
            plt.ylabel("Episodes Completed\nOver {0} Steps".format(bin_size), fontsize=18, rotation=0, labelpad=100)
            plt.ylim(ylim)
            if save_plots:
                plt.savefig(exp_name + og_names[j] + '.svg', dpi=200)
            else:
                plt.show()
            plt.close()


def stepsize_methods_plots(fake_features=False, save_plots=False, plot_learning_curves=False, plot_stepsizes=False,
                           bin_size=5):
    downsample = 1000
    methods = ['idbd', 'autostep', 'adam', 'sidbd']
    Xaxis = np.arange(NUM_SAMPLES - bin_size + 1) + 1

    experiment_names = ['add_good_feats', 'add_bad_feats', 'add_5_good_5_bad', 'add_5_good_20_bad',
                        'add_5_good_100_bad', 'continuously_add_bad']
    if fake_features:
        task_name = 'mountain_car_task_add_fake_features'
    else:
        task_name = 'mountain_car_task_add_features'
        # experiment_names.append('add_good_feats')

    results_dir = os.path.join(os.getcwd(), 'results', task_name)

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

    if plot_learning_curves:
        """ Plot Learning Curves """
        ylim = (0.0, 8 * (bin_size // 1000))
        for exp_name in experiment_names:
            for i, m in enumerate(methods):

                avg, ste = load_avg_learning_curve(results_dirpath=os.path.join(results_dir, m, exp_name),
                                                   bin_size=bin_size, results_name='results.p', method_name=m,
                                                   learning_curve_name='avg_learning_curve_bins'+str(bin_size)+'.p',
                                                   num_samples=NUM_SAMPLES, secondary_results_name='reward_per_step',
                                                   use_moving_sume=True)

                plt.plot(Xaxis[::downsample], avg[::downsample], color=methods_colors[i], label=m)
                plt.fill_between(Xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
                                 color=methods_lighter_colors[i])

            avg, ste = load_avg_learning_curve(results_dirpath=os.path.join(results_dir, 'rescaled_sgd', exp_name),
                                               bin_size=bin_size, results_name='results.p', method_name='rescaled_sgd',
                                               learning_curve_name='avg_learning_curve_bins'+str(bin_size)+'.p',
                                               num_samples=NUM_SAMPLES, secondary_results_name='reward_per_step',
                                               use_moving_sume=True)

            plt.plot(Xaxis[::downsample], avg[::downsample], color=RESCALED_SGD_COLORS['solid'], label='rescaled_sgd',
                     linestyle='dotted')
            plt.fill_between(Xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
                             color=RESCALED_SGD_COLORS['lighter'])

            plt.legend()
            plt.xlabel("Training Examples", fontsize=18)
            plt.ylabel("Episodes Completed\nOver {0} Steps".format(bin_size), fontsize=18, rotation=0, labelpad=100)
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
    if plot_stepsizes:
        methods = ['idbd', 'autostep', 'adam', 'rescaled_sgd', 'sidbd']
        checkpoint = 1000
        Xaxis = (np.arange(NUM_SAMPLES // checkpoint) + 1) * 1000

        ylim = (0,0.23)
        #     {
        #     'mountain_car_task_add_features': {'add_good_feats': (0, 0.23), 'add_bad_feats': (0, 0.15),
        #                                       'add_5_good_5_bad': (0, 0.20), 'add_5_good_20_bad': (0, 0.15),
        #                                       'add_5_good_100_bad': (0, 0.15), 'continuously_add_bad': (0, 0.15)},
        #     'mountain_car_task_add_fake_features': {'add_good_feats': (0, 0.23), 'add_bad_feats': (0, 0.15),
        #                                             'add_5_good_5_bad': (0, 0.23), 'add_5_good_20_bad': (0, 0.23),
        #                                             'add_5_good_100_bad': (0, 0.20), 'continuously_add_bad': (0, 0.15)}
        # }

        # ylim = (0, 0.23)
        for exp_name in experiment_names:
            for i, m in enumerate(methods):

                avg_stepsizes, _ = load_avg_learning_curve(results_dirpath=os.path.join(results_dir, m, exp_name),
                                                           bin_size=1, results_name='results.p',
                                                           method_name='stepsizes',
                                                           learning_curve_name='avg_stepsize_bins'+str(1)+'.p',
                                                           num_samples=NUM_SAMPLES)

                num_features = get_num_features(exp_name)
                current_features = 0
                serial_name = 0
                for j, nf in enumerate(num_features):
                    for k in range(current_features, current_features + nf):
                        plt.plot(Xaxis, avg_stepsizes[:, k], color=stepsize_colors[j])
                        # plt.ylim(ylim[m][exp_name])
                        # plt.savefig(str(serial_name) + '.png')
                        # plt.show()
                        # plt.close()
                        serial_name += 1
                    current_features += nf
                plt.title('{0}, {1}'.format(exp_name, m))
                plt.xlabel("Training Examples", fontsize=18)
                plt.ylabel("Stepsize", fontsize=18, rotation=0, labelpad=50)
                if exp_name != 'continuously_add_bad':
                    plt.vlines(x=MIDPOINT, ymin=0, ymax=100, colors='#BD2A4E', linestyles='dashed')
                plt.ylim(ylim)#ylim[task_name][exp_name])
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
    parser.add_argument('-ff', '--fake_features', action='store_true', default=False)
    plot_parameters = parser.parse_args()

    if plot_parameters.sgd_plots:
        sgd_plots(bin_size=plot_parameters.bin_size, save_plots=plot_parameters.save_plots,
                  fake_features=plot_parameters.fake_features)
    if plot_parameters.stepsize_methods_plots:
        stepsize_methods_plots(bin_size=plot_parameters.bin_size, save_plots=plot_parameters.save_plots,
                               plot_learning_curves=plot_parameters.plot_learning_curves,
                               plot_stepsizes=plot_parameters.plot_stepsizes,
                               fake_features=plot_parameters.fake_features)

    # color_blind_palette = {
    #     'blue-ish': '#332288',   # I'm not good at naming colors
    #     'green-ish': '#117733',
    #     'light green-ish': '#44AA99',
    #     'light blue-ish': '#88CCEE',
    #     'red-ish': '#CC6677',
    #     'purple-ish': '#CC6677',
    #     'dark purple-ish': '#882255'
    # }

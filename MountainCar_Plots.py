import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle

from src import get_average_and_standard_error, load_results, moving_sum, load_avg_learning_curve

NUM_SAMPLES = 200000
EXTENDED_NUM_SAMPLES = NUM_SAMPLES * 2
MIDPOINT = 100000
RESCALED_SGD_COLORS = {'solid': '#3B3B3B', 'lighter': '#c4c4c4'}

weights_and_stepsize_colors = [
    "#44AA99",   # greenish
    "#DDCC77",   # yellowish
    "#CC6677"    # pinkish
]

methods_colors = {
    'idbd': "#332288",                  # blue
    'autostep': "#DDCC77",              # yellow
    'adam': "#AA4499",                  # purple
    'sidbd': "#117733",                 # green-ish
    'slow_adam': '#882255',             # dark purple-ish
    'rescaled_stepsize': '#3B3B3B',     # black
}

methods_lighter_colors = {
    'idbd': "#b3add3",                  # blue
    'autostep': "#f2ebcc",              # yellow
    'adam' :"#ebd3e7",                  # purple
    'sidbd':"#b8d5c2",                  # green-ish
    'slow_adam': '#e0c4d2',             # dark purple-ish
    'rescaled_stepsize': '#c4c4c4',     # black
}

sgd_colors = [
    "#78b5d2",  # blue
    "#ffdd8e",  # yellow
    "#b696d7",  # purple
]

sgd_lighter_colors = [
    "#c6dfec",  # blue
    "#fff0cc",  # yellow
    "#d9c8ea",  # purple
]


def load_learning_curves(results_dirpath, method_name, exp_type, bin_size=1000,
                         exclude_diverging=False, num_samples=NUM_SAMPLES, recompute=False):

    results_filename = "learning_curve_bs" + str(bin_size)
    if exclude_diverging: results_filename += "_excluding_diverging_runs"
    results_filename += ".p"

    learning_curve_path = os.path.join(results_dirpath, method_name, exp_type, results_filename)
    if os.path.isfile(learning_curve_path) and not recompute:
        num_diverging, avg_learning_curve, ste_learning_curve = load_results(learning_curve_path)
    else:
        raw_results_path = os.path.join(results_dirpath, method_name, exp_type, 'results.p')
        results_dict = load_results(raw_results_path)

        diverging_runs = results_dict[method_name]['diverging_runs']
        num_diverging = np.sum(diverging_runs)
        reward_per_step = results_dict[method_name]['reward_per_step']
        if exclude_diverging:
            reward_per_step = reward_per_step[diverging_runs == 0, :]

        ms_reward_per_step = np.apply_along_axis(lambda a: moving_sum(a, bin_size), 1, reward_per_step) + bin_size
        avg_learning_curve, ste_learning_curve = get_average_and_standard_error(data_array=ms_reward_per_step,
                                                                                bin_size=1, num_samples=num_samples)

        with open(learning_curve_path, mode='wb') as learning_curve_file:
            pickle.dump((num_diverging, avg_learning_curve, ste_learning_curve), learning_curve_file)

    return num_diverging, avg_learning_curve, ste_learning_curve


def load_stepsizes_and_weights(results_dirpath, method_name, exp_type, exclude_diverging=False,
                               recompute=False, agg_actions=False, secondary_name='small_stepsize'):

    results_filename = "stepsize_and_weights"
    if exclude_diverging: results_filename += "_excluding_diverging_runs"
    if agg_actions: results_filename += '_aggregated_actions'
    if method_name == 'sgd': results_filename += "_" + secondary_name
    results_filename += ".p"

    learning_curve_path = os.path.join(results_dirpath, method_name, exp_type, results_filename)
    if os.path.isfile(learning_curve_path) and not recompute:
        avg_stepsizes, avg_weights = load_results(learning_curve_path)
    else:
        raw_results_path = os.path.join(results_dirpath, method_name, exp_type, 'results.p')
        if method_name != 'sgd':
            results_dict = load_results(raw_results_path)[method_name]
        else:
            results_dict = load_results(raw_results_path)[secondary_name]

        diverging_runs = results_dict['diverging_runs']
        action_counter = results_dict['action_counter']
        avg_stepsizes = None
        weight_sum_per_checkpoint = results_dict['weight_sum_per_checkpoint']
        if method_name != 'sgd': stepsize_sum_per_checkpoint = results_dict['stepsize_sum_per_checkpoint']

        if exclude_diverging:
            action_counter = action_counter[diverging_runs == 0]
            weight_sum_per_checkpoint = weight_sum_per_checkpoint[diverging_runs == 0]
            if method_name != 'sgd': stepsize_sum_per_checkpoint = stepsize_sum_per_checkpoint[diverging_runs == 0]
        if agg_actions:
            action_counter = np.sum(action_counter, axis=2)
            weight_sum_per_checkpoint = np.sum(weight_sum_per_checkpoint, axis=2)
            if method_name != 'sgd': stepsize_sum_per_checkpoint = np.sum(stepsize_sum_per_checkpoint, axis=2)

        action_counter = action_counter.reshape(action_counter.shape + (1,))
        avg_weight_per_checkpoint = weight_sum_per_checkpoint / action_counter
        avg_weights = np.average(avg_weight_per_checkpoint, axis=0)

        if method_name != 'sgd':
            avg_stepsizes_per_checkpoint = stepsize_sum_per_checkpoint / action_counter
            avg_stepsizes = np.average(avg_stepsizes_per_checkpoint, axis=0)

        with open(learning_curve_path, mode='wb') as learning_curve_file:
            pickle.dump((avg_stepsizes, avg_weights), learning_curve_file)

    return avg_stepsizes, avg_weights


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


def sgd_plots(save_plots=False, fake_features=False, bin_size=1000, plot_weights=False,
              exp_names=('add_good_feats', ), extended_training=False, recompute=False):
    assert NUM_SAMPLES % bin_size == 0
    downsample = 1000
    checkpoint = 1000

    num_samples = NUM_SAMPLES if not extended_training else EXTENDED_NUM_SAMPLES
    lc_xaxis = np.arange(num_samples - bin_size + 1) + 1
    ws_xaxis = (np.arange(num_samples // checkpoint) + 1) * 1000

    method = 'sgd'
    ms_func = lambda a: moving_sum(a, bin_size)

    if fake_features:
        task_name = 'mountain_car_control_task_add_fake_features'
    else:
        task_name = 'mountain_car_control_task_add_features'
    if extended_training:
        task_name += '_extended_training'

    default_names = ['add_good_feats', 'add_bad_feats', 'continuously_add_bad']
    experiment_names = list(set(default_names) & set(exp_names))
    results_dir = os.path.join(os.getcwd(), 'results', task_name)
    ylim = (0.0, 8 * (bin_size // 1000))

    for exp_name in experiment_names:

        results_dict = load_results(os.path.join(results_dir, method, exp_name, 'results.p'))

        names = [k for k in results_dict.keys() if k != 'sample_size']
        for i, n in enumerate(names):
            moving_sum_per_run = np.apply_along_axis(ms_func, 1, results_dict[n]['reward_per_step']) + bin_size
            avg, ste = get_average_and_standard_error(data_array=moving_sum_per_run, bin_size=1,
                                                      num_samples=num_samples)
            plt.plot(lc_xaxis[::downsample], avg[::downsample], color=sgd_colors[i], label=n)
            plt.fill_between(lc_xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
                             color=sgd_lighter_colors[i])

        if not extended_training:
            avg, ste = load_avg_learning_curve(results_dirpath=os.path.join(results_dir, 'rescaled_sgd', exp_name),
                                               bin_size=bin_size, results_name='results.p', method_name='rescaled_sgd',
                                               learning_curve_name='avg_learning_curve_bins' + str(bin_size) + '.p',
                                               num_samples=num_samples, secondary_results_name='reward_per_step',
                                               use_moving_sume=True)
            plt.plot(lc_xaxis[::downsample], avg[::downsample], color=RESCALED_SGD_COLORS['solid'], linestyle='dotted',
                     label='rescaled_sgd')
            plt.fill_between(lc_xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
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

        if plot_weights:
            weights_ylim = (-80,40)
            for n in names:
                _, avg_weights = load_stepsizes_and_weights(results_dirpath=results_dir, method_name='sgd',
                                                            exp_type=exp_name, exclude_diverging=True,
                                                            recompute=recompute, agg_actions=True, secondary_name=n)
                num_features = get_num_features(exp_name)
                current_features = 0
                for j, nf in enumerate(num_features):
                    for k in range(current_features, current_features + nf):
                        plt.plot(ws_xaxis, avg_weights[:,k], color=weights_and_stepsize_colors[j])
                    current_features += nf
                plt.title('{1}, {0}, sgd'.format(n, exp_name))
                plt.xlabel("Training Examples", fontsize=18)
                plt.ylabel("Average Weight", fontsize=18, rotation=0, labelpad=70)
                plt.vlines(x=MIDPOINT, ymin=weights_ylim[0], ymax=weights_ylim[1], colors='#BD2A4E',linestyles='dashed')
                xmin, xmax  = plt.gca().get_xlim()
                plt.hlines(y=0, xmin=xmin, xmax=xmax, color='#6B6B6B', linestyles='dashed')
                plt.xlim((xmin, xmax))
                plt.ylim(weights_ylim)
                if save_plots:
                    plt.savefig(n + "_weights_plot_" + exp_name + '_sgd' + '.svg', dpi=200)
                else:
                    plt.show()
                plt.close()

    ########################################
    # add_one_good_100_bad data #####
    ########################################
    default_names = ['add_5_good_5_bad', 'add_5_good_20_bad', 'add_5_good_100_bad']
    experiment_names = list(set(default_names) & set(exp_names))
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
                                               num_samples=num_samples, secondary_results_name='reward_per_step',
                                               use_moving_sume=True)
            plt.plot(lc_xaxis[::downsample], avg[::downsample], color=RESCALED_SGD_COLORS['solid'], linestyle='dotted',
                     label='rescaled_sgd')
            plt.fill_between(lc_xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
                             color=RESCALED_SGD_COLORS['lighter'])

            i = 0
            for k in range(n):
                name_index = j * n + k
                moving_sum_per_run = np.apply_along_axis(ms_func, 1,
                                                         results_dict[names[name_index]]['reward_per_step']) + bin_size
                avg, ste = get_average_and_standard_error(data_array=moving_sum_per_run, bin_size=1,
                                                          num_samples=NUM_SAMPLES)
                plt.plot(lc_xaxis[::downsample], avg[::downsample], color=sgd_colors[i], label=names[name_index])
                plt.fill_between(lc_xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
                                 color=sgd_lighter_colors[i])
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
                           plot_weights=False, bin_size=5, exclude_diverging_runs=False, agg_actions=False,
                           experiment_types=('add_good_feats',), methods=('adam',), extended_training=False,
                           recompute=False):
    downsample = 1000
    num_actions = 3
    num_samples = NUM_SAMPLES if not extended_training else EXTENDED_NUM_SAMPLES
    lc_xaxis = np.arange(num_samples - bin_size + 1) + 1
    checkpoint = 1000
    ws_xaxis = (np.arange(num_samples // checkpoint) + 1) * 1000

    # experiment_types = ['add_good_feats', 'add_bad_feats', 'add_5_good_5_bad', 'add_5_good_20_bad',
    #                     'add_5_good_100_bad', 'continuously_add_bad']
    if fake_features:
        task_name = 'mountain_car_control_task_add_fake_features'
    else:
        task_name = 'mountain_car_control_task_add_features'
    if extended_training:
        task_name += '_extended_training'

    results_dir = os.path.join(os.getcwd(), 'results', task_name)

    if plot_learning_curves:
        """ Plot Learning Curves """
        ylim = (0.0, 8 * (bin_size // 1000))
        for exp_name in experiment_types:
            print("Experiment name: {0}".format(exp_name))
            for i, m in enumerate(methods):

                num_dvg, avg, ste = load_learning_curves(results_dirpath=results_dir,
                                                         method_name=m, exp_type=exp_name,
                                                         bin_size=bin_size, num_samples=NUM_SAMPLES,
                                                         exclude_diverging=exclude_diverging_runs)

                print("\tMethod: {0}\n\tNumber of Diverging Runs: {1}".format(m, num_dvg))

                plt.plot(lc_xaxis[::downsample], avg[::downsample], color=methods_colors[m], label=m)
                plt.fill_between(lc_xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
                                 color=methods_lighter_colors[m])

            if not extended_training and False:
                avg, ste = load_avg_learning_curve(results_dirpath=os.path.join(results_dir, 'rescaled_sgd', exp_name),
                                                   bin_size=bin_size, results_name='results.p', method_name='rescaled_sgd',
                                                   learning_curve_name='avg_learning_curve_bins'+str(bin_size)+'.p',
                                                   num_samples=NUM_SAMPLES, secondary_results_name='reward_per_step',
                                                   use_moving_sume=True)

                plt.plot(lc_xaxis[::downsample], avg[::downsample], color=RESCALED_SGD_COLORS['solid'], label='rescaled_sgd',
                         linestyle='dotted')
                plt.fill_between(lc_xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
                                 color=RESCALED_SGD_COLORS['lighter'])

            plt.legend()
            plt.xlabel("Training Examples", fontsize=18)
            plt.ylabel("Episodes Completed\nOver {0} Steps".format(bin_size), fontsize=18, rotation=0, labelpad=100)
            plt.title('{0}'.format(exp_name))
            if exp_name != 'continuously_add_bad':
                plt.vlines(x=MIDPOINT, ymin=0, ymax=100, colors='#BD2A4E', linestyles='dashed')
            plt.ylim(ylim)
            if save_plots:
                plot_name = 'learning_curve_' + exp_name
                if exclude_diverging_runs: plot_name += 'excluding_diverging_runs'
                plot_name += '.svg'
                plt.savefig(plot_name, dpi=200)
            else:
                plt.show()
            plt.close()

    """ Plot Stepsizes """
    if plot_stepsizes:
        ylim = (0,0.23)

        for exp_name in experiment_types:
            for i, m in enumerate(methods):

                avg_stepsizes, avg_weights = load_stepsizes_and_weights(results_dirpath=results_dir, method_name=m,
                                                                        exp_type=exp_name,
                                                                        exclude_diverging=exclude_diverging_runs,
                                                                        agg_actions=agg_actions, recompute=True)

                num_features = get_num_features(exp_name)
                current_features = 0
                serial_name = 0
                for j, nf in enumerate(num_features):
                    for k in range(current_features, current_features + nf):
                        if agg_actions:
                            plt.plot(ws_xaxis, avg_stepsizes[:, k], color=weights_and_stepsize_colors[j])
                        else:
                            for a in range(num_actions):
                                plt.plot(ws_xaxis, avg_stepsizes[:, a, k], color=weights_and_stepsize_colors[j])
                        # plt.ylim(ylim[m][exp_name])
                        # plt.savefig(str(serial_name) + '.png')
                        # plt.show()
                        # plt.close()
                        # serial_name += 1
                    current_features += nf
                plt.title('{0}, {1}'.format(exp_name, m))
                plt.xlabel("Training Examples", fontsize=18)
                plt.ylabel("Stepsize", fontsize=18, rotation=0, labelpad=50)
                if exp_name != 'continuously_add_bad':
                    plt.vlines(x=MIDPOINT, ymin=0, ymax=100, colors='#BD2A4E', linestyles='dashed')
                plt.ylim(ylim)
                if save_plots:
                    plt.savefig("stepsize_plot_" + exp_name + '_' + m + '.svg', dpi=200)
                else:
                    plt.show()
                plt.close()

    """ Plot Weights """
    if plot_weights:
        for exp_name in experiment_types:
            for i, m in enumerate(methods):
                avg_stepsizes, avg_weights = load_stepsizes_and_weights(results_dirpath=results_dir, method_name=m,
                                                                        exp_type=exp_name,
                                                                        exclude_diverging=exclude_diverging_runs,
                                                                        agg_actions=agg_actions, recompute=True)
                num_features = get_num_features(exp_name)
                current_features = 0
                for j, nf in enumerate(num_features):
                    for k in range(current_features, current_features + nf):
                        if agg_actions:
                            plt.plot(ws_xaxis, avg_weights[:, k], color=weights_and_stepsize_colors[j])
                        else:
                            for a in range(num_actions):
                                plt.plot(ws_xaxis, avg_weights[:, a, k], color=weights_and_stepsize_colors[j])
                    current_features += nf
                plt.title('{0}, {1}'.format(exp_name, m))
                plt.xlabel("Training Examples", fontsize=18)
                plt.ylabel("Average Weight", fontsize=18, rotation=0, labelpad=70)
                if exp_name != 'continuously_add_bad':
                    plt.vlines(x=MIDPOINT, ymin=0, ymax=100, colors='#BD2A4E', linestyles='dashed')
                if save_plots:
                    plt.savefig("stepsize_plot_" + exp_name + '_' + m + '.svg', dpi=200)
                else:
                    plt.show()
                plt.close()


def baseline_plots(bin_size=1000, save_plots=False, plot_learning_curves=False, plot_weights=False,
                   exclude_diverging_runs=False, methods=('adam',), extended_training=False, recompute=False):

    results_dir = os.path.join(os.getcwd(), 'results', 'mountain_car_control_baseline')
    ms_func = lambda a: moving_sum(a, bin_size)
    downsample = 1000
    lc_xaxis = np.arange(NUM_SAMPLES - bin_size + 1) + 1
    checkpoint = 1000
    ws_xaxis = (np.arange(NUM_SAMPLES // checkpoint) + 1) * 1000

    if plot_learning_curves:

        ylim = (0.0, 8 * (bin_size // 1000))

        for i, m in enumerate(methods):

            if m == 'sgd':
                results_dict = load_results(os.path.join(results_dir, m, 'results.p'))
                for n, ss in enumerate(['small_stepsize', 'med_stepsize', 'large_stepsize']):
                    moving_sum_per_run = np.apply_along_axis(ms_func, 1,
                                                             results_dict[ss]['reward_per_step']) + bin_size
                    avg, ste = get_average_and_standard_error(data_array=moving_sum_per_run, bin_size=1,
                                                              num_samples=NUM_SAMPLES)
                    plt.plot(lc_xaxis[::downsample], avg[::downsample], color=sgd_colors[n], label=ss)
                    plt.fill_between(lc_xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
                                     color=sgd_lighter_colors[n])
            else:
                num_dvg, avg, ste = load_learning_curves(results_dirpath=results_dir, method_name=m, exp_type='',
                                                         bin_size=bin_size, num_samples=NUM_SAMPLES,
                                                         exclude_diverging=exclude_diverging_runs)

                print("\tMethod: {0}\n\tNumber of Diverging Runs: {1}".format(m, num_dvg))

                plt.plot(lc_xaxis[::downsample], avg[::downsample], color=methods_colors[m], label=m)
                plt.fill_between(lc_xaxis[::downsample], (avg + ste)[::downsample], (avg - ste)[::downsample],
                                 color=methods_lighter_colors[m])

        plt.legend()
        plt.xlabel("Training Examples", fontsize=18)
        plt.ylabel("Episodes Completed\nOver {0} Steps".format(bin_size), fontsize=18, rotation=0, labelpad=100)
        plt.title('Baseline')
        plt.vlines(x=MIDPOINT, ymin=0, ymax=100, colors='#BD2A4E', linestyles='dashed')
        plt.ylim(ylim)
        if save_plots:
            plot_name = 'baseline_'
            if exclude_diverging_runs: plot_name += 'excluding_diverging_runs'
            plot_name += '.svg'
            plt.savefig(plot_name, dpi=200)
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sgd_plots', action='store_true', default=False)
    parser.add_argument('--stepsize_methods_plots', action='store_true', default=False)
    parser.add_argument('--plot_learning_curves', action='store_true', default=False)
    parser.add_argument('--plot_stepsizes', action='store_true', default=False)
    parser.add_argument('--plot_weights', action='store_true', default=False)
    parser.add_argument('--plot_baselines', action='store_true', default=False)
    parser.add_argument('-sp', '--save_plots', action='store_true', default=False)
    parser.add_argument('-bs', '--bin_size', action='store', default=1, type=int)
    parser.add_argument('-ff', '--fake_features', action='store_true', default=False)
    parser.add_argument('-edr', '--exclude_diverging_runs', action='store_true', default=False)
    parser.add_argument('-aa', '--aggregate_actions', action='store_true', default=False)
    parser.add_argument('-et', '--experiment_types', action='store', nargs='+', type=str,
                        default=['add_good_feats', 'add_bad_feats', 'add_5_good_5_bad', 'add_5_good_20_bad',
                                 'add_5_good_100_bad', 'continuously_add_bad'])
    parser.add_argument('-etr', '--extended_training', action='store_true', default=False)
    parser.add_argument('-rc', '--recompute', action='store_true', default=False)
    parser.add_argument('-m', '--methods', action='store', nargs='+', type=str, required=False, default='adam')
    plot_parameters = parser.parse_args()

    if plot_parameters.sgd_plots:
        sgd_plots(bin_size=plot_parameters.bin_size, save_plots=plot_parameters.save_plots,
                  fake_features=plot_parameters.fake_features, plot_weights=plot_parameters.plot_weights,
                  exp_names=plot_parameters.experiment_types, extended_training=plot_parameters.extended_training,
                  recompute=plot_parameters.recompute)

    if plot_parameters.stepsize_methods_plots:
        stepsize_methods_plots(bin_size=plot_parameters.bin_size, save_plots=plot_parameters.save_plots,
                               plot_learning_curves=plot_parameters.plot_learning_curves,
                               plot_stepsizes=plot_parameters.plot_stepsizes,
                               plot_weights=plot_parameters.plot_weights,
                               fake_features=plot_parameters.fake_features,
                               exclude_diverging_runs=plot_parameters.exclude_diverging_runs,
                               agg_actions=plot_parameters.aggregate_actions,
                               experiment_types=plot_parameters.experiment_types,
                               methods=plot_parameters.methods,
                               extended_training=plot_parameters.extended_training)

    if plot_parameters.plot_baselines:
        baseline_plots(bin_size=plot_parameters.bin_size, save_plots=plot_parameters.save_plots,
                       plot_learning_curves=plot_parameters.plot_learning_curves,
                       exclude_diverging_runs=plot_parameters.exclude_diverging_runs,
                       plot_weights=plot_parameters.plot_weights, methods=plot_parameters.methods)

    # color_blind_palette = {
    #     'blue-ish': '#332288',   # I'm not good at naming colors
    #     'green-ish': '#117733',
    #     'light green-ish': '#44AA99',
    #     'light blue-ish': '#88CCEE',
    #     'red-ish': '#CC6677',
    #     'purple-ish': '#CC6677',
    #     'dark purple-ish': '#882255'
    # }

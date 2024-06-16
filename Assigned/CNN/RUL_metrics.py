
import numpy as np
import matplotlib.pyplot as plt
import sys
import random as rd
import tensorflow as tf
import tqdm
import math
import pickle


def compute_coverage(true_RULs, RUL_distributions, alpha):
    """
    Computes the coverage and mean width of the credible interval for a given alpha.

    The coverage indicates the proportion of true RUL values that fall within the 
    credible interval defined by alpha. A coverage closer to alpha suggests better
    calibration. The mean width measures the average size of these intervals.

    Parameters:
        true_RULs (dict): A dictionary mapping test instances to their true RUL values.
        RUL_distributions (dict): A dictionary mapping test instances to a list of 
            predicted RUL values for that instance.
        alpha (float): The desired probability mass of the credible interval. Must be between 0 and 1.

    Returns:
        coverage (float): The fraction of true RUL values falling within the interval. Between 0 and 1.
        mean_width (float): The average width of the credible intervals across all instances.
    """

    # Initialize the parameters of the credible interval
    total_width = 0
    in_ci = 0  # The number of components for which the true RUL falls within the credible interval
    percentile_lower = 0.5 - 0.5 * alpha
    percentile_higher = 0.5 + 0.5 * alpha

    # Check for each test instance i if the true RUL falls inside or outside the credible interval.
    for i in true_RULs.keys():
        # Get the probability dstributions of the RUL test instance i, and the true RUL.
        distribution = RUL_distributions.get(i)
        true_RUL = true_RULs.get(i)
        distribution.sort()
        number_of_predictions = len(distribution)

        # Get the indices of the RUL predictions belonging to the considered percenticles.
        # We use -1, since a list in python starts at 0 instead of 1
        index_lower = max(0, int(percentile_lower * number_of_predictions) - 1)
        index_higher = int(percentile_higher * number_of_predictions) - 1
        lower_bound_ci = distribution[index_lower]
        upper_bound_ci = distribution[index_higher]

        # Check if the true RUL is within the credible interval
        if lower_bound_ci <= true_RUL <= upper_bound_ci:
            in_ci += 1

        # Update the total width of all credible interval
        total_width = total_width + (upper_bound_ci - lower_bound_ci)

    # Calculate the coverage and the mean width of the credible interval.
    coverage = in_ci / len(true_RULs.keys())
    mean_width = total_width / len(true_RULs.keys())
    return coverage, mean_width


def compute_mean_variance(RUL_distributions, number_of_runs):
    """

    """
    mean_var = 0
    mean_std = 0

    for i in RUL_distributions.keys():
        distribution = RUL_distributions[i]
        mean = np.mean(np.array(distribution))

        # Get the variance of the prediction
        var = 0
        for j in range(0, len(distribution), 1):
            var = var + (distribution[j] - mean) ** 2

        var = var / number_of_runs
        std = math.sqrt(var)

        mean_var = mean_var + var
        mean_std = mean_std + std

    mean_var = mean_var / len(RUL_distributions.keys())
    mean_std = mean_std / len(RUL_distributions.keys())

    return mean_var, mean_std


def compute_area_under(x1, x2, f1, f2):
    """Computes the area between the ideal curve and the reliability curve, between x1 and x2. 

    Parameters
    ----------
    x1 : float
        The start value of alpha.
    x2 : float
        The end value of alpha.
    f1 : float
        The coverage at alpha = x1.
    f2 : float
        The coverage at alpha = x2.

    Returns
    -------
    area : float
        The area between the ideal curve and the reliability curve, between x1 and x2. This area
        contributes to the underestimation part of the reliability score.
    """

    area = (x2 - f2) * (x2 - x1) - 0.5 * (x2 - x1) * (x2 - x1)
    area += 0.5 * (x2 - x1) * (f2 - f1)
    return area


def compute_area_above(x1, x2, f1, f2):
    """Computes the area between the ideal curve and the reliability curve, between x1 and x2. Here, the reliability
    curve is above the ideal curve between x1 and x2.

    Parameters
    ----------
    x1 : float
        The start value of alpha.
    x2 : float
        The end value of alpha.
    f1 : float
        The coverage at alpha = x1.
    f2 : float
        The coverage at alpha = x2 .
    Returns
    -------
    area : Float
        The area between the ideal curve and the reliability curve, between x1 and x2. This area
        contributes to the overestimation part of the reliability score.
    """
    area = (f1 - x1) * (x2 - x1) - 0.5 * (x2 - x1) * (x2 - x1)
    area += 0.5 * (x2 - x1) * (f2 - f1)
    return area


def compute_reliability_score(true_RULs, RUL_distributions, name):
    """Computes the reliability scores (under, over, and total), and optionally plots the reliability diagram.

    Parameters
    ----------
    true_RULs: dictionary
        A dictionary with for each test instance (key, integer), the true RUL (value). 
    RUL_distributions : dictionary
        A dictionary with for each test instance (key, integer), a list (value) with all RUL predictions of this test
        instance. true_RULs and RUL distributions should have the same set of keys.
    name : str
        e


    Returns
    -------
    RS_total : float
        The total reliability score, representing the sum of the underestimation and overestimation scores.
        A lower score indicates better reliability.
    RS_under : float
        The reliability score due to underestimation of uncertainty. A lower score indicates better reliability.
    RS_over : float
        The reliability score due to overestimation of uncertainty. A lower score indicates
    """

    step_size = 0.01

    # Calculate the reliability and ideal curves.
    reliability_curve = []
    for alpha in np.arange(0, 1 + sys.float_info.epsilon, step_size):  # one is included
        alpha_coverage = compute_coverage(true_RULs, RUL_distributions, alpha)[0]
        reliability_curve.append(alpha_coverage)
    ideal_curve = list(np.arange(0, 1 + sys.float_info.epsilon, step_size))  # ideal curve, where y = x

    plot_reliability_diagram(ideal_curve, reliability_curve, name)

    # Calculate the reliability score.
    RS_under = 0  # underestimation of the uncertainty
    RS_over = 0  # overestimation of the uncertainty

    for alpha in np.arange(0, 1, step_size):
        next_alpha = alpha + step_size
        coverage_alpha = compute_coverage(true_RULs, RUL_distributions, alpha)[0]
        coverage_next_alpha = compute_coverage(true_RULs, RUL_distributions, next_alpha)[0]

        # If the reliability curve is beneath the ideal curve:
        if coverage_alpha <= alpha and coverage_next_alpha <= next_alpha:
            surface = compute_area_under(alpha, next_alpha, coverage_alpha, coverage_next_alpha)
            RS_under += surface

        # If the reliability curve is above the ideal curve:
        elif coverage_alpha >= alpha and coverage_next_alpha >= next_alpha:
            surface = compute_area_above(alpha, next_alpha, coverage_alpha, coverage_next_alpha)
            RS_over += surface

        # If the reliability curve starts under the ideal curve, and ends above the ideal curve:
        elif coverage_alpha <= alpha and coverage_next_alpha >= next_alpha:
            # Find the place where the reliability curve crosses the ideal curve.
            dy = coverage_next_alpha - coverage_alpha
            a = dy / step_size
            alpha_cross = (coverage_alpha - a * alpha) / (1 - a)
            coverage_cross = alpha_cross

            # Calculate the surface under the ideal curve.
            surface_under = compute_area_under(alpha, alpha_cross, coverage_alpha, coverage_cross)
            RS_under += surface_under
            # Calculate the surface above the ideal curve.
            surface_above = compute_area_above(alpha_cross, next_alpha, coverage_cross, coverage_next_alpha)
            RS_over += surface_above

        # If the reliability curve starts above the ideal curve, and ends under the ideal curve:
        elif coverage_alpha >= alpha and coverage_next_alpha <= next_alpha:
            dy = coverage_next_alpha - coverage_alpha
            a = dy / step_size
            alpha_cross = (coverage_alpha - a * alpha) / (1 - a)
            coverage_cross = alpha_cross

            surface_above = compute_area_above(alpha_cross, next_alpha, coverage_cross, coverage_next_alpha)
            RS_over += surface_above
            surface_under = compute_area_under(alpha, alpha_cross, coverage_alpha, coverage_cross)
            RS_under += surface_under

    RS_total = RS_under + RS_over
    return RS_total, RS_under, RS_over


def plot_reliability_diagram(ideal_curve, reliability_curve, name):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if name != "blablabla":
        if name == "FD001":
            ax.plot(ideal_curve, reliability_curve, label=name, color="blue", linestyle="dashed", lw=3)
        elif name == "FD002":
            ax.plot(ideal_curve, reliability_curve, label=name, color="green", linestyle="dotted", lw=3)
        elif name == "FD003":
            ax.plot(ideal_curve, reliability_curve, label=name, color="chocolate", linestyle='dashdot', lw=3)
        elif name == "FD004":
            ax.plot(ideal_curve, reliability_curve, label=name, color="fuchsia", linestyle=(0, (3, 1, 1, 1, 1, 1)), lw=3)

    ax.legend(fontsize=16)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Probability')
    ax.set_title('Reliability Diagram')

    # Display the plot
    plt.show()


# this function computes the output values of the model and uses some KPI's to show the results
def test_montecarlo_output(model, input_array, target_array, number_of_runs, plot_hist, name):
    """

    Parameters
    ----------
    model
    input_array
    target_array
    number_of_runs
    """
    mc_predictions = []
    true_RULs, RUL_distributions, mean_RULs = {}, {}, {}
    RMSE, MAE = 0, 0

    rd.seed(7042018)
    tf.random.set_seed(7042018)

    # predict the RUL value MC_runs times
    for _ in tqdm.tqdm(range(number_of_runs)):
        y_p = np.rint(model.predict(input_array))
        mc_predictions.append(y_p)

    # capture the mean of the predictions, the actual predictions and the standard deviation of the predictions
    for i in range(len(mc_predictions[0])):
        all_predictions = []
        for j in range(len(mc_predictions)):
            all_predictions.append(mc_predictions[j][i][0])
        RUL_distributions[i] = all_predictions
        true_RULs[i] = target_array[i]
        mean_RULs[i] = np.mean(np.array(all_predictions))
        RMSE = RMSE + (target_array[i] - np.mean(np.array(all_predictions))) ** 2
        MAE = MAE + abs(target_array[i] - np.mean(np.array(all_predictions)))

    coverage_0_5, mean_width_0_5 = compute_coverage(true_RULs, RUL_distributions, 0.5)
    coverage_0_9, mean_width_0_9 = compute_coverage(true_RULs, RUL_distributions, 0.9)
    coverage_0_95, mean_width_0_95 = compute_coverage(true_RULs, RUL_distributions, 0.95)
    RS_total, RS_under, RS_over = compute_reliability_score(true_RULs, RUL_distributions, name)

    mean_var, mean_std = compute_mean_variance(RUL_distributions, number_of_runs)

# Summary of Metrics (FD001):
# - Total Predictions: 100
# - Reliability Score (under): 0.0314 (Lower is better; indicates minimal underestimation of uncertainty)
# - Reliability Score (over): 0.0192 (Lower is better; indicates minimal overestimation of uncertainty)
# - Total Reliability Score: 0.0506 (Sum of under and over; lower is better)
# - Coverage at alpha = 0.5: 0.49 (Close to 0.5 indicates good calibration)
# - Mean Width at alpha = 0.5: 15.14 (Smaller widths are better for precision)
# - Coverage at alpha = 0.9: 0.86 (Close to 0.9 indicates good calibration)
# - Mean Width at alpha = 0.9: 37.06
# - Coverage at alpha = 0.95: 0.86 (Close to 0.95 indicates good calibration)
# - Mean Width at alpha = 0.95: 37.06
# - RMSE: 13.12 (Lower is better; indicates accuracy of predictions)
# - MAE: 9.92 (Lower is better; indicates accuracy of predictions)
# - Mean Variance: 128.14 (Lower is better; indicates consistency of predictions)
# - Mean Standard Deviation: 11.10 (Lower is better; indicates precision of predictions)
#
# Overall, these metrics suggest the model has good calibration, reliability, and predictive accuracy.
    print("\nThere are a total of " + str(len(target_array)) + " predictions.")
    print("The reliability score (under) is " + str(RS_under))
    print("The reliability score (over) is " + str(RS_over))
    print("The total reliability score is " + str(RS_total))
    print("The coverage at alpha = 0.5 is " + str(coverage_0_5))
    print("The mean width at 0.5 is " + str(mean_width_0_5))
    print("The coverage at 0.alpha = 0.9 is " + str(coverage_0_9))
    print("The mean width at 0.9 is " + str(mean_width_0_9))
    print("The coverage at 0.alpha = 0.95 is " + str(coverage_0_95))
    print("The mean width at 0.95 is " + str(mean_width_0_95))
    RMSE = math.sqrt(RMSE / len(target_array))
    print("The RMSE is " + str(RMSE))
    print("The MAE is " + str(MAE / len(target_array)))
    print("The mean variance is ", mean_var)
    print("The mean std is ", mean_std)

    ##################################################
    # -----------------PLot the histograms------------#
    ##################################################

    for i in plot_hist:
        fig, ax = plt.subplots()
        fs = 16

        # get the true RULS
        true_RUL = true_RULs[i]

        # Get the predictions
        predictions = RUL_distributions[i]

        # PLot a histogram with the predictions
        bin_width = 5
        ax.hist(predictions, density=True, bins=np.arange(min(predictions), max(predictions) + bin_width, bin_width),
                color="lightcoral", ec="lightcoral")

        # Get the mean RUL prediction
        mean_RUL = mean_RULs[i]
        print("the true RUL is ", true_RUL, " and the mean RUL is ", mean_RUL)

        # Plot a vertical line at the true RUL
        ax.axvline(x=true_RUL, lw=3.2, label="Actual RUL", c='b')
        ax.axvline(x=mean_RUL, lw=3.2, label="Mean predicted RUL", c='r')
        ax.set_ylabel('Probability', fontsize=fs)
        ax.set_xlabel('RUL (flight cycles)', fontsize=fs)

        right_side = ax.spines["right"]
        right_side.set_visible(False)
        upper_side = ax.spines["top"]
        upper_side.set_visible(False)

        ax.tick_params(axis='x', labelsize=fs)
        ax.tick_params(axis='y', labelsize=fs)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=fs - 2)

        fig.tight_layout()
        name_fig = "./hist_" + name + "_" + str(i)
        plt.savefig(name_fig, dpi=400)

    engine_all_predictions = {}
    engine_mean_prediction = {}
    engine_true_RUL = {}

    engine_number = 1
    engine = name + "_" + str(engine_number)
    engine_all_predictions[engine] = []
    engine_mean_prediction[engine] = []
    engine_true_RUL[engine] = []

    
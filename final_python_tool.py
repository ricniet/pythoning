#!/usr/bin/env python

# This script contains the final automated tool (Prototype B: Candidate Filtering) built for
# my internship project. The problem essentially involved an engineering team looking for a data-driven
# approach to identifying how to specify upwards of 2,000 inputs in order to observe an expected change
# in one of 50 targets.
#
# A large number of randomly generated input configurations are fed to a pre-trained
# model, and the associated predictions are filtered based on 1) a user-specified change in the target
# and a threshold for selecting the top candidates.
#
# Each target involved its own unique data pre-processing and engineering steps and thus
# this script automates for whatever target is specified by the user.

#######################################################################
#
# Pipeline flow
#
# 1. Call generate_candidates()
#   - Generate input configuration candidates
#
# 2. Call data_preprocess()
#   - Specify target and candidate inputs
#   - Performs necessary target-specific data preprocessing & engineering in preparation for predictions
#
# 3. Call generate_top_candidates()
#   - Specify target, target (lower) threshold, top k to return, and paths to directories
#   - Obtain prepared candidates (input) and pre-trained model
#   - Generate target predictions and 90% prediction interval from candidates and pre-trained model
#   - Returns csv file and figure of top k predictions
#
#######################################################################

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.base import clone
import argparse

def data_preprocess(pmu, X_candidates, pmu_dict):

    preprocess_function_path = path_to_pmus + "preprocess_function_dictionary.pkl"
    function_dict = joblib.load(preprocess_function_path)

    #drop features
    X_candidates = function_dict[current_pmu_dict['drop_code']](X_candidates)

    # group features
    X_candidates = function_dict[current_pmu_dict['group_code']](X_candidates)

    # normalize
    X_candidates = function_dict['normalize_fun'](X_candidates)

    return(X_candidates)

def generate_top_candidates(pmu, pmu_threshold, top_k, path_to_candidates, path_to_model, path_to_save):
    """"
    Generate predictions for a specified PMU based on generated input candidates, and
    pre-trained model. Based on desired range for a specified PMU, return filtered candidates.
    Candidates are filtered then filtered, and returned.
    :param str pmu: performance monitoring unit (PMU) of interest
    :param int64 pmu_threshold: minimum value of expected PMU range
    :param int64 top_k: number of filtered candidates to return
    :param str path_to_candidates: directory for candidates
    :param str path_to_model: directory for model
    :param str path_to_save: directory in which to save output
    :return: file and figure with information about top k predictions and prediction interval
    """

    pmu_dictionary_path = path_to_pmus + "pmu_dictionary.pkl"
    pmu_dict = joblib.load(pmu_dictionary_path)

    current_pmu_dict = pmu_dict[pmu]

    if (pmu != current_pmu_dict['pmu_code']=='rejected_pmu'):
        print('You have entered an illegal PMU. Please try again.')
        exit()

    if (pmu_threshold <= current_pmu_dict['percentile_5']):
        input('Warning: The current threshold may produce unstable results; consider increasing pmu_threshold. Escape or press Enter to continue with current value: ')

    # Load candidates
    candidate_path = path_to_candidates + "candidate_input_configurations.pkl"

    print("Loading candidates from {}".format(candidate_path))
    X_candidates = joblib.load(candidate_path)

    # Load pre-trained model for user-specified PMU
    model_path = path_to_model + "final_model_June30_{}.pkl".format(pmu)

    print("Loading model for PMU {} from {}".format(pmu, model_path))

    # Create models for 90% credible interval limits. Since it uses the primary model
    # fit using least squares loss, it's just a matter of setting new parameters.
    primary_model = joblib.load(model_path)
    upper_limit_model = clone(primary_model).set_params(clf__loss='quantile', clf__alpha=0.95)
    lower_limit_model = clone(primary_model).set_params(clf__loss='quantile', clf__alpha=0.05)

    # generate predictions & add to X_candidates
    primary_prediction = primary_model.predict(X_candidates)
    upper_limit_prediction = upper_limit_model.predict(X_candidates)
    lower_limit_prediction = lower_limit_model.predict(X_candidates)

    preds = dict(primary_prediction=primary_prediction,
                 lower_limit_prediction=lower_limit_prediction,
                 upper_limit_prediction=upper_limit_prediction)

    X_candidates = pd.concat([pd.DataFrame(preds).set_index(X_candidates.index), X_candidates], axis=1)

    # omit predictions below threshold
    X_candidates = X_candidates[X_candidates['primary_prediction'] >= pmu_threshold]

    # sort dataframe by `predecited_pmu` and return top `top_k` candidates
    X_candidates.sort_values(by=['primary_prediction'], ascending=False, inplace=True)

    # in case top_k > number of candidates
    if (top_k <= X_candidates.shape[0]):
        top_candidates = X_candidates.iloc[0:top_k,:]
    else:
        top_candidates = X_candidates

    # calculate error from main prediction for figure
    high_error = top_candidates.upper_limit_prediction - top_candidates.primary_prediction
    low_error = top_candidates.primary_prediction - top_candidates.lower_limit_prediction

    # must sort in descending or else figure doesn't order them correctly
    top_candidates.sort_values(by=['primary_prediction'], ascending=True, inplace=True)

    plt.figure(figsize=(8,5))
    plt.errorbar(top_candidates.primary_prediction, top_candidates.index, xerr=[low_error, high_error],
                 fmt='o', linestyle='none',label="Predicted value")
    plt.axvline(x=pmu_threshold, color='r', linestyle='--', label="Specified threshold")
    plt.xlabel("Prediction", fontsize=12)
    plt.ylabel("Candidate index", fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    plt.legend(fontsize=14)
    plt.title("Top {} predictions with 90% prediction intervals for\nPMU: ".format(top_k, pmu), fontsize=14);

    print("Saving figure to {}".format(path_to_save))
    plt.savefig(path_to_save + "top_candidates_figure.png", dpi=96, bbox_inches='tight')

    # save top candidate summary table
    data = {'candidate_index': top_candidates.index,
            'prediction': top_candidates.primary_prediction,
            'lower_bound': top_candidates.lower_limit_prediction,
            'upper_bound': top_candidates.upper_limit_prediction}
    print("Saving summary table to {}".format(path_to_save))
    pd.DataFrame(data).to_csv(path_to_save + "top_candidates_table.csv", index=False)


# Get current path
pwd = os.path.dirname(os.path.abspath(__file__))

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument("pmu",
                    help="Specify PMU of interest")
parser.add_argument("pmu_threshold", type=int,
                    help="Minimum expected PMU value")
parser.add_argument("top_k", type=int, default=10,
                    help="The number of candidates to return. Default is 10.")
parser.add_argument("path_to_candidates",
                    help="Path to generated input candidate file. Default is ../Candidates",
                    default="../Candidates")
parser.add_argument("path_to_model",
                    help="Path to trained models. Default is ../Models",
                    default="../Models")
parser.add_argument("path_to_save",
                    help="Path to save output. Default is ../Results",
                    default="../Results")
opts = parser.parse_args()

pmu = opts.pmu
pmu_threshold = opts.pmu_threshold
top_k = opts.top_k
candidate_path = opts.path_to_candidates
model_path = opts.path_to_model
save_path = opts.path_to_save

generate_top_candidates(pmu=pmu, pmu_threshold=pmu_threshold,
                        top_k=top_k, path_to_candidates=candidate_path,
                        path_to_model=model_path, path_to_save=save_path)
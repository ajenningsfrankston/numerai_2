#
# Select from a set of classifiers the most consistent, then bag.
#

import numpy as np
import pandas as pd

from make_prediction import make_prediction

from numerapi import NumerAPI
import zipfile



data_archive = NumerAPI().download_current_dataset(dest_path='./tmp',unzip=False)

with zipfile.ZipFile(data_archive,"r") as zip_ref:
    zip_ref.extractall("./tmp/numerai_datasets")

competitions = NumerAPI().get_tournaments()
comp_names = list()
for comp in competitions: comp_names.append(comp["name"])

print(comp_names)

print("# Loading data...")
# The training data is used to train your model how to predict the targets.
train = pd.read_csv('./tmp/numerai_datasets/numerai_training_data.csv', header=0)
# The tournament data is the data that Numerai uses to evaluate your model.
tournament = pd.read_csv('./tmp/numerai_datasets/numerai_tournament_data.csv', header=0)

# The tournament data contains validation data, test data and live data.
# Validation is used to test your model locally so we separate that.

validation = tournament[tournament['data_type'] == 'validation']


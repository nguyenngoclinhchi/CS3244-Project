# -*- coding: utf-8 -*-
"""Linh_Chi.ipynb
Original file is located at
    https://colab.research.google.com/drive/13SEnWoVQlVhCtoKFGTQQbbwpF0k3MBcD
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 100)
from tqdm import tqdm_notebook as tqdm
# for preprocessing the data
from sklearn.preprocessing import StandardScaler
# the model
from sklearn.svm import SVC
from sklearn.decomposition import PCA
# for combining the preprocess with model training
from sklearn.pipeline import Pipeline
# for optimizing the hyperparameters of the pipeline
from sklearn.model_selection import GridSearchCV

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Import files
train_values = pd.read_csv('train_values.csv', index_col='building_id')
train_labels = pd.read_csv('train_labels.csv', index_col='building_id')
selected_features = ['foundation_type',
                     'area_percentage',
                     'height_percentage',
                     'count_floors_pre_eq',
                     'land_surface_condition',
                     'has_superstructure_cement_mortar_stone']
train_values_subset = train_values[selected_features]

train_values_subset = pd.get_dummies(train_values_subset)

# Train the model
pipe_steps = [('scaler', StandardScaler()), ('pca', PCA()), ('SupVM', SVC(kernel='rbf'))]
pipe = Pipeline(pipe_steps)
param_grid = {'pca__n_components': [2],
              'SupVM__C'         : [0.1, 0.5, 1, 10, 30, 40, 50, 70, 100, 500, 1000],
              'SupVM__gamma'     : [0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50],
              }
print('Start fitting training data')

for num_cv in tqdm(range(4, 7)):
    gs = GridSearchCV(pipe, param_grid, cv=num_cv)
    gs.fit(train_values_subset, train_labels.values.ravel())
    print("Best fit parameter for %d fold CV" % num_cv, gs.best_params_)

    # Evaluate the model
    from sklearn.metrics import f1_score

    in_sample_predict = gs.predict(train_values_subset)
    f1_score(train_labels, in_sample_predict, average='micro')

    # Read values then output Results
    test_values = pd.read_csv('test_values.csv', index_col='building_id')
    test_values_subset = test_values[selected_features]
    test_values_subset = pd.get_dummies(test_values_subset)
    predictions = gs.predict(test_values_subset)
    submission_format = pd.read_csv('submission_format' + '_rbf' + '_cv=' + str(num_cv) + '.csv',
                                    index_col='building_id')
    my_submission = pd.DataFrame(data=predictions,
                                 columns=submission_format.columns,
                                 index=submission_format.index)
    my_submission.head()
    my_submission.to_csv('submission.csv')

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 23:28:45 2023

@author: Neil Yu
"""

import pandas as pd
from scipy import stats
import numpy as np

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data.drop('Id', axis = 1, inplace = True)
test_ids = test_data["Id"].copy()


#remove null column over a 15 percent in both train and test dataset, which is only 6 variables
train_null_percentage = train_data.isnull().sum() / train_data.shape[0]
train_drop_list = list(train_null_percentage[train_null_percentage > 0.15].index)
train_data = train_data.drop(train_drop_list, axis=1)

# After observing the dataset, rows with multiple missing values (3/4 of them) were prevalent.
# This suggests potential issues with data collection or quality for these rows.
# To maintain data integrity and avoid imputation bias, these rows are dropped.
train_data = train_data.dropna(axis = 0)

#change all string data to lowr cases
train_data = train_data.applymap(lambda x: x.lower() if isinstance(x, str) else x)
test_data = test_data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# separate numerical and categorical data
numeric_data = train_data._get_numeric_data()
object_data = train_data.select_dtypes(include=['object'])

#find out variables with object type who has high p_value
large_pvalues = []
for feature in object_data:
    groups = train_data.groupby(feature)['SalePrice'].apply(list)
    f_value, p_value = stats.f_oneway(*groups)
    if p_value > 0.05:
        large_pvalues.append(feature)
    
object_data = object_data.drop(large_pvalues, axis=1)
train_data = train_data.drop(large_pvalues, axis=1)

#drop columns in test dataset that has been dropped by train dataset
remaining_cols = train_data.columns
remaining_cols = [col for col in remaining_cols if col != 'SalePrice']
test_data = test_data[remaining_cols]


#substitute all the categorical data with their accoring smoothed median SalePrice with random noise

# Constants
WEIGHT = 5
NOISE_RANGE = 0.05  # +/- 5%

# Create the smoothed median function
def get_smoothed_median(data, col, weight):
    overall_median = data['SalePrice'].median()
    category_medians = data.groupby(col)['SalePrice'].median()
    category_counts = data[col].value_counts()

    smoothed_medians = {}
    for category in category_medians.index:
        smoothed_medians[category] = (category_medians[category]*category_counts[category] + weight*overall_median) / (category_counts[category] + weight)

    return smoothed_medians

for col in train_data.columns:
    if train_data[col].dtype == 'object':
        smoothed_medians = get_smoothed_median(train_data, col, WEIGHT)
        
        # Apply the smoothed median
        train_data[col] = train_data[col].map(smoothed_medians)
        
        # Apply random noise for each row in the column
        train_data[col] *= (1 + np.random.uniform(-NOISE_RANGE, NOISE_RANGE, size=len(train_data)))

        # For test data, map with the same smoothed medians and apply noise. 
        # Unseen categories are replaced with the overall train median.
        test_data[col] = test_data[col].map(smoothed_medians).fillna(train_data['SalePrice'].median())
        test_data[col] *= (1 + np.random.uniform(-NOISE_RANGE, NOISE_RANGE, size=len(test_data)))

##
##
## feature selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

# Split your data into features and target
X = train_data.drop("SalePrice", axis=1)
y = train_data["SalePrice"]

# Split into training and validation sets for evaluation purposes
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Random Forest to get feature importances
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances and sort them
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features_rf = feature_importances.nlargest(40).index  # Let's say we want top 40 features based on importance

# Use RFE for feature selection
estimator = RandomForestRegressor(n_estimators=50, random_state=42)
selector = RFE(estimator, n_features_to_select=40, step=1)  # Here, we aim to select 40 features
selector = selector.fit(X_train, y_train)


# Get the selected features from RFE
selected_features_rfe = X.columns[selector.support_]

# Find the overlap between the two methods
overlap_features = set(top_features_rf).intersection(set(selected_features_rfe))

print("Features selected by Random Forest:", top_features_rf)
print("Features selected by RFE:", selected_features_rfe)
print("Overlap between the two methods:", overlap_features)


####
####
from sklearn.model_selection import train_test_split

X = train_data[overlap_features]  # assuming selected_features are your overlapping 36 features
y = train_data['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['auto', 'sqrt', 'log2']
}

model = GradientBoostingRegressor()

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_log_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

from sklearn.metrics import mean_squared_log_error

y_pred = grid_search.best_estimator_.predict(X_val)
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
print("RMSLE on validation set:", rmsle)


######
# Fit the model with the best parameters on the entire training set
best_model = GradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=4,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=300,
    subsample=0.8,
    random_state=42
)

best_model.fit(X_train, y_train)

# Select the same features as in the training data
test_data = test_data[overlap_features]

# Replace NaN values in each column with the mode of that column
for column in test_data.columns:
    mode_val = test_data[column].mode()[0]
    test_data[column].fillna(mode_val, inplace=True)


# Now, predict on the processed test set
predictions = best_model.predict(test_data)

# Create a DataFrame for submission
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": predictions
})

# Save the DataFrame to a CSV file
submission.to_csv("submission_new.csv", index=False)

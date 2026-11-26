# %% [markdown]
# ## Instructions
# 
# The first few code blocks contain the necessary code to run the models, as well as pre-processing steps. These include importing the necessary packages, reading the CSV file, splitting the data, defining the features and target variable, and defining the regression metrics function.
# 
# Following the initial setup, the subsequent blocks of code explore various models in a logical order, starting from linear models (such as linear, lasso, polynomial, etc.), followed by nonlinear models (such as XG Boost, Random Forest,etc). Each model is explored using pipelines, allowing for certain preprocessing steps like scaling and imputation to be integrated into the model definition. Hyperparameter optimization is also included within each the respective models' block.  Each model produces a DataFrame containing metric information like the MAE, MSE, Max_Error and R^2 values, allowing for performance evaluation.
# 
# Block #8 produces a summary DataFrame that compares the predictive performance of all models. The best three-performing models, XG Boost, Kernel Ridge Regression and K Nearest Neighbours, were then combined into a custom ensemble model. 
# 
# Parity plots are generated for the baseline DummyRegressor models and the top performing models. The final block contains a summary table that provides metric information for all tested models, including the custom ensemble model
# 
# 

# %%
'''
Models trained and evaluated on clean_data.csv with 
all NaN values imputed using median imputation.
'''

# 0. Programming Environment Setup

# Numerical computing and data handling
import numpy as np
import pandas as pd
from pathlib import Path

# Modelling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import max_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from math import degrees
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor


# Dummy modelling
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    max_error,
    mean_absolute_percentage_error,
    root_mean_squared_error
)


# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
%matplotlib inline


# %%
#1.1 Data Preparation Part 1

#Fix Random Seed
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)



#drop features, use eda_df since we're using a pipeline that will impute missing values
PROJECT_DIR = Path(__file__).resolve().parent.parent
EDA_DIR = PROJECT_DIR / "EDA"
INPUT_FILE = EDA_DIR / "cleaned_data.csv"

eda_df = pd.read_csv(INPUT_FILE, keep_default_na=False, na_values=[""])
eda_df = eda_df.drop(columns=['Z', 'N'])
eda_df.head(10)

# %%
#1.2 Data Preparation Part 2

#splitting: uses a random split since the data is not skewed 
df_train, df_test = train_test_split(
    eda_df,
    test_size=0.2,  # 20% for testing
    random_state= RANDOM_SEED  # Ensure reproducibility
)
print(df_train.shape)
df_train.head()

#Define features
features = [
    "A",
    "N/Z",
    "Q_beta_keV",
    "Q_alpha_keV",
    "S_n_keV",
    "S_p_keV"
]

#Define target
TARGET = df_train['log_half_life'].name

#Regression metrics function from tutorial 4
def get_regression_metrics(model, X, y_true):

    y_predicted = model.predict(X)

    mae = mean_absolute_error(y_true, y_predicted)
    mse = mean_squared_error(y_true, y_predicted)
    maximum_error = max_error(y_true, y_predicted)
    rmse = root_mean_squared_error(y_true, y_predicted)

    metrics_dict = {
        'mae': mae,
        'mse': mse,
        'max_error': maximum_error,
        'rmse': rmse,
        'r^2': r2_score(y_true, y_predicted)
    }

    return metrics_dict

# %%
#2.1 Build Baseline Model  (Mean)

#baseline model using a mean dummy regressor 
dummyregressor_mean = make_pipeline(
    SimpleImputer(strategy='median'),
    DummyRegressor(strategy='mean')
)

dummyregressor_mean.fit(df_train[features], df_train[TARGET])

dummyregressor_mean_results_train = get_regression_metrics(
    dummyregressor_mean,
    df_train[features],
    df_train[TARGET]
)

# Testing results
dummyregressor_mean_results_test = get_regression_metrics(
    dummyregressor_mean,
    df_test[features],
    df_test[TARGET]
)

# Create dataframe
dummyregressor_mean_results_df = pd.DataFrame(
    [dummyregressor_mean_results_train, dummyregressor_mean_results_test],
    index=['Train', 'Test']
)

# Round to 4 decimal places
dummyregressor_mean_results_df = dummyregressor_mean_results_df.round(4)

# Display dataframe
display(dummyregressor_mean_results_df)

# %%
#2.2 Build baseline model using a median dummy regressor 
dummyregressor_median = make_pipeline(
    SimpleImputer(strategy='median'),
    DummyRegressor(strategy='median')
)

dummyregressor_median.fit(df_train[features], df_train[TARGET])

dummyregressor_median_results_train = get_regression_metrics(
    dummyregressor_median,
    df_train[features],
    df_train[TARGET]
)

# Testing results
dummyregressor_median_results_test = get_regression_metrics(
    dummyregressor_median,
    df_test[features],
    df_test[TARGET]
)

# Create dataframe
dummyregressor_median_results_df = pd.DataFrame(
    [dummyregressor_median_results_train, dummyregressor_median_results_test],
    index=['Train', 'Test']
)

# Round to 4 decimal places
dummyregressor_median_results_df = dummyregressor_median_results_df.round(4)

# Display dataframe
display(dummyregressor_median_results_df)

# %%
#3.1 Linear Regression
linear_model = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    LinearRegression()
)

# Train model
linear_model.fit(df_train[features], df_train[TARGET])

# Calculate metrics
linear_model_results_train = get_regression_metrics(
    linear_model, df_train[features], df_train[TARGET]
)

linear_model_results_test = get_regression_metrics(
    linear_model, df_test[features], df_test[TARGET]
)

# Create dataframe
linear_model_results_df = pd.DataFrame(
    [linear_model_results_train, linear_model_results_test],
    index=['Train', 'Test']
)

# Round to 4 decimal places
linear_model_results_df = linear_model_results_df.round(4)

# Add model name
linear_model_results_df.insert(0, 'Model', 'Linear Regression')

# Display dataframe
display(linear_model_results_df)

# %%
#3.2 Ridge Regression
ridge_model = make_pipeline(
    SimpleImputer(strategy=('median')),
    StandardScaler(),
    Ridge(alpha=1.0)
)
ridge_model.fit(df_train[features],df_train[TARGET])
ridge_model_results_train = get_regression_metrics(ridge_model, df_train[features], df_train[TARGET])
ridge_model_results_test = get_regression_metrics(ridge_model, df_test[features], df_test[TARGET])

# Create dataframe
ridge_model_results_df = pd.DataFrame(
    [ridge_model_results_train, ridge_model_results_test],
    index=['Train', 'Test'])

# Round to 4 decimal places
ridge_model_results_df = ridge_model_results_df.round(4)
ridge_model_results_df.insert(0, 'Model', 'Ridge Regression')

# Display dataframe
display(ridge_model_results_df)

# %%
#3.3 LASSO 
lasso_model = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    Lasso(alpha=0.1, max_iter=10000)
)

lasso_model.fit(df_train[features],df_train[TARGET])
lasso_model_results_train = get_regression_metrics(lasso_model, df_train[features], df_train[TARGET])
lasso_model_results_test = get_regression_metrics(lasso_model, df_test[features], df_test[TARGET])

lasso_model_results_df = pd.DataFrame(
    [lasso_model_results_train, lasso_model_results_test],
    index=['Train', 'Test']
)
# Round to 4 decimal places
lasso_model_results_df = lasso_model_results_df.round(4)
lasso_model_results_df.insert(0, 'Model', 'Lasso Regression')

# Display dataframe
display(lasso_model_results_df)

# %%
#3.4 Polynomial Models Part 1
polynomial_results = []

for degree in [2,3,4]: #testing various polynomial degrees
    polynomial_model = make_pipeline(
        SimpleImputer(strategy='median'),
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )

    polynomial_model.fit(df_train[features],df_train[TARGET])
    polynomial_model_results_train = get_regression_metrics(polynomial_model, df_train[features], df_train[TARGET])
    polynomial_model_results_test = get_regression_metrics(polynomial_model, df_test[features], df_test[TARGET])

    polynomial_results.append({
    "Degree": degree,

    **{f"Train {k.upper()}": v
       for k, v in polynomial_model_results_train.items()},

    **{f"Test {k.upper()}": v
       for k, v in polynomial_model_results_test.items()}
})


polynomial_results_df = pd.DataFrame(polynomial_results)

polynomial_results_df = polynomial_results_df.copy()

# Rename and organize columns into Train and Test groups
polynomial_results_df = polynomial_results_df.set_index('Degree')

polynomial_results_df = polynomial_results_df[
    [
        'Train MAE', 'Train MSE', 'Train RMSE',
        'Train R^2', 'Train MAX_ERROR',
        'Test MAE', 'Test MSE', 'Test RMSE',
        'Test R^2', 'Test MAX_ERROR'
    ]
]

# Create grouped column headings
polynomial_results_df.columns = pd.MultiIndex.from_tuples([
    ('Train', 'MAE'),
    ('Train', 'MSE'),
    ('Train', 'RMSE'),
    ('Train', 'R²'),
    ('Train', 'Max Error'),
    ('Test', 'MAE'),
    ('Test', 'MSE'),
    ('Test', 'RMSE'),
    ('Test', 'R²'),
    ('Test', 'Max Error')
])

# Round values
polynomial_results_df = polynomial_results_df.round(4)

display(polynomial_results_df)

# %%
#3.5 using a polynomial model with degree 3
polynomial_deg3_model = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression()
)

polynomial_deg3_model.fit(df_train[features],df_train[TARGET])
polynomial_deg3_results_train = get_regression_metrics(polynomial_deg3_model, df_train[features], df_train[TARGET])
polynomial_deg3_results_test = get_regression_metrics(polynomial_deg3_model, df_test[features], df_test[TARGET])

polynomial_deg3_results_df = pd.DataFrame(
    [polynomial_deg3_results_train, polynomial_deg3_results_test],
    index=['Train', 'Test']
)

# Round to 4 decimal places
polynomial_deg3_results_df = polynomial_deg3_results_df.round(4)
polynomial_deg3_results_df.insert(0, 'Model', 'Polynomial (3rd Degree)')

# Display dataframe
display(polynomial_deg3_results_df)

# %%
#Kernel Ridge Regression Model
krr_baseline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaling', StandardScaler()),
    ('krr', KernelRidge(
        alpha=1.0,
        kernel='rbf',
        gamma=0.1
    ))
])

#Train baseline model
krr_baseline.fit(df_train[features], df_train[TARGET])

#Define hyperparameter grid
krr_param_grid = {
    'scaling': [StandardScaler(), MinMaxScaler()],
    'krr__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],
    'krr__gamma': [0.001, 0.01, 0.1, 1.0, 10.0]
}


#Optimize KRR using GridSearchCV
grid_krr = GridSearchCV(
    estimator=krr_baseline,
    param_grid=krr_param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

#Fit using training data only
grid_krr.fit(df_train[features], df_train[TARGET])

#Display best hyperparameters
print("Best parameters:", grid_krr.best_params_)
print("Best CV MAE:", -grid_krr.best_score_)


#Retrieve optimized KRR model
krr_optimized = grid_krr.best_estimator_


#Evaluate baseline and optimized models

#Baseline model metrics
krr_baseline_results_train = get_regression_metrics(
    krr_baseline, df_train[features], df_train[TARGET]
)

krr_baseline_results_test = get_regression_metrics(
    krr_baseline, df_test[features], df_test[TARGET]
)

# Optimized model metrics
krr_optimized_results_train = get_regression_metrics(
    krr_optimized, df_train[features], df_train[TARGET]
)

krr_optimized_results_test = get_regression_metrics(
    krr_optimized, df_test[features], df_test[TARGET]
)


#Create comparison DataFrame
#Create comparison DataFrames
df_krr_train_results = pd.DataFrame.from_dict(
    {
        'KRR - Baseline': krr_baseline_results_train,
        'KRR - Optimized': krr_optimized_results_train
    },
    orient='index'
)

df_krr_test_results = pd.DataFrame.from_dict(
    {
        'KRR - Baseline': krr_baseline_results_test,
        'KRR - Optimized': krr_optimized_results_test
    },
    orient='index'
)

#Combine training and testing results side-by-side
df_krr_comparison = pd.concat(
    {
        'Train': df_krr_train_results,
        'Test': df_krr_test_results
    },
    axis=1
)

df_krr_comparison.index.name = 'Model'

#Display results
print("\nComparison of Kernel Ridge Regression Model Performance\n")

with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', 2000
):
    display(df_krr_comparison.round(4))


# %%
#5. XGBoost 

#Train baseline model
xgb_baseline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaling', StandardScaler()),
    ('xgb', XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_SEED
    ))
])

xgb_baseline.fit(df_train[features], df_train[TARGET])

xgb_param_grid = {
    'scaling': [StandardScaler(), MinMaxScaler()],
    'xgb__n_estimators': [100, 200, 300],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__max_depth': [2, 3, 5]
}

grid_xgb = GridSearchCV(
    estimator=xgb_baseline,
    param_grid=xgb_param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=2,
    verbose=1
)


#Define hyperparameter grid

#Optimize KRR using GridSearchCV
#Fit using training data only
xgb_param_grid = {
    'scaling': [StandardScaler(), MinMaxScaler()],
    'xgb__n_estimators': [100, 200, 300],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__max_depth': [2, 3, 5]
}

grid_xgb = GridSearchCV(
    estimator=xgb_baseline,
    param_grid=xgb_param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=2,
    verbose=1
)

grid_xgb.fit(df_train[features], df_train[TARGET])

# Display best hyperparameters
print("Best parameters:", grid_xgb.best_params_)
print("Best CV MAE:", -grid_xgb.best_score_)


#Retrieve optimized KRR model

xgb_optimized = grid_xgb.best_estimator_


#Evaluate baseline and optimized models

#Baseline model metrics
xgb_baseline_results_train = get_regression_metrics(
    xgb_baseline, df_train[features], df_train[TARGET]
)

xgb_baseline_results_test = get_regression_metrics(
    xgb_baseline, df_test[features], df_test[TARGET]
)

#Optimized model metrics
xgb_optimized_results_train = get_regression_metrics(
    xgb_optimized, df_train[features], df_train[TARGET]
)

xgb_optimized_results_test = get_regression_metrics(
    xgb_optimized, df_test[features], df_test[TARGET]
)


#Create comparison DataFrame
df_xgb_train_results = pd.DataFrame.from_dict(
    {
        'XGB - Baseline': xgb_baseline_results_train,
        'XGB - Optimized': xgb_optimized_results_train
    },
    orient='index'
)

df_xgb_test_results = pd.DataFrame.from_dict(
    {
        'XGB - Baseline': xgb_baseline_results_test,
        'XGB - Optimized': xgb_optimized_results_test
    },
    orient='index'
)

# Combine training and testing results side-by-side
df_xgb_comparison = pd.concat(
    {
        'Train': df_xgb_train_results,
        'Test': df_xgb_test_results
    },
    axis=1
)

df_xgb_comparison.index.name = 'Model'

# Display results
print("\nComparison of XGBoost Performance\n")

with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', 2000
):
    display(df_xgb_comparison.round(4))
    


# %%
#6. Random Forest Model

# Train baseline model
rf_baseline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaling', StandardScaler()),
    ('rf', RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_SEED
    ))
])

rf_baseline.fit(df_train[features], df_train[TARGET])

param_grid_rf = {
    'randomforestregressor__n_estimators': [100, 200, 300],
    'randomforestregressor__max_depth': [None, 10, 20, 30],
    'randomforestregressor__min_samples_split': [2, 5, 10],
    'randomforestregressor__min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(
    estimator=rf_baseline,
    param_grid=param_grid_rf,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)


rf_baseline.fit(df_train[features], df_train[TARGET])

param_grid_rf = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(
    estimator=rf_baseline,
    param_grid=param_grid_rf,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

grid_rf.fit(
    df_train[features],
    df_train[TARGET]
)

# Display best hyperparameters
print("Best parameters:", grid_rf.best_params_)
print("Best CV MAE:", -grid_rf.best_score_)


# 4. Retrieve optimized KRR model

rf_optimized = grid_rf.best_estimator_


# 5. Evaluate baseline and optimized models

# Baseline model metrics
rf_baseline_results_train = get_regression_metrics(
    rf_baseline, df_train[features], df_train[TARGET]
)

rf_baseline_results_test = get_regression_metrics(
    rf_baseline, df_test[features], df_test[TARGET]
)

# Optimized model metrics
rf_optimized_results_train = get_regression_metrics(
    rf_optimized, df_train[features], df_train[TARGET]
)

rf_optimized_results_test = get_regression_metrics(
    rf_optimized, df_test[features], df_test[TARGET]
)


# 6. Create comparison DataFrame
df_rf_train_results = pd.DataFrame.from_dict(
    {
        'Random Forest - Baseline': rf_baseline_results_train,
        'Random Forest - Optimized': rf_optimized_results_train
    },
    orient='index'
)

df_rf_test_results = pd.DataFrame.from_dict(
    {
        'Random Forest - Baseline': rf_baseline_results_test,
        'Random Forest - Optimized': rf_optimized_results_test
    },
    orient='index'
)

# Combine training and testing results side-by-side
df_rf_comparison = pd.concat(
    {
        'Train': df_rf_train_results,
        'Test': df_rf_test_results
    },
    axis=1
)

df_rf_comparison.index.name = 'Model'

# Display results
print("\nComparison of Random Forest Performance\n")

with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', 2000
):
    display(df_rf_comparison.round(4))
    

# %%
#7. K-Nearest Neighbours Model

knn_baseline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaling', StandardScaler()),
    ('knn', KNeighborsRegressor(
        n_neighbors=5
    ))
])

# Train baseline model
knn_baseline.fit(df_train[features], df_train[TARGET])


#Define hyperparameter grid
param_grid_knn = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

#Optimize KNN using GridSearchCV

grid_knn = GridSearchCV(
    estimator=knn_baseline,
    param_grid=param_grid_knn,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

grid_knn.fit(
    df_train[features],
    df_train[TARGET]
)

#Display best hyperparameters
print("Best parameters:", grid_knn.best_params_)
print("Best CV MAE:", -grid_knn.best_score_)


#Retrieve optimized KNN model

knn_optimized = grid_knn.best_estimator_


# Evaluate baseline and optimized models

# Baseline model metrics
knn_baseline_results_train = get_regression_metrics(
    knn_baseline, df_train[features], df_train[TARGET]
)

knn_baseline_results_test = get_regression_metrics(
    knn_baseline, df_test[features], df_test[TARGET]
)

# Optimized model metrics
knn_optimized_results_train = get_regression_metrics(
    knn_optimized, df_train[features], df_train[TARGET]
)

knn_optimized_results_test = get_regression_metrics(
    knn_optimized, df_test[features], df_test[TARGET]
)


#Create comparison DataFrames

df_knn_train_results = pd.DataFrame.from_dict(
    {
        'KNN - Baseline': knn_baseline_results_train,
        'KNN - Optimized': knn_optimized_results_train
    },
    orient='index'
)

df_knn_test_results = pd.DataFrame.from_dict(
    {
        'KNN - Baseline': knn_baseline_results_test,
        'KNN - Optimized': knn_optimized_results_test
    },
    orient='index'
)

# Combine training and testing results side-by-side
df_knn_comparison = pd.concat(
    {
        'Train': df_knn_train_results,
        'Test': df_knn_test_results
    },
    axis=1
)

df_knn_comparison.index.name = 'Model'


#Display results

print("\nComparison of KNN Performance\n")

with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', 2000
):
    display(df_knn_comparison.round(4))


# %%
#8. Summary Dataframe of all models created so far

# Create dataframe containing all test results
all_results_test_df = pd.DataFrame({
    'Dummy (Mean)': dummyregressor_mean_results_test,
    'Dummy (Median)': dummyregressor_median_results_test,
    'OLS - Baseline': linear_model_results_test,
    'Ridge Regression': ridge_model_results_test,
    'LASSO': lasso_model_results_test,
    'Polynomial Degree 3': polynomial_deg3_results_test,
    'Kernel Ridge Regression': krr_optimized_results_test,
    'XGBoost': xgb_optimized_results_test,
    'Random Forest': rf_optimized_results_test,
    'K Nearest Neighbors': knn_optimized_results_test,
})

# Transpose dataframe
all_results_test_df = all_results_test_df.T

# Add model categories
all_results_test_df.insert(0, 'Model Category', [
    'Dummy Regression',
    'Dummy Regression',
    'Linear Regression',
    'Linear Regression',
    'Linear Regression',
    'Polynomial Regression',
    'Kernel Regression',
    'Tree-Based Regression',
    'Tree-Based Regression',
    'Distance-Based Regression'
])

# Add model names
all_results_test_df.insert(1, 'Model', all_results_test_df.index)

# Reset index
all_results_test_df = all_results_test_df.reset_index(drop=True)

# Round to 4 decimal places
all_results_test_df = all_results_test_df.round(4)


# Display test results
print("\nTEST SET MODEL PERFORMANCE")
display(all_results_test_df)



# Create dataframe containing all training results
all_results_train_df = pd.DataFrame({
    'Dummy (Mean)': dummyregressor_mean_results_train,
    'Dummy (Median)': dummyregressor_median_results_train,
    'OLS - Baseline': linear_model_results_train,
    'Ridge Regression': ridge_model_results_train,
    'LASSO': lasso_model_results_train,
    'Polynomial Degree 3': polynomial_deg3_results_train,
    'Kernel Ridge Regression': krr_optimized_results_train,
    'XGBoost': xgb_optimized_results_train,
    'Random Forest': rf_optimized_results_train,
    'K Nearest Neighbors': knn_optimized_results_train,
})

# Transpose dataframe
all_results_train_df = all_results_train_df.T

# Add model categories
all_results_train_df.insert(0, 'Model Category', [
    'Dummy Regression',
    'Dummy Regression',
    'Linear Regression',
    'Linear Regression',
    'Linear Regression',
    'Polynomial Regression',
    'Kernel Regression',
    'Tree-Based Regression',
    'Tree-Based Regression',
    'Distance-Based Regression'
])

# Add model names
all_results_train_df.insert(1, 'Model', all_results_train_df.index)

# Reset index
all_results_train_df = all_results_train_df.reset_index(drop=True)

# Round to 4 decimal places
all_results_train_df = all_results_train_df.round(4)

# Display results
print("\nTRAINING SET MODEL PERFORMANCE")
display(all_results_train_df)


# %%
#9. Create a custom ensemble model out of the top 3 performing models from first round of testing
# ensemble model = random forest + xgb + krr, all with equal weight
ensemble_model = VotingRegressor(
    estimators=[
        ('random forest', rf_optimized),
        ('xgb', xgb_optimized),
        ('krr', krr_optimized),
    ]
)

ensemble_model.fit(df_train[features], df_train[TARGET])
ensemble_model_results_train = get_regression_metrics(ensemble_model, df_train[features], df_train[TARGET])
ensemble_model_results_test = get_regression_metrics(ensemble_model, df_test[features], df_test[TARGET])

# Create dataframe
ensemble_model_results_df = pd.DataFrame(
    [ensemble_model_results_train, ensemble_model_results_test],
    index=['Train', 'Test']
)

# Round to 4 decimal places
ensemble_model_results_df = ensemble_model_results_df.round(4)

# Add model name
ensemble_model_results_df.insert(0, 'Model', 'Ensemble')

# Display dataframe
display(ensemble_model_results_df)

# %%
#10. Parity plots of dummy models + best performing models
# Define models
models = {
    'Dummy (Mean)': dummyregressor_mean,
    'Dummy (Median)': dummyregressor_median,
    'Kernel Ridge': krr_optimized,
    'XGBoost': xgb_optimized,
    'Random Forest': rf_optimized,
    'KNN': knn_optimized,
    'Ensemble': ensemble_model
}

# Define parity plot function
def hist_density(y_true, y_pred, xlabel='True log half-life',
                 ylabel='Predicted log half-life', title=None, bins=50):

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]

    mae = np.mean(np.abs(y_true - y_pred))

    # Create individual figure
    fig, ax = plt.subplots(figsize=(6, 4.5))

    h = ax.hist2d(
        y_true, y_pred,
        bins=bins,
        range=[lims, lims],
        cmin=1,
        cmap='viridis',
        norm=LogNorm()
    )

    # Add colour bar
    fig.colorbar(h[3], ax=ax, label='Number of isotopes')

    # Add parity line
    ax.plot(lims, lims, 'k--', lw=1, label='y = x')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title} (MAE = {mae:.2f})" if title else f"MAE = {mae:.2f}"
    )

    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


# Generate individual parity plots for all models
for name, model in models.items():

    y_pred = model.predict(df_test[features])

    hist_density(
        df_test[TARGET],
        y_pred,
        title=f'Test - {name}'
    )


# %%
#11. Summary dataframe of all models 

# Create dataframe containing all test results
all_results_test_df = pd.DataFrame({
    'Dummy (Mean)': dummyregressor_mean_results_test,
    'Dummy (Median)': dummyregressor_median_results_test,
    'OLS - Baseline': linear_model_results_test,
    'Ridge Regression': ridge_model_results_test,
    'LASSO': lasso_model_results_test,
    'Polynomial Degree 3': polynomial_deg3_results_test,
    'Kernel Ridge Regression': krr_optimized_results_test,
    'XGBoost': xgb_optimized_results_test,
    'Random Forest': rf_optimized_results_test,
    'K Nearest Neighbors': knn_optimized_results_test,
    'Ensemble Model': ensemble_model_results_test
})

# Transpose dataframe
all_results_test_df = all_results_test_df.T

# Add model categories
all_results_test_df.insert(0, 'Model Category', [
    'Dummy Regression',
    'Dummy Regression',
    'Linear Regression',
    'Linear Regression',
    'Linear Regression',
    'Polynomial Regression',
    'Kernel Regression',
    'Ensemble',
    'Ensemble',
    'Distance-Based Regression',
    'Ensemble'
])

# Add model names
all_results_test_df.insert(1, 'Model', all_results_test_df.index)

# Reset index
all_results_test_df = all_results_test_df.reset_index(drop=True)

# Round to 4 decimal places
all_results_test_df = all_results_test_df.round(4)


# Display test results
print("\nTEST SET MODEL PERFORMANCE")
display(all_results_test_df)



# Create dataframe containing all training results
all_results_train_df = pd.DataFrame({
    'Dummy (Mean)': dummyregressor_mean_results_train,
    'Dummy (Median)': dummyregressor_median_results_train,
    'OLS - Baseline': linear_model_results_train,
    'Ridge Regression': ridge_model_results_train,
    'LASSO': lasso_model_results_train,
    'Polynomial Degree 3': polynomial_deg3_results_train,
    'Kernel Ridge Regression': krr_optimized_results_train,
    'XGBoost': xgb_optimized_results_train,
    'Random Forest': rf_optimized_results_train,
    'K Nearest Neighbors': knn_optimized_results_train,
    'Ensemble Model': ensemble_model_results_train
})

# Transpose dataframe
all_results_train_df = all_results_train_df.T

# Add model categories
all_results_train_df.insert(0, 'Model Category', [
    'Dummy Regression',
    'Dummy Regression',
    'Linear Regression',
    'Linear Regression',
    'Linear Regression',
    'Polynomial Regression',
    'Kernel Regression',
    'Ensemble',
    'Ensemble',
    'Distance-Based Regression',
    'Ensemble'
])

# Add model names
all_results_train_df.insert(1, 'Model', all_results_train_df.index)

# Reset index
all_results_train_df = all_results_train_df.reset_index(drop=True)

# Round to 4 decimal places
all_results_train_df = all_results_train_df.round(4)

# Display results
print("\nTRAINING SET MODEL PERFORMANCE")
display(all_results_train_df)




# %% [markdown]
# # Nuclear Half-Life Prediction — Included NaN
# 
# This notebook evaluates regression models that can natively handle missing feature values without imputation or row removal.
# 
# Models:
# - Dummy Regressor
# - Decision Tree Regressor
# - Random Forest Regressor
# - XGBoost Regressor

# %%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_validate
)

from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    max_error
)

from xgboost import XGBRegressor
import xgboost as xgb


# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# Environment check
print("Python:", sys.version)
print("scikit-learn:", sklearn.__version__)
print("XGBoost:", xgb.__version__)
print("Random seed:", RANDOM_SEED)

# %% [markdown]
# ## 2. Load Dataset
# 
# Load the extracted ENSDF dataset while preserving missing feature values.

# %%
# Import Data
PROJECT_DIR = Path(__file__).resolve().parent.parent
EXTRACTION_DIR = PROJECT_DIR / "Extraction"
INPUT_FILE = EXTRACTION_DIR / "extracted_data.csv"

df = pd.read_csv(INPUT_FILE, keep_default_na=False, na_values=[""])

print("Dataset shape:", df.shape)
display(df.head())

# %%
# 3. Prepare target
# Keep finite half-lives and use log10(seconds) as the target.

reg_df = df[df["half_life_status"] == "finite"].copy()

reg_df["log_half_life"] = np.log10(
    reg_df["half_life_seconds"]
)

print("Finite half-life samples:", len(reg_df))
display(reg_df[["nuclide", "half_life_seconds", "log_half_life"]].head())

# %%
# 4. Define features
# Use A, N/Z, Qβ, Qα, Sn, and Sp. Keep NaN values unchanged.

reg_df["N/Z"] = reg_df["N"] / reg_df["Z"]

FEATURES = [
    "A",
    "N/Z",
    "Q_beta_keV",
    "Q_alpha_keV",
    "S_n_keV",
    "S_p_keV"
]

TARGET = "log_half_life"

X = reg_df[FEATURES].copy()
y = reg_df[TARGET].copy()

print("Features:", FEATURES)
print("X shape:", X.shape)
print("y shape:", y.shape)

display(X.head())

# %%
# 5. Check missing values
# Confirm that NaN values are retained in the Included-NaN dataset.

missing_summary = pd.DataFrame({
    "Missing count": X.isna().sum(),
    "Missing (%)": X.isna().mean() * 100
})

display(missing_summary)

# %%
# 6. Train-test split
# Use the same 80:20 split and random seed for reproducibility.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

print("\nMissing values in training set:")
display(X_train.isna().sum())

print("Missing values in test set:")
display(X_test.isna().sum())

# %%
# 7. Evaluation function
# Use the same metrics for every model.

def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    maximum_error = max_error(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Max Error": maximum_error
    }

# %%
# 8. Dummy baselines
# Evaluate mean and median baselines using 5-fold CV on the training set.

dummy_mean = DummyRegressor(strategy="mean")
dummy_median = DummyRegressor(strategy="median")

dummy_mean_cv = cross_validate(
    dummy_mean,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error"
)

dummy_median_cv = cross_validate(
    dummy_median,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error"
)

dummy_results = pd.DataFrame({
    "Model": ["Dummy Mean", "Dummy Median"],
    "CV MAE": [
        -dummy_mean_cv["test_score"].mean(),
        -dummy_median_cv["test_score"].mean()
    ]
})

display(dummy_results)

# %%
# 9. Decision Tree baseline
# Evaluate the default tree using 5-fold CV on the training set.

dt_baseline = DecisionTreeRegressor(
    random_state=RANDOM_SEED
)

dt_baseline_cv = cross_validate(
    dt_baseline,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error"
)

dt_baseline_cv_mae = -dt_baseline_cv["test_score"].mean()

print("Decision Tree baseline CV MAE:")
print(dt_baseline_cv_mae)

# %%
# 10. Decision Tree tuning
# Use 5-fold CV and MAE to control tree complexity.

dt_param_grid = {
    "max_depth": [3, 5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5, 10]
}

dt_grid = GridSearchCV(
    estimator=DecisionTreeRegressor(
        random_state=RANDOM_SEED
    ),
    param_grid=dt_param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

dt_grid.fit(X_train, y_train)

print("Best parameters:")
print(dt_grid.best_params_)

print("\nBest CV MAE:")
print(-dt_grid.best_score_)

# %%
# 11. Select tuned Decision Tree
# Store the best model selected by training-set cross-validation.

dt_best = dt_grid.best_estimator_

print("Best Decision Tree parameters:")
print(dt_grid.best_params_)

print("\nBest Decision Tree CV MAE:")
print(-dt_grid.best_score_)

# %%
# 12. Random Forest baseline
# Use the course baseline settings and evaluate with 5-fold CV.

rf_baseline = RandomForestRegressor(
    n_estimators=100,
    max_features=0.5,
    min_samples_leaf=3,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

rf_baseline_cv = cross_validate(
    rf_baseline,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

rf_baseline_cv_mae = -rf_baseline_cv["test_score"].mean()

print("Random Forest baseline CV MAE:")
print(rf_baseline_cv_mae)

# %%
# 13. Random Forest tuning
# Tune the main RF hyperparameters using 5-fold CV and MAE.

rf_param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [10, 20, None],
    "max_features": [0.3, 0.5],
    "min_samples_leaf": [1, 2]
}

rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    param_grid=rf_param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

print("Best parameters:")
print(rf_grid.best_params_)

print("\nBest CV MAE:")
print(-rf_grid.best_score_)

# %%
# 14. Refine Random Forest tuning
# Search around the best parameters from the first grid.

rf_refined_grid = {
    "n_estimators": [500, 800],
    "max_depth": [30, None],
    "max_features": [0.5, 0.7],
    "min_samples_leaf": [1, 2]
}

rf_refined = GridSearchCV(
    estimator=RandomForestRegressor(
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    param_grid=rf_refined_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

rf_refined.fit(X_train, y_train)

print("Best parameters:")
print(rf_refined.best_params_)

print("\nBest CV MAE:")
print(-rf_refined.best_score_)

# %%
# 15. Select tuned Random Forest
# Store the best RF selected by training-set cross-validation.

rf_best = rf_refined.best_estimator_

print("Best Random Forest parameters:")
print(rf_refined.best_params_)

print("\nBest Random Forest CV MAE:")
print(-rf_refined.best_score_)

# %%
# 16. XGBoost baseline
# Evaluate the default XGBoost model using 5-fold CV.

xgb_baseline = XGBRegressor(
    objective="reg:squarederror",
    random_state=RANDOM_SEED,
    n_jobs=-1
)

xgb_baseline_cv = cross_validate(
    xgb_baseline,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

xgb_baseline_cv_mae = -xgb_baseline_cv["test_score"].mean()

print("XGBoost baseline CV MAE:")
print(xgb_baseline_cv_mae)

# %%
# 17. XGBoost tuning
# Tune the main XGBoost hyperparameters using 5-fold CV and MAE.

xgb_param_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1]
}

xgb_grid = GridSearchCV(
    estimator=XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=1
    ),
    param_grid=xgb_param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train, y_train)

print("Best parameters:")
print(xgb_grid.best_params_)

print("\nBest CV MAE:")
print(-xgb_grid.best_score_)

# %%
# 18. Refine XGBoost tuning
# Search around the best parameters from the first grid.

xgb_refined_grid = {
    "n_estimators": [800, 1200],
    "max_depth": [7, 9],
    "learning_rate": [0.03, 0.05, 0.07]
}

xgb_refined = GridSearchCV(
    estimator=XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=1
    ),
    param_grid=xgb_refined_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

xgb_refined.fit(X_train, y_train)

print("Best refined parameters:")
print(xgb_refined.best_params_)

print("\nBest refined CV MAE:")
print(-xgb_refined.best_score_)

# %%
# 19. Final XGBoost test evaluation
# Evaluate the CV-selected XGBoost model once on the test set.

xgb_best = xgb_refined.best_estimator_

xgb_best_train_pred = xgb_best.predict(X_train)
xgb_best_test_pred = xgb_best.predict(X_test)

xgb_best_results = pd.DataFrame(
    {
        "Train": evaluate_regression(y_train, xgb_best_train_pred),
        "Test": evaluate_regression(y_test, xgb_best_test_pred)
    }
).T

display(xgb_best_results)

print("Best XGBoost parameters:")
print(xgb_refined.best_params_)

print("\nBest XGBoost CV MAE:")
print(-xgb_refined.best_score_)

# %%
# 20. Final model comparison
# Compare the tuned models on the same test set.

dt_final_pred = dt_best.predict(X_test)
rf_final_pred = rf_best.predict(X_test)
xgb_final_pred = xgb_best.predict(X_test)

final_results = pd.DataFrame({
    "Decision Tree": evaluate_regression(y_test, dt_final_pred),
    "Random Forest": evaluate_regression(y_test, rf_final_pred),
    "XGBoost": evaluate_regression(y_test, xgb_final_pred)
}).T

final_results = final_results.sort_values("MAE")

display(final_results)

# %%
# 21. Parity plots
# Compare actual and predicted test values for all tuned models.

predictions = {
    "Decision Tree": dt_final_pred,
    "Random Forest": rf_final_pred,
    "XGBoost": xgb_final_pred
}

all_values = np.concatenate([
    y_test.to_numpy(),
    dt_final_pred,
    rf_final_pred,
    xgb_final_pred
])

plot_min = all_values.min()
plot_max = all_values.max()

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.5),
    sharex=True,
    sharey=True
)

for ax, (model_name, y_pred) in zip(axes, predictions.items()):
    results = evaluate_regression(y_test, y_pred)

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.6,
        s=25
    )

    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        linestyle="--"
    )

    ax.set_title(
        f"{model_name}\n"
        f"MAE = {results['MAE']:.3f}, "
        f"$R^2$ = {results['R2']:.3f}"
    )

    ax.set_xlabel("Actual log10(Half-Life [s])")
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.grid(alpha=0.25)

axes[0].set_ylabel("Predicted log10(Half-Life [s])")

plt.tight_layout()
plt.show()

# %%
# 22. Feature importance
# Compare the relative feature importance of the tuned tree-based models.

models = {
    "Decision Tree": dt_best,
    "Random Forest": rf_best,
    "XGBoost": xgb_best
}

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.5),
    sharey=True
)

for ax, (model_name, model) in zip(axes, models.items()):
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=True
    )

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.set_title(model_name)
    ax.set_xlabel("Feature Importance")
    ax.grid(
        axis="x",
        alpha=0.25
    )

axes[0].set_ylabel("Feature")

plt.tight_layout()
plt.show()

# %%
# 23. Test MAE comparison
# Compare final test MAE across the tuned models.

mae_comparison = final_results["MAE"].sort_values()

plt.figure(figsize=(7, 4.5))

bars = plt.bar(
    mae_comparison.index,
    mae_comparison.values
)

plt.ylabel("Test MAE")
plt.title("Final Test MAE by Model")
plt.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, mae_comparison.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()

# %%
# 24. Final summary
# Summarize the best models, CV performance, and final test results.

summary_table = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],
    "Best CV MAE": [
        -dt_grid.best_score_,
        -rf_refined.best_score_,
        -xgb_refined.best_score_
    ],
    "Test MAE": [
        final_results.loc["Decision Tree", "MAE"],
        final_results.loc["Random Forest", "MAE"],
        final_results.loc["XGBoost", "MAE"]
    ],
    "Test RMSE": [
        final_results.loc["Decision Tree", "RMSE"],
        final_results.loc["Random Forest", "RMSE"],
        final_results.loc["XGBoost", "RMSE"]
    ],
    "Test R2": [
        final_results.loc["Decision Tree", "R2"],
        final_results.loc["Random Forest", "R2"],
        final_results.loc["XGBoost", "R2"]
    ],
    "Test Max Error": [
        final_results.loc["Decision Tree", "Max Error"],
        final_results.loc["Random Forest", "Max Error"],
        final_results.loc["XGBoost", "Max Error"]
    ]
})

summary_table = summary_table.sort_values(
    "Test MAE"
).reset_index(drop=True)

display(summary_table)

print("\nBest model on the Included-NaN test set:")
print(summary_table.loc[0, "Model"])



# IMPORTS

import os
import duckdb
import numpy as np
import pandas as pd
import optuna

# SCIKIT-LEARN

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


from catboost import CatBoostRegressor

from src.Preprocessor.preprocessor import data_preprocessor
from src.Analyze_data.export_analyse import load_features_from_db

# =======================================================
# DATA PREPARATION
# =======================================================
df = load_features_from_db()
X_train_proc, X_test_proc, y_train, y_test, paths_train, paths_test, ids_train, ids_test, preprocessor = data_preprocessor(df)


# =======================================================
# OPTUNA STUDY
# =======================================================
# def objective(trial):
#     params = {
#         'iterations': trial.suggest_int('iterations', 10, 200, step=10),
#         'depth': trial.suggest_int('depth', 1, 10),
#         'loss_function': 'RMSE',
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
#         'task_type': 'GPU',
#         'verbose': 0
#     }
#
#     model = CatBoostRegressor(**params, random_state=42)
#     r2_scores = cross_val_score(model, X_train_proc, y_train, cv=5, scoring="r2", n_jobs=1)
#     return r2_scores.mean()
#
#
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=30)
#
# print("BEST R2 VIA CROSS-VAL:", study.best_value)
# print("Best params from Optuna:", study.best_params)


# =======================================================
# FINAL RESULTS (TRAINING & SAVING)
# =======================================================

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'CatBoost', 'catboost_best_model.cbm')

# final_model = CatBoostRegressor(
#     **study.best_params,
#     loss_function='RMSE',
#     task_type='GPU',
#     random_state=42,
#     logging_level='Verbose'
# )
#
# final_model.fit(X_train_proc, y_train, eval_set=(X_test_proc, y_test), plot=False)
# final_model.save_model(model_path)
# print(f"Model saved in: {model_path}")


# =======================================================
# TESTING LOADING FROM EXISTING CBM
# =======================================================

loaded_model = CatBoostRegressor()
loaded_model.load_model(model_dir)
best_model_params = loaded_model.get_all_params()
print('\nLoaded model params:', best_model_params)
y_pred = loaded_model.predict(X_test_proc)
test_r2 = r2_score(y_test, y_pred)
print(f"R2 SCORE on VALIDATION (Loaded Model): {test_r2:.4f}")

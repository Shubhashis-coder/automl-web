"""
Configuration file for models, parameter grids, and tuning settings.
"""

# -------------------------------------------------------------------
# TUNING LEVELS
# -------------------------------------------------------------------
TUNING_LEVELS = {
    'quick': {'n_trials': 20, 'cv': 3, 'n_iter': 10},
    'medium': {'n_trials': 50, 'cv': 5, 'n_iter': 30},
    'extensive': {'n_trials': 100, 'cv': 10, 'n_iter': 70}
}

TUNING_METHODS = ['optuna', 'random', 'grid', 'halving']

# -------------------------------------------------------------------
# PARAMETER GRIDS FOR ALL 10 MODELS
# -------------------------------------------------------------------
PARAM_GRIDS = {
    'Linear Regression': {},
    'ElasticNet': {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_iter': [1000, 2000, 5000]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.6, 0.8, 1.0],
        'max_features': ['sqrt', 'log2', None]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9, 12],
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.5, 0.7, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        'reg_alpha': [0, 0.001, 0.01, 0.1, 1.0, 5.0],
        'reg_lambda': [0, 0.001, 0.01, 0.1, 1.0, 5.0],
        'min_child_weight': [1, 3, 5, 7, 10]
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [-1, 5, 10, 15, 20],
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        'num_leaves': [31, 50, 70, 100, 150],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.001, 0.01, 0.1, 1.0],
        'reg_lambda': [0, 0.001, 0.01, 0.1, 1.0],
        'min_child_samples': [5, 10, 20, 30, 50]
    },
    'CatBoost': {
        'iterations': [50, 100, 200, 300, 500],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5, 7, 10],
        'border_count': [32, 64, 128],
        'bagging_temperature': [0, 0.5, 1.0, 2.0],
        'random_strength': [1, 2, 5, 10]
    },
    'KNN': {
        'n_neighbors': [2, 3, 5, 7, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    },
    'SVR': {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'epsilon': [0.001, 0.01, 0.05, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4]
    },
    'Gaussian Process': {
        'alpha': [1e-10, 1e-8, 1e-5, 1e-3, 0.01, 0.1, 1.0],
        'kernel': ['RBF', 'Matern', 'RationalQuadratic'],
        'normalize_y': [True, False],
        'n_restarts_optimizer': [0, 3, 5, 10]
    },
    'ANN': {
        'units_1': [16, 32, 64, 128, 256],
        'units_2': [8, 16, 32, 64, 128],
        'units_3': [4, 8, 16, 32, 64],
        'dropout_1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'dropout_2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2],
        'l2_reg': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
        'batch_size': [8, 16, 32, 64],
        'epochs': [50, 100, 150, 200, 300],
        'activation': ['relu', 'elu', 'selu', 'tanh']
    }
}

# -------------------------------------------------------------------
# DEFAULT PARAMETERS FOR MANUAL TRAINING
# -------------------------------------------------------------------
DEFAULT_PARAMS = {
    'Linear Regression': {},
    'ElasticNet': {'alpha': 1.0, 'l1_ratio': 0.5, 'max_iter': 2000},
    'Random Forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1},
    'Gradient Boosting': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
    'XGBoost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
    'LightGBM': {'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.1, 'num_leaves': 31},
    'CatBoost': {'iterations': 100, 'depth': 6, 'learning_rate': 0.1},
    'KNN': {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto', 'p': 2},
    'SVR': {'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf'},
    'Gaussian Process': {'alpha': 1e-5, 'kernel': 'Matern', 'normalize_y': True},
    'ANN': {
        'units_1': 64, 'units_2': 32, 'units_3': 16,
        'dropout_1': 0.2, 'dropout_2': 0.2,
        'learning_rate': 0.001, 'l2_reg': 0.001,
        'batch_size': 16, 'epochs': 100, 'activation': 'relu'
    }
}
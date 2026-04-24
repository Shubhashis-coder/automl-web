import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Try importing TensorFlow (optional)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. ANN model will be skipped.")

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def ensure_numeric(X):
    """Convert any remaining object/string columns to numeric using LabelEncoder."""
    X = X.copy()
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col].astype(str))
    return X

def train_test_scale(X, y, test_size=0.2, random_state=42):
    """Split and scale data."""
    X = ensure_numeric(X)
    if len(X) < 10:
        raise ValueError(f"Too few samples ({len(X)}) for reliable training.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# -------------------------------------------------------------------
# ANN training
# -------------------------------------------------------------------
def _train_ann(X_train, y_train, params, X_val=None, y_val=None):
    """Build and train ANN model."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed. Cannot train ANN.")
    
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(params['units_1'], activation=params.get('activation', 'relu'),
                    kernel_regularizer=l2(params.get('l2_reg', 0.001))))
    model.add(BatchNormalization())
    model.add(Dropout(params.get('dropout_1', 0.3)))
    model.add(Dense(params['units_2'], activation=params.get('activation', 'relu')))
    model.add(Dropout(params.get('dropout_2', 0.2)))
    if 'units_3' in params and params['units_3'] > 0:
        model.add(Dense(params['units_3'], activation=params.get('activation', 'relu')))
        model.add(Dropout(params.get('dropout_2', 0.2)))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=params.get('learning_rate', 0.001))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss', min_lr=1e-6)
    
    validation_data = (X_val, y_val) if X_val is not None else None
    validation_split = 0.2 if validation_data is None else 0.0
    
    history = model.fit(
        X_train, y_train,
        epochs=params.get('epochs', 100),
        batch_size=params.get('batch_size', 16),
        validation_split=validation_split,
        validation_data=validation_data,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    return model, history

# -------------------------------------------------------------------
# Optuna objective functions
# -------------------------------------------------------------------
def _objective_rf(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return r2_score(y_val, model.predict(X_val))

def _objective_xgb(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }
    model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return r2_score(y_val, model.predict(X_val))

def _objective_lgb(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', -1, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 150),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }
    model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    return r2_score(y_val, model.predict(X_val))

def _objective_cat(trial, X_train, y_train, X_val, y_val):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
    }
    model = CatBoostRegressor(**params, random_state=42, verbose=False, allow_writing_files=False)
    model.fit(X_train, y_train)
    return r2_score(y_val, model.predict(X_val))

def _objective_knn(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 2, 20),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'p': trial.suggest_int('p', 1, 2),
    }
    model = KNeighborsRegressor(**params, n_jobs=-1)
    model.fit(X_train, y_train)
    return r2_score(y_val, model.predict(X_val))

def _objective_svr(trial, X_train, y_train, X_val, y_val):
    params = {
        'C': trial.suggest_float('C', 0.01, 100.0, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.001, 0.5, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
    }
    model = SVR(**params)
    model.fit(X_train, y_train)
    return r2_score(y_val, model.predict(X_val))

def _objective_elasticnet(trial, X_train, y_train, X_val, y_val):
    params = {
        'alpha': trial.suggest_float('alpha', 0.0001, 10.0, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
    }
    model = ElasticNet(**params, random_state=42, max_iter=5000)
    model.fit(X_train, y_train)
    return r2_score(y_val, model.predict(X_val))

def _objective_gp(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for Gaussian Process Regression."""
    kernel_choice = trial.suggest_categorical('kernel_type', ['RBF', 'Matern', 'RationalQuadratic'])
    
    # Create kernel based on choice
    if kernel_choice == 'RBF':
        length_scale = trial.suggest_float('rbf_length_scale', 0.01, 10.0, log=True)
        kernel = RBF(length_scale=length_scale)
    elif kernel_choice == 'Matern':
        length_scale = trial.suggest_float('matern_length_scale', 0.01, 10.0, log=True)
        nu = trial.suggest_categorical('matern_nu', [0.5, 1.5, 2.5])
        kernel = Matern(length_scale=length_scale, nu=nu)
    else:
        length_scale = trial.suggest_float('rq_length_scale', 0.01, 10.0, log=True)
        alpha_param = trial.suggest_float('rq_alpha', 0.1, 10.0, log=True)
        kernel = RationalQuadratic(length_scale=length_scale, alpha=alpha_param)
    
    alpha_val = trial.suggest_float('alpha', 1e-10, 1.0, log=True)
    
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha_val,
        random_state=42,
        n_restarts_optimizer=5,
        normalize_y=True
    )
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return r2_score(y_val, y_pred)
    except Exception:
        return -1.0  # return poor score on failure

def _objective_ann(trial, X_train, y_train, X_val, y_val):
    if not TF_AVAILABLE:
        return -1.0
    
    params = {
        'units_1': trial.suggest_int('units_1', 16, 256),
        'units_2': trial.suggest_int('units_2', 8, 128),
        'units_3': trial.suggest_int('units_3', 4, 64),
        'dropout_1': trial.suggest_float('dropout_1', 0.1, 0.5),
        'dropout_2': trial.suggest_float('dropout_2', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'l2_reg': trial.suggest_float('l2_reg', 1e-7, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
        'epochs': trial.suggest_int('epochs', 50, 200),
        'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'selu']),
    }
    try:
        model, _ = _train_ann(X_train, y_train, params, X_val, y_val)
        y_pred = model.predict(X_val, verbose=0).ravel()
        return r2_score(y_val, y_pred)
    except Exception:
        return -1.0

# Mapping from model name to objective function
OBJECTIVE_FUNCTIONS = {
    'Random Forest': _objective_rf,
    'XGBoost': _objective_xgb,
    'LightGBM': _objective_lgb,
    'CatBoost': _objective_cat,
    'KNN': _objective_knn,
    'SVR': _objective_svr,
    'ElasticNet': _objective_elasticnet,
    'Gaussian Process': _objective_gp,
    'ANN': _objective_ann,
}

# -------------------------------------------------------------------
# Default parameters for manual training (fallback)
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

# -------------------------------------------------------------------
# Main tuning function
# -------------------------------------------------------------------
def tune_model(model_name, X_train, y_train, n_trials=50, progress_callback=None):
    """
    Tune model hyperparameters using Optuna.
    
    Returns: (trained_model, history, best_params_dict)
    """
    # Linear Regression doesn't need tuning
    if model_name == 'Linear Regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, None, {}
    
    # Gradient Boosting not in Optuna objectives; use manual with default
    if model_name == 'Gradient Boosting':
        params = DEFAULT_PARAMS['Gradient Boosting']
        model = GradientBoostingRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        return model, None, params
    
    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Get objective function
    objective_fn = OBJECTIVE_FUNCTIONS.get(model_name)
    if objective_fn is None:
        raise ValueError(f"Model '{model_name}' not supported for tuning.")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Optimize
    objective = lambda trial: objective_fn(trial, X_tr, y_tr, X_val, y_val)
    
    callbacks = []
    if progress_callback:
        callbacks.append(progress_callback)
    
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
    
    best_params = study.best_params
    
    # Train final model with best parameters
    if model_name == 'Random Forest':
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        history = None
    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor(**best_params, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        history = None
    elif model_name == 'LightGBM':
        model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        history = None
    elif model_name == 'CatBoost':
        model = CatBoostRegressor(**best_params, random_state=42, verbose=False, allow_writing_files=False)
        model.fit(X_train, y_train)
        history = None
    elif model_name == 'KNN':
        model = KNeighborsRegressor(**best_params, n_jobs=-1)
        model.fit(X_train, y_train)
        history = None
    elif model_name == 'SVR':
        model = SVR(**best_params)
        model.fit(X_train, y_train)
        history = None
    elif model_name == 'ElasticNet':
        model = ElasticNet(**best_params, random_state=42, max_iter=5000)
        model.fit(X_train, y_train)
        history = None
    elif model_name == 'Gaussian Process':
        # Extract kernel from best params
        kernel_type = best_params.pop('kernel_type', 'Matern')
        if kernel_type == 'RBF':
            length_scale = best_params.pop('rbf_length_scale', 1.0)
            kernel = RBF(length_scale=length_scale)
        elif kernel_type == 'Matern':
            length_scale = best_params.pop('matern_length_scale', 1.0)
            nu = best_params.pop('matern_nu', 2.5)
            kernel = Matern(length_scale=length_scale, nu=nu)
        else:
            length_scale = best_params.pop('rq_length_scale', 1.0)
            alpha_param = best_params.pop('rq_alpha', 1.0)
            kernel = RationalQuadratic(length_scale=length_scale, alpha=alpha_param)
        
        alpha_val = best_params.pop('alpha', 1e-5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha_val,
            random_state=42,
            n_restarts_optimizer=5,
            normalize_y=True
        )
        model.fit(X_train, y_train)
        history = None
    elif model_name == 'ANN':
        if TF_AVAILABLE:
            model, history = _train_ann(X_train, y_train, best_params)
        else:
            raise ImportError("TensorFlow not available")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, history, best_params


# -------------------------------------------------------------------
# Manual training function (without Optuna)
# -------------------------------------------------------------------
def train_manual(model_name, X_train, y_train, params):
    """
    Train model with manually specified parameters.
    
    Returns: (model, history)
    """
    if model_name == 'Linear Regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, None
    
    elif model_name == 'ElasticNet':
        model = ElasticNet(**params, random_state=42, max_iter=5000)
        model.fit(X_train, y_train)
        return model, None
    
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model, None
    
    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        return model, None
    
    elif model_name == 'LightGBM':
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        return model, None
    
    elif model_name == 'CatBoost':
        model = CatBoostRegressor(**params, random_state=42, verbose=False, allow_writing_files=False)
        model.fit(X_train, y_train)
        return model, None
    
    elif model_name == 'KNN':
        model = KNeighborsRegressor(**params, n_jobs=-1)
        model.fit(X_train, y_train)
        return model, None
    
    elif model_name == 'SVR':
        model = SVR(**params)
        model.fit(X_train, y_train)
        return model, None
    
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        return model, None
    
    elif model_name == 'Gaussian Process':
        # Handle kernel parameter
        kernel_map = {
            'RBF': RBF(),
            'Matern': Matern(nu=2.5),
            'RationalQuadratic': RationalQuadratic()
        }
        kernel_name = params.get('kernel', 'Matern')
        kernel = kernel_map.get(kernel_name, Matern(nu=2.5))
        
        gp_params = {
            'kernel': kernel,
            'alpha': params.get('alpha', 1e-5),
            'random_state': 42,
            'n_restarts_optimizer': params.get('n_restarts_optimizer', 5),
            'normalize_y': params.get('normalize_y', True)
        }
        model = GaussianProcessRegressor(**gp_params)
        model.fit(X_train, y_train)
        return model, None
    
    elif model_name == 'ANN':
        if TF_AVAILABLE:
            model, history = _train_ann(X_train, y_train, params)
            return model, history
        else:
            raise ImportError("TensorFlow not available")
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


# -------------------------------------------------------------------
# Evaluation function
# -------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model on test data.
    
    Returns: dict with MAE, RMSE, R2, y_pred
    """
    y_pred = model.predict(X_test)
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'y_pred': y_pred
    }
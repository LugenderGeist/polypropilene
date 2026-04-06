import optuna
import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


def save_best_params_to_json(best_params, model_name, save_folder):
    """Сохраняет лучшие параметры в JSON файл"""
    os.makedirs(save_folder, exist_ok=True)
    json_path = os.path.join(save_folder, f'{model_name.lower()}_best_params.json')

    # Преобразуем numpy типы в Python типы для JSON
    params_serializable = {}
    for key, value in best_params.items():
        if isinstance(value, np.integer):
            params_serializable[key] = int(value)
        elif isinstance(value, np.floating):
            params_serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            params_serializable[key] = value.tolist()
        else:
            params_serializable[key] = value

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params_serializable, f, indent=4, ensure_ascii=False)

    print(f"📁 Параметры сохранены: {json_path}")
    return json_path


def load_best_params_from_json(model_name, save_folder):
    """Загружает лучшие параметры из JSON файла"""
    json_path = os.path.join(save_folder, f'{model_name.lower()}_best_params.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def objective_rf_cv(trial, X, y, cv_folds=5):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1
    }

    # Важно: oob_score не используем при кросс-валидации
    # oob_score работает только с bootstrap=True, но для CV он не нужен
    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
    return scores.mean()


def objective_xgb(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_val)
    return r2_score(y_val, y_pred)


def objective_mlp(trial, X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes',
                                                        [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.01, log=True),
        'batch_size': trial.suggest_categorical('batch_size', ['auto', 32, 64, 128]),
        'max_iter': 500,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'random_state': 42,
        'verbose': False
    }
    model = MLPRegressor(**params)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    return r2_score(y_val, y_pred)


def optimize_random_forest(X, y, n_trials=50, cv_folds=5, save_folder=None):
    print("\n" + "=" * 60)
    print("🌲 ОПТИМИЗАЦИЯ RANDOM FOREST")
    print("=" * 60)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective_rf_cv(trial, X, y, cv_folds), n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    print(f"\n✅ Лучший R² (CV): {best_score:.4f}")
    for param, value in best_params.items():
        print(f"   {param}: {value}")

    if save_folder:
        save_best_params_to_json(best_params, "RandomForest", save_folder)

    return best_params, study


def optimize_xgboost(X, y, n_trials=50, save_folder=None):
    print("\n" + "=" * 60)
    print("🚀 ОПТИМИЗАЦИЯ XGBOOST")
    print("=" * 60)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective_xgb(trial, X_train, y_train, X_val, y_val),
                   n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    print(f"\n✅ Лучший R² (val): {best_score:.4f}")
    for param, value in best_params.items():
        print(f"   {param}: {value}")

    if save_folder:
        save_best_params_to_json(best_params, "XGBoost", save_folder)

    return best_params, study


def optimize_mlp(X, y, n_trials=50, save_folder=None):
    print("\n" + "=" * 60)
    print("🧠 ОПТИМИЗАЦИЯ НЕЙРОСЕТИ (MLP)")
    print("=" * 60)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective_mlp(trial, X_train, y_train, X_val, y_val),
                   n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    print(f"\n✅ Лучший R² (val): {best_score:.4f}")
    for param, value in best_params.items():
        print(f"   {param}: {value}")

    if save_folder:
        save_best_params_to_json(best_params, "MLP", save_folder)

    return best_params, study


def plot_optimization_history(study, model_name, save_folder=None):
    import matplotlib.pyplot as plt

    trials = [t for t in study.trials if t.value is not None]
    values = [t.value for t in trials]
    numbers = [t.number for t in trials]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(numbers, values, 'b-', alpha=0.7, linewidth=1)
    axes[0].scatter(numbers, values, c='steelblue', s=30, alpha=0.5)
    axes[0].set_xlabel('Номер испытания')
    axes[0].set_ylabel('R²')
    axes[0].set_title(f'{model_name} - Прогресс оптимизации')
    axes[0].grid(True, alpha=0.3)

    best_values = np.maximum.accumulate(values)
    axes[1].plot(numbers, best_values, 'g-', linewidth=2)
    axes[1].set_xlabel('Номер испытания')
    axes[1].set_ylabel('Лучший R²')
    axes[1].set_title(f'{model_name} - Лучшее значение')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Optuna - Оптимизация {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, f'{model_name.lower()}_optimization_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 График сохранен: {save_path}")

    plt.show()
    plt.close(fig)
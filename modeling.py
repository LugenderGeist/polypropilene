import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
from config import (TEST_SIZE, RANDOM_STATE, RF_PARAMS, XGB_PARAMS, MLP_PARAMS, TOP_FEATURES_TO_SHOW)

warnings.filterwarnings('ignore')
plt.ioff()


def _prepare_data(df, input_columns, output_columns):
    """Подготовка данных для обучения"""
    target = output_columns[0]
    X = df[input_columns].copy()
    y = df[target].copy()

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    return X, y, target


def _print_metrics(model_name, r2_train, r2_test):
    """Вывод метрик в терминал"""
    print(f"\n{'=' * 60}")
    print(f"{model_name}")
    print(f"{'=' * 60}")
    print(f"  R² (train): {r2_train:.4f}")
    print(f"  R² (test):  {r2_test:.4f}")


def _print_top_features(feature_importance, input_columns, top_n):
    """Вывод топ-N важных признаков"""
    print(f"\n  Топ-{top_n} важных признаков:")
    for i, (idx, row) in enumerate(feature_importance.head(top_n).iterrows()):
        print(f"    {i + 1}. {row['feature']}: {row['importance']:.4f} ({row['importance'] * 100:.2f}%)")


def _save_plots(results, feature_importance, save_folder, model_name):
    """Сохранение графиков в файлы"""
    os.makedirs(save_folder, exist_ok=True)

    # График R²
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Обучающая', 'Тестовая']
    r2_values = [results['r2_train'], results['r2_test']]
    bars = ax.bar(labels, r2_values, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02, f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('R²')
    ax.set_title(f'{model_name} - Сравнение R²')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'{model_name.lower()}_r2.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # График важности признаков
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_importance) * 0.3)))
    top_n = min(15, len(feature_importance))
    top_features = feature_importance.head(top_n)
    imp_col = 'importance'
    colors = plt.cm.RdYlGn_r(top_features[imp_col].values / (top_features[imp_col].max() + 1e-10))
    ax.barh(range(len(top_features)), top_features[imp_col].values, color=colors, alpha=0.8, edgecolor='black')
    for i, (_, row) in enumerate(top_features.iterrows()):
        ax.text(row[imp_col] + 0.01, i, f"{row[imp_col]:.4f} ({row[imp_col] * 100:.2f}%)", va='center', fontsize=9)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=9)
    ax.set_xlabel('Важность')
    ax.set_title(f'{model_name} - Важность признаков')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'{model_name.lower()}_importance.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # График предсказаний
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y, y_pred, title, color in zip(axes,
                                           [results['y_train'], results['y_test']],
                                           [results['y_train_pred'], results['y_test_pred']],
                                           ['Обучающая выборка', 'Тестовая выборка'],
                                           ['steelblue', 'coral']):
        ax.scatter(y, y_pred, alpha=0.5, s=20, c=color, edgecolors='k', linewidth=0.5)
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5)
        ax.set_xlabel('Реальные')
        ax.set_ylabel('Предсказанные')
        ax.set_title(f'{title} (R² = {results["r2_train" if "Обучающая" in title else "r2_test"]:.4f})')
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'{model_name} - Предсказания')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'{model_name.lower()}_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _save_report(results, feature_importance, save_folder, model_name):
    """Сохранение текстового отчета"""
    os.makedirs(save_folder, exist_ok=True)
    report_path = os.path.join(save_folder, f'{model_name.lower()}_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"ОТЧЕТ ПО МОДЕЛИ {model_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Целевая переменная: {results['target']}\n\n")
        f.write("ПАРАМЕТРЫ МОДЕЛИ\n")
        f.write("-" * 40 + "\n")
        for param, value in results['model_params'].items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nРазмер выборок: обучающая={results['n_train']}, тестовая={results['n_test']}\n\n")
        f.write("МЕТРИКИ КАЧЕСТВА\n")
        f.write("-" * 40 + "\n")
        f.write(f"Обучающая выборка:\n")
        f.write(f"  R² = {results['r2_train']:.6f}\n")
        f.write(f"  RMSE = {results['rmse_train']:.6f}\n")
        f.write(f"  MAE = {results['mae_train']:.6f}\n\n")
        f.write(f"Тестовая выборка:\n")
        f.write(f"  R² = {results['r2_test']:.6f}\n")
        f.write(f"  RMSE = {results['rmse_test']:.6f}\n")
        f.write(f"  MAE = {results['mae_test']:.6f}\n\n")
        f.write("ВАЖНОСТЬ ПРИЗНАКОВ\n")
        f.write("-" * 40 + "\n")
        for i, row in feature_importance.iterrows():
            f.write(f"{i + 1:2d}. {row['feature']}: {row['importance']:.6f} ({row['importance'] * 100:.2f}%)\n")
    return report_path


def build_random_forest_model(df, input_columns, output_columns, save_folder=None):
    """Random Forest"""
    X, y, target = _prepare_data(df, input_columns, output_columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    _print_metrics("🌲 RANDOM FOREST", r2_train, r2_test)

    feature_importance = pd.DataFrame({
        'feature': input_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    _print_top_features(feature_importance, input_columns, TOP_FEATURES_TO_SHOW)

    results = {
        'model_type': 'random_forest',
        'r2_train': r2_train, 'r2_test': r2_test,
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae_train': mean_absolute_error(y_train, y_train_pred),
        'mae_test': mean_absolute_error(y_test, y_test_pred),
        'feature_importance': feature_importance,
        'target': target,
        'y_test': y_test, 'y_test_pred': y_test_pred,
        'y_train': y_train, 'y_train_pred': y_train_pred,
        'model_params': RF_PARAMS,
        'n_train': len(X_train), 'n_test': len(X_test)
    }

    if save_folder:
        _save_plots(results, feature_importance, save_folder, "RandomForest")
        _save_report(results, feature_importance, save_folder, "RandomForest")

    return results, model, feature_importance


def build_xgboost_model(df, input_columns, output_columns, save_folder=None):
    """XGBoost"""
    X, y, target = _prepare_data(df, input_columns, output_columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    _print_metrics("🚀 XGBOOST", r2_train, r2_test)

    feature_importance = pd.DataFrame({
        'feature': input_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    _print_top_features(feature_importance, input_columns, TOP_FEATURES_TO_SHOW)

    results = {
        'model_type': 'xgboost',
        'r2_train': r2_train, 'r2_test': r2_test,
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae_train': mean_absolute_error(y_train, y_train_pred),
        'mae_test': mean_absolute_error(y_test, y_test_pred),
        'feature_importance': feature_importance,
        'target': target,
        'y_test': y_test, 'y_test_pred': y_test_pred,
        'y_train': y_train, 'y_train_pred': y_train_pred,
        'model_params': XGB_PARAMS,
        'n_train': len(X_train), 'n_test': len(X_test)
    }

    if save_folder:
        _save_plots(results, feature_importance, save_folder, "XGBoost")
        _save_report(results, feature_importance, save_folder, "XGBoost")

    return results, model, feature_importance


def build_mlp_model(df, input_columns, output_columns, save_folder=None):
    """Нейросеть MLP"""
    X, y, target = _prepare_data(df, input_columns, output_columns)

    # Нормализация для нейросети
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = MLPRegressor(**MLP_PARAMS)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    _print_metrics("🧠 НЕЙРОСЕТЬ (MLP)", r2_train, r2_test)

    # Для нейросети - используем средние абсолютные веса первого слоя
    feature_importance = pd.DataFrame({
        'feature': input_columns,
        'importance': np.abs(model.coefs_[0]).mean(axis=1)
    }).sort_values('importance', ascending=False)

    _print_top_features(feature_importance, input_columns, TOP_FEATURES_TO_SHOW)

    results = {
        'model_type': 'mlp',
        'r2_train': r2_train, 'r2_test': r2_test,
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae_train': mean_absolute_error(y_train, y_train_pred),
        'mae_test': mean_absolute_error(y_test, y_test_pred),
        'feature_importance': feature_importance,
        'target': target,
        'y_test': y_test, 'y_test_pred': y_test_pred,
        'y_train': y_train, 'y_train_pred': y_train_pred,
        'model_params': MLP_PARAMS,
        'n_train': len(X_train), 'n_test': len(X_test),
        'n_iter': model.n_iter_
    }

    if save_folder:
        _save_plots(results, feature_importance, save_folder, "MLP")
        _save_report(results, feature_importance, save_folder, "MLP")

    return results, model, feature_importance


def compare_models(results_dict, save_folder=None):
    """Сравнение моделей и выбор лучшей"""
    print("\n" + "=" * 60)
    print("📊 СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    print(f"{'Модель':<20} {'R² (test)':<12}")
    print("-" * 35)

    best_model = None
    best_r2 = -np.inf

    for name, res in results_dict.items():
        r2 = res['r2_test']
        print(f"{name:<20} {r2:<12.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_model = name

    print("-" * 35)
    print(f"\n🎉 Лучшая модель: {best_model} (R² = {best_r2:.4f})")

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        comp_path = os.path.join(save_folder, 'models_comparison.txt')
        with open(comp_path, 'w', encoding='utf-8') as f:
            f.write("СРАВНЕНИЕ МОДЕЛЕЙ\n")
            f.write("=" * 60 + "\n\n")
            for name, res in results_dict.items():
                f.write(f"{name}:\n")
                f.write(f"  R²: {res['r2_test']:.6f}\n")
                f.write(f"  RMSE: {res['rmse_test']:.6f}\n")
                f.write(f"  MAE: {res['mae_test']:.6f}\n\n")
            f.write(f"Лучшая модель: {best_model}\n")

    return best_model, best_r2
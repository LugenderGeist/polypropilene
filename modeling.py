import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Попытка импорта XGBoost
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost не установлен. Для использования XGBoost выполните: pip install xgboost")


def build_random_forest_model(df, input_columns, output_columns, save_folder=None,
                              test_size=0.2, random_state=42):
    """
    Построение модели Random Forest для прогнозирования
    """
    if not output_columns:
        print("Нет выходных столбцов для моделирования")
        return None, None, None

    target = output_columns[0]

    print("\n" + "=" * 80)
    print(f"ПОСТРОЕНИЕ МОДЕЛИ RANDOM FOREST ДЛЯ {target}")
    print("=" * 80)

    # Подготовка данных
    X = df[input_columns].copy()
    y = df[target].copy()

    # Проверка на достаточность данных
    if len(X) < 10:
        print("Ошибка: недостаточно данных для построения модели")
        return None, None, None

    # Заполняем пропуски средними значениями
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Проверка на дисперсию
    if y.std() == 0:
        print("Ошибка: целевая переменная не имеет вариации (все значения одинаковы)")
        return None, None, None

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\nРазмер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")

    # Параметры для Random Forest с регуляризацией
    rf_params = {
        'n_estimators': 200,
        'max_depth': 8,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'bootstrap': True,
        'oob_score': True,
        'random_state': random_state,
        'n_jobs': -1
    }

    print("\nПараметры Random Forest:")
    for param, value in rf_params.items():
        print(f"  {param}: {value}")

    # Обучаем Random Forest
    print("\nОбучение Random Forest...")
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)

    # Out-of-Bag оценка
    oob_score = rf.oob_score_
    print(f"\nRandom Forest Out-of-Bag R²: {oob_score:.4f}")

    # Предсказания Random Forest
    y_train_pred_rf = rf.predict(X_train)
    y_test_pred_rf = rf.predict(X_test)

    # Метрики Random Forest
    r2_train_rf = r2_score(y_train, y_train_pred_rf)
    r2_test_rf = r2_score(y_test, y_test_pred_rf)
    rmse_train_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
    rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
    mae_train_rf = mean_absolute_error(y_train, y_train_pred_rf)
    mae_test_rf = mean_absolute_error(y_test, y_test_pred_rf)

    print(f"\nRandom Forest - Обучающая выборка:")
    print(f"  R² = {r2_train_rf:.4f}")
    print(f"  RMSE = {rmse_train_rf:.4f}")
    print(f"  MAE = {mae_train_rf:.4f}")

    print(f"\nRandom Forest - Тестовая выборка:")
    print(f"  R² = {r2_test_rf:.4f}")
    print(f"  RMSE = {rmse_test_rf:.4f}")
    print(f"  MAE = {mae_test_rf:.4f}")

    # Важность признаков Random Forest
    feature_importance_rf = pd.DataFrame({
        'feature': input_columns,
        'importance_rf': rf.feature_importances_
    }).sort_values('importance_rf', ascending=False)

    print("\n" + "=" * 80)
    print("ВАЖНОСТЬ ПРИЗНАКОВ (Random Forest)")
    print("=" * 80)
    for i, row in feature_importance_rf.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance_rf']:.4f} ({row['importance_rf'] * 100:.2f}%)")

    results = {
        'model_type': 'random_forest',
        'r2_train': r2_train_rf,
        'r2_test': r2_test_rf,
        'rmse_train': rmse_train_rf,
        'rmse_test': rmse_test_rf,
        'mae_train': mae_train_rf,
        'mae_test': mae_test_rf,
        'oob_score': oob_score,
        'feature_importance': feature_importance_rf,
        'target': target,
        'y_test': y_test,
        'y_test_pred': y_test_pred_rf,
        'y_train': y_train,
        'y_train_pred': y_train_pred_rf,
        'model_params': rf_params
    }

    # Визуализация
    plot_r2_comparison(results, save_folder, model_name="Random Forest")
    plot_metrics_comparison(results, save_folder, model_name="Random Forest")
    plot_feature_importance(feature_importance_rf, save_folder, model_name="Random Forest")
    plot_predictions_vs_actual(results, save_folder, model_name="Random Forest")

    # Сохраняем результаты
    if save_folder:
        save_model_results(results, rf, feature_importance_rf, save_folder, model_name="Random Forest")

    return results, rf, feature_importance_rf


def build_xgboost_model(df, input_columns, output_columns, save_folder=None,
                        test_size=0.2, random_state=42):
    """
    Построение модели XGBoost для прогнозирования с визуализацией
    """
    if not XGB_AVAILABLE:
        print("XGBoost не установлен. Пропуск...")
        return None, None, None

    if not output_columns:
        print("Нет выходных столбцов для моделирования")
        return None, None, None

    target = output_columns[0]

    print("\n" + "=" * 80)
    print(f"ПОСТРОЕНИЕ МОДЕЛИ XGBOOST ДЛЯ {target}")
    print("=" * 80)

    # Подготовка данных
    X = df[input_columns].copy()
    y = df[target].copy()

    # Проверка на достаточность данных
    if len(X) < 10:
        print("Ошибка: недостаточно данных для построения модели")
        return None, None, None

    # Заполняем пропуски средними значениями
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Проверка на дисперсию
    if y.std() == 0:
        print("Ошибка: целевая переменная не имеет вариации (все значения одинаковы)")
        return None, None, None

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\nРазмер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")

    # Параметры для XGBoost с регуляризацией
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,  # L1 регуляризация
        'reg_lambda': 1.0,  # L2 регуляризация
        'random_state': random_state,
        'n_jobs': -1,
        'verbosity': 0
    }

    print("\nПараметры XGBoost:")
    for param, value in xgb_params.items():
        print(f"  {param}: {value}")

    # Обучаем XGBoost
    print("\nОбучение XGBoost...")
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)

    # Предсказания XGBoost
    y_train_pred_xgb = xgb_model.predict(X_train)
    y_test_pred_xgb = xgb_model.predict(X_test)

    # Метрики XGBoost
    r2_train_xgb = r2_score(y_train, y_train_pred_xgb)
    r2_test_xgb = r2_score(y_test, y_test_pred_xgb)
    rmse_train_xgb = np.sqrt(mean_squared_error(y_train, y_train_pred_xgb))
    rmse_test_xgb = np.sqrt(mean_squared_error(y_test, y_test_pred_xgb))
    mae_train_xgb = mean_absolute_error(y_train, y_train_pred_xgb)
    mae_test_xgb = mean_absolute_error(y_test, y_test_pred_xgb)

    print(f"\nXGBoost - Обучающая выборка:")
    print(f"  R² = {r2_train_xgb:.4f}")
    print(f"  RMSE = {rmse_train_xgb:.4f}")
    print(f"  MAE = {mae_train_xgb:.4f}")

    print(f"\nXGBoost - Тестовая выборка:")
    print(f"  R² = {r2_test_xgb:.4f}")
    print(f"  RMSE = {rmse_test_xgb:.4f}")
    print(f"  MAE = {mae_test_xgb:.4f}")

    # Важность признаков XGBoost
    feature_importance_xgb = pd.DataFrame({
        'feature': input_columns,
        'importance_xgb': xgb_model.feature_importances_
    }).sort_values('importance_xgb', ascending=False)

    print("\n" + "=" * 80)
    print("ВАЖНОСТЬ ПРИЗНАКОВ (XGBoost)")
    print("=" * 80)
    for i, row in feature_importance_xgb.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance_xgb']:.4f} ({row['importance_xgb'] * 100:.2f}%)")

    results = {
        'model_type': 'xgboost',
        'r2_train': r2_train_xgb,
        'r2_test': r2_test_xgb,
        'rmse_train': rmse_train_xgb,
        'rmse_test': rmse_test_xgb,
        'mae_train': mae_train_xgb,
        'mae_test': mae_test_xgb,
        'feature_importance': feature_importance_xgb,
        'target': target,
        'y_test': y_test,
        'y_test_pred': y_test_pred_xgb,
        'y_train': y_train,
        'y_train_pred': y_train_pred_xgb,
        'model_params': xgb_params
    }

    # Визуализация для XGBoost
    plot_r2_comparison(results, save_folder, model_name="XGBoost")
    plot_metrics_comparison(results, save_folder, model_name="XGBoost")
    plot_feature_importance(feature_importance_xgb, save_folder, model_name="XGBoost")
    plot_predictions_vs_actual(results, save_folder, model_name="XGBoost")

    # Сохраняем результаты
    if save_folder:
        save_model_results(results, xgb_model, feature_importance_xgb, save_folder, model_name="XGBoost")

    return results, xgb_model, feature_importance_xgb


def plot_r2_comparison(results, save_folder=None, model_name="Model"):
    """
    График сравнения R² на обучающей и тестовой выборках
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Данные для графика
    labels = ['Обучающая выборка', 'Тестовая выборка']
    r2_values = [results['r2_train'], results['r2_test']]
    colors = ['steelblue', 'coral']

    # Столбцы
    bars = ax.bar(labels, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    # Добавляем значения на столбцы
    for bar, value in zip(bars, r2_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Настройки
    ax.set_ylabel('Коэффициент детерминации (R²)', fontsize=12)
    ax.set_title(f'{model_name} - Сравнение R² на выборках', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Хороший результат (0.7)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Средний результат (0.5)')
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Слабый результат (0.3)')
    ax.legend(loc='lower right')

    # Добавляем пояснение о переобучении
    overfitting_gap = results['r2_train'] - results['r2_test']
    if overfitting_gap > 0.1:
        ax.text(0.5, -0.15,
                f'⚠️ Внимание: разница R² = {overfitting_gap:.4f} (возможно переобучение)',
                transform=ax.transAxes, ha='center', fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, f'{model_name.lower().replace(" ", "_")}_r2_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График R² для {model_name} сохранен: {save_path}")

    plt.show()
    plt.close(fig)


def plot_metrics_comparison(results, save_folder=None, model_name="Model"):
    """
    График сравнения RMSE и MAE на обучающей и тестовой выборках
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. RMSE
    ax1 = axes[0]
    labels = ['Обучающая', 'Тестовая']
    rmse_values = [results['rmse_train'], results['rmse_test']]
    colors = ['steelblue', 'coral']

    bars1 = ax1.bar(labels, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    for bar, value in zip(bars1, rmse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.02,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=11)

    ax1.set_ylabel('Значение', fontsize=12)
    ax1.set_title(f'{model_name} - Среднеквадратичная ошибка (RMSE)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. MAE
    ax2 = axes[1]
    mae_values = [results['mae_train'], results['mae_test']]

    bars2 = ax2.bar(labels, mae_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    for bar, value in zip(bars2, mae_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.02,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=11)

    ax2.set_ylabel('Значение', fontsize=12)
    ax2.set_title(f'{model_name} - Средняя абсолютная ошибка (MAE)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'{model_name} - Сравнение ошибок на обучающей и тестовой выборках', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, f'{model_name.lower().replace(" ", "_")}_metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График метрик для {model_name} сохранен: {save_path}")

    plt.show()
    plt.close(fig)


def plot_feature_importance(feature_importance, save_folder=None, model_name="Model", top_n=15):
    """
    График важности признаков
    """
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_importance) * 0.3)))

    # Берем топ-N
    n_features = min(top_n, len(feature_importance))
    top_features = feature_importance.head(n_features)

    # Определяем колонку с важностью в зависимости от модели
    if 'importance_rf' in top_features.columns:
        importance_col = 'importance_rf'
    elif 'importance_xgb' in top_features.columns:
        importance_col = 'importance_xgb'
    else:
        importance_col = top_features.columns[1]

    # Цвета: от зеленого к красному
    colors = plt.cm.RdYlGn_r(top_features[importance_col].values / (top_features[importance_col].max() + 1e-10))

    bars = ax.barh(range(len(top_features)), top_features[importance_col].values,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Добавляем значения на график
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row[importance_col] + 0.01, i,
                f"{row[importance_col]:.4f} ({row[importance_col] * 100:.2f}%)",
                va='center', fontsize=9)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=10)
    ax.set_xlabel('Важность признака (относительная)', fontsize=12)
    ax.set_title(f'{model_name} - Важность признаков\n(сумма всех важностей = 1.00)',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Добавляем пояснение
    ax.text(0.5, -0.1,
            "Важность показывает, насколько признак влияет на прогноз.\n"
            "Значение 0.3 означает, что признак объясняет 30% предсказательной силы модели.",
            transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График важности признаков для {model_name} сохранен: {save_path}")

    plt.show()
    plt.close(fig)


def plot_predictions_vs_actual(results, save_folder=None, model_name="Model"):
    """
    График предсказанных vs реальных значений
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Обучающая выборка
    ax1 = axes[0]
    ax1.scatter(results['y_train'], results['y_train_pred'], alpha=0.5, s=30, c='steelblue', edgecolors='k',
                linewidth=0.5)

    # Линия идеального предсказания
    min_val = min(results['y_train'].min(), results['y_train_pred'].min())
    max_val = max(results['y_train'].max(), results['y_train_pred'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное предсказание')

    ax1.set_xlabel('Реальные значения', fontsize=12)
    ax1.set_ylabel('Предсказанные значения', fontsize=12)
    ax1.set_title(f'{model_name} - Обучающая выборка (R² = {results["r2_train"]:.4f})', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Тестовая выборка
    ax2 = axes[1]
    ax2.scatter(results['y_test'], results['y_test_pred'], alpha=0.5, s=30, c='coral', edgecolors='k', linewidth=0.5)
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное предсказание')

    ax2.set_xlabel('Реальные значения', fontsize=12)
    ax2.set_ylabel('Предсказанные значения', fontsize=12)
    ax2.set_title(f'{model_name} - Тестовая выборка (R² = {results["r2_test"]:.4f})', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{model_name} - Сравнение реальных и предсказанных значений', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, f'{model_name.lower().replace(" ", "_")}_predictions_vs_actual.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График предсказаний для {model_name} сохранен: {save_path}")

    plt.show()
    plt.close(fig)


def save_model_results(results, model, feature_importance, save_folder, model_name="Model"):
    """
    Сохранение результатов модели
    """
    # Сохраняем текстовый отчет
    report_path = os.path.join(save_folder, f'{model_name.lower().replace(" ", "_")}_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"ОТЧЕТ ПО МОДЕЛИ {model_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Целевая переменная: {results['target']}\n\n")

        f.write("ПАРАМЕТРЫ МОДЕЛИ (РЕГУЛЯРИЗАЦИЯ)\n")
        f.write("-" * 40 + "\n")
        for param, value in results['model_params'].items():
            f.write(f"{param}: {value}\n")

        f.write("\nМЕТРИКИ КАЧЕСТВА\n")
        f.write("-" * 40 + "\n")
        f.write(f"Обучающая выборка:\n")
        f.write(f"  R² = {results['r2_train']:.4f}\n")
        f.write(f"  RMSE = {results['rmse_train']:.4f}\n")
        f.write(f"  MAE = {results['mae_train']:.4f}\n\n")

        f.write(f"Тестовая выборка:\n")
        f.write(f"  R² = {results['r2_test']:.4f}\n")
        f.write(f"  RMSE = {results['rmse_test']:.4f}\n")
        f.write(f"  MAE = {results['mae_test']:.4f}\n\n")

        if 'oob_score' in results and results['oob_score'] is not None:
            f.write(f"Out-of-Bag R²: {results['oob_score']:.4f}\n\n")

        f.write("ОЦЕНКА ПЕРЕОБУЧЕНИЯ\n")
        f.write("-" * 40 + "\n")
        overfitting_gap = results['r2_train'] - results['r2_test']
        f.write(f"Разница R² (train - test): {overfitting_gap:.4f}\n")
        if overfitting_gap > 0.1:
            f.write("ВНИМАНИЕ: Обнаружено потенциальное переобучение!\n\n")
        elif overfitting_gap > 0.05:
            f.write("Небольшое переобучение, модель приемлема.\n\n")
        else:
            f.write("Переобучение не обнаружено, модель хорошо обобщает.\n\n")

        f.write("ВАЖНОСТЬ ПРИЗНАКОВ\n")
        f.write("-" * 40 + "\n")
        f.write("(Сумма всех важностей = 1.00)\n\n")

        importance_col = 'importance_rf' if 'importance_rf' in feature_importance.columns else 'importance_xgb'
        for i, row in feature_importance.iterrows():
            f.write(f"{i + 1:2d}. {row['feature']}: {row[importance_col]:.4f} ({row[importance_col] * 100:.2f}%)\n")

    print(f"\nОтчет для {model_name} сохранен в: {report_path}")

    # Сохраняем важность признаков в CSV
    importance_path = os.path.join(save_folder, f'{model_name.lower().replace(" ", "_")}_feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')
    print(f"Важность признаков для {model_name} сохранена в: {importance_path}")


def build_ensemble_model(df, input_columns, output_columns, save_folder=None,
                         test_size=0.2, random_state=42):
    """
    Построение ансамбля моделей (Random Forest + XGBoost)
    """
    if not output_columns:
        print("Нет выходных столбцов для моделирования")
        return None

    target = output_columns[0]

    print("\n" + "=" * 80)
    print(f"ПОСТРОЕНИЕ АНСАМБЛЯ МОДЕЛЕЙ ДЛЯ {target}")
    print("=" * 80)

    # Подготовка данных
    X = df[input_columns].copy()
    y = df[target].copy()

    # Заполняем пропуски
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\nРазмер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")

    # Обучаем Random Forest
    print("\nОбучение Random Forest...")
    rf_params = {
        'n_estimators': 200,
        'max_depth': 8,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': random_state,
        'n_jobs': -1
    }
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)

    # Обучаем XGBoost
    if XGB_AVAILABLE:
        print("\nОбучение XGBoost...")
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'random_state': random_state,
            'verbosity': 0
        }
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        r2_xgb = r2_score(y_test, y_pred_xgb)

        # Ансамбль (среднее арифметическое)
        y_pred_ensemble = (y_pred_rf + y_pred_xgb) / 2
        r2_ensemble = r2_score(y_test, y_pred_ensemble)
        rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
        mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)

        print(f"\nРезультаты:")
        print(f"  Random Forest R²: {r2_rf:.4f}")
        print(f"  XGBoost R²: {r2_xgb:.4f}")
        print(f"  Ансамбль R²: {r2_ensemble:.4f}")
        print(f"  Ансамбль RMSE: {rmse_ensemble:.4f}")
        print(f"  Ансамбль MAE: {mae_ensemble:.4f}")

        # Визуализация ансамбля
        if save_folder:
            plot_ensemble_results(r2_rf, r2_xgb, r2_ensemble, save_folder)

        return {
            'r2_ensemble': r2_ensemble,
            'rmse_ensemble': rmse_ensemble,
            'mae_ensemble': mae_ensemble,
            'r2_rf': r2_rf,
            'r2_xgb': r2_xgb,
            'rf_model': rf,
            'xgb_model': xgb_model
        }
    else:
        print("XGBoost не установлен, ансамбль не построен")
        return None


def plot_ensemble_results(r2_rf, r2_xgb, r2_ensemble, save_folder=None):
    """
    Визуализация результатов ансамбля
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    models = ['Random Forest', 'XGBoost', 'Ансамбль']
    r2_values = [r2_rf, r2_xgb, r2_ensemble]
    colors = ['steelblue', 'coral', 'green']

    bars = ax.bar(models, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    for bar, value in zip(bars, r2_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Коэффициент детерминации (R²)', fontsize=12)
    ax.set_title('Сравнение моделей: Ансамбль vs отдельные модели', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Хороший результат (0.7)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Средний результат (0.5)')
    ax.legend()

    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, 'ensemble_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График ансамбля сохранен: {save_path}")

    plt.show()
    plt.close(fig)


def plot_model_comparison(rf_results, xgb_results, save_folder=None):
    """
    Сравнение метрик Random Forest и XGBoost
    """
    if rf_results is None and xgb_results is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['R²', 'RMSE', 'MAE']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        values = []
        labels = []

        if rf_results is not None:
            if metric == 'R²':
                values.append(rf_results['r2_test'])
            elif metric == 'RMSE':
                values.append(rf_results['rmse_test'])
            else:
                values.append(rf_results['mae_test'])
            labels.append('Random Forest')

        if xgb_results is not None:
            if metric == 'R²':
                values.append(xgb_results['r2_test'])
            elif metric == 'RMSE':
                values.append(xgb_results['rmse_test'])
            else:
                values.append(xgb_results['mae_test'])
            labels.append('XGBoost')

        colors = ['steelblue', 'coral'][:len(values)]
        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'Сравнение {metric}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Сравнение моделей: Random Forest vs XGBoost', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сравнения моделей сохранен: {save_path}")

    plt.show()
    plt.close(fig)


def plot_feature_importance_comparison(rf_importance, xgb_importance, save_folder=None, top_n=10):
    """
    Сравнение важности признаков Random Forest и XGBoost
    """
    if rf_importance is None and xgb_importance is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, top_n * 0.4)))

    # 1. Random Forest
    if rf_importance is not None:
        ax1 = axes[0]
        top_rf = rf_importance.head(top_n)
        colors_rf = plt.cm.RdYlGn_r(top_rf['importance_rf'].values / (top_rf['importance_rf'].max() + 1e-10))

        bars1 = ax1.barh(range(len(top_rf)), top_rf['importance_rf'].values,
                         color=colors_rf, alpha=0.8, edgecolor='black', linewidth=0.5)

        for i, (idx, row) in enumerate(top_rf.iterrows()):
            ax1.text(row['importance_rf'] + 0.01, i,
                     f"{row['importance_rf']:.4f} ({row['importance_rf'] * 100:.2f}%)",
                     va='center', fontsize=9)

        ax1.set_yticks(range(len(top_rf)))
        ax1.set_yticklabels(top_rf['feature'].values, fontsize=10)
        ax1.set_xlabel('Важность', fontsize=12)
        ax1.set_title('Random Forest', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')

    # 2. XGBoost
    if xgb_importance is not None:
        ax2 = axes[1]
        top_xgb = xgb_importance.head(top_n)
        colors_xgb = plt.cm.RdYlGn_r(top_xgb['importance_xgb'].values / (top_xgb['importance_xgb'].max() + 1e-10))

        bars2 = ax2.barh(range(len(top_xgb)), top_xgb['importance_xgb'].values,
                         color=colors_xgb, alpha=0.8, edgecolor='black', linewidth=0.5)

        for i, (idx, row) in enumerate(top_xgb.iterrows()):
            ax2.text(row['importance_xgb'] + 0.01, i,
                     f"{row['importance_xgb']:.4f} ({row['importance_xgb'] * 100:.2f}%)",
                     va='center', fontsize=9)

        ax2.set_yticks(range(len(top_xgb)))
        ax2.set_yticklabels(top_xgb['feature'].values, fontsize=10)
        ax2.set_xlabel('Важность', fontsize=12)
        ax2.set_title('XGBoost', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Сравнение важности признаков: Random Forest vs XGBoost', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, 'feature_importance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сравнения важности признаков сохранен: {save_path}")

    plt.show()
    plt.close(fig)
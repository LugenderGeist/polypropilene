import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from config import TEST_SIZE, RANDOM_STATE, RF_PARAMS, XGB_PARAMS

warnings.filterwarnings('ignore')
plt.ioff()

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def build_random_forest_model(df, input_columns, output_columns, save_folder=None):
    """Построение модели Random Forest для прогнозирования"""
    if not output_columns:
        return None, None, None

    target = output_columns[0]

    print("\n" + "=" * 80)
    print(f"RANDOM FOREST - {target}")
    print("=" * 80)

    # Подготовка данных
    X = df[input_columns].copy()
    y = df[target].copy()

    if len(X) < 10:
        print("Ошибка: недостаточно данных")
        return None, None, None

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    if y.std() == 0:
        print("Ошибка: целевая переменная не имеет вариации")
        return None, None, None

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Обучаем Random Forest
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)

    # Предсказания
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # Метрики
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # ============= ВЫВОД В ТЕРМИНАЛ ТОЛЬКО R² =============
    print(f"\nРазмер выборок: обучающая={len(X_train)}, тестовая={len(X_test)}")
    print(f"\n📊 R² на обучающей выборке: {r2_train:.4f}")
    print(f"📊 R² на тестовой выборке: {r2_test:.4f}")

    overfitting_gap = r2_train - r2_test
    if overfitting_gap > 0.1:
        print(f"\n⚠️ Внимание: разница R² = {overfitting_gap:.4f} (возможно переобучение)")
    elif overfitting_gap > 0.05:
        print(f"\n⚠️ Небольшое переобучение: разница R² = {overfitting_gap:.4f}")
    else:
        print(f"\n✅ Переобучение не обнаружено (разница R² = {overfitting_gap:.4f})")

    # Важность признаков (только топ-5 в терминал)
    feature_importance = pd.DataFrame({
        'feature': input_columns,
        'importance_rf': rf.feature_importances_
    }).sort_values('importance_rf', ascending=False)

    results = {
        'model_type': 'random_forest',
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'oob_score': rf.oob_score_,
        'feature_importance': feature_importance,
        'target': target,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_train': y_train,
        'y_train_pred': y_train_pred,
        'model_params': RF_PARAMS,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    # Сохраняем графики и отчеты в папку
    if save_folder:
        save_plots_to_files(results, feature_importance, save_folder, model_name="Random Forest")
        save_model_results(results, rf, feature_importance, save_folder, model_name="Random Forest")

    return results, rf, feature_importance


def build_xgboost_model(df, input_columns, output_columns, save_folder=None):
    """Построение модели XGBoost для прогнозирования"""
    if not XGB_AVAILABLE:
        print("XGBoost не установлен. Для установки: pip install xgboost")
        return None, None, None

    if not output_columns:
        return None, None, None

    target = output_columns[0]

    print("\n" + "=" * 80)
    print(f"XGBOOST - {target}")
    print("=" * 80)

    # Подготовка данных
    X = df[input_columns].copy()
    y = df[target].copy()

    if len(X) < 10:
        print("Ошибка: недостаточно данных")
        return None, None, None

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    if y.std() == 0:
        print("Ошибка: целевая переменная не имеет вариации")
        return None, None, None

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Обучаем XGBoost
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train, verbose=False)

    # Предсказания
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)

    # Метрики
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # ============= ВЫВОД В ТЕРМИНАЛ ТОЛЬКО R² =============
    print(f"\nРазмер выборок: обучающая={len(X_train)}, тестовая={len(X_test)}")
    print(f"\n📊 R² на обучающей выборке: {r2_train:.4f}")
    print(f"📊 R² на тестовой выборке: {r2_test:.4f}")

    overfitting_gap = r2_train - r2_test
    if overfitting_gap > 0.1:
        print(f"\n⚠️ Внимание: разница R² = {overfitting_gap:.4f} (возможно переобучение)")
    elif overfitting_gap > 0.05:
        print(f"\n⚠️ Небольшое переобучение: разница R² = {overfitting_gap:.4f}")
    else:
        print(f"\n✅ Переобучение не обнаружено (разница R² = {overfitting_gap:.4f})")

    # Важность признаков (только топ-5 в терминал)
    feature_importance = pd.DataFrame({
        'feature': input_columns,
        'importance_xgb': xgb_model.feature_importances_
    }).sort_values('importance_xgb', ascending=False)

    results = {
        'model_type': 'xgboost',
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'feature_importance': feature_importance,
        'target': target,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_train': y_train,
        'y_train_pred': y_train_pred,
        'model_params': XGB_PARAMS,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    # Сохраняем графики и отчеты в папку
    if save_folder:
        save_plots_to_files(results, feature_importance, save_folder, model_name="XGBoost")
        save_model_results(results, xgb_model, feature_importance, save_folder, model_name="XGBoost")

    return results, xgb_model, feature_importance


def save_plots_to_files(results, feature_importance, save_folder, model_name="Model"):
    """Сохранение графиков в файлы (без отображения на экране)"""

    # График R²
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    labels = ['Обучающая выборка', 'Тестовая выборка']
    r2_values = [results['r2_train'], results['r2_test']]
    colors = ['steelblue', 'coral']

    bars = ax1.bar(labels, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    for bar, value in zip(bars, r2_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Коэффициент детерминации (R²)', fontsize=12)
    ax1.set_title(f'{model_name} - Сравнение R² на выборках', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Хороший результат (0.7)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Средний результат (0.5)')
    ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Слабый результат (0.3)')
    ax1.legend(loc='lower right')

    plt.tight_layout()
    save_path1 = os.path.join(save_folder, f'{model_name.lower().replace(" ", "_")}_r2_comparison.png')
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # График метрик (RMSE и MAE)
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE
    labels = ['Обучающая', 'Тестовая']
    rmse_values = [results['rmse_train'], results['rmse_test']]
    colors = ['steelblue', 'coral']

    bars1 = axes[0].bar(labels, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    for bar, value in zip(bars1, rmse_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height + height * 0.02,
                     f'{value:.4f}', ha='center', va='bottom', fontsize=11)
    axes[0].set_ylabel('Значение', fontsize=12)
    axes[0].set_title(f'{model_name} - RMSE', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # MAE
    mae_values = [results['mae_train'], results['mae_test']]
    bars2 = axes[1].bar(labels, mae_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    for bar, value in zip(bars2, mae_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height + height * 0.02,
                     f'{value:.4f}', ha='center', va='bottom', fontsize=11)
    axes[1].set_ylabel('Значение', fontsize=12)
    axes[1].set_title(f'{model_name} - MAE', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'{model_name} - Сравнение ошибок', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path2 = os.path.join(save_folder, f'{model_name.lower().replace(" ", "_")}_metrics_comparison.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # График важности признаков
    fig3, ax3 = plt.subplots(figsize=(10, max(6, len(feature_importance) * 0.3)))

    n_features = min(15, len(feature_importance))
    top_features = feature_importance.head(n_features)

    if 'importance_rf' in top_features.columns:
        importance_col = 'importance_rf'
    else:
        importance_col = 'importance_xgb'

    colors = plt.cm.RdYlGn_r(top_features[importance_col].values / (top_features[importance_col].max() + 1e-10))
    bars = ax3.barh(range(len(top_features)), top_features[importance_col].values,
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax3.text(row[importance_col] + 0.01, i,
                 f"{row[importance_col]:.4f} ({row[importance_col] * 100:.2f}%)",
                 va='center', fontsize=9)

    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'].values, fontsize=10)
    ax3.set_xlabel('Важность признака (относительная)', fontsize=12)
    ax3.set_title(f'{model_name} - Важность признаков', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_path3 = os.path.join(save_folder, f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # График предсказаний
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

    # Обучающая выборка
    axes4[0].scatter(results['y_train'], results['y_train_pred'], alpha=0.5, s=30,
                     c='steelblue', edgecolors='k', linewidth=0.5)
    min_val = min(results['y_train'].min(), results['y_train_pred'].min())
    max_val = max(results['y_train'].max(), results['y_train_pred'].max())
    axes4[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное предсказание')
    axes4[0].set_xlabel('Реальные значения', fontsize=12)
    axes4[0].set_ylabel('Предсказанные значения', fontsize=12)
    axes4[0].set_title(f'{model_name} - Обучающая выборка (R² = {results["r2_train"]:.4f})', fontsize=12,
                       fontweight='bold')
    axes4[0].legend()
    axes4[0].grid(True, alpha=0.3)

    # Тестовая выборка
    axes4[1].scatter(results['y_test'], results['y_test_pred'], alpha=0.5, s=30,
                     c='coral', edgecolors='k', linewidth=0.5)
    axes4[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное предсказание')
    axes4[1].set_xlabel('Реальные значения', fontsize=12)
    axes4[1].set_ylabel('Предсказанные значения', fontsize=12)
    axes4[1].set_title(f'{model_name} - Тестовая выборка (R² = {results["r2_test"]:.4f})', fontsize=12,
                       fontweight='bold')
    axes4[1].legend()
    axes4[1].grid(True, alpha=0.3)

    plt.suptitle(f'{model_name} - Сравнение реальных и предсказанных значений', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path4 = os.path.join(save_folder, f'{model_name.lower().replace(" ", "_")}_predictions_vs_actual.png')
    plt.savefig(save_path4, dpi=300, bbox_inches='tight')
    plt.close(fig4)


def save_model_results(results, model, feature_importance, save_folder, model_name="Model"):
    """Сохранение результатов модели в файл (все детали)"""

    report_path = os.path.join(save_folder, f'{model_name.lower().replace(" ", "_")}_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"ОТЧЕТ ПО МОДЕЛИ {model_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Целевая переменная: {results['target']}\n\n")

        f.write("ПАРАМЕТРЫ МОДЕЛИ\n")
        f.write("-" * 40 + "\n")
        for param, value in results['model_params'].items():
            f.write(f"{param}: {value}\n")

        f.write("\nРАЗМЕР ВЫБОРОК\n")
        f.write("-" * 40 + "\n")
        f.write(f"Обучающая выборка: {results['n_train']} строк\n")
        f.write(f"Тестовая выборка: {results['n_test']} строк\n")

        f.write("\nМЕТРИКИ КАЧЕСТВА\n")
        f.write("-" * 40 + "\n")
        f.write(f"Обучающая выборка:\n")
        f.write(f"  R² = {results['r2_train']:.6f}\n")
        f.write(f"  RMSE = {results['rmse_train']:.6f}\n")
        f.write(f"  MAE = {results['mae_train']:.6f}\n\n")

        f.write(f"Тестовая выборка:\n")
        f.write(f"  R² = {results['r2_test']:.6f}\n")
        f.write(f"  RMSE = {results['rmse_test']:.6f}\n")
        f.write(f"  MAE = {results['mae_test']:.6f}\n\n")

        if 'oob_score' in results and results['oob_score'] is not None:
            f.write(f"Out-of-Bag R²: {results['oob_score']:.6f}\n\n")

        f.write("ОЦЕНКА ПЕРЕОБУЧЕНИЯ\n")
        f.write("-" * 40 + "\n")
        overfitting_gap = results['r2_train'] - results['r2_test']
        f.write(f"Разница R² (train - test): {overfitting_gap:.6f}\n")
        if overfitting_gap > 0.1:
            f.write("ВНИМАНИЕ: Обнаружено потенциальное переобучение!\n")
        elif overfitting_gap > 0.05:
            f.write("Небольшое переобучение, модель приемлема.\n")
        else:
            f.write("Переобучение не обнаружено, модель хорошо обобщает.\n")

        f.write("\nВАЖНОСТЬ ПРИЗНАКОВ\n")
        f.write("-" * 40 + "\n")
        f.write("(Сумма всех важностей = 1.00)\n\n")

        importance_col = 'importance_rf' if 'importance_rf' in feature_importance.columns else 'importance_xgb'
        for i, row in feature_importance.iterrows():
            f.write(f"{i + 1:2d}. {row['feature']}: {row[importance_col]:.6f} ({row[importance_col] * 100:.2f}%)\n")

    # Сохраняем важность признаков в CSV
    importance_path = os.path.join(save_folder, f'{model_name.lower().replace(" ", "_")}_feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')

    print(f"📁 Полный отчет сохранен в: {save_folder}")
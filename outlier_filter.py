import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.signal import savgol_filter, find_peaks

def detect_outliers_iqr(data, multiplier=1.5):
    """Метод межквартильного размаха (IQR)"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    return outlier_mask, lower_bound, upper_bound


def detect_outliers_mad(data, threshold=3.5):
    """Метод медианного абсолютного отклонения (MAD) - устойчив к выбросам"""
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad
    outlier_mask = np.abs(modified_z_scores) > threshold
    lower_bound = median - threshold * mad / 0.6745
    upper_bound = median + threshold * mad / 0.6745
    return outlier_mask, lower_bound, upper_bound


def detect_outliers_rolling(data, window_size=10, threshold=3):
    """Метод скользящего окна - для временных рядов"""
    rolling_mean = pd.Series(data).rolling(window=window_size, center=True, min_periods=1).mean().values
    rolling_std = pd.Series(data).rolling(window=window_size, center=True, min_periods=1).std().values
    rolling_std[rolling_std == 0] = 1e-10
    z_scores = np.abs((data - rolling_mean) / rolling_std)
    outlier_mask = z_scores > threshold
    return outlier_mask, None, None


def detect_outliers_derivative(data, threshold_multiplier=5):
    """Метод производной - ищет резкие скачки"""
    diff = np.diff(data, prepend=data[0])
    diff_abs = np.abs(diff)
    diff_threshold = np.nanmean(diff_abs) + threshold_multiplier * np.nanstd(diff_abs)
    outlier_mask = diff_abs > diff_threshold
    return outlier_mask, None, None


def detect_outliers_peak(data, prominence=0.5, distance=10):
    """Метод поиска пиков - для изолированных выбросов"""
    # Находим пики
    peaks, properties = find_peaks(data, prominence=prominence, distance=distance)
    valleys, _ = find_peaks(-data, prominence=prominence, distance=distance)

    outlier_mask = np.zeros(len(data), dtype=bool)
    outlier_mask[peaks] = True
    outlier_mask[valleys] = True

    return outlier_mask, None, None


def detect_outliers_savgol(data, window_length=21, polyorder=3, threshold=3):
    """Метод фильтрации Савицкого-Голая - сглаживание и поиск отклонений"""
    try:
        smoothed = savgol_filter(data, window_length=window_length, polyorder=polyorder)
        residuals = data - smoothed
        residual_std = np.nanstd(residuals)
        outlier_mask = np.abs(residuals) > threshold * residual_std
        return outlier_mask, None, None
    except:
        return np.zeros(len(data), dtype=bool), None, None


def detect_outliers_isolation_forest(df, column, contamination=0.1):
    """Метод Isolation Forest - машинное обучение для обнаружения аномалий"""
    try:
        from sklearn.ensemble import IsolationForest

        data = df[column].values.reshape(-1, 1)
        # Удаляем NaN для обучения
        mask_valid = ~np.isnan(data).flatten()
        data_valid = data[mask_valid]

        if len(data_valid) < 10:
            return np.zeros(len(data), dtype=bool), None, None

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(data_valid)

        # Создаем маску для всех данных
        outlier_mask = np.zeros(len(data), dtype=bool)
        outlier_mask[mask_valid] = (predictions == -1)

        return outlier_mask, None, None
    except ImportError:
        print("Isolation Forest требует scikit-learn. Установите: pip install scikit-learn")
        return np.zeros(len(data), dtype=bool), None, None
    except Exception as e:
        print(f"Ошибка в Isolation Forest: {e}")
        return np.zeros(len(data), dtype=bool), None, None


def apply_outlier_filter(df, column, method='iqr', **kwargs):
    """
    Применение фильтра выбросов к столбцу

    Parameters:
    - df: DataFrame
    - column: имя столбца
    - method: метод фильтрации
        'iqr' - межквартильный размах
        'mad' - медианное абсолютное отклонение
        'rolling' - скользящее окно
        'derivative' - производная (резкие скачки)
        'peak' - поиск пиков
        'savgol' - фильтр Савицкого-Голая
        'isolation_forest' - Isolation Forest (ML)
    """
    data = pd.to_numeric(df[column], errors='coerce')
    original_count = len(data)
    original_mean = data.mean()
    original_std = data.std()

    if method == 'iqr':
        multiplier = kwargs.get('multiplier', 1.5)
        outlier_mask, lower_bound, upper_bound = detect_outliers_iqr(data, multiplier)
        bounds = [lower_bound, upper_bound]
        method_name = f"IQR (множитель={multiplier})"

    elif method == 'mad':
        threshold = kwargs.get('threshold', 3.5)
        outlier_mask, lower_bound, upper_bound = detect_outliers_mad(data, threshold)
        bounds = [lower_bound, upper_bound]
        method_name = f"MAD (порог={threshold})"

    elif method == 'rolling':
        window_size = kwargs.get('window_size', 10)
        threshold = kwargs.get('threshold', 3)
        outlier_mask, _, _ = detect_outliers_rolling(data, window_size, threshold)
        bounds = None
        method_name = f"Скользящее окно (окно={window_size}, порог={threshold}σ)"

    elif method == 'derivative':
        threshold_multiplier = kwargs.get('threshold_multiplier', 5)
        outlier_mask, _, _ = detect_outliers_derivative(data, threshold_multiplier)
        bounds = None
        method_name = f"Производная (порог={threshold_multiplier}×σ)"

    elif method == 'peak':
        prominence = kwargs.get('prominence', 0.5)
        distance = kwargs.get('distance', 10)
        outlier_mask, _, _ = detect_outliers_peak(data, prominence, distance)
        bounds = None
        method_name = f"Поиск пиков (prominence={prominence}, distance={distance})"

    elif method == 'savgol':
        window_length = kwargs.get('window_length', 21)
        polyorder = kwargs.get('polyorder', 3)
        threshold = kwargs.get('threshold', 3)
        outlier_mask, _, _ = detect_outliers_savgol(data, window_length, polyorder, threshold)
        bounds = None
        method_name = f"Фильтр Савицкого-Голая (окно={window_length}, порог={threshold}σ)"

    elif method == 'isolation_forest':
        contamination = kwargs.get('contamination', 0.1)
        outlier_mask, _, _ = detect_outliers_isolation_forest(df, column, contamination)
        bounds = None
        method_name = f"Isolation Forest (contamination={contamination})"

    else:
        raise ValueError(f"Неизвестный метод: {method}")

    # Создаем копию данных с выбросами, замененными на NaN
    filtered_data = data.copy()
    filtered_data[outlier_mask] = np.nan

    outlier_count = outlier_mask.sum()
    outlier_percent = (outlier_count / original_count) * 100

    statis = {
        'original_count': original_count,
        'outlier_count': outlier_count,
        'outlier_percent': outlier_percent,
        'original_mean': original_mean,
        'original_std': original_std,
        'filtered_mean': filtered_data.mean(),
        'filtered_std': filtered_data.std(),
        'bounds': bounds,
        'method': method_name
    }

    return filtered_data, outlier_mask, bounds, statis


def visualize_outlier_filter(df, column, filtered_data, outlier_mask, bounds, statis, save_folder=None):
    """
    Визуализация результата фильтрации выбросов
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    data = pd.to_numeric(df[column], errors='coerce')

    # 1. Исходные данные с выделением выбросов
    ax1 = axes[0, 0]
    ax1.plot(df.index, data, color='blue', alpha=0.5, linewidth=1.5, label='Данные')

    if outlier_mask.sum() > 0:
        ax1.scatter(df.index[outlier_mask], data[outlier_mask],
                    color='red', s=50, label=f'Выбросы ({statis["outlier_count"]})', zorder=5)

    if bounds:
        ax1.axhline(y=bounds[0], color='orange', linestyle='--', alpha=0.7, label=f'Нижняя граница = {bounds[0]:.4f}')
        ax1.axhline(y=bounds[1], color='orange', linestyle='--', alpha=0.7, label=f'Верхняя граница = {bounds[1]:.4f}')
        ax1.fill_between(df.index, bounds[0], bounds[1], alpha=0.1, color='green')

    ax1.set_title(f'{column} - Исходные данные с выделением выбросов', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Индекс строки')
    ax1.set_ylabel('Значение')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. Данные после фильтрации
    ax2 = axes[0, 1]
    ax2.plot(df.index, filtered_data, color='green', alpha=0.7, linewidth=1.5, marker='.', markersize=2)
    if bounds:
        ax2.axhline(y=bounds[0], color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=bounds[1], color='orange', linestyle='--', alpha=0.7)
        ax2.fill_between(df.index, bounds[0], bounds[1], alpha=0.1, color='green')
    ax2.set_title(f'{column} - После фильтрации (выбросы удалены)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Индекс строки')
    ax2.set_ylabel('Значение')
    ax2.grid(True, alpha=0.3)

    # 3. Распределение до фильтрации
    ax3 = axes[1, 0]
    data_clean = data[~outlier_mask]
    ax3.hist(data_clean, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Нормальные значения')
    if outlier_mask.sum() > 0:
        ax3.hist(data[outlier_mask], bins=30, alpha=0.7, color='red', edgecolor='black', label='Выбросы')
    if bounds:
        ax3.axvline(x=bounds[0], color='orange', linestyle='--', alpha=0.7, label=f'Нижняя граница')
        ax3.axvline(x=bounds[1], color='orange', linestyle='--', alpha=0.7, label=f'Верхняя граница')
    ax3.set_title(f'{column} - Распределение до фильтрации', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Значение')
    ax3.set_ylabel('Частота')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # 4. Статистика
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    СТАТИСТИКА ФИЛЬТРАЦИИ
    {'=' * 40}

    Метод: {statis['method']}

    Исходные данные:
      Количество: {statis['original_count']}
      Среднее: {statis['original_mean']:.4f}
      Стд: {statis['original_std']:.4f}

    Обнаружено выбросов:
      Количество: {statis['outlier_count']}
      Процент: {statis['outlier_percent']:.2f}%

    После фильтрации:
      Среднее: {statis['filtered_mean']:.4f}
      Стд: {statis['filtered_std']:.4f}
    """

    if bounds:
        stats_text += f"""

    Границы:
      Нижняя: {bounds[0]:.4f}
      Верхняя: {bounds[1]:.4f}
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle(f'Анализ выбросов для столбца: {column}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_folder:
        safe_name = column.replace('/', '_').replace('\\', '_').replace(':', '_')
        save_path = os.path.join(save_folder, f'outlier_analysis_{safe_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График анализа выбросов сохранен: {save_path}")

    plt.show()
"""
Файл конфигурации проекта
Здесь собраны все настраиваемые параметры
"""

# ============= ПУТИ К ФАЙЛАМ =============
INPUT_FILE = 'ПП.csv'  # Путь к исходному файлу с данными

# ============= ПАРАМЕТРЫ ЗАГРУЗКИ =============
ENCODINGS_TO_TRY = ['cp1251', 'windows-1251', 'cp1252', 'latin1', 'iso-8859-1', 'utf-8-sig']

# ============= ПАРАМЕТРЫ ГРАНИЦ (по умолчанию) =============
DEFAULT_BOUNDS_PERCENT = 50  # Границы ±X% от среднего

# ============= ПАРАМЕТРЫ ПОИСКА ОКНА =============
MIN_WINDOW_SIZE = 2000  # Минимальный размер окна для поиска

# ============= ПАРАМЕТРЫ МОДЕЛЕЙ =============
# Общие параметры
TEST_SIZE = 0.2  # Доля тестовой выборки
RANDOM_STATE = 42  # Seed для воспроизводимости

# Random Forest параметры
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'bootstrap': True,
    'oob_score': True,
    'n_jobs': 1  # 1 поток для избежания проблем
}

# XGBoost параметры
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_jobs': 1,
    'verbosity': 0
}

# ============= ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ =============
FIGURE_DPI = 300  # Качество сохраняемых графиков
PLOT_SIZE = (15, 4)  # Базовый размер графиков

# ============= ПАРАМЕТРЫ ФИЛЬТРАЦИИ =============
IQR_MULTIPLIER = 1.5  # Множитель для IQR метода
MAD_THRESHOLD = 3.5  # Порог для MAD метода
ROLLING_WINDOW = 10  # Размер окна для скользящего метода
ROLLING_THRESHOLD = 3  # Порог для скользящего метода
DERIVATIVE_MULTIPLIER = 5  # Множитель для метода производной
PEAK_PROMINENCE = 0.5  # Prominence для поиска пиков
PEAK_DISTANCE = 10  # Расстояние между пиками
SAVGOL_WINDOW = 21  # Длина окна для фильтра Савицкого-Голая
SAVGOL_POLYORDER = 3  # Порядок полинома для фильтра Савицкого-Голая
ISOLATION_FOREST_CONTAMINATION = 0.1  # Доля выбросов для Isolation Forest
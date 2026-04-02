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
    'n_jobs': 1
}

# XGBoost параметры
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_jobs': 1,
    'verbosity': 0
}

# ============= НЕЙРОСЕТЬ (MLP) ПАРАМЕТРЫ =============
MLP_PARAMS = {
    'hidden_layer_sizes': (100, 50, 50),  # Два скрытых слоя: 100 и 50 нейронов
    'activation': 'relu',              # Функция активации
    'solver': 'adam',                  # Оптимизатор
    'alpha': 0.0001,                   # L2 регуляризация
    'batch_size': 'auto',              # Размер батча
    'learning_rate': 'adaptive',       # Адаптивная скорость обучения
    'learning_rate_init': 0.001,       # Начальная скорость обучения
    'max_iter': 500,                   # Максимальное количество эпох
    'early_stopping': True,            # Ранняя остановка
    'validation_fraction': 0.1,        # Доля валидационной выборки
    'tol': 0.0001,                     # Допуск для остановки
    'random_state': RANDOM_STATE,
    'verbose': False
}

# ============= ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ =============
FIGURE_DPI = 300
PLOT_SIZE = (15, 4)

# ============= ПАРАМЕТРЫ ФИЛЬТРАЦИИ =============
IQR_MULTIPLIER = 1.5
MAD_THRESHOLD = 3.5
ROLLING_WINDOW = 10
ROLLING_THRESHOLD = 3
DERIVATIVE_MULTIPLIER = 5
PEAK_PROMINENCE = 0.5
PEAK_DISTANCE = 10
SAVGOL_WINDOW = 21
SAVGOL_POLYORDER = 3
ISOLATION_FOREST_CONTAMINATION = 0.1
"""
Файл конфигурации проекта
"""

# ============= ПУТИ К ФАЙЛАМ =============
INPUT_FILE = 'ПП.csv'

# ============= ПАРАМЕТРЫ ЗАГРУЗКИ =============
ENCODINGS_TO_TRY = ['cp1251', 'windows-1251', 'cp1252', 'latin1', 'iso-8859-1', 'utf-8-sig']

# ============= ПАРАМЕТРЫ ГРАНИЦ (по умолчанию) =============
DEFAULT_BOUNDS_PERCENT = 50

# ============= ПАРАМЕТРЫ ПОИСКА ОКНА =============
MIN_WINDOW_SIZE = 2000

# ============= ПАРАМЕТРЫ МОДЕЛЕЙ =============
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_FEATURES_TO_SHOW = 10  # Количество важных признаков для вывода в терминал

# Random Forest параметры
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'bootstrap': True,
    'oob_score': True,
    'n_jobs': -1,
    'random_state': 42
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
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0
}

# Нейросеть (MLP) параметры
MLP_PARAMS = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'batch_size': 'auto',
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'tol': 0.0001,
    'random_state': 42,
    'verbose': False
}

# ============= ПАРАМЕТРЫ ОПТИМИЗАЦИИ =============
OPTIMIZATION_TOP_FEATURES = 8      # Количество важных признаков (если не выбраны все)
OPTIMIZATION_POP_SIZE = 50
OPTIMIZATION_GENERATIONS = 100
OPTIMIZATION_MUTATION_RATE = 0.1
OPTIMIZATION_CROSSOVER_RATE = 0.7
OPTIMIZATION_ELITISM = 2

# ============= ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ =============
FIGURE_DPI = 300

# ============= ПАРАМЕТРЫ ФИЛЬТРАЦИИ =============
IQR_MULTIPLIER = 1.5
MAD_THRESHOLD = 3.5
DERIVATIVE_MULTIPLIER = 5
PEAK_PROMINENCE = 0.5
PEAK_DISTANCE = 10
SAVGOL_WINDOW = 21
SAVGOL_POLYORDER = 3
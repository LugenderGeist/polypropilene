import os

# ============= ПУТИ К ФАЙЛАМ =============
INPUT_FILE = 'data/ПП.csv'

# Создаем папку results, если её нет
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    'bootstrap': True,      # bootstrap должен быть True для oob_score
    'oob_score': True,      # out-of-bag оценка
    'n_jobs': -1,
    'random_state': 42
}

# XGBoost параметры
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "learning_rate": 0.05651254898379137,
    "subsample": 0.8327847839841903,
    "colsample_bytree": 0.8016559265627492,
    "min_child_weight": 3,
    "reg_alpha": 0.19186616934726605,
    "reg_lambda": 0.3066518709056766
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
OPTIMIZATION_TOP_FEATURES = 8
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

# ============= ПАРАМЕТРЫ ГЕНЕРАЦИИ НАБОРОВ =============
GENERATION_NUM_SAMPLES = 100        # Количество генерируемых наборов
GENERATION_METHOD = 'latin'         # Метод генерации: 'random', 'latin', 'grid'
GENERATION_USE_TOP_FEATURES = True  # Использовать только важные признаки (True) или все (False)
GENERATION_TOP_FEATURES = 8         # Количество важных признаков для генерации (если USE_TOP_FEATURES=True)

# ============= ПАРАМЕТРЫ OPTUNA =============
OPTUNA_N_TRIALS = 500              # Количество испытаний для оптимизации
OPTUNA_CV_FOLDS = 7               # Количество фолдов для кросс-валидации
OPTUNA_USE_OPTIMIZED_PARAMS = False  # Использовать оптимизированные параметры (True/False)
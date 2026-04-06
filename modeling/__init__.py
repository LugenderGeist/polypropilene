"""
Моделирование и оптимизация
"""

# Экспортируем функции для обучения моделей
from .modeling import (
    build_random_forest_model,
    build_xgboost_model,
    build_mlp_model,
    compare_models
)

# Экспортируем функции для оптимизации гиперпараметров
from .hyperopt import (
    optimize_random_forest,
    optimize_xgboost,
    optimize_mlp,
    plot_optimization_history
)

# Экспортируем функции для генетической оптимизации
from .optimization import (
    run_optimization,
    GeneticOptimizer
)

# Экспортируем функции для генерации наборов
from .generation import (
    generate_samples,
    generate_random_samples,
    generate_latin_hypercube_samples,
    generate_grid_samples
)

__all__ = [
    # Модели
    'build_random_forest_model',
    'build_xgboost_model',
    'build_mlp_model',
    'compare_models',

    # Оптимизация гиперпараметров
    'optimize_random_forest',
    'optimize_xgboost',
    'optimize_mlp',
    'plot_optimization_history',

    # Генетическая оптимизация
    'run_optimization',
    'GeneticOptimizer',

    # Генерация
    'generate_samples',
    'generate_random_samples',
    'generate_latin_hypercube_samples',
    'generate_grid_samples'
]
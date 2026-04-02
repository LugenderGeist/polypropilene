import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import warnings
from config import (OPTIMIZATION_POP_SIZE, OPTIMIZATION_GENERATIONS,
                    OPTIMIZATION_MUTATION_RATE, OPTIMIZATION_CROSSOVER_RATE,
                    OPTIMIZATION_ELITISM)

warnings.filterwarnings('ignore')


class GeneticOptimizer:
    """
    Генетический алгоритм для оптимизации выходного параметра
    """

    def __init__(self, model, input_columns, scaler=None,
                 pop_size=None, generations=None, mutation_rate=None,
                 crossover_rate=None, elitism=None):
        """
        Parameters:
        - model: обученная модель
        - input_columns: список входных признаков
        - scaler: нормализатор данных (если использовался)
        - pop_size: размер популяции (берется из config, если не указан)
        - generations: количество поколений (берется из config, если не указан)
        - mutation_rate: вероятность мутации (берется из config, если не указан)
        - crossover_rate: вероятность скрещивания (берется из config, если не указан)
        - elitism: количество лучших особей (берется из config, если не указан)
        """
        self.model = model
        self.input_columns = input_columns
        self.scaler = scaler

        # Используем параметры из config, если не переданы явно
        self.pop_size = pop_size if pop_size is not None else OPTIMIZATION_POP_SIZE
        self.generations = generations if generations is not None else OPTIMIZATION_GENERATIONS
        self.mutation_rate = mutation_rate if mutation_rate is not None else OPTIMIZATION_MUTATION_RATE
        self.crossover_rate = crossover_rate if crossover_rate is not None else OPTIMIZATION_CROSSOVER_RATE
        self.elitism = elitism if elitism is not None else OPTIMIZATION_ELITISM

        # Границы для каждого признака
        self.bounds = {}

    def set_bounds_from_data(self, df, input_columns):
        """Устанавливает границы на основе реальных данных"""
        for col in input_columns:
            data = df[col].dropna()
            self.bounds[col] = {
                'min': data.min(),
                'max': data.max(),
                'mean': data.mean(),
                'std': data.std()
            }
        return self.bounds

    def set_bounds_manual(self, bounds_dict):
        """Ручная установка границ"""
        self.bounds = bounds_dict

    def create_individual(self):
        """Создает одну особь (случайный набор параметров)"""
        individual = {}
        for col in self.input_columns:
            min_val = self.bounds[col]['min']
            max_val = self.bounds[col]['max']
            individual[col] = np.random.uniform(min_val, max_val)
        return individual

    def create_population(self):
        """Создает начальную популяцию"""
        return [self.create_individual() for _ in range(self.pop_size)]

    def evaluate_fitness(self, individual):
        """Оценивает приспособленность особи (значение выходного параметра)"""
        X = pd.DataFrame([individual])[self.input_columns]

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        y_pred = self.model.predict(X_scaled)[0]
        return y_pred

    def evaluate_population(self, population):
        """Оценивает всю популяцию"""
        fitness_scores = []
        for individual in population:
            fitness = self.evaluate_fitness(individual)
            fitness_scores.append(fitness)
        return fitness_scores

    def select_parents(self, population, fitness_scores):
        """Турнирная селекция для выбора родителей"""
        parents = []
        for _ in range(2):
            tournament_size = 3
            candidates_idx = np.random.choice(len(population), tournament_size, replace=False)
            best_idx = candidates_idx[np.argmax([fitness_scores[i] for i in candidates_idx])]
            parents.append(population[best_idx])
        return parents

    def crossover(self, parent1, parent2):
        """Одноточечное скрещивание"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = {}
        child2 = {}

        split_point = np.random.randint(1, len(self.input_columns))
        split_col = self.input_columns[split_point]

        crossover_started = False
        for col in self.input_columns:
            if col == split_col:
                crossover_started = True

            if crossover_started:
                child1[col] = parent2[col]
                child2[col] = parent1[col]
            else:
                child1[col] = parent1[col]
                child2[col] = parent2[col]

        return child1, child2

    def mutate(self, individual):
        """Мутация особи"""
        for col in self.input_columns:
            if np.random.random() < self.mutation_rate:
                std = self.bounds[col]['std'] * 0.1
                mutation = np.random.normal(0, std)
                new_val = individual[col] + mutation
                individual[col] = np.clip(new_val,
                                          self.bounds[col]['min'],
                                          self.bounds[col]['max'])
        return individual

    def run(self, verbose=True):
        """Запуск генетического алгоритма"""
        population = self.create_population()
        best_fitness_history = []
        mean_fitness_history = []

        best_individual = None
        best_fitness = -np.inf

        for generation in range(self.generations):
            fitness_scores = self.evaluate_population(population)

            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            best_fitness_history.append(current_best_fitness)
            mean_fitness_history.append(np.mean(fitness_scores))

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

            if verbose and generation % 20 == 0:
                print(f"Поколение {generation}: Лучший = {best_fitness:.4f}, Средний = {np.mean(fitness_scores):.4f}")

            # Элитизм
            elite_indices = np.argsort(fitness_scores)[-self.elitism:]
            new_population = [population[i].copy() for i in elite_indices]

            # Создаем остальных особей
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            population = new_population[:self.pop_size]

        # Сортируем финальную популяцию
        final_fitness = self.evaluate_population(population)
        sorted_idx = np.argsort(final_fitness)[::-1]

        results = []
        for idx in sorted_idx[:10]:
            results.append({
                'fitness': final_fitness[idx],
                'parameters': population[idx]
            })

        if verbose:
            print(f"\n" + "=" * 80)
            print("ЛУЧШИЕ РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
            print("=" * 80)
            print(f"\nЛучшее найденное значение: {best_fitness:.4f}")
            print("\nОптимальные параметры:")
            for col, value in best_individual.items():
                print(f"  {col}: {value:.4f}")

        return {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'results': results,
            'history': {
                'best': best_fitness_history,
                'mean': mean_fitness_history
            }
        }


def run_optimization(df_original, model, input_columns, output_columns,
                     n_top_features=None, save_folder=None):
    """
    Запуск оптимизации для максимизации выходного параметра

    Parameters:
    - df_original: исходный DataFrame (для границ)
    - model: обученная модель
    - input_columns: все входные столбцы
    - output_columns: выходные столбцы
    - n_top_features: количество важных признаков (берется из config, если не указан)
    - save_folder: папка для сохранения результатов
    """
    from config import OPTIMIZATION_TOP_FEATURES

    if n_top_features is None:
        n_top_features = OPTIMIZATION_TOP_FEATURES

    target = output_columns[0]

    print("\n" + "=" * 80)
    print(f"ОПТИМИЗАЦИЯ ВХОДНЫХ ПАРАМЕТРОВ ДЛЯ МАКСИМИЗАЦИИ {target}")
    print("=" * 80)

    print(f"\n📊 Параметры оптимизации (из config.py):")
    print(f"   - Важных признаков: {OPTIMIZATION_TOP_FEATURES}")
    print(f"   - Размер популяции: {OPTIMIZATION_POP_SIZE}")
    print(f"   - Количество поколений: {OPTIMIZATION_GENERATIONS}")
    print(f"   - Вероятность мутации: {OPTIMIZATION_MUTATION_RATE}")
    print(f"   - Вероятность скрещивания: {OPTIMIZATION_CROSSOVER_RATE}")
    print(f"   - Элитизм: {OPTIMIZATION_ELITISM}")

    # ============= ВЫБОР ВАЖНЫХ ПРИЗНАКОВ =============
    feature_importance = None

    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        print("\n📊 Используется важность признаков из модели")
    elif hasattr(model, 'coefs_'):
        feature_importance = np.abs(model.coefs_[0]).mean(axis=1)
        print("\n📊 Используются веса первого слоя нейросети")
    else:
        feature_importance = np.ones(len(input_columns))
        print("\n📊 Важность признаков не найдена, используются все признаки")

    importance_df = pd.DataFrame({
        'feature': input_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print(f"\nТоп-{n_top_features} наиболее важных признаков для оптимизации:")
    top_features = importance_df.head(n_top_features)['feature'].tolist()
    for i, (idx, row) in enumerate(importance_df.head(n_top_features).iterrows()):
        print(f"  {i + 1}. {row['feature']}: {row['importance']:.4f}")

    use_all = input("\nИспользовать все признаки для оптимизации? (да/нет): ").strip().lower()
    if use_all in ['да', 'yes', 'y', 'д']:
        optimize_features = input_columns
        print(f"\n✅ Оптимизируемые признаки ({len(optimize_features)}): все признаки")
    else:
        optimize_features = top_features
        print(f"\n✅ Оптимизируемые признаки ({len(optimize_features)}): {optimize_features}")

    # ============= СОЗДАНИЕ ОПТИМИЗАТОРА =============
    optimizer = GeneticOptimizer(
        model=model,
        input_columns=optimize_features,
        scaler=None
        # Параметры pop_size, generations и т.д. берутся из config автоматически
    )

    bounds = optimizer.set_bounds_from_data(df_original, optimize_features)

    print("\n📊 Границы поиска:")
    for col in optimize_features:
        print(f"  {col}: [{bounds[col]['min']:.4f}, {bounds[col]['max']:.4f}]")

    # ============= ФИКСАЦИЯ ОСТАЛЬНЫХ ПРИЗНАКОВ =============
    fixed_params = {}
    for col in input_columns:
        if col not in optimize_features:
            fixed_params[col] = df_original[col].mean()

    if fixed_params:
        print(f"\nФиксированные параметры (средние значения):")
        for col, val in fixed_params.items():
            print(f"  {col}: {val:.4f}")

    # Модифицируем метод evaluate_fitness
    def evaluate_with_fixed(individual):
        full_params = fixed_params.copy()
        full_params.update(individual)
        X = pd.DataFrame([full_params])[input_columns]

        if hasattr(model, 'predict'):
            try:
                y_pred = model.predict(X)[0]
            except:
                y_pred = model.predict(X.values)[0]
        else:
            y_pred = model.predict(X)[0]
        return y_pred

    optimizer.evaluate_fitness = evaluate_with_fixed

    # ============= ЗАПУСК ОПТИМИЗАЦИИ =============
    print("\n" + "=" * 80)
    print("ЗАПУСК ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
    print("=" * 80)

    result = optimizer.run(verbose=True)

    # ============= ВИЗУАЛИЗАЦИЯ И СОХРАНЕНИЕ =============
    if save_folder:
        plot_optimization_results(result, optimizer.bounds, optimize_features,
                                  target, save_folder)
        save_optimization_results(result, optimize_features, target, save_folder)
        save_optimization_details(result, optimizer.bounds, optimize_features, target, save_folder)

    return result

def plot_optimization_results(result, bounds, optimize_features, target, save_folder=None):
    """Визуализация результатов оптимизации - каждый график в отдельном файле"""

    # 1. История оптимизации (график сходимости)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    generations = range(len(result['history']['best']))
    ax1.plot(generations, result['history']['best'], 'b-', linewidth=2, label='Лучшее значение')
    ax1.plot(generations, result['history']['mean'], 'r--', linewidth=1.5, label='Среднее по популяции')
    ax1.set_xlabel('Поколение', fontsize=12)
    ax1.set_ylabel(f'Значение {target}', fontsize=12)
    ax1.set_title('Сходимость генетического алгоритма', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_folder:
        save_path1 = os.path.join(save_folder, '01_convergence_history.png')
        plt.savefig(save_path1, dpi=300, bbox_inches='tight')
        print(f"📁 График сходимости сохранен: {save_path1}")
    plt.show()
    plt.close(fig1)

    # 2. Топ-10 лучших решений
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    top_fitness = [r['fitness'] for r in result['results'][:10]]
    bars = ax2.barh(range(len(top_fitness)), top_fitness, color='green', alpha=0.7, edgecolor='black')

    # Добавляем значения на бары
    for i, (bar, value) in enumerate(zip(bars, top_fitness)):
        ax2.text(value + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{value:.4f}', va='center', fontsize=9)

    ax2.set_xlabel(f'Значение {target}', fontsize=12)
    ax2.set_ylabel('Ранг решения', fontsize=12)
    ax2.set_title('Топ-10 лучших решений', fontsize=14, fontweight='bold')
    ax2.set_yticks(range(len(top_fitness)))
    ax2.set_yticklabels([f'Решение {i + 1}' for i in range(len(top_fitness))])
    ax2.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_folder:
        save_path2 = os.path.join(save_folder, '02_top_10_solutions.png')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"📁 График топ-10 решений сохранен: {save_path2}")
    plt.show()
    plt.close(fig2)

    # 3. Оптимальные значения параметров
    fig3, ax3 = plt.subplots(figsize=(12, max(6, len(optimize_features) * 0.4)))
    best_individual = result['best_individual']
    param_names = list(best_individual.keys())
    param_values = list(best_individual.values())

    # Нормализуем значения для цветовой шкалы (относительно границ)
    colors = []
    for col in param_names:
        min_val = bounds[col]['min']
        max_val = bounds[col]['max']
        normalized = (best_individual[col] - min_val) / (max_val - min_val)
        colors.append(plt.cm.RdYlGn(normalized))

    bars = ax3.barh(range(len(param_values)), param_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Добавляем значения на бары
    for i, (bar, value, col) in enumerate(zip(bars, param_values, param_names)):
        ax3.text(value + (bounds[col]['max'] - bounds[col]['min']) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f'{value:.4f}', va='center', fontsize=9)

    ax3.set_yticks(range(len(param_values)))
    ax3.set_yticklabels(param_names, fontsize=10)
    ax3.set_xlabel('Значение параметра', fontsize=12)
    ax3.set_title('Оптимальные значения параметров', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_folder:
        save_path3 = os.path.join(save_folder, '03_optimal_parameters.png')
        plt.savefig(save_path3, dpi=300, bbox_inches='tight')
        print(f"📁 График оптимальных параметров сохранен: {save_path3}")
    plt.show()
    plt.close(fig3)

    # 4. Сравнение с исходными данными (гистограмма распределения)
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    # Собираем статистику по каждому оптимизируемому параметру
    param_comparison = []
    for col in optimize_features:
        opt_value = best_individual[col]
        min_val = bounds[col]['min']
        max_val = bounds[col]['max']
        mean_val = bounds[col]['mean']

        # Относительное положение оптимума в диапазоне
        relative_pos = (opt_value - min_val) / (max_val - min_val) * 100

        param_comparison.append({
            'param': col,
            'optimal': opt_value,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'relative_pos': relative_pos
        })

    # Сортируем по относительной позиции
    param_comparison.sort(key=lambda x: x['relative_pos'])

    x_pos = range(len(param_comparison))
    relative_positions = [p['relative_pos'] for p in param_comparison]
    param_names_short = [p['param'][:20] + '...' if len(p['param']) > 20 else p['param']
                         for p in param_comparison]

    colors = ['red' if pos < 30 else 'green' if pos > 70 else 'orange' for pos in relative_positions]
    bars = ax4.barh(x_pos, relative_positions, color=colors, alpha=0.7, edgecolor='black')

    # Добавляем значения
    for i, (bar, pos, param) in enumerate(zip(bars, relative_positions, param_comparison)):
        ax4.text(pos + 1, bar.get_y() + bar.get_height() / 2,
                 f'{pos:.1f}%', va='center', fontsize=9)

    ax4.set_yticks(x_pos)
    ax4.set_yticklabels(param_names_short, fontsize=9)
    ax4.set_xlabel('Позиция оптимума в диапазоне (%)', fontsize=12)
    ax4.set_title('Распределение оптимальных значений в границах', fontsize=14, fontweight='bold')
    ax4.axvline(x=50, color='blue', linestyle='--', alpha=0.5, label='Середина диапазона')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_folder:
        save_path4 = os.path.join(save_folder, '04_optimal_distribution.png')
        plt.savefig(save_path4, dpi=300, bbox_inches='tight')
        print(f"📁 График распределения оптимальных значений сохранен: {save_path4}")
    plt.show()
    plt.close(fig4)

    # 5. Тепловая карта корреляций оптимальных параметров (если их достаточно)
    if len(optimize_features) >= 3:
        fig5, ax5 = plt.subplots(figsize=(10, 8))

        # Собираем топ-10 решений для анализа
        solutions_data = []
        for r in result['results'][:10]:
            solution = [r['parameters'][col] for col in optimize_features]
            solutions_data.append(solution)

        # Нормализуем и вычисляем корреляции
        solutions_array = np.array(solutions_data)
        corr_matrix = np.corrcoef(solutions_array.T)

        # Рисуем тепловую карту
        im = ax5.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

        # Добавляем значения в ячейки
        for i in range(len(optimize_features)):
            for j in range(len(optimize_features)):
                text = ax5.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                ha="center", va="center", color="black", fontsize=8)

        ax5.set_xticks(range(len(optimize_features)))
        ax5.set_yticks(range(len(optimize_features)))
        ax5.set_xticklabels(optimize_features, rotation=45, ha='right', fontsize=8)
        ax5.set_yticklabels(optimize_features, fontsize=8)
        ax5.set_title('Корреляция параметров в лучших решениях', fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax5, label='Коэффициент корреляции')
        plt.tight_layout()

        if save_folder:
            save_path5 = os.path.join(save_folder, '05_parameters_correlation.png')
            plt.savefig(save_path5, dpi=300, bbox_inches='tight')
            print(f"📁 Тепловая карта корреляций сохранена: {save_path5}")
        plt.show()
        plt.close(fig5)

    # 6. Эволюция лучшего решения (дополнительный график)
    fig6, ax6 = plt.subplots(figsize=(10, 6))

    # Показываем изменение лучшего значения относительно начального
    best_history = result['history']['best']
    improvement = [b - best_history[0] for b in best_history]

    ax6.plot(generations, improvement, 'g-', linewidth=2)
    ax6.fill_between(generations, 0, improvement, alpha=0.3, color='green')
    ax6.set_xlabel('Поколение', fontsize=12)
    ax6.set_ylabel(f'Улучшение {target}', fontsize=12)
    ax6.set_title('Динамика улучшения результата', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_folder:
        save_path6 = os.path.join(save_folder, '06_improvement_dynamics.png')
        plt.savefig(save_path6, dpi=300, bbox_inches='tight')
        print(f"📁 График динамики улучшения сохранен: {save_path6}")
    plt.show()
    plt.close(fig6)


def save_optimization_details(result, optimize_features, target, save_folder):
    """Сохранение детальных результатов оптимизации в CSV"""

    # Сохраняем историю оптимизации
    history_df = pd.DataFrame({
        'generation': range(len(result['history']['best'])),
        'best_fitness': result['history']['best'],
        'mean_fitness': result['history']['mean']
    })
    history_path = os.path.join(save_folder, 'optimization_history.csv')
    history_df.to_csv(history_path, index=False, encoding='utf-8-sig')
    print(f"📁 История оптимизации сохранена: {history_path}")

    # Сохраняем топ-100 решений
    top_solutions = []
    for i, r in enumerate(result['results'][:100]):
        solution = {'rank': i + 1, 'fitness': r['fitness']}
        for col, val in r['parameters'].items():
            solution[col] = val
        top_solutions.append(solution)

    solutions_df = pd.DataFrame(top_solutions)
    solutions_path = os.path.join(save_folder, 'top_solutions.csv')
    solutions_df.to_csv(solutions_path, index=False, encoding='utf-8-sig')
    print(f"📁 Топ-100 решений сохранены: {solutions_path}")

    # Сохраняем лучшее решение отдельно
    best_solution = {'parameter': [], 'optimal_value': [], 'min_bound': [], 'max_bound': [], 'mean_value': []}
    for col in optimize_features:
        best_solution['parameter'].append(col)
        best_solution['optimal_value'].append(result['best_individual'][col])
        best_solution['min_bound'].append(bounds[col]['min'])
        best_solution['max_bound'].append(bounds[col]['max'])
        best_solution['mean_value'].append(bounds[col]['mean'])

    best_df = pd.DataFrame(best_solution)
    best_path = os.path.join(save_folder, 'best_solution.csv')
    best_df.to_csv(best_path, index=False, encoding='utf-8-sig')
    print(f"📁 Лучшее решение сохранено: {best_path}")

def save_optimization_results(result, optimize_features, target, save_folder):
    """Сохранение результатов оптимизации в файл"""
    results_path = os.path.join(save_folder, 'optimization_results.txt')

    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Целевая функция: максимизация {target}\n\n")

        f.write("ЛУЧШЕЕ НАЙДЕННОЕ РЕШЕНИЕ\n")
        f.write("-" * 40 + "\n")
        f.write(f"Значение {target}: {result['best_fitness']:.6f}\n\n")
        f.write("Оптимальные параметры:\n")
        for col, val in result['best_individual'].items():
            f.write(f"  {col}: {val:.6f}\n")

        f.write("\nТОП-10 ЛУЧШИХ РЕШЕНИЙ\n")
        f.write("-" * 40 + "\n")
        for i, r in enumerate(result['results'][:10]):
            f.write(f"\n{i + 1}. Значение = {r['fitness']:.6f}\n")
            for col, val in r['parameters'].items():
                f.write(f"    {col}: {val:.6f}\n")

        f.write("\nПАРАМЕТРЫ ОПТИМИЗАЦИИ\n")
        f.write("-" * 40 + "\n")
        f.write(f"Количество поколений: {len(result['history']['best'])}\n")
        f.write(f"Лучшее значение в начале: {result['history']['best'][0]:.6f}\n")
        f.write(f"Лучшее значение в конце: {result['history']['best'][-1]:.6f}\n")
        f.write(f"Улучшение: {result['history']['best'][-1] - result['history']['best'][0]:.6f}\n")

    print(f"📁 Результаты оптимизации сохранены в: {results_path}")
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Пробуем импортировать qmc для латинского гиперкуба
try:
    from scipy.stats import qmc

    QMC_AVAILABLE = True
except ImportError:
    QMC_AVAILABLE = False
    print("⚠️ Для метода 'latin' требуется scipy>=1.8.0. Установите: pip install scipy --upgrade")


def generate_random_samples(bounds, n_samples, optimize_features, fixed_params=None, random_state=42):
    """
    Генерация случайных наборов параметров
    """
    np.random.seed(random_state)
    samples = []

    for _ in range(n_samples):
        sample = {}
        for col in optimize_features:
            min_val = bounds[col]['min']
            max_val = bounds[col]['max']
            sample[col] = np.random.uniform(min_val, max_val)

        if fixed_params:
            sample.update(fixed_params)
        samples.append(sample)

    return samples


def generate_latin_hypercube_samples(bounds, n_samples, optimize_features, fixed_params=None, random_state=42):
    """
    Генерация наборов методом латинского гиперкуба (равномерное покрытие пространства)
    """
    if not QMC_AVAILABLE:
        print("⚠️ Библиотека qmc не найдена. Используется случайная генерация.")
        return generate_random_samples(bounds, n_samples, optimize_features, fixed_params, random_state)

    np.random.seed(random_state)

    # Создаем латинский гиперкуб
    sampler = qmc.LatinHypercube(d=len(optimize_features), seed=random_state)
    sample = sampler.random(n=n_samples)

    # Масштабируем на границы
    samples = []
    for i in range(n_samples):
        sample_dict = {}
        for j, col in enumerate(optimize_features):
            min_val = bounds[col]['min']
            max_val = bounds[col]['max']
            sample_dict[col] = min_val + sample[i, j] * (max_val - min_val)

        if fixed_params:
            sample_dict.update(fixed_params)
        samples.append(sample_dict)

    return samples


def generate_grid_samples(bounds, optimize_features, fixed_params=None, points_per_dim=3):
    """
    Генерация сеточных наборов (полный факторный эксперимент)
    Внимание: количество образцов = points_per_dim ^ количество признаков
    """
    # Создаем сетку для каждого признака
    grids = []
    for col in optimize_features:
        min_val = bounds[col]['min']
        max_val = bounds[col]['max']
        grids.append(np.linspace(min_val, max_val, points_per_dim))

    # Создаем декартово произведение
    mesh = np.array(np.meshgrid(*grids)).T.reshape(-1, len(optimize_features))

    samples = []
    for i in range(len(mesh)):
        sample_dict = {}
        for j, col in enumerate(optimize_features):
            sample_dict[col] = mesh[i, j]

        if fixed_params:
            sample_dict.update(fixed_params)
        samples.append(sample_dict)

    return samples


def generate_samples(df_original, model, input_columns, output_columns,
                     n_top_features=None, n_samples=100, method='latin',
                     save_folder=None):
    """
    Генерация наборов входных параметров и предсказание выходов
    """
    target = output_columns[0]

    print("\n" + "=" * 80)
    print(f"ГЕНЕРАЦИЯ НАБОРОВ ВХОДНЫХ ПАРАМЕТРОВ")
    print("=" * 80)

    # ============= 1. ОПРЕДЕЛЕНИЕ ВАЖНЫХ ПРИЗНАКОВ =============
    if n_top_features is not None and n_top_features < len(input_columns):
        # Получаем важность признаков из модели
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coefs_'):
            importance = np.abs(model.coefs_[0]).mean(axis=1)
        else:
            importance = np.ones(len(input_columns))

        importance_df = pd.DataFrame({
            'feature': input_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        optimize_features = importance_df.head(n_top_features)['feature'].tolist()
        print(f"\n📊 Для генерации используются топ-{n_top_features} важных признаков:")
        for i, row in importance_df.head(n_top_features).iterrows():
            print(f"   {i + 1}. {row['feature']}: {row['importance']:.4f}")
    else:
        optimize_features = input_columns
        print(f"\n📊 Для генерации используются все {len(input_columns)} признаков")

    # ============= 2. НАСТРОЙКА ГРАНИЦ =============
    bounds = {}
    for col in optimize_features:
        data = df_original[col].dropna()
        bounds[col] = {
            'min': data.min(),
            'max': data.max(),
            'mean': data.mean(),
            'std': data.std()
        }

    print(f"\n📊 Границы генерации:")
    for col in optimize_features:
        print(f"   {col}: [{bounds[col]['min']:.4f}, {bounds[col]['max']:.4f}]")

    # ============= 3. ФИКСАЦИЯ ОСТАЛЬНЫХ ПРИЗНАКОВ =============
    fixed_params = {}
    for col in input_columns:
        if col not in optimize_features:
            fixed_params[col] = df_original[col].mean()

    if fixed_params:
        print(f"\n📌 Фиксированные параметры (средние значения):")
        for col, val in fixed_params.items():
            print(f"   {col}: {val:.4f}")

    # ============= 4. ГЕНЕРАЦИЯ НАБОРОВ =============
    print(f"\n🚀 Генерация {n_samples} наборов методом '{method}'...")

    if method == 'random':
        samples = generate_random_samples(bounds, n_samples, optimize_features, fixed_params)
    elif method == 'latin':
        samples = generate_latin_hypercube_samples(bounds, n_samples, optimize_features, fixed_params)
    elif method == 'grid':
        # Для сетки нужно рассчитать количество точек
        points_per_dim = max(2, int(np.power(n_samples, 1 / len(optimize_features))))
        samples = generate_grid_samples(bounds, optimize_features, fixed_params, points_per_dim)
        print(f"   Сгенерировано {len(samples)} наборов (сетка {points_per_dim}×{points_per_dim})")
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    # ============= 5. ПРЕДСКАЗАНИЕ ВЫХОДОВ =============
    print("\n📊 Предсказание выходных значений...")

    inputs_list = []
    predictions_list = []

    for i, sample in enumerate(samples):
        # Создаем DataFrame для предсказания
        X = pd.DataFrame([sample])[input_columns]

        # Предсказываем
        try:
            y_pred = model.predict(X)[0]
        except:
            y_pred = model.predict(X.values)[0]

        # Сохраняем
        inputs_list.append(sample)
        predictions_list.append({
            'sample_id': i + 1,
            **sample,
            f'predicted_{target}': y_pred
        })

        if (i + 1) % 20 == 0:
            print(f"   Обработано {i + 1}/{len(samples)} наборов...")

    # ============= 6. СОЗДАНИЕ DATAFRAME =============
    inputs_df = pd.DataFrame(inputs_list)
    predictions_df = pd.DataFrame(predictions_list)

    # ============= 7. ВИЗУАЛИЗАЦИЯ =============
    plot_generated_samples(inputs_df, predictions_df, target, optimize_features, save_folder)

    # ============= 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ =============
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)

        # Сохраняем только входные параметры
        inputs_path = os.path.join(save_folder, f'generated_inputs_{method}_{n_samples}.csv')
        inputs_df.to_csv(inputs_path, index=False, encoding='utf-8-sig')
        print(f"\n📁 Входные параметры сохранены в: {inputs_path}")

        # Сохраняем входы + предсказанные выходы
        full_path = os.path.join(save_folder, f'generated_full_{method}_{n_samples}.csv')
        predictions_df.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f"📁 Полные данные (входы + предсказания) сохранены в: {full_path}")

        # Сохраняем информацию о генерации
        info_path = os.path.join(save_folder, f'generation_info.txt')
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("ИНФОРМАЦИЯ О ГЕНЕРАЦИИ НАБОРОВ\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Метод генерации: {method}\n")
            f.write(f"Количество наборов: {n_samples}\n")
            f.write(f"Целевая переменная: {target}\n\n")
            f.write("ГЕНЕРИРУЕМЫЕ ПРИЗНАКИ:\n")
            f.write("-" * 40 + "\n")
            for col in optimize_features:
                f.write(f"  {col}: [{bounds[col]['min']:.6f}, {bounds[col]['max']:.6f}]\n")
            if fixed_params:
                f.write("\nФИКСИРОВАННЫЕ ПРИЗНАКИ:\n")
                f.write("-" * 40 + "\n")
                for col, val in fixed_params.items():
                    f.write(f"  {col}: {val:.6f}\n")

        print(f"📁 Информация о генерации сохранена в: {info_path}")

    return inputs_df, predictions_df


def plot_generated_samples(inputs_df, predictions_df, target, optimize_features, save_folder=None):
    """Визуализация сгенерированных наборов"""

    # Определяем количество графиков
    n_features = len(optimize_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(optimize_features):
        ax = axes[i]

        # Гистограмма распределения
        ax.hist(inputs_df[col], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Частота', fontsize=10)
        ax.set_title(f'Распределение {col}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Скрываем лишние подграфики
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Распределение сгенерированных параметров',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, 'generated_distributions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 График распределений сохранен: {save_path}")

    plt.show()
    plt.close(fig)

    # Второй график - предсказанные значения
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    pred_col = f'predicted_{target}'
    ax2.hist(predictions_df[pred_col], bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel(f'Предсказанный {target}', fontsize=12)
    ax2.set_ylabel('Частота', fontsize=12)
    ax2.set_title(f'Распределение предсказанных значений {target}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, 'generated_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 График предсказаний сохранен: {save_path}")

    plt.show()
    plt.close(fig2)
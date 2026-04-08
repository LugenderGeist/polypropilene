import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')


def plot_nonlinear_dependencies(df, input_columns, output_columns, save_folder=None, top_n=5):
    target = output_columns[0]

    print("\n" + "=" * 80)
    print("ВИЗУАЛИЗАЦИЯ НЕЛИНЕЙНЫХ ЗАВИСИМОСТЕЙ")
    print("=" * 80)

    # Вычисляем обе корреляции для каждого признака
    dependencies = []
    for col in input_columns:
        pearson = df[col].corr(df[target])

        data1 = pd.to_numeric(df[col], errors='coerce').dropna().values
        data2 = pd.to_numeric(df[target], errors='coerce').dropna().values
        min_len = min(len(data1), len(data2))
        if min_len > 0:
            dist_corr = distance_correlation(data1[:min_len], data2[:min_len])
        else:
            dist_corr = 0.0

        # Разница показывает нелинейность
        nonlinearity = dist_corr - abs(pearson)

        dependencies.append({
            'feature': col,
            'pearson': abs(pearson),
            'distance_corr': dist_corr,
            'nonlinearity': nonlinearity
        })

    # Сортируем по силе нелинейности
    dependencies.sort(key=lambda x: x['nonlinearity'], reverse=True)
    top_features = dependencies[:top_n]

    print(f"\nТоп-{top_n} признаков с наибольшей нелинейной зависимостью от {target}:")
    for i, dep in enumerate(top_features):
        print(f"  {i + 1}. {dep['feature']}: "
              f"линейная = {dep['pearson']:.3f}, "
              f"нелинейная = {dep['distance_corr']:.3f}, "
              f"разница = {dep['nonlinearity']:.3f}")

    # Создаём графики
    n_rows = (top_n + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes[0], axes[1], axes[2]]

    for i, dep in enumerate(top_features):
        ax = axes[i]
        col = dep['feature']

        x = df[col].dropna().values
        y = df[target].loc[df[col].dropna().index].values

        # Сортируем для гладких кривых
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Точки данных
        ax.scatter(x, y, alpha=0.3, s=20, c='steelblue', label='Данные')

        # 1. Линейная регрессия (пунктир)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x_sorted, p(x_sorted), 'r--', linewidth=2, label=f'Линейная (r={dep["pearson"]:.2f})')

        # 2. Нелинейное сглаживание (lowess) через изотонную регрессию
        from scipy.interpolate import UnivariateSpline
        try:
            # Пробуем сплайн
            spline = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted) * 0.05)
            ax.plot(x_sorted, spline(x_sorted), 'g-', linewidth=2,
                    label=f'Нелинейная (dCorr={dep["distance_corr"]:.2f})')
        except:
            # Если сплайн не работает, используем скользящее среднее
            window = max(5, len(x_sorted) // 50)
            smoothed = np.convolve(y_sorted, np.ones(window) / window, mode='same')
            ax.plot(x_sorted, smoothed, 'g-', linewidth=2,
                    label=f'Нелинейная (dCorr={dep["distance_corr"]:.2f})')

        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel(target, fontsize=11)
        ax.set_title(f'{col}\n(dCorr={dep["distance_corr"]:.3f}, |Pearson|={dep["pearson"]:.3f})',
                     fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Скрываем лишние подграфики
    for i in range(len(top_features), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Нелинейные зависимости от {target}\n'
                 f'(зелёная линия — нелинейная аппроксимация, красная пунктир — линейная)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, 'nonlinear_dependencies.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📁 Графики нелинейных зависимостей сохранены: {save_path}")

    plt.show()
    plt.close(fig)

    # Пробуем подобрать формулы
    print("\n" + "=" * 80)
    print("ПОПЫТКА ПОДОБРАТЬ ФОРМУЛЫ ДЛЯ НЕЛИНЕЙНЫХ ЗАВИСИМОСТЕЙ")
    print("=" * 80)

    for dep in top_features[:3]:  # только топ-3
        col = dep['feature']
        x = df[col].dropna().values
        y = df[target].loc[df[col].dropna().index].values

        # Нормализуем для численной стабильности
        x_norm = (x - x.mean()) / x.std()
        y_norm = (y - y.mean()) / y.std()

        # Сортируем
        sort_idx = np.argsort(x_norm)
        x_sorted = x_norm[sort_idx]
        y_sorted = y_norm[sort_idx]

        # Пробуем разные функции
        def func_poly2(x, a, b, c):
            return a * x ** 2 + b * x + c

        def func_poly3(x, a, b, c, d):
            return a * x ** 3 + b * x ** 2 + c * x + d

        def func_exp(x, a, b, c):
            return a * np.exp(b * x) + c

        def func_log(x, a, b, c):
            return a * np.log(np.abs(x) + 0.1) + b * x + c

        def func_sin(x, a, b, c, d):
            return a * np.sin(b * x + c) + d

        best_r2 = -np.inf
        best_formula = None
        best_params = None

        models = [
            ('Полином 2-й степени', func_poly2, [-10, -10, -10], [10, 10, 10]),
            ('Полином 3-й степени', func_poly3, [-10, -10, -10, -10], [10, 10, 10, 10]),
            ('Экспонента', func_exp, [-10, -2, -10], [10, 2, 10]),
            ('Логарифм', func_log, [-10, -10, -10], [10, 10, 10]),
            ('Синус', func_sin, [-2, 0.5, -3.14, -2], [2, 3, 3.14, 2])
        ]

        for name, func, bounds_lower, bounds_upper in models:
            try:
                popt, _ = curve_fit(func, x_sorted, y_sorted,
                                    bounds=(bounds_lower, bounds_upper),
                                    maxfev=5000)
                y_pred = func(x_sorted, *popt)
                r2 = 1 - np.sum((y_sorted - y_pred) ** 2) / np.sum((y_sorted - np.mean(y_sorted)) ** 2)

                if r2 > best_r2 and r2 > 0.3:
                    best_r2 = r2
                    best_formula = name
                    best_params = popt
            except:
                continue

        if best_formula:
            print(f"\n📐 {col} → {target}:")
            print(f"   Лучшая аппроксимация: {best_formula} (R² = {best_r2:.3f})")
            if best_formula == 'Полином 2-й степени' and best_params:
                a, b, c = best_params
                print(f"   Формула (в нормализованных координатах): y = {a:.4f}·x² + {b:.4f}·x + {c:.4f}")
        else:
            print(f"\n📐 {col} → {target}: сложная нелинейность, простая формула не подобрана")

    return top_features

def distance_correlation(X, Y):
    X = np.array(X).flatten()
    Y = np.array(Y).flatten()
    n = len(X)

    # Евклидовы расстояния
    a = squareform(pdist(X.reshape(-1, 1)))
    b = squareform(pdist(Y.reshape(-1, 1)))

    # Двойное центрирование
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, np.newaxis] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, np.newaxis] + b.mean()

    # Distance covariance и variance
    dCov = np.sqrt((A * B).sum() / (n ** 2))
    dVarX = np.sqrt((A * A).sum() / (n ** 2))
    dVarY = np.sqrt((B * B).sum() / (n ** 2))

    if dVarX * dVarY > 0:
        return dCov / np.sqrt(dVarX * dVarY)
    else:
        return 0.0


def plot_distance_correlation_heatmap(df, input_columns, output_columns, save_folder=None):
    """
    Построение тепловой карты distance correlation.
    В отличие от корреляции Пирсона, обнаруживает ЛЮБЫЕ зависимости (не только линейные).
    """
    # Все интересующие нас столбцы
    all_cols = input_columns + output_columns
    n_cols = len(all_cols)

    print(f"\nВычисление distance correlation для {n_cols} столбцов...")
    print("(Это может занять 1-2 минуты для 10 000 строк × 20 столбцов)")

    # Вычисляем матрицу distance correlation
    dist_corr_matrix = np.zeros((n_cols, n_cols))

    for i, col1 in enumerate(all_cols):
        for j, col2 in enumerate(all_cols):
            data1 = pd.to_numeric(df[col1], errors='coerce').dropna().values
            data2 = pd.to_numeric(df[col2], errors='coerce').dropna().values

            # Берем минимальную общую длину
            min_len = min(len(data1), len(data2))
            if min_len > 0:
                dist_corr_matrix[i, j] = distance_correlation(data1[:min_len], data2[:min_len])
            else:
                dist_corr_matrix[i, j] = 0.0

            # Прогресс (для длинных вычислений)
            if (i * n_cols + j) % 50 == 0:
                print(f"   Прогресс: {i * n_cols + j + 1}/{n_cols * n_cols}")

    print("✅ Вычисление завершено!")

    # Настройка размера графика
    fig_size = max(10, min(20, n_cols * 0.5))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Цветовая схема: viridis хорошо показывает градиенты
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Рисуем тепловую карту
    sns.heatmap(dist_corr_matrix,
                annot=True,
                fmt='.2f',
                cmap=cmap,
                center=0.5,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Distance Correlation"},
                ax=ax,
                annot_kws={'size': max(6, min(10, 14 - n_cols * 0.3))})

    # Подписываем оси
    ax.set_xticklabels(all_cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(all_cols, fontsize=9)

    # Выделяем входные и выходные столбцы цветом
    for tick in ax.get_xticklabels():
        col_name = tick.get_text()
        if col_name in input_columns:
            tick.set_color('blue')
            tick.set_weight('bold')
        elif col_name in output_columns:
            tick.set_color('red')
            tick.set_weight('bold')

    for tick in ax.get_yticklabels():
        col_name = tick.get_text()
        if col_name in input_columns:
            tick.set_color('blue')
            tick.set_weight('bold')
        elif col_name in output_columns:
            tick.set_color('red')
            tick.set_weight('bold')

    ax.set_title('Distance Correlation Matrix\n(обнаруживает ЛЮБЫЕ зависимости, не только линейные)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Сохраняем график
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, 'distance_correlation_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 Distance correlation тепловая карта сохранена: {save_path}")

    plt.show()
    plt.close(fig)

    # Выводим статистику
    print("\n" + "=" * 80)
    print("СТАТИСТИКА DISTANCE CORRELATION")
    print("=" * 80)
    print("\nDistance correlation измеряет ЛЮБУЮ зависимость между переменными.")
    print("Значение 0 означает независимость, 1 — функциональную зависимость.")
    print("\nСамые сильные выявленные зависимости:")

    # Находим топ-10 самых сильных зависимостей (не включая диагональ)
    pairs = []
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            pairs.append({
                'pair': f"{all_cols[i]} - {all_cols[j]}",
                'distance_corr': dist_corr_matrix[i, j]
            })

    pairs.sort(key=lambda x: x['distance_corr'], reverse=True)

    for p in pairs[:10]:
        print(f"  {p['pair']}: {p['distance_corr']:.4f}")

    # Отдельно выводим зависимости входных с выходными
    if output_columns:
        print("\nЗависимости входных признаков с выходными:")
        target = output_columns[0]
        target_idx = all_cols.index(target)

        target_corrs = []
        for i, col in enumerate(input_columns):
            target_corrs.append({
                'feature': col,
                'distance_corr': dist_corr_matrix[i, target_idx]
            })

        target_corrs.sort(key=lambda x: x['distance_corr'], reverse=True)

        for tc in target_corrs[:10]:
            print(f"  {tc['feature']}: {tc['distance_corr']:.4f}")

    return dist_corr_matrix


def compare_correlations(df, input_columns, output_columns, save_folder=None):
    """
    Сравнение классической корреляции Пирсона и Distance Correlation.
    Показывает, какие зависимости были пропущены линейной корреляцией.
    """
    all_cols = input_columns + output_columns
    target = output_columns[0]

    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ КОРРЕЛЯЦИЙ: Пирсон vs Distance Correlation")
    print("=" * 80)

    results = []
    for col in input_columns:
        # Корреляция Пирсона (только линейная)
        pearson = df[col].corr(df[target])

        # Distance correlation (любая зависимость)
        data1 = pd.to_numeric(df[col], errors='coerce').dropna().values
        data2 = pd.to_numeric(df[target], errors='coerce').dropna().values
        min_len = min(len(data1), len(data2))
        if min_len > 0:
            dist_corr = distance_correlation(data1[:min_len], data2[:min_len])
        else:
            dist_corr = 0.0

        difference = dist_corr - abs(pearson)

        results.append({
            'feature': col,
            'pearson': pearson,
            'distance_corr': dist_corr,
            'difference': difference,
            'note': '⚠️ Нелинейная!' if difference > 0.2 else ''
        })

    results_df = pd.DataFrame(results).sort_values('distance_corr', ascending=False)

    print("\nРезультаты (отсортированы по distance correlation):")
    print("-" * 80)
    print(f"{'Признак':<25} {'Пирсон':<12} {'Distance':<12} {'Разница':<12} {'Примечание'}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        note = row['note']
        print(f"{row['feature']:<25} {row['pearson']:<12.4f} {row['distance_corr']:<12.4f} "
              f"{row['difference']:<12.4f} {note}")

    # Визуализация сравнения
    fig, ax = plt.subplots(figsize=(12, max(6, len(input_columns) * 0.3)))

    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(x - width / 2, results_df['pearson'].abs().values, width,
           label='|Корреляция Пирсона|', color='steelblue', alpha=0.7)
    ax.bar(x + width / 2, results_df['distance_corr'].values, width,
           label='Distance Correlation', color='coral', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(results_df['feature'].values, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Сила зависимости', fontsize=12)
    ax.set_title('Сравнение: линейная корреляция vs универсальная distance correlation',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, 'correlation_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📁 График сравнения сохранен: {save_path}")

    plt.show()
    plt.close(fig)

    return results_df


def plot_raw_data(df, input_columns, output_columns, save_folder=None):
    n_cols = len(df.columns)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, column in enumerate(df.columns):
        ax = axes[idx]

        if column in input_columns:
            color = 'blue'
            data_type = 'Входные'
        elif column in output_columns:
            color = 'red'
            data_type = 'Выходные'
        else:
            color = 'green'
            data_type = 'Другие'

        try:
            data = pd.to_numeric(df[column], errors='coerce')
            if data.isna().all():
                ax.text(0.5, 0.5, f'Столбец "{column}"\nне содержит числовых данных',
                        ha='center', va='center', transform=ax.transAxes)
            else:
                ax.plot(df.index, data, color=color, alpha=0.7, linewidth=1.5, marker='.', markersize=2)
                ax.set_title(f'{column}\n({data_type})', fontsize=10, fontweight='bold')
                ax.set_xlabel('Индекс строки')
                ax.set_ylabel('Значение')
                ax.grid(True, alpha=0.3)

                mean_val = data.mean()
                std_val = data.std()
                ax.text(0.02, 0.95, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                        transform=ax.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            ax.text(0.5, 0.5, f'Ошибка: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{column}\n({data_type})', fontsize=10, fontweight='bold')

    for idx in range(len(df.columns), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Сырые данные (без границ)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, 'all_raw_plots.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Графики сырых данных сохранены в: {save_path}")

    # Только сохраняем, не показываем
    plt.close(fig)


def plot_correlation_heatmap(df, input_columns, output_columns, save_folder=None, title="Тепловая карта корреляций"):
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) == 0:
        print("ОШИБКА: Нет числовых данных для построения корреляционной матрицы")
        return None

    corr_matrix = numeric_df.corr()
    n_features = len(corr_matrix.columns)
    fig_size = max(10, min(20, n_features * 0.6))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Коэффициент корреляции"},
                ax=ax, annot_kws={'size': max(6, min(10, 14 - n_features * 0.3))})

    for tick in ax.get_xticklabels():
        col_name = tick.get_text()
        if col_name in input_columns:
            tick.set_color('blue')
            tick.set_weight('bold')
        elif col_name in output_columns:
            tick.set_color('red')
            tick.set_weight('bold')

    for tick in ax.get_yticklabels():
        col_name = tick.get_text()
        if col_name in input_columns:
            tick.set_color('blue')
            tick.set_weight('bold')
        elif col_name in output_columns:
            tick.set_color('red')
            tick.set_weight('bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, 'correlation_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Тепловая карта сохранена в: {save_path}")

    # Только сохраняем, не показываем
    plt.close(fig)
    return corr_matrix


def plot_single_column(df, column, data_type, lower_bound=None, upper_bound=None, mean_val=None, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    color = 'blue' if data_type == 'Входные' else 'red'

    try:
        data = pd.to_numeric(df[column], errors='coerce')

        if data.isna().all():
            ax.text(0.5, 0.5, f'Столбец "{column}"\nне содержит числовых данных',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{column}\n({data_type})', fontsize=12, fontweight='bold')
        else:
            if mean_val is None:
                mean_val = data.mean()

            if lower_bound is None:
                lower_bound = mean_val * 0.5
            if upper_bound is None:
                upper_bound = mean_val * 1.5

            ax.plot(df.index, data, color=color, alpha=0.7, linewidth=1.5, marker='.', markersize=2, label='Данные')
            ax.axhline(y=mean_val, color='black', linestyle='--', alpha=0.5, linewidth=1.5,
                       label=f'Среднее = {mean_val:.2f}')
            ax.axhline(y=lower_bound, color='orange', linestyle=':', alpha=0.7, linewidth=1.5,
                       label=f'Нижняя граница = {lower_bound:.2f}')
            ax.axhline(y=upper_bound, color='orange', linestyle=':', alpha=0.7, linewidth=1.5,
                       label=f'Верхняя граница = {upper_bound:.2f}')
            ax.fill_between(df.index, lower_bound, upper_bound, alpha=0.1, color='green')

            outlier_mask = (data < lower_bound) | (data > upper_bound)
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                ax.scatter(df.index[outlier_mask], data[outlier_mask],
                           color='red', s=50, label=f'Выбросы ({outlier_count})', zorder=5)

            ax.set_title(f'{column} ({data_type})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Индекс строки', fontsize=12)
            ax.set_ylabel('Значение', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)

            std_val = data.std()
            outlier_percent = (outlier_count / len(data)) * 100
            stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}\nВыбросы: {outlier_count} ({outlier_percent:.1f}%)'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    except Exception as e:
        ax.text(0.5, 0.5, f'Ошибка: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{column}\n({data_type})', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ График сохранен: {save_path}")

    plt.show()
    plt.close(fig)

    return lower_bound, upper_bound, mean_val


def plot_all_columns(df, bounds_config, input_columns, output_columns, save_folder=None):
    n_cols = len(df.columns)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, column in enumerate(df.columns):
        ax = axes[idx]

        if column in input_columns:
            color = 'blue'
            data_type = 'Входные'
        elif column in output_columns:
            color = 'red'
            data_type = 'Выходные'
        else:
            color = 'green'
            data_type = 'Другие'

        try:
            data = pd.to_numeric(df[column], errors='coerce')

            if data.isna().all():
                ax.text(0.5, 0.5, f'Столбец "{column}"\nне содержит числовых данных',
                        ha='center', va='center', transform=ax.transAxes)
            else:
                if column in bounds_config:
                    mean_val = bounds_config[column]['mean']
                    lower_bound = bounds_config[column]['lower']
                    upper_bound = bounds_config[column]['upper']
                else:
                    mean_val = data.mean()
                    lower_bound = mean_val * 0.5
                    upper_bound = mean_val * 1.5

                ax.plot(df.index, data, color=color, alpha=0.7, linewidth=1.5, marker='.', markersize=2)
                ax.axhline(y=mean_val, color='black', linestyle='--', alpha=0.5, linewidth=1)
                ax.axhline(y=lower_bound, color='orange', linestyle=':', alpha=0.7, linewidth=1)
                ax.axhline(y=upper_bound, color='orange', linestyle=':', alpha=0.7, linewidth=1)
                ax.fill_between(df.index, lower_bound, upper_bound, alpha=0.1, color='green')

                outlier_mask = (data < lower_bound) | (data > upper_bound)
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    ax.scatter(df.index[outlier_mask], data[outlier_mask],
                               color='red', s=20, zorder=5)

                ax.set_title(f'{column}\n({data_type})', fontsize=10, fontweight='bold')
                ax.set_xlabel('Индекс строки')
                ax.set_ylabel('Значение')
                ax.grid(True, alpha=0.3)

                std_val = data.std()
                ax.text(0.02, 0.95, f'μ={mean_val:.2f}\nσ={std_val:.2f}\nВыбросы: {outlier_count}',
                        transform=ax.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        except Exception as e:
            ax.text(0.5, 0.5, f'Ошибка: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{column}\n({data_type})', fontsize=10, fontweight='bold')

    for idx in range(len(df.columns), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Все графики с текущими границами', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, 'all_plots_current.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Общий график сохранен в: {save_path}")

    plt.show()
    plt.close(fig)


def plot_correlation_with_target(df, target_columns, input_columns, save_folder=None):
    numeric_df = df.select_dtypes(include=[np.number])

    for target in target_columns:
        if target not in numeric_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        correlations = {}
        for col in input_columns:
            if col in numeric_df.columns and col != target:
                corr = numeric_df[col].corr(numeric_df[target])
                correlations[col] = corr

        if not correlations:
            print(f"Нет данных для построения корреляций с {target}")
            plt.close()
            continue

        sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
        colors = ['red' if v < 0 else 'green' for v in sorted_corr.values()]
        bars = ax.bar(range(len(sorted_corr)), sorted_corr.values(), color=colors, alpha=0.7)

        for i, (col, corr) in enumerate(sorted_corr.items()):
            ax.text(i, corr + (0.02 if corr >= 0 else -0.08),
                    f'{corr:.3f}', ha='center', va='bottom' if corr >= 0 else 'top',
                    fontsize=9)

        ax.set_xticks(range(len(sorted_corr)))
        ax.set_xticklabels(sorted_corr.keys(), rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Коэффициент корреляции Пирсона', fontsize=12)
        ax.set_xlabel('Входные признаки', fontsize=12)
        ax.set_title(f'Корреляция входных признаков с {target}', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Сильная положительная (0.5)')
        ax.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5, label='Сильная отрицательная (-0.5)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            safe_name = target.replace('/', '_').replace('\\', '_').replace(':', '_')
            save_path = os.path.join(save_folder, f'correlation_with_{safe_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ График корреляций для {target} сохранен в: {save_path}")

        plt.close(fig)
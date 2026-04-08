import matplotlib
matplotlib.use('TkAgg')  # Для отображения графиков с границами
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns


def plot_raw_data(df, input_columns, output_columns, save_folder=None):
    """Построение графиков сырых данных без границ (только сохранение, без показа)"""
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
    """Построение тепловой карты корреляций (только сохранение, без показа)"""
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
    """Построение графика для одного столбца (показывается при очистке)"""
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

    # Показываем график (нужен при интерактивной настройке)
    plt.show()
    plt.close(fig)

    return lower_bound, upper_bound, mean_val


def plot_all_columns(df, bounds_config, input_columns, output_columns, save_folder=None):
    """Построение всех графиков с текущими границами (показывается)"""
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

    # Показываем график (нужен при интерактивной настройке)
    plt.show()
    plt.close(fig)


def plot_correlation_with_target(df, target_columns, input_columns, save_folder=None):
    """Построение графиков корреляции входных признаков с целевыми переменными (только сохранение)"""
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

        # Только сохраняем, не показываем
        plt.close(fig)
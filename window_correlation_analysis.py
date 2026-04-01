import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')


def find_best_window(df, input_columns, output_columns, min_window_size=2000):
    """
    Находит лучшее окно данных по средней корреляции

    Parameters:
    - df: DataFrame с данными
    - input_columns: список входных столбцов
    - output_columns: список выходных столбцов
    - min_window_size: минимальный размер окна

    Returns:
    - best_window: информация о лучшем окне
    - window_data: данные лучшего окна
    """
    if not output_columns:
        print("Нет выходных столбцов для анализа")
        return None, None

    # Берем первый выходной столбец
    target = output_columns[0]

    # Подготавливаем данные
    data = df[input_columns + [target]].copy().dropna()

    if len(data) < min_window_size:
        print(f"Недостаточно данных (нужно минимум {min_window_size} строк)")
        return None, None

    total_rows = len(data)
    print(f"\nВсего строк: {total_rows}")
    print(f"Минимальный размер окна: {min_window_size}")

    # Определяем размеры окон
    window_sizes = [min_window_size]
    if total_rows >= min_window_size * 1.5:
        window_sizes.append(int(min_window_size * 1.5))
    if total_rows >= min_window_size * 2:
        window_sizes.append(min_window_size * 2)

    # Шаги сдвига
    step_sizes = [int(min_window_size * 0.2), int(min_window_size * 0.3)]
    step_sizes = [s for s in step_sizes if s > 0]

    print(f"Размеры окон: {window_sizes}")
    print(f"Шаги сдвига: {step_sizes}")

    # Базовая корреляция
    base_corr = np.mean([data[col].corr(data[target]) for col in input_columns])
    print(f"\nБазовая средняя корреляция: {base_corr:.4f}")

    # Поиск лучшего окна
    best_window = None
    best_score = -np.inf

    for window_size in window_sizes:
        for step in step_sizes:
            start = 0
            while start + window_size <= total_rows:
                end = start + window_size
                window_data = data.iloc[start:end]

                # Средняя корреляция в окне
                mean_corr = np.mean([window_data[col].corr(window_data[target]) for col in input_columns])

                if mean_corr > best_score:
                    best_score = mean_corr
                    best_window = {
                        'start_row': start,
                        'end_row': end,
                        'window_size': window_size,
                        'step': step,
                        'mean_correlation': mean_corr,
                        'improvement': mean_corr - base_corr
                    }

                start += step

    if best_window:
        print("\n" + "=" * 80)
        print("ЛУЧШЕЕ ОКНО ДАННЫХ")
        print("=" * 80)
        print(f"Строки: {best_window['start_row']} - {best_window['end_row']}")
        print(f"Размер окна: {best_window['window_size']}")
        print(f"Средняя корреляция: {best_window['mean_correlation']:.4f}")
        print(f"Улучшение: +{best_window['improvement']:.4f}")

        window_data = data.iloc[best_window['start_row']:best_window['end_row']]
        return best_window, window_data

    return None, None


def plot_best_window_heatmap(df, best_window, input_columns, output_columns, save_folder=None):
    """Строит тепловую карту для лучшего окна"""
    if best_window is None:
        return

    target = output_columns[0]
    start = best_window['start_row']
    end = best_window['end_row']
    window_data = df.iloc[start:end]

    # Корреляционная матрица
    corr_matrix = window_data[input_columns + [target]].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap=cmap,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Коэффициент корреляции"},
                ax=ax,
                annot_kws={'size': 9})

    # Выделяем входные и выходные столбцы
    for tick in ax.get_xticklabels():
        col_name = tick.get_text()
        if col_name in input_columns:
            tick.set_color('blue')
            tick.set_weight('bold')
        elif col_name == target:
            tick.set_color('red')
            tick.set_weight('bold')

    for tick in ax.get_yticklabels():
        col_name = tick.get_text()
        if col_name in input_columns:
            tick.set_color('blue')
            tick.set_weight('bold')
        elif col_name == target:
            tick.set_color('red')
            tick.set_weight('bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)

    ax.set_title(f'Тепловая карта корреляций\nЛучшее окно (строки {start}-{end})',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, 'best_window_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_window_raw_data(df, best_window, input_columns, output_columns, save_folder=None):
    if best_window is None:
        return

    target = output_columns[0]
    start = best_window['start_row']
    end = best_window['end_row']
    window_data = df.iloc[start:end]

    # Все столбцы для отображения
    all_cols = input_columns + [target]
    n_cols = len(all_cols)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, column in enumerate(all_cols):
        ax = axes[idx]

        if column in input_columns:
            color = 'blue'
            data_type = 'Входные'
        else:
            color = 'red'
            data_type = 'Выходные'

        try:
            data = pd.to_numeric(window_data[column], errors='coerce')
            if data.isna().all():
                ax.text(0.5, 0.5, f'Столбец "{column}"\nне содержит числовых данных',
                        ha='center', va='center', transform=ax.transAxes)
            else:
                ax.plot(window_data.index, data, color=color, alpha=0.7, linewidth=1.5, marker='.', markersize=2)
                ax.set_title(f'{column}\n({data_type})', fontsize=10, fontweight='bold')
                ax.set_xlabel('Индекс строки')
                ax.set_ylabel('Значение')
                ax.grid(True, alpha=0.3)

                mean_val = data.mean()
                std_val = data.std()
                ax.axhline(y=mean_val, color='black', linestyle='--', alpha=0.5, linewidth=1)
                ax.text(0.02, 0.95, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                        transform=ax.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            ax.text(0.5, 0.5, f'Ошибка: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{column}\n({data_type})', fontsize=10, fontweight='bold')

    # Убираем лишние графики
    for idx in range(len(all_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Сырые данные лучшего окна (строки {start}-{end})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_folder:
        save_path = os.path.join(save_folder, 'best_window_raw_data.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_best_window_data(df, best_window, input_columns, output_columns, save_folder):
    if best_window is None:
        return None

    start = best_window['start_row']
    end = best_window['end_row']
    window_data = df.iloc[start:end]

    # Сохраняем CSV
    csv_file = os.path.join(save_folder, f'best_window_rows_{start}_{end}.csv')
    window_data.to_csv(csv_file, index=False, encoding='utf-8-sig')

    # Сохраняем мета-информацию
    info_file = os.path.join(save_folder, f'best_window_info_{start}_{end}.txt')
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("ИНФОРМАЦИЯ О ЛУЧШЕМ ОКНЕ\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("ПАРАМЕТРЫ ОКНА:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Строки: {start} - {end}\n")
        f.write(f"Размер окна: {best_window['window_size']}\n")
        f.write(f"Шаг сдвига: {best_window['step']}\n")
        f.write(f"Средняя корреляция: {best_window['mean_correlation']:.6f}\n")
        f.write(f"Улучшение: +{best_window['improvement']:.6f}\n\n")

        f.write("КОРРЕЛЯЦИИ В ОКНЕ:\n")
        f.write("-" * 40 + "\n")
        target = output_columns[0] if output_columns else None
        if target:
            for col in input_columns:
                corr = window_data[col].corr(window_data[target])
                f.write(f"{col} → {target}: {corr:.6f}\n")

        f.write("\nСТАТИСТИКА:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Строк: {len(window_data)}\n")
        f.write(f"Столбцов: {len(window_data.columns)}\n\n")

        f.write("ВХОДНЫЕ ПРИЗНАКИ:\n")
        for col in input_columns:
            f.write(f"  {col}: μ={window_data[col].mean():.6f}, σ={window_data[col].std():.6f}\n")

        f.write("\nВЫХОДНЫЕ ПРИЗНАКИ:\n")
        for col in output_columns:
            f.write(f"  {col}: μ={window_data[col].mean():.6f}, σ={window_data[col].std():.6f}\n")

    return csv_file
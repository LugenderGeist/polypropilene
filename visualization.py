import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_single_column(df, column, data_type, lower_bound=None, upper_bound=None, mean_val=None, save_path=None):
    """
    Построение графика для одного столбца
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Определяем цвет
    color = 'blue' if data_type == 'Входные' else 'red'

    try:
        # Преобразуем в числовой формат
        data = pd.to_numeric(df[column], errors='coerce')

        if data.isna().all():
            ax.text(0.5, 0.5, f'Столбец "{column}"\nне содержит числовых данных',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{column}\n({data_type})', fontsize=12, fontweight='bold')
        else:
            # Вычисляем среднее если не передано
            if mean_val is None:
                mean_val = data.mean()

            # Вычисляем границы если не переданы
            if lower_bound is None:
                lower_bound = mean_val * 0.5
            if upper_bound is None:
                upper_bound = mean_val * 1.5

            # Строим график
            ax.plot(df.index, data, color=color, alpha=0.7, linewidth=1.5, marker='.', markersize=2, label='Данные')

            # Добавляем линию среднего значения
            ax.axhline(y=mean_val, color='black', linestyle='--', alpha=0.5, linewidth=1.5,
                       label=f'Среднее = {mean_val:.2f}')

            # Добавляем границы
            ax.axhline(y=lower_bound, color='orange', linestyle=':', alpha=0.7, linewidth=1.5,
                       label=f'Нижняя граница = {lower_bound:.2f}')
            ax.axhline(y=upper_bound, color='orange', linestyle=':', alpha=0.7, linewidth=1.5,
                       label=f'Верхняя граница = {upper_bound:.2f}')

            # Закрашиваем область между границами
            ax.fill_between(df.index, lower_bound, upper_bound, alpha=0.1, color='green')

            # Подсвечиваем точки за пределами границ
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                ax.scatter(df.index[outlier_mask], data[outlier_mask],
                           color='red', s=50, label=f'Выбросы ({outlier_count})', zorder=5)

            # Настройка графика
            ax.set_title(f'{column} ({data_type})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Индекс строки', fontsize=12)
            ax.set_ylabel('Значение', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)

            # Добавляем статистику
            std_val = data.std()
            outlier_percent = (outlier_count / len(data)) * 100

            stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}\nВыбросы: {outlier_count} ({outlier_percent:.1f}%)'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    except Exception as e:
        ax.text(0.5, 0.5, f'Ошибка: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{column}\n({data_type})', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Сохраняем график если указан путь
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {save_path}")

    plt.show()

    return lower_bound, upper_bound, mean_val


def plot_all_columns(df, bounds_config, input_columns, output_columns, save_folder=None):
    """Построение всех графиков с текущими границами"""
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
                # Получаем границы из конфигурации
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

                # Подсвечиваем выбросы
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

    # Сохраняем общий график если указана папка
    if save_folder:
        save_path = os.path.join(save_folder, 'all_plots_current.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Общий график сохранен: {save_path}")

    plt.show()


def save_initial_plots(df, input_columns, output_columns, save_folder):
    """Сохраняет начальные графики с границами ±50%"""
    print("\n" + "=" * 80)
    print("СОХРАНЕНИЕ НАЧАЛЬНЫХ ГРАФИКОВ")
    print("=" * 80)

    # Создаем подпапку для начальных графиков
    initial_folder = os.path.join(save_folder, 'initial_plots')
    if not os.path.exists(initial_folder):
        os.makedirs(initial_folder)

    # Сохраняем общий график всех столбцов
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

    plt.suptitle('НАЧАЛЬНЫЕ ГРАФИКИ (границы ±50% от среднего)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Сохраняем общий график
    all_plots_path = os.path.join(initial_folder, 'all_initial_plots.png')
    plt.savefig(all_plots_path, dpi=300, bbox_inches='tight')
    print(f"Общий начальный график сохранен: {all_plots_path}")
    plt.show()

    # Сохраняем отдельные графики для каждого столбца
    print("\nСохранение отдельных графиков...")
    for column in df.columns:
        if column in input_columns:
            data_type = 'Входные'
        elif column in output_columns:
            data_type = 'Выходные'
        else:
            continue

        try:
            data = pd.to_numeric(df[column], errors='coerce')
            if not data.isna().all():
                fig, ax = plt.subplots(figsize=(12, 6))

                color = 'blue' if data_type == 'Входные' else 'red'
                mean_val = data.mean()
                lower_bound = mean_val * 0.5
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

                ax.set_title(f'{column} ({data_type}) - Начальные границы ±50%', fontsize=14, fontweight='bold')
                ax.set_xlabel('Индекс строки', fontsize=12)
                ax.set_ylabel('Значение', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=10)

                std_val = data.std()
                outlier_percent = (outlier_count / len(data)) * 100
                stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}\nВыбросы: {outlier_count} ({outlier_percent:.1f}%)'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                plt.tight_layout()

                # Сохраняем отдельный график
                safe_name = column.replace('/', '_').replace('\\', '_').replace(':', '_')
                single_path = os.path.join(initial_folder, f'{safe_name}_initial.png')
                plt.savefig(single_path, dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"Не удалось сохранить график для {column}: {e}")

    print(f"Все начальные графики сохранены в папку: {initial_folder}")

def plot_comparison_before_after(df_before, df_after, column, data_type, bounds_config, save_path=None):
    """Строит сравнительные графики до и после удаления выбросов"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    color = 'blue' if data_type == 'Входные' else 'red'

    # График до удаления
    data_before = pd.to_numeric(df_before[column], errors='coerce')
    config = bounds_config[column]

    ax1.plot(df_before.index, data_before, color=color, alpha=0.7, linewidth=1.5, marker='.', markersize=2)
    ax1.axhline(y=config['mean'], color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.axhline(y=config['lower'], color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
    ax1.axhline(y=config['upper'], color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
    ax1.fill_between(df_before.index, config['lower'], config['upper'], alpha=0.1, color='green')

    outlier_mask_before = (data_before < config['lower']) | (data_before > config['upper'])
    if outlier_mask_before.sum() > 0:
        ax1.scatter(df_before.index[outlier_mask_before], data_before[outlier_mask_before],
                    color='red', s=30, label=f'Выбросы ({outlier_mask_before.sum()})', zorder=5)

    ax1.set_title(f'{column} ({data_type}) - ДО УДАЛЕНИЯ', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Индекс строки')
    ax1.set_ylabel('Значение')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График после удаления
    data_after = pd.to_numeric(df_after[column], errors='coerce')

    ax2.plot(df_after.index, data_after, color=color, alpha=0.7, linewidth=1.5, marker='.', markersize=2)
    ax2.axhline(y=config['mean'], color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.axhline(y=config['lower'], color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
    ax2.axhline(y=config['upper'], color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
    ax2.fill_between(df_after.index, config['lower'], config['upper'], alpha=0.1, color='green')

    outlier_mask_after = (data_after < config['lower']) | (data_after > config['upper'])
    if outlier_mask_after.sum() > 0:
        ax2.scatter(df_after.index[outlier_mask_after], data_after[outlier_mask_after],
                    color='red', s=30, label=f'Выбросы ({outlier_mask_after.sum()})', zorder=5)

    ax2.set_title(f'{column} ({data_type}) - ПОСЛЕ УДАЛЕНИЯ', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Индекс строки')
    ax2.set_ylabel('Значение')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f'Сравнение данных: {column}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Сравнительный график сохранен: {save_path}")

    plt.show()
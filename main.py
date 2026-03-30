import os
import pandas as pd
import numpy as np
from utils import (load_data, create_plots_folder, setup_columns,
                   save_bounds_config, remove_outliers, save_cleaned_data)
from visualization import (plot_all_columns, save_initial_plots,
                           plot_correlation_heatmap, plot_correlation_with_target)
from interactive_menu import interactive_bounds_adjustment


def print_data_info(df):
    """Выводит информацию о данных"""
    print("\n" + "=" * 60)
    print("ИНФОРМАЦИЯ О ДАННЫХ")
    print("=" * 60)
    print(f"Форма данных: {df.shape}")
    print(f"Столбцы: {list(df.columns)}")
    print(f"\nПервые 5 строк:")
    print(df.head())
    print(f"\nТипы данных:")
    print(df.dtypes)


def print_final_statistics(df_original, df_cleaned, bounds_config, all_columns, removed_indices):
    """Выводит итоговую статистику по выбросам"""
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)

    print(f"\nИсходное количество строк: {len(df_original)}")
    print(f"Количество строк после удаления: {len(df_cleaned)}")
    print(f"Удалено строк: {len(removed_indices)} ({len(removed_indices) / len(df_original) * 100:.2f}%)")

    if len(removed_indices) > 0:
        print(f"\nУдаленные индексы: {removed_indices[:20]}" +
              ("..." if len(removed_indices) > 20 else ""))

    # Статистика по каждому столбцу
    print("\n" + "=" * 80)
    print("СТАТИСТИКА ПО СТОЛБЦАМ ПОСЛЕ УДАЛЕНИЯ")
    print("=" * 80)

    for col in all_columns:
        if col in bounds_config:
            try:
                data_original = pd.to_numeric(df_original[col], errors='coerce')
                data_cleaned = pd.to_numeric(df_cleaned[col], errors='coerce')
                config = bounds_config[col]

                mean_before = data_original.mean()
                mean_after = data_cleaned.mean()
                std_before = data_original.std()
                std_after = data_cleaned.std()

                print(f"\n{col} ({config['data_type']}):")
                print(f"  Границы: [{config['lower']:.4f}, {config['upper']:.4f}]")
                print(f"  До удаления - Среднее: {mean_before:.4f}, Стд: {std_before:.4f}")
                print(f"  После удаления - Среднее: {mean_after:.4f}, Стд: {std_after:.4f}")
                if mean_before != 0:
                    print(f"  Изменение среднего: {((mean_after - mean_before) / mean_before * 100):.2f}%")
                if std_before != 0:
                    print(f"  Изменение стд: {((std_after - std_before) / std_before * 100):.2f}%")

            except Exception as e:
                print(f"\n{col}: Ошибка - {str(e)}")


def main():
    # Загрузка данных
    print("=" * 60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)

    file_path = 'ПП.csv'
    df, encoding = load_data(file_path)

    if df is None:
        return

    # Вывод информации о данных
    print_data_info(df)

    # Настройка входных и выходных столбцов
    input_columns, output_columns = setup_columns(df)
    all_data_columns = input_columns + output_columns

    # Создаем папку для сохранения графиков
    plots_folder = create_plots_folder()
    print(f"\nСоздана папка для сохранения графиков: {plots_folder}")

    # Сохраняем начальные графики
    save_initial_plots(df, input_columns, output_columns, plots_folder)

    # Строим тепловую карту корреляций для исходных данных
    print("\n" + "=" * 80)
    print("ТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИЙ (ИСХОДНЫЕ ДАННЫЕ)")
    print("=" * 80)
    input("Нажмите Enter, чтобы построить тепловую карту корреляций...")

    correlation_folder = os.path.join(plots_folder, 'correlation_analysis')
    if not os.path.exists(correlation_folder):
        os.makedirs(correlation_folder)

    # Общая тепловая карта
    plot_correlation_heatmap(df, input_columns, output_columns,
                             save_folder=correlation_folder,
                             title="Тепловая карта корреляций (исходные данные)")

    # Корреляции входных с выходными
    if output_columns:
        plot_correlation_with_target(df, output_columns, input_columns,
                                     save_folder=correlation_folder)

    # Интерактивная настройка границ
    print("\n" + "=" * 60)
    print("ИНТЕРАКТИВНАЯ НАСТРОЙКА ГРАНИЦ")
    print("=" * 60)
    print("Теперь вы можете итеративно настраивать границы для каждого столбца.")
    print("Начнем с границ ±50% от среднего значения.")
    print("Вы сможете:")
    print("  - Изменять нижнюю или верхнюю границу по отдельности")
    print("  - Просматривать отдельные графики")
    print("  - Сохранять настроенные графики")
    print("  - Просматривать все графики с текущими настройками")
    print("  - Настраивать каждый столбец индивидуально")

    input("\nНажмите Enter, чтобы начать настройку...")

    # Запускаем интерактивную настройку
    bounds_config = interactive_bounds_adjustment(df, all_data_columns, input_columns, output_columns, plots_folder)

    # Показываем финальные графики до удаления
    print("\n" + "=" * 80)
    print("ФИНАЛЬНЫЕ ГРАФИКИ С НАСТРОЕННЫМИ ГРАНИЦАМИ (ДО УДАЛЕНИЯ)")
    print("=" * 80)
    input("Нажмите Enter, чтобы показать финальные графики до удаления...")

    # Создаем папку для финальных графиков
    final_folder = os.path.join(plots_folder, 'final_plots_before_removal')
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)

    plot_all_columns(df, bounds_config, input_columns, output_columns, final_folder)

    # Спрашиваем, нужно ли удалить выбросы
    print("\n" + "=" * 80)
    print("УДАЛЕНИЕ ВЫБРОСОВ")
    print("=" * 80)
    remove_choice = input("Хотите удалить строки с выбросами (точки за пределами границ)? (да/нет): ").strip().lower()

    if remove_choice in ['да', 'yes', 'y', 'д']:
        # Удаляем выбросы
        df_cleaned, removed_indices, removal_report = remove_outliers(df, bounds_config, all_data_columns)

        if len(removed_indices) > 0:
            # Строим тепловую карту для очищенных данных
            print("\n" + "=" * 80)
            print("ТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИЙ (ОЧИЩЕННЫЕ ДАННЫЕ)")
            print("=" * 80)
            input("Нажмите Enter, чтобы построить тепловую карту для очищенных данных...")

            plot_correlation_heatmap(df_cleaned, input_columns, output_columns,
                                     save_folder=correlation_folder,
                                     title="Тепловая карта корреляций (очищенные данные)")

            # Сохраняем очищенные данные
            save_choice = input("\nСохранить очищенные данные в файл? (да/нет): ").strip().lower()
            if save_choice in ['да', 'yes', 'y', 'д']:
                cleaned_path, info_path = save_cleaned_data(df_cleaned, 'ПП.csv', plots_folder)

                # Сохраняем информацию об удалении
                with open(info_path, 'w', encoding='utf-8') as f:
                    f.write("ИНФОРМАЦИЯ ОБ УДАЛЕНИИ ВЫБРОСОВ\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Исходное количество строк: {len(df)}\n")
                    f.write(f"Удалено строк: {len(removed_indices)}\n")
                    f.write(f"Осталось строк: {len(df_cleaned)}\n\n")
                    f.write("Удаленные индексы:\n")
                    f.write(str(removed_indices) + "\n\n")
                    f.write("Детали по столбцам:\n")
                    for col, report in removal_report.items():
                        f.write(f"\n{col}:\n")
                        f.write(f"  Количество выбросов: {report['outlier_count']}\n")
                        f.write(f"  Процент выбросов: {report['outlier_percent']:.2f}%\n")
                        f.write(f"  Индексы выбросов: {report['indices']}\n")

                print(f"Информация об удалении сохранена в: {info_path}")

            # Показываем графики после удаления
            print("\n" + "=" * 80)
            print("ГРАФИКИ ПОСЛЕ УДАЛЕНИЯ ВЫБРОСОВ")
            print("=" * 80)
            input("Нажмите Enter, чтобы показать графики после удаления...")

            after_removal_folder = os.path.join(plots_folder, 'final_plots_after_removal')
            if not os.path.exists(after_removal_folder):
                os.makedirs(after_removal_folder)

            plot_all_columns(df_cleaned, bounds_config, input_columns, output_columns, after_removal_folder)

            # Выводим итоговую статистику
            print_final_statistics(df, df_cleaned, bounds_config, all_data_columns, removed_indices)

            # Сохраняем конфигурацию границ
            save_config = input("\nСохранить конфигурацию границ в файл? (да/нет): ").strip().lower()
            if save_config in ['да', 'yes', 'y', 'д']:
                config_file = save_bounds_config(bounds_config, plots_folder)
                print(f"Конфигурация сохранена в файл: {config_file}")

            print("\n" + "=" * 60)
            print(f"ВСЕ ГРАФИКИ СОХРАНЕНЫ В ПАПКУ: {plots_folder}")
            print("ГОТОВО!")
            print("=" * 60)

        else:
            print("\nВыбросов не обнаружено. Очистка не требуется.")
    else:
        # Сохраняем конфигурацию границ без удаления
        save_config = input("\nСохранить конфигурацию границ в файл? (да/нет): ").strip().lower()
        if save_config in ['да', 'yes', 'y', 'д']:
            config_file = save_bounds_config(bounds_config, plots_folder)
            print(f"Конфигурация сохранена в файл: {config_file}")

        print("\n" + "=" * 60)
        print(f"ВСЕ ГРАФИКИ СОХРАНЕНЫ В ПАПКУ: {plots_folder}")
        print("ГОТОВО!")
        print("=" * 60)


if __name__ == "__main__":
    main()
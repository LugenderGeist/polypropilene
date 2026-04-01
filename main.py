import os
import pandas as pd
from interactive_menu import interactive_bounds_adjustment
from utils import (load_data, create_plots_folder, setup_columns,
                   save_bounds_config, remove_outliers, save_cleaned_data)
from visualization import (plot_all_columns, plot_correlation_heatmap)
from window_correlation_analysis import (find_best_window, plot_best_window_heatmap,
                                         plot_window_raw_data)


def print_final_statistics(df_original, df_cleaned, bounds_config, all_columns, removed_indices):
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

    # Настройка входных и выходных столбцов
    input_columns, output_columns = setup_columns(df)
    all_data_columns = input_columns + output_columns

    # Создаем папку для сохранения графиков
    plots_folder = create_plots_folder()
    print(f"\nСоздана папка для сохранения графиков: {plots_folder}")

    # ============= 1. СЫРЫЕ ДАННЫЕ БЕЗ ГРАНИЦ =============
    print("\n" + "=" * 80)
    print("1. СЫРЫЕ ДАННЫЕ")
    print("=" * 80)
    input("Нажмите Enter, чтобы показать графики сырых данных...")

    # Создаем папку для сырых графиков
    raw_plots_folder = os.path.join(plots_folder, 'raw_plots')
    if not os.path.exists(raw_plots_folder):
        os.makedirs(raw_plots_folder)

    # Функция plot_raw_data должна быть добавлена в visualization.py
    # Она строит графики без границ
    from visualization import plot_raw_data
    plot_raw_data(df, input_columns, output_columns, save_folder=raw_plots_folder)

    # ============= 3. ТЕПЛОВАЯ КАРТА =============
    print("\n" + "=" * 80)
    print("3. ТЕПЛОВАЯ КАРТА")
    print("=" * 80)
    input("Нажмите Enter, чтобы построить тепловую карту...")

    correlation_folder = os.path.join(plots_folder, 'correlation_analysis')
    if not os.path.exists(correlation_folder):
        os.makedirs(correlation_folder)

    # Общая тепловая карта
    plot_correlation_heatmap(df, input_columns, output_columns,
                             save_folder=correlation_folder,
                             title="Тепловая карта")

    # ============= 4. ПОИСК ЛУЧШЕГО ОКНА =============
    print("\n" + "=" * 80)
    print("4. ПОИСК ЛУЧШЕГО ОКНА ДАННЫХ")
    print("=" * 80)
    print("Этот анализ поможет найти участок данных, где корреляции между")
    print("входными и выходными параметрами наиболее сильные.")
    print("Будет найдено окно размером от 2000 строк с максимальной средней корреляцией.")

    search_window = input("\nВыполнить поиск лучшего окна данных? (да/нет): ").strip().lower()

    if search_window in ['да', 'yes', 'y', 'д']:
        print("\n" + "=" * 80)
        print("ПОИСК ЛУЧШЕГО ОКНА ДАННЫХ")
        print("=" * 80)

        best_window, best_window_data = find_best_window(df, input_columns, output_columns, min_window_size=2000)

        if best_window is not None:
            # Тепловая карта для лучшего окна
            plot_best_window_heatmap(df, best_window, input_columns, output_columns, save_folder=correlation_folder)

            # Графики сырых данных для лучшего окна
            plot_window_raw_data(df, best_window, input_columns, output_columns, save_folder=correlation_folder)

            # Сохраняем лучшее окно в CSV
            start = best_window['start_row']
            end = best_window['end_row']
            window_file = os.path.join(plots_folder, f'best_window_rows_{start}_{end}.csv')
            best_window_data.to_csv(window_file, index=False, encoding='utf-8-sig')
            print(f"\nДанные лучшего окна сохранены в: {window_file}")

            # Информация о лучшем окне
            info_file = os.path.join(correlation_folder, 'best_window_info.txt')
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write("ИНФОРМАЦИЯ О ЛУЧШЕМ ОКНЕ\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Строки: {start} - {end}\n")
                f.write(f"Размер окна: {best_window['window_size']}\n")
                f.write(f"Средняя корреляция: {best_window['mean_correlation']:.4f}\n")
                f.write(f"Улучшение относительно всех данных: +{best_window['improvement']:.4f}\n")
            print(f"Информация сохранена в: {info_file}")
        else:
            print("\nНе удалось найти подходящее окно. Возможно, недостаточно данных.")
    else:
        print("\nПоиск лучшего окна пропущен.")

        # Создаем временную конфигурацию с границами 50%
        temp_config = {}
        for col in all_data_columns:
            try:
                data = pd.to_numeric(df[col], errors='coerce')
                if not data.isna().all():
                    mean_val = data.mean()
                    temp_config[col] = {
                        'mean': mean_val,
                        'lower': mean_val * 0.5,
                        'upper': mean_val * 1.5,
                        'data_type': 'Входные' if col in input_columns else 'Выходные'
                    }
            except:
                pass

        # Создаем папку для начальных графиков с границами
        initial_plots_folder = os.path.join(plots_folder, 'initial_plots')
        if not os.path.exists(initial_plots_folder):
            os.makedirs(initial_plots_folder)

    # ============= 5. ЧИСТКА ДАННЫХ =============
    print("\n" + "=" * 80)
    print("5. ВЫБОР МЕТОДОВ ДЛЯ ЧИСТКИ ДАННЫХ")
    print("=" * 80)
    print("Вы можете вручную выбрать границы, по которым можно очистить данные,")
    print("или воспользоваться одним из фильтров.")

    input("\nНажмите Enter, чтобы начать чистку...")

    # Запускаем интерактивную настройку
    bounds_config = interactive_bounds_adjustment(df, all_data_columns, input_columns, output_columns, plots_folder)

    # ============= 6. ОЧИЩЕННЫЕ ДАННЫЕ =============
    print("\n" + "=" * 80)
    print("6. ВЫДЕЛЕНИЕ ИСКЛЮЧАЕМЫХ ДАННЫХ")
    print("=" * 80)
    input("Нажмите Enter, чтобы показать графики после применения фильтров...")

    final_folder = os.path.join(plots_folder, 'final_plots_before_removal')
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)

    plot_all_columns(df, bounds_config, input_columns, output_columns, final_folder)

    # ============= 7. УДАЛЕНИЕ ВЫБРОСОВ =============
    print("\n" + "=" * 80)
    print("7. УДАЛЕНИЕ ВЫБРОСОВ")
    print("=" * 80)
    remove_choice = input("Хотите удалить отфильтрованные данные? (да/нет): ").strip().lower()

    if remove_choice in ['да', 'yes', 'y', 'д']:
        # Удаляем выбросы
        df_cleaned, removed_indices, removal_report = remove_outliers(df, bounds_config, all_data_columns)

        if len(removed_indices) > 0:
            # ============= 8. ОЧИЩЕННЫЕ ДАННЫЕ =============
            print("\n" + "=" * 80)
            print("8. ОЧИЩЕННЫЕ ДАННЫЕ")
            print("=" * 80)
            input("Нажмите Enter, чтобы показать графики очищенных данных...")

            after_removal_folder = os.path.join(plots_folder, 'final_plots_after_removal')
            if not os.path.exists(after_removal_folder):
                os.makedirs(after_removal_folder)

            plot_all_columns(df_cleaned, bounds_config, input_columns, output_columns, after_removal_folder)

            # Сохраняем очищенные данные
            save_choice = input("\nСохранить очищенные данные в файл? (да/нет): ").strip().lower()
            if save_choice in ['да', 'yes', 'y', 'д']:
                cleaned_path, info_path = save_cleaned_data(df_cleaned, 'ПП_part.csv', plots_folder)

                # Сохраняем информацию об удалении
                with open(info_path, 'w', encoding='utf-8') as f:
                    f.write("ИНФОРМАЦИЯ ОБ УДАЛЕНИИ ВЫБРОСОВ\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Исходное количество строк: {len(df)}\n")
                    f.write(f"Удалено строк: {len(removed_indices)}\n")
                    f.write(str(removed_indices) + "\n\n")
                    f.write("Детали по столбцам:\n")
                    for col, report in removal_report.items():
                        f.write(f"\n{col}:\n")
                        f.write(f"  Количество выбросов: {report['outlier_count']}\n")
                        f.write(f"  Процент выбросов: {report['outlier_percent']:.2f}%\n")

                print(f"Информация об удалении сохранена в: {info_path}")

            # Выводим итоговую статистику
            print_final_statistics(df, df_cleaned, bounds_config, all_data_columns, removed_indices)

            # Сохраняем конфигурацию границ
            save_config = input("\nСохранить конфигурацию границ в файл? (да/нет): ").strip().lower()
            if save_config in ['да', 'yes', 'y', 'д']:
                config_file = save_bounds_config(bounds_config, plots_folder)
                print(f"Конфигурация сохранена в файл: {config_file}")

        else:
            print("\nВыбросов не обнаружено. Очистка не требуется.")
            save_config = input("\nСохранить конфигурацию границ в файл? (да/нет): ").strip().lower()
            if save_config in ['да', 'yes', 'y', 'д']:
                config_file = save_bounds_config(bounds_config, plots_folder)
                print(f"Конфигурация сохранена в файл: {config_file}")
    else:
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
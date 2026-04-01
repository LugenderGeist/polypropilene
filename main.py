import os
import pandas as pd
import numpy as np
from utils import (load_data, create_plots_folder, setup_columns,
                   save_bounds_config, remove_outliers, save_cleaned_data)
from visualization import (plot_all_columns, plot_raw_data, save_initial_plots,
                           plot_correlation_heatmap, plot_correlation_with_target)
from interactive_menu import interactive_bounds_adjustment
from window_correlation_analysis import (find_best_window, plot_best_window_heatmap,
                                         plot_window_raw_data, save_best_window_data)


def main():
    # ============= ЗАГРУЗКА ДАННЫХ =============
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

    # Создаем основную папку для сохранения графиков
    main_plots_folder = create_plots_folder()

    # ============= 1. ИСХОДНЫЕ ДАННЫЕ =============
    print("\n" + "=" * 80)
    print("1. ИСХОДНЫЕ ДАННЫЕ")
    print("=" * 80)

    # Создаем папку для исходных данных
    raw_data_folder = os.path.join(main_plots_folder, '01_raw_data')
    if not os.path.exists(raw_data_folder):
        os.makedirs(raw_data_folder)

    print("\n1.1. Графики сырых данных")
    input("Нажмите Enter, чтобы показать графики...")
    plot_raw_data(df, input_columns, output_columns, save_folder=raw_data_folder)

    print("\n1.2. Тепловая карта корреляций")
    input("Нажмите Enter, чтобы построить тепловую карту...")
    plot_correlation_heatmap(df, input_columns, output_columns,
                             save_folder=raw_data_folder,
                             title="Тепловая карта корреляций (исходные данные)")

    # ============= 2. ВЫБОР РЕЖИМА РАБОТЫ =============
    print("\n" + "=" * 80)
    print("2. ВЫБОР РЕЖИМА РАБОТЫ")
    print("=" * 80)
    print("Выберите режим:")
    print("1. Полная обработка данных (настройка границ, фильтрация выбросов)")
    print("2. Быстрый режим (пропустить обработку, сразу к поиску окна и модели)")

    mode_choice = input("\nВаш выбор (1/2): ").strip()

    # Инициализация переменных
    bounds_config = None
    df_processed = df.copy()
    processed_data_folder = None

    if mode_choice == '1':
        # ============= ПОЛНЫЙ РЕЖИМ: ОБРАБОТКА ДАННЫХ =============
        print("\n" + "=" * 80)
        print("ПОЛНЫЙ РЕЖИМ: ОБРАБОТКА ДАННЫХ")
        print("=" * 80)

        # Создаем папку для обработанных данных
        processed_data_folder = os.path.join(main_plots_folder, '02_processed_data')
        if not os.path.exists(processed_data_folder):
            os.makedirs(processed_data_folder)

        print("\n2.1. Графики с границами ±50%")
        input("Нажмите Enter, чтобы показать графики...")

        # Показываем графики с границами 50%
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

        plot_all_columns(df, temp_config, input_columns, output_columns,
                         save_folder=processed_data_folder)

        # Интерактивная настройка границ и фильтрация
        bounds_config = interactive_bounds_adjustment(df, all_data_columns, input_columns, output_columns,
                                                      processed_data_folder)

        # Показываем графики с новыми границами
        print("\n2.2. Графики с настроенными границами")
        input("Нажмите Enter, чтобы показать графики...")

        plot_all_columns(df, bounds_config, input_columns, output_columns,
                         save_folder=processed_data_folder)

        # Тепловая карта после настройки границ
        print("\n2.3. Тепловая карта после настройки границ")
        input("Нажмите Enter, чтобы построить тепловую карту...")

        plot_correlation_heatmap(df, input_columns, output_columns,
                                 save_folder=processed_data_folder,
                                 title="Тепловая карта корреляций (после настройки границ)")

        # Удаление выбросов
        print("\n2.4. Удаление выбросов")
        remove_choice = input(
            "Хотите удалить строки с выбросами (точки за пределами границ)? (да/нет): ").strip().lower()

        if remove_choice in ['да', 'yes', 'y', 'д']:
            df_cleaned, removed_indices, removal_report = remove_outliers(df, bounds_config, all_data_columns)

            if len(removed_indices) > 0:
                df_processed = df_cleaned

                print("\n2.5. Графики очищенных данных")
                input("Нажмите Enter, чтобы показать графики...")

                plot_all_columns(df_cleaned, bounds_config, input_columns, output_columns,
                                 save_folder=processed_data_folder)

                print(f"\nУдалено строк: {len(removed_indices)}")
                print(f"Осталось строк: {len(df_cleaned)}")
            else:
                print("\nВыбросов не обнаружено. Очистка не требуется.")

        # Сохраняем конфигурацию границ
        if bounds_config:
            save_config = input("\nСохранить конфигурацию границ в файл? (да/нет): ").strip().lower()
            if save_config in ['да', 'yes', 'y', 'д']:
                config_file = save_bounds_config(bounds_config, processed_data_folder)
                print(f"Конфигурация сохранена в файл: {config_file}")

        print(f"\n✅ Все графики обработанных данных сохранены в: {processed_data_folder}")

    elif mode_choice == '2':
        # ============= БЫСТРЫЙ РЕЖИМ: ПРОПУСК ОБРАБОТКИ =============
        print("\n" + "=" * 80)
        print("БЫСТРЫЙ РЕЖИМ: ПРОПУСК ОБРАБОТКИ ДАННЫХ")
        print("=" * 80)
        print("Вы пропустили этапы настройки границ и фильтрации выбросов.")
        print("Данные будут использованы в исходном виде.")

        df_processed = df.copy()

        # Создаем базовую конфигурацию границ для отображения (если понадобится)
        bounds_config = {}
        for col in all_data_columns:
            try:
                data = pd.to_numeric(df[col], errors='coerce')
                if not data.isna().all():
                    mean_val = data.mean()
                    bounds_config[col] = {
                        'mean': mean_val,
                        'lower': mean_val * 0.5,
                        'upper': mean_val * 1.5,
                        'data_type': 'Входные' if col in input_columns else 'Выходные'
                    }
            except:
                pass

        print("\n✅ Переход к поиску лучшего окна и построению модели...")

    else:
        print("Неверный выбор. Используется полный режим по умолчанию.")
        mode_choice = '1'
        df_processed = df.copy()
        bounds_config = {}
        for col in all_data_columns:
            try:
                data = pd.to_numeric(df[col], errors='coerce')
                if not data.isna().all():
                    mean_val = data.mean()
                    bounds_config[col] = {
                        'mean': mean_val,
                        'lower': mean_val * 0.5,
                        'upper': mean_val * 1.5,
                        'data_type': 'Входные' if col in input_columns else 'Выходные'
                    }
            except:
                pass

    # ============= 3. ПОИСК ЛУЧШЕГО ОКНА =============
    print("\n" + "=" * 80)
    print("3. ПОИСК ЛУЧШЕГО ОКНА ДАННЫХ")
    print("=" * 80)

    search_window = input("\nВыполнить поиск лучшего окна данных? (да/нет): ").strip().lower()

    best_window = None
    best_window_data = None
    best_window_folder = None

    if search_window in ['да', 'yes', 'y', 'д']:
        # Создаем папку для лучшего окна
        best_window_folder = os.path.join(main_plots_folder, '03_best_window')
        if not os.path.exists(best_window_folder):
            os.makedirs(best_window_folder)

        print("\n" + "=" * 80)
        print("ПОИСК ЛУЧШЕГО ОКНА ДАННЫХ")
        print("=" * 80)

        best_window, best_window_data = find_best_window(df_processed, input_columns, output_columns,
                                                         min_window_size=2000)

        if best_window is not None:
            start = best_window['start_row']
            end = best_window['end_row']

            # Сохраняем данные и графики окна
            save_best_window_data(df_processed, best_window, input_columns, output_columns, best_window_folder)

            # Визуализация
            plot_best_window_heatmap(df_processed, best_window, input_columns, output_columns,
                                     save_folder=best_window_folder)
            plot_window_raw_data(df_processed, best_window, input_columns, output_columns,
                                 save_folder=best_window_folder)

            print(f"\n✅ Все данные и графики лучшего окна сохранены в: {best_window_folder}")
        else:
            print("\n❌ Не удалось найти подходящее окно. Возможно, недостаточно данных.")
    else:
        print("\nПоиск лучшего окна пропущен.")

    # ============= 4. ПОСТРОЕНИЕ МОДЕЛИ =============
    # Создаем папку для результатов моделирования
    modeling_folder = os.path.join(main_plots_folder, '04_modeling_results')
    if not os.path.exists(modeling_folder):
        os.makedirs(modeling_folder)

    # Цикл выбора модели
    while True:
        print("\n" + "=" * 80)
        print("4. ПОСТРОЕНИЕ МОДЕЛИ")
        print("=" * 80)

        # Выбор данных для моделирования
        if best_window is not None:
            use_window = input(
                f"\nИспользовать лучшее окно (строки {best_window['start_row']}-{best_window['end_row']}) для моделирования? (да/нет): ").strip().lower()
            if use_window in ['да', 'yes', 'y', 'д']:
                data_for_model = best_window_data
                print(f"\nМодель будет построена на данных лучшего окна ({len(data_for_model)} строк)")
            else:
                data_for_model = df_processed
                print(f"\nМодель будет построена на всех данных ({len(data_for_model)} строк)")
        else:
            data_for_model = df_processed
            print(f"\nМодель будет построена на всех данных ({len(data_for_model)} строк)")

        print("\nВыберите действие:")
        print("1. Обучить Random Forest")
        print("2. Обучить XGBoost")
        print("3. Завершить работу")

        model_choice = input("\nВаш выбор (1/2/3): ").strip()

        if model_choice == '1':
            # Random Forest
            try:
                from modeling import build_random_forest_model

                print("\n" + "=" * 80)
                print("ПОСТРОЕНИЕ МОДЕЛИ RANDOM FOREST")
                print("=" * 80)

                results, model, importance = build_random_forest_model(
                    data_for_model, input_columns, output_columns,
                    save_folder=modeling_folder,
                    test_size=0.2,
                    random_state=42
                )

                if results:
                    print(f"\n✅ Модель Random Forest обучена")
                    print(f"   Результаты сохранены в: {modeling_folder}")

            except ImportError as e:
                print(f"Ошибка импорта: {e}")
            except Exception as e:
                print(f"Ошибка при построении модели: {e}")

        elif model_choice == '2':
            # XGBoost
            try:
                from modeling import build_xgboost_model

                print("\n" + "=" * 80)
                print("ПОСТРОЕНИЕ МОДЕЛИ XGBOOST")
                print("=" * 80)

                results, model, importance = build_xgboost_model(
                    data_for_model, input_columns, output_columns,
                    save_folder=modeling_folder,
                    test_size=0.2,
                    random_state=42
                )

                if results:
                    print(f"\n✅ Модель XGBoost обучена")
                    print(f"   Результаты сохранены в: {modeling_folder}")

            except ImportError as e:
                print(f"Ошибка импорта: {e}")
                print("Для XGBoost выполните: pip install xgboost")
            except Exception as e:
                print(f"Ошибка при построении модели: {e}")

        elif model_choice == '3':
            print("\nЗавершение работы с моделями...")
            break

        else:
            print("Неверный выбор. Пожалуйста, выберите 1, 2 или 3.")

    # ============= ИТОГОВАЯ ИНФОРМАЦИЯ =============
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТРУКТУРА СОХРАНЕННЫХ ДАННЫХ")
    print("=" * 60)
    print(f"\nОсновная папка: {main_plots_folder}")
    print(f"  ├── 01_raw_data/              # Исходные данные (графики, тепловая карта)")
    if processed_data_folder:
        print(f"  ├── 02_processed_data/        # Обработанные данные (графики с границами)")
    if best_window_folder:
        print(f"  ├── 03_best_window/           # Лучшее окно (данные CSV, графики)")
    print(f"  └── 04_modeling_results/       # Результаты моделирования")

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)


if __name__ == "__main__":
    main()
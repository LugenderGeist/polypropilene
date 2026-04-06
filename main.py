import os
import pandas as pd
import numpy as np
from config import (INPUT_FILE, MIN_WINDOW_SIZE, DEFAULT_BOUNDS_PERCENT,
                    OPTIMIZATION_TOP_FEATURES)
from utils import (load_data, create_plots_folder, setup_columns,
                   save_bounds_config, remove_outliers, save_cleaned_data)
from visualization import (plot_all_columns, plot_raw_data,
                           plot_correlation_heatmap, plot_correlation_with_target)
from interactive_menu import interactive_bounds_adjustment
from window_correlation_analysis import (find_best_window, plot_best_window_heatmap,
                                         plot_window_raw_data, save_best_window_data)


def main():
    # ============= ЗАГРУЗКА ДАННЫХ =============
    print("=" * 60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)

    df, encoding = load_data(INPUT_FILE)

    if df is None:
        return

    print(f"\n📊 Всего столбцов в файле: {df.shape[1]}")

    # Настройка входных и выходных столбцов
    input_columns, output_columns = setup_columns(df)
    all_data_columns = input_columns + output_columns

    # Создаем основную папку для сохранения
    main_plots_folder = create_plots_folder()

    # ============= 1. ИСХОДНЫЕ ДАННЫЕ =============
    print("\n" + "=" * 80)
    print("1. ИСХОДНЫЕ ДАННЫЕ")
    print("=" * 80)

    raw_data_folder = os.path.join(main_plots_folder, '01_raw_data')
    os.makedirs(raw_data_folder, exist_ok=True)

    print("\n1.1. Графики сырых данных")
    plot_raw_data(df, input_columns, output_columns, save_folder=raw_data_folder)

    print("\n1.2. Тепловая карта корреляций")
    plot_correlation_heatmap(df, input_columns, output_columns,
                             save_folder=raw_data_folder,
                             title="Тепловая карта корреляций (исходные данные)")

    # Сохраняем исходные данные
    original_csv_path = os.path.join(raw_data_folder, 'original_data.csv')
    df.to_csv(original_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📁 Исходные данные сохранены в: {original_csv_path}")

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
    cleaned_data_saved = False

    if mode_choice == '1':
        # ============= ПОЛНЫЙ РЕЖИМ: ОБРАБОТКА ДАННЫХ =============
        print("\n" + "=" * 80)
        print("ПОЛНЫЙ РЕЖИМ: ОБРАБОТКА ДАННЫХ")
        print("=" * 80)

        # Создаем папку для обработанных данных
        processed_data_folder = os.path.join(main_plots_folder, '02_processed_data')
        os.makedirs(processed_data_folder, exist_ok=True)

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
                        'lower': mean_val * (1 - DEFAULT_BOUNDS_PERCENT / 100),
                        'upper': mean_val * (1 + DEFAULT_BOUNDS_PERCENT / 100),
                        'data_type': 'Входные' if col in input_columns else 'Выходные'
                    }
            except:
                pass

        plot_all_columns(df, temp_config, input_columns, output_columns,
                         save_folder=processed_data_folder)

        # Интерактивная настройка границ и фильтрация
        bounds_config = interactive_bounds_adjustment(df, all_data_columns, input_columns, output_columns,
                                                      processed_data_folder)

        # Удаление выбросов
        print("\n2.2. Удаление выбросов")
        remove_choice = input(
            "Хотите удалить строки с выбросами (точки за пределами границ)? (да/нет): ").strip().lower()

        if remove_choice in ['да', 'yes', 'y', 'д']:
            df_cleaned, removed_indices, removal_report = remove_outliers(df, bounds_config, all_data_columns)

            if len(removed_indices) > 0:
                df_processed = df_cleaned

                print("\n2.3. Графики очищенных данных")
                input("Нажмите Enter, чтобы показать графики...")

                plot_all_columns(df_cleaned, bounds_config, input_columns, output_columns,
                                 save_folder=processed_data_folder)

                print(f"\nУдалено строк: {len(removed_indices)}")
                print(f"Осталось строк: {len(df_cleaned)}")

                # ============= ТЕПЛОВАЯ КАРТА ДЛЯ ОЧИЩЕННЫХ ДАННЫХ =============
                print("\n2.4. Тепловая карта корреляций (очищенные данные)")
                input("Нажмите Enter, чтобы построить тепловую карту для очищенных данных...")

                plot_correlation_heatmap(df_cleaned, input_columns, output_columns,
                                         save_folder=processed_data_folder,
                                         title="Тепловая карта корреляций (очищенные данные)")

                # ============= СОХРАНЕНИЕ ОЧИЩЕННЫХ ДАННЫХ =============
                print("\n" + "=" * 80)
                print("СОХРАНЕНИЕ ОЧИЩЕННЫХ ДАННЫХ")
                print("=" * 80)

                # Сохраняем очищенные данные
                cleaned_path, info_path = save_cleaned_data(df_cleaned, 'ПП.csv', processed_data_folder)
                cleaned_data_saved = True

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

                print(f"📁 Очищенные данные сохранены в: {cleaned_path}")
                print(f"📄 Информация об удалении сохранена в: {info_path}")

            else:
                print("\nВыбросов не обнаружено. Очистка не требуется.")
                save_current = input(
                    "\nСохранить текущие данные (с настроенными границами) в CSV? (да/нет): ").strip().lower()
                if save_current in ['да', 'yes', 'y', 'д']:
                    current_path = os.path.join(processed_data_folder, 'data_with_bounds.csv')
                    df.to_csv(current_path, index=False, encoding='utf-8-sig')
                    print(f"📁 Данные сохранены в: {current_path}")
                    cleaned_data_saved = True
        else:
            # Пользователь не захотел удалять выбросы
            save_current = input(
                "\nСохранить текущие данные (с настроенными границами) в CSV? (да/нет): ").strip().lower()
            if save_current in ['да', 'yes', 'y', 'д']:
                current_path = os.path.join(processed_data_folder, 'data_with_bounds.csv')
                df.to_csv(current_path, index=False, encoding='utf-8-sig')
                print(f"📁 Данные сохранены в: {current_path}")
                cleaned_data_saved = True

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

        print("\nСохранить исходные данные в отдельный файл?")
        save_original = input("(да/нет): ").strip().lower()
        if save_original in ['да', 'yes', 'y', 'д']:
            quick_folder = os.path.join(main_plots_folder, 'quick_mode_data')
            os.makedirs(quick_folder, exist_ok=True)
            quick_path = os.path.join(quick_folder, 'original_data.csv')
            df.to_csv(quick_path, index=False, encoding='utf-8-sig')
            print(f"📁 Исходные данные сохранены в: {quick_path}")

        df_processed = df.copy()

        # Создаем базовую конфигурацию границ для отображения
        bounds_config = {}
        for col in all_data_columns:
            try:
                data = pd.to_numeric(df[col], errors='coerce')
                if not data.isna().all():
                    mean_val = data.mean()
                    bounds_config[col] = {
                        'mean': mean_val,
                        'lower': mean_val * (1 - DEFAULT_BOUNDS_PERCENT / 100),
                        'upper': mean_val * (1 + DEFAULT_BOUNDS_PERCENT / 100),
                        'data_type': 'Входные' if col in input_columns else 'Выходные'
                    }
            except:
                pass

        print("\n✅ Переход к поиску лучшего окна и построению модели...")

    else:
        print("Неверный выбор. Используется полный режим по умолчанию.")
        df_processed = df.copy()
        bounds_config = {}
        for col in all_data_columns:
            try:
                data = pd.to_numeric(df[col], errors='coerce')
                if not data.isna().all():
                    mean_val = data.mean()
                    bounds_config[col] = {
                        'mean': mean_val,
                        'lower': mean_val * (1 - DEFAULT_BOUNDS_PERCENT / 100),
                        'upper': mean_val * (1 + DEFAULT_BOUNDS_PERCENT / 100),
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
        best_window_folder = os.path.join(main_plots_folder, '03_best_window')
        os.makedirs(best_window_folder, exist_ok=True)

        print("\n" + "=" * 80)
        print("ПОИСК ЛУЧШЕГО ОКНА ДАННЫХ")
        print("=" * 80)

        best_window, best_window_data = find_best_window(df_processed, input_columns, output_columns,
                                                         min_window_size=MIN_WINDOW_SIZE)

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

    # ============= 4. ПОСТРОЕНИЕ МОДЕЛЕЙ И ОПТИМИЗАЦИЯ =============
    modeling_folder = os.path.join(main_plots_folder, '04_modeling_results')
    os.makedirs(modeling_folder, exist_ok=True)

    # Выбор данных для моделирования
    if best_window is not None:
        use_window = input(
            f"\nИспользовать лучшее окно (строки {best_window['start_row']}-{best_window['end_row']}) для моделирования? (да/нет): ").strip().lower()
        if use_window in ['да', 'yes', 'y', 'д']:
            data_for_model = best_window_data
            print(f"\n✅ Модели будут построены на данных лучшего окна ({len(data_for_model)} строк)")
        else:
            data_for_model = df_processed
            print(f"\n✅ Модели будут построены на всех данных ({len(data_for_model)} строк)")
    else:
        data_for_model = df_processed
        print(f"\n✅ Модели будут построены на всех данных ({len(data_for_model)} строк)")

    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)

    results_dict = {}
    models_dict = {}

    # 1. Random Forest
    try:
        from modeling import build_random_forest_model
        results, model, _ = build_random_forest_model(
            data_for_model, input_columns, output_columns,
            save_folder=modeling_folder
        )
        if results:
            results_dict['Random Forest'] = results
            models_dict['Random Forest'] = model
    except Exception as e:
        print(f"❌ Ошибка Random Forest: {e}")

    # 2. XGBoost
    try:
        from modeling import build_xgboost_model
        results, model, _ = build_xgboost_model(
            data_for_model, input_columns, output_columns,
            save_folder=modeling_folder
        )
        if results:
            results_dict['XGBoost'] = results
            models_dict['XGBoost'] = model
    except Exception as e:
        print(f"❌ Ошибка XGBoost: {e}")

    # 3. Нейросеть
    try:
        from modeling import build_mlp_model
        results, model, _ = build_mlp_model(
            data_for_model, input_columns, output_columns,
            save_folder=modeling_folder
        )
        if results:
            results_dict['MLP'] = results
            models_dict['MLP'] = model
    except Exception as e:
        print(f"❌ Ошибка нейросети: {e}")

    # Сравнение моделей
    if results_dict:
        from modeling import compare_models
        best_model_name, best_r2 = compare_models(results_dict, save_folder=modeling_folder)

        # Выбор модели для оптимизации
        print("\n" + "=" * 80)
        print("ОПТИМИЗАЦИЯ")
        print("=" * 80)

        print("\nДоступные модели:")
        for i, name in enumerate(results_dict.keys(), 1):
            print(f"  {i}. {name} (R² = {results_dict[name]['r2_test']:.4f})")
        print(f"  {len(results_dict) + 1}. Использовать лучшую ({best_model_name})")

        opt_choice = input(f"\nВыберите модель для оптимизации (1-{len(results_dict) + 1}): ").strip()

        try:
            opt_idx = int(opt_choice) - 1
            if opt_idx == len(results_dict):
                selected_model_name = best_model_name
            elif 0 <= opt_idx < len(results_dict):
                selected_model_name = list(results_dict.keys())[opt_idx]
            else:
                selected_model_name = best_model_name
        except:
            selected_model_name = best_model_name

        selected_model = models_dict[selected_model_name]
        print(f"\n✅ Для оптимизации выбрана: {selected_model_name}")

        # Выбор признаков для оптимизации
        print("\nВыберите признаки для оптимизации:")
        print("  1. Только наиболее важные (из config.py)")
        print("  2. Все входные параметры")

        feat_choice = input("Ваш выбор (1/2): ").strip()
        use_all_features = feat_choice == '2'

        # Запуск оптимизации
        optimize_choice = input(
            f"\nПровести оптимизацию для максимизации {output_columns[0]}? (да/нет): ").strip().lower()

        if optimize_choice in ['да', 'yes', 'y', 'д']:
            try:
                from optimization import run_optimization
                from config import OPTIMIZATION_TOP_FEATURES

                optimization_folder = os.path.join(modeling_folder, 'optimization')
                os.makedirs(optimization_folder, exist_ok=True)

                n_features = None if use_all_features else OPTIMIZATION_TOP_FEATURES

                opt_result = run_optimization(
                    df_original=data_for_model,
                    model=selected_model,
                    input_columns=input_columns,
                    output_columns=output_columns,
                    n_top_features=n_features,
                    save_folder=optimization_folder
                )

                if opt_result:
                    print(f"\n✅ Оптимизация завершена!")
                    print(f"   Лучшее значение: {opt_result['best_fitness']:.4f}")

            except Exception as e:
                print(f"Ошибка при оптимизации: {e}")
    else:
        print("\n❌ Не удалось обучить ни одну модель!")
if __name__ == "__main__":
    main()
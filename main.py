import os
import pandas as pd
import json
from datetime import datetime

# Импорты из config
from config import RESULTS_DIR, INPUT_FILE, MIN_WINDOW_SIZE, DEFAULT_BOUNDS_PERCENT

# Импорты из utils
from utils.utils import (load_data, setup_columns,
                              remove_outliers, save_cleaned_data)
from utils.visualization import (plot_all_columns, plot_raw_data,
                                       plot_correlation_heatmap)

# Импорты из preprocessing
from preprocessing.interactive_menu import interactive_bounds_adjustment
from preprocessing.window_analysis import (find_best_window, plot_best_window_heatmap,
                                           plot_window_raw_data, save_best_window_data)

# Импорты из modeling
from modeling.modeling import (build_random_forest_model, build_xgboost_model,
                             build_catboost_model, compare_models)
from modeling.hyperopt import (optimize_random_forest, optimize_xgboost, optimize_catboost,
                               plot_optimization_history, load_best_params_from_json)
from modeling.optimization import run_optimization
from modeling.generation import generate_samples

import matplotlib
matplotlib.use('TkAgg')

def create_plots_folder():
    folder_name = f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    folder_path = os.path.join(RESULTS_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_optimized_params_to_json(params_dict, save_folder):
    json_path = os.path.join(save_folder, 'optimized_params.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params_dict, f, indent=4, ensure_ascii=False)
    print(f"📁 Оптимизированные параметры сохранены: {json_path}")
    return json_path


def main():
    # ============= ЗАГРУЗКА ДАННЫХ =============
    print("=" * 60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)

    df, encoding = load_data(INPUT_FILE)

    if df is None:
        return

    print(f"\n📊 Форма данных: {df.shape}")
    print(f"📊 Всего столбцов: {df.shape[1]}")
    print(f"📊 Всего строк: {df.shape[0]}")

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

    print("\nГрафики сырых данных сохраняются в папке проекта")
    plot_raw_data(df, input_columns, output_columns, save_folder=raw_data_folder)

    print("\nТепловая карта корреляций сохраняется в папке проекта")
    plot_correlation_heatmap(df, input_columns, output_columns,
                             save_folder=raw_data_folder,
                             title="Тепловая карта корреляций (исходные данные)")

    # Сохраняем исходные данные
    original_csv_path = os.path.join(raw_data_folder, 'original_data.csv')
    df.to_csv(original_csv_path, index=False, encoding='utf-8-sig')

    # ============= ВЫБОР РЕЖИМА РАБОТЫ =============
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
        # ============= ОБРАБОТКА ДАННЫХ =============
        print("\n" + "=" * 80)
        print("ОБРАБОТКА ДАННЫХ")
        print("=" * 80)

        processed_data_folder = os.path.join(main_plots_folder, '02_processed_data')
        os.makedirs(processed_data_folder, exist_ok=True)

        print("\n2.1. Графики с границами ±50% сохраняются в папке проекта")

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

        bounds_config = interactive_bounds_adjustment(df, all_data_columns, input_columns, output_columns,
                                                      processed_data_folder)

        # Проверка выбросов
        print("\n" + "=" * 80)
        print("УДАЛЕНИЕ ВЫБРАННЫХ ДАННЫХ")
        print("=" * 80)

        outlier_info = {}
        has_outliers = False

        for col in all_data_columns:
            if col in bounds_config:
                try:
                    data = pd.to_numeric(df[col], errors='coerce')
                    config = bounds_config[col]
                    outlier_mask = (data < config['lower']) | (data > config['upper'])
                    outlier_count = outlier_mask.sum()
                    if outlier_count > 0:
                        has_outliers = True
                        outlier_info[col] = {'count': outlier_count, 'percent': (outlier_count / len(data)) * 100}
                except:
                    pass

        if has_outliers:
            remove_choice = input("\nУдалить строки с выбросами? (да/нет): ").strip().lower()

            if remove_choice in ['да', 'yes', 'y', 'д']:
                df_cleaned, removed_indices, removal_report = remove_outliers(df, bounds_config, all_data_columns)

                if len(removed_indices) > 0:
                    df_processed = df_cleaned
                    print(f"\n✅ Удалено строк: {len(removed_indices)}")
                    print(f"   Осталось строк: {len(df_cleaned)}")

                    print("\nГрафики после очистки сохранены в папке проекта")
                    plot_all_columns(df_cleaned, bounds_config, input_columns, output_columns,
                                     save_folder=processed_data_folder)

                    print("\nТепловая карта после очистки сохранена в папке проекта")
                    plot_correlation_heatmap(df_cleaned, input_columns, output_columns,
                                             save_folder=processed_data_folder,
                                             title="Тепловая карта корреляций (очищенные данные)")

                    cleaned_path, info_path = save_cleaned_data(df_cleaned, 'ПП.csv', processed_data_folder)
                else:
                    print("\nВыбросов не обнаружено.")
            else:
                print("\nОчистка данных пропущена.")
        else:
            print("\n✅ Выбросов за пределами границ не обнаружено.")

        print(f"\n✅ Обработка данных завершена. Результаты сохранены в: {processed_data_folder}")

    elif mode_choice == '2':
        print("Данные будут использованы в исходном виде.")
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
    else:
        print("Неверный выбор. Используется быстрый режим.")
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

    # ============= ПОИСК ЛУЧШЕГО ОКНА =============
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

        best_window, best_window_data = find_best_window(df_processed, input_columns, output_columns,
                                                         min_window_size=MIN_WINDOW_SIZE)

        if best_window is not None:
            start = best_window['start_row']
            end = best_window['end_row']

            print(f"\n✅ Найдено лучшее окно:")
            print(f"   Строки: {start} - {end}")
            print(f"   Размер: {best_window['window_size']} строк")
            print(f"   Средняя корреляция: {best_window['mean_correlation']:.4f}")

            save_best_window_data(df_processed, best_window, input_columns, output_columns, best_window_folder)
            plot_best_window_heatmap(df_processed, best_window, input_columns, output_columns,
                                     save_folder=best_window_folder)
            plot_window_raw_data(df_processed, best_window, input_columns, output_columns,
                                 save_folder=best_window_folder)

            print(f"\n📁 Данные лучшего окна сохранены в: {best_window_folder}")
        else:
            print("\n❌ Не удалось найти подходящее окно.")
    else:
        print("\nПоиск лучшего окна пропущен.")

    # ============= ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ =============
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

    # Спрашиваем про оптимизацию гиперпараметров
    print("\n" + "=" * 80)
    print("ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ (OPTUNA)")
    print("=" * 80)

    use_optuna = input("\nВыполнить поиск оптимальных гиперпараметров для моделей? (да/нет): ").strip().lower()

    optimized_params = {}
    optuna_folder = os.path.join(modeling_folder, 'optuna_results')

    if use_optuna in ['да', 'yes', 'y', 'д']:
        os.makedirs(optuna_folder, exist_ok=True)

        x_opt = data_for_model[input_columns].copy().fillna(data_for_model[input_columns].mean())
        y_opt = data_for_model[output_columns[0]].copy().fillna(data_for_model[output_columns[0]].mean())

        # Оптимизация Random Forest
        rf_choice = input("\n🌲 Оптимизировать Random Forest? (да/нет): ").strip().lower()
        if rf_choice in ['да', 'yes', 'y', 'д']:
            from config import OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS
            best_params, study = optimize_random_forest(
                x_opt, y_opt, n_trials=OPTUNA_N_TRIALS, cv_folds=OPTUNA_CV_FOLDS,
                save_folder=optuna_folder
            )
            optimized_params['Random Forest'] = best_params
            plot_optimization_history(study, "Random Forest", save_folder=optuna_folder)

        # Оптимизация XGBoost
        xgb_choice = input("\n🚀 Оптимизировать XGBoost? (да/нет): ").strip().lower()
        if xgb_choice in ['да', 'yes', 'y', 'д']:
            from config import OPTUNA_N_TRIALS
            best_params, study = optimize_xgboost(
                x_opt, y_opt, n_trials=OPTUNA_N_TRIALS,
                save_folder=optuna_folder
            )
            optimized_params['XGBoost'] = best_params
            plot_optimization_history(study, "XGBoost", save_folder=optuna_folder)

        # Оптимизация CatBoost (исправленный отступ - на одном уровне с XGBoost)
        cat_choice = input("\n🐱 Оптимизировать CatBoost? (да/нет): ").strip().lower()
        if cat_choice in ['да', 'yes', 'y', 'д']:
            from config import OPTUNA_N_TRIALS
            best_params, study = optimize_catboost(
                x_opt, y_opt, n_trials=OPTUNA_N_TRIALS,
                save_folder=optuna_folder
            )
            optimized_params['CatBoost'] = best_params
            plot_optimization_history(study, "CatBoost", save_folder=optuna_folder)

        if optimized_params:
            save_optimized_params_to_json(optimized_params, optuna_folder)
            print(f"\n✅ Оптимизация гиперпараметров завершена!")
            print(f"   Результаты сохранены в: {optuna_folder}")
    else:
        print("\nОптимизация гиперпараметров пропущена.")

    # ============= ОБУЧЕНИЕ МОДЕЛЕЙ =============
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)

    use_optimized = False
    if optimized_params:
        use_optimized_input = input("\nИспользовать оптимизированные параметры для моделей? (да/нет): ").strip().lower()
        use_optimized = use_optimized_input in ['да', 'yes', 'y', 'д']

    if use_optimized:
        print("\n✅ Будут использованы оптимизированные параметры")
    else:
        print("\n✅ Будут использованы параметры из config.py")

    results_dict = {}
    models_dict = {}

    # Random Forest
    try:
        if use_optimized and 'Random Forest' in optimized_params:
            from config import RF_PARAMS
            original_params = RF_PARAMS.copy()
            RF_PARAMS.update(optimized_params['Random Forest'])
            print("\n🌲 Random Forest с оптимизированными параметрами")

        results, model, _ = build_random_forest_model(
            data_for_model, input_columns, output_columns,
            save_folder=modeling_folder
        )
        if results:
            results_dict['Random Forest'] = results
            models_dict['Random Forest'] = model

        if use_optimized and 'Random Forest' in optimized_params:
            RF_PARAMS.clear()
            RF_PARAMS.update(original_params)
    except Exception as e:
        print(f"❌ Ошибка Random Forest: {e}")

    # XGBoost
    try:
        if use_optimized and 'XGBoost' in optimized_params:
            from config import XGB_PARAMS
            original_params = XGB_PARAMS.copy()
            XGB_PARAMS.update(optimized_params['XGBoost'])
            print("\n🚀 XGBoost с оптимизированными параметрами")

        results, model, _ = build_xgboost_model(
            data_for_model, input_columns, output_columns,
            save_folder=modeling_folder
        )
        if results:
            results_dict['XGBoost'] = results
            models_dict['XGBoost'] = model

        if use_optimized and 'XGBoost' in optimized_params:
            XGB_PARAMS.clear()
            XGB_PARAMS.update(original_params)
    except Exception as e:
        print(f"❌ Ошибка XGBoost: {e}")

    # CatBoost
    try:
        if use_optimized and 'CatBoost' in optimized_params:
            from config import CATBOOST_PARAMS
            original_params = CATBOOST_PARAMS.copy()
            CATBOOST_PARAMS.update(optimized_params['CatBoost'])
            print("\n🐱 CatBoost с оптимизированными параметрами")

        results, model, _ = build_catboost_model(
            data_for_model, input_columns, output_columns,
            save_folder=modeling_folder
        )
        if results:
            results_dict['CatBoost'] = results
            models_dict['CatBoost'] = model

        if use_optimized and 'CatBoost' in optimized_params:
            CATBOOST_PARAMS.clear()
            CATBOOST_PARAMS.update(original_params)
    except Exception as e:
        print(f"❌ Ошибка CatBoost: {e}")

    # Сравнение моделей
    if results_dict:
        best_model_name, best_r2 = compare_models(results_dict, save_folder=modeling_folder)

        # ============= ЦИКЛ ВЫБОРА ДЕЙСТВИЯ =============
        while True:
            print("\n" + "=" * 80)
            print("ВЫБОР ДЕЙСТВИЯ")
            print("=" * 80)
            print("Что вы хотите сделать?")
            print("1. Оптимизация (поиск лучших входных параметров)")
            print("2. Генерация наборов входных параметров")
            print("3. Завершить работу")

            action_choice = input("\nВаш выбор (1/2/3): ").strip()

            if action_choice == '1':
                # ============= ОПТИМИЗАЦИЯ =============
                print("\n" + "=" * 80)
                print("ОПТИМИЗАЦИЯ ВХОДНЫХ ПАРАМЕТРОВ")
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

                print("\nВыберите признаки для оптимизации:")
                print("  1. Только наиболее важные (из config.py)")
                print("  2. Все входные параметры")

                feat_choice = input("Ваш выбор (1/2): ").strip()
                use_all_features = feat_choice == '2'

                optimize_choice = input(
                    f"\nПровести оптимизацию для максимизации {output_columns[0]}? (да/нет): ").strip().lower()

                if optimize_choice in ['да', 'yes', 'y', 'д']:
                    from config import OPTIMIZATION_TOP_FEATURES

                    optimization_folder = os.path.join(main_plots_folder, '05_optimization')
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
                        print(f"   Результаты сохранены в: {optimization_folder}")

                    # После завершения оптимизации возвращаемся в меню
                    input("\nНажмите Enter, чтобы вернуться к выбору действия...")
                    continue
                else:
                    print("\nОптимизация отменена.")
                    continue

            elif action_choice == '2':
                # ============= ГЕНЕРАЦИЯ =============
                print("\n" + "=" * 80)
                print("ГЕНЕРАЦИЯ НАБОРОВ ВХОДНЫХ ПАРАМЕТРОВ")
                print("=" * 80)

                print("\nДоступные модели:")
                for i, name in enumerate(results_dict.keys(), 1):
                    print(f"  {i}. {name} (R² = {results_dict[name]['r2_test']:.4f})")
                print(f"  {len(results_dict) + 1}. Использовать лучшую ({best_model_name})")

                gen_choice = input(f"\nВыберите модель для генерации (1-{len(results_dict) + 1}): ").strip()

                try:
                    gen_idx = int(gen_choice) - 1
                    if gen_idx == len(results_dict):
                        selected_model_name = best_model_name
                    elif 0 <= gen_idx < len(results_dict):
                        selected_model_name = list(results_dict.keys())[gen_idx]
                    else:
                        selected_model_name = best_model_name
                except:
                    selected_model_name = best_model_name

                selected_model = models_dict[selected_model_name]
                print(f"\n✅ Для генерации выбрана: {selected_model_name}")

                from config import GENERATION_NUM_SAMPLES, GENERATION_METHOD

                print(f"\n📊 Параметры генерации (из config.py):")
                print(f"   - Количество наборов: {GENERATION_NUM_SAMPLES}")
                print(f"   - Метод: {GENERATION_METHOD}")

                # Выбор признаков для генерации
                print("\nВыберите признаки для генерации:")
                print("  1. Только наиболее важные (из config.py)")
                print("  2. Все входные параметры")

                feat_choice = input("Ваш выбор (1/2): ").strip()
                use_all_features = feat_choice == '2'

                if use_all_features:
                    print(f"   - Используются все {len(input_columns)} признаков")
                    n_features = None
                else:
                    from config import GENERATION_TOP_FEATURES
                    print(f"   - Используются топ-{GENERATION_TOP_FEATURES} важных признаков")
                    n_features = GENERATION_TOP_FEATURES

                generate_choice = input(f"\nЗапустить генерацию? (да/нет): ").strip().lower()

                if generate_choice in ['да', 'yes', 'y', 'д']:
                    generation_folder = os.path.join(main_plots_folder, '06_generation')
                    os.makedirs(generation_folder, exist_ok=True)

                    inputs_df, predictions_df = generate_samples(
                        df_original=data_for_model,
                        model=selected_model,
                        input_columns=input_columns,
                        output_columns=output_columns,
                        n_top_features=n_features,
                        n_samples=GENERATION_NUM_SAMPLES,
                        method=GENERATION_METHOD,
                        save_folder=generation_folder
                    )

                    print(f"\n✅ Генерация завершена!")
                    print(f"   Создано {len(inputs_df)} наборов")
                    print(f"   Результаты сохранены в: {generation_folder}")

                    # После завершения генерации возвращаемся в меню
                    input("\nНажмите Enter, чтобы вернуться к выбору действия...")
                    continue
                else:
                    print("\nГенерация отменена.")
                    continue
    else:
        print("\n❌ Не удалось обучить ни одну модель!")

    # ИТОГОВАЯ ИНФОРМАЦИЯ
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТРУКТУРА СОХРАНЕННЫХ ДАННЫХ")
    print("=" * 60)
    print(f"\nОсновная папка: {main_plots_folder}")

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)


if __name__ == "__main__":
    main()
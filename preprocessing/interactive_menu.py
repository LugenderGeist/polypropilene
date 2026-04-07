import os
import pandas as pd
from utils.visualization import plot_single_column, plot_all_columns
from preprocessing.outlier_filter import apply_outlier_filter, visualize_outlier_filter


def interactive_bounds_adjustment(df, all_columns, input_columns, output_columns, save_folder):
    """Интерактивное изменение границ для каждого столбца"""

    # Инициализация конфигурации границ
    bounds_config = {}
    for col in all_columns:
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

    # Создаем папку для сохранения настроенных графиков
    adjusted_folder = os.path.join(save_folder, 'adjusted_plots')
    if not os.path.exists(adjusted_folder):
        os.makedirs(adjusted_folder)

    # Создаем папку для фильтров
    filter_folder = os.path.join(save_folder, 'outlier_filters')
    if not os.path.exists(filter_folder):
        os.makedirs(filter_folder)

    # Интерактивный цикл изменения границ
    while True:
        print("\n" + "=" * 80)
        print("ЧИСТКА ДАННЫХ")
        print("=" * 80)
        print("\nДоступные столбцы для настройки:")

        # Показываем список столбцов с текущими границами
        for i, col in enumerate(all_columns, 1):
            if col in bounds_config:
                config = bounds_config[col]
                print(f"{i}. {col} ({config['data_type']}) - "
                      f"Нижняя: {config['lower']:.4f}, "
                      f"Верхняя: {config['upper']:.4f}, "
                      f"Среднее: {config['mean']:.4f}")

        # Выводим пункт "Показать все графики" перед завершением
        print(f"\n{len(all_columns) + 1}. Показать все графики с текущими настройками")
        print(f"0. Завершить настройку и показать финальные графики")

        # Выбор столбца
        try:
            choice = input("\nВыберите номер столбца для настройки (0 для выхода): ").strip()
            if choice == '0':
                break

            idx = int(choice) - 1

            # Проверяем, не выбран ли пункт "Показать все графики"
            if idx == len(all_columns):
                print("\nПоказываю все графики с текущими границами...")
                plot_all_columns(df, bounds_config, input_columns, output_columns, adjusted_folder)
                continue

            if 0 <= idx < len(all_columns):
                col = all_columns[idx]
                if col not in bounds_config:
                    print("Этот столбец не содержит числовых данных")
                    continue

                config = bounds_config[col]

                print(f"\nНастройка столбца: {col} ({config['data_type']})")
                print(f"Текущие значения:")
                print(f"  Среднее: {config['mean']:.4f}")
                print(f"  Нижняя граница: {config['lower']:.4f}")
                print(f"  Верхняя граница: {config['upper']:.4f}")

                print("\nВыберите действие:")
                print("1. Изменить нижнюю границу")
                print("2. Изменить верхнюю границу")
                print("3. Показать график этого столбца")
                print("4. Применить автоматический фильтр выбросов")
                print("5. Вернуться к выбору столбца")

                action = input("Ваш выбор (1-5): ").strip()

                if action == '1':
                    try:
                        new_lower = float(
                            input(f"Введите новое значение нижней границы (текущая: {config['lower']:.4f}): "))
                        config['lower'] = new_lower
                        print(f"Нижняя граница изменена на {new_lower:.4f}")

                        print("\nПоказываю обновленный график...")
                        plot_single_column(df, col, config['data_type'],
                                           config['lower'], config['upper'], config['mean'])

                    except ValueError:
                        print("Ошибка: введите корректное число")

                elif action == '2':
                    try:
                        new_upper = float(
                            input(f"Введите новое значение верхней границы (текущая: {config['upper']:.4f}): "))
                        config['upper'] = new_upper
                        print(f"Верхняя граница изменена на {new_upper:.4f}")

                        print("\nПоказываю обновленный график...")
                        plot_single_column(df, col, config['data_type'],
                                           config['lower'], config['upper'], config['mean'])

                    except ValueError:
                        print("Ошибка: введите корректное число")

                elif action == '3':
                    # Показываем график этого столбца с текущими границами
                    plot_single_column(df, col, config['data_type'],
                                       config['lower'], config['upper'], config['mean'])

                elif action == '4':
                    # Применяем автоматический фильтр выбросов
                    print("\n" + "=" * 60)
                    print(f"АВТОМАТИЧЕСКАЯ ФИЛЬТРАЦИЯ ДЛЯ {col}")
                    print("=" * 60)
                    print("\nДоступные методы фильтрации:")
                    print("1. IQR (межквартильный размах) - для стационарных данных")
                    print("2. MAD (медианное абсолютное отклонение) - устойчив к выбросам")
                    print("3. Производная - для поиска резких скачков")
                    print("4. Поиск пиков - для изолированных пиков")
                    print("5. Фильтр Савицкого-Голая - сглаживание и поиск отклонений")

                    method_choice = input("\nВыберите метод (1-5): ").strip()

                    try:
                        if method_choice == '1':
                            multiplier = float(input("Множитель IQR (1.5-3, по умолч. 1.5): ") or "1.5")
                            filtered_data, outlier_mask, bounds, stats = apply_outlier_filter(
                                df, col, method='iqr', multiplier=multiplier
                            )
                        elif method_choice == '2':
                            threshold = float(input("Порог MAD (3-3.5, по умолч. 3.5): ") or "3.5")
                            filtered_data, outlier_mask, bounds, stats = apply_outlier_filter(
                                df, col, method='mad', threshold=threshold
                            )
                        elif method_choice == '3':
                            threshold_multiplier = float(input("Множитель порога (3-7, по умолч. 5): ") or "5")
                            filtered_data, outlier_mask, bounds, stats = apply_outlier_filter(
                                df, col, method='derivative', threshold_multiplier=threshold_multiplier
                            )
                        elif method_choice == '4':
                            prominence = float(input("Prominence (0.3-1, по умолч. 0.5): ") or "0.5")
                            distance = int(input("Мин. расстояние между пиками (5-20, по умолч. 10): ") or "10")
                            filtered_data, outlier_mask, bounds, stats = apply_outlier_filter(
                                df, col, method='peak', prominence=prominence, distance=distance
                            )
                        elif method_choice == '5':
                            window_length = int(input("Длина окна (нечетное, 11-31, по умолч. 21): ") or "21")
                            polyorder = int(input("Порядок полинома (2-5, по умолч. 3): ") or "3")
                            threshold = float(input("Порог (2-4, по умолч. 3): ") or "3")
                            filtered_data, outlier_mask, bounds, stats = apply_outlier_filter(
                                df, col, method='savgol', window_length=window_length,
                                polyorder=polyorder, threshold=threshold
                            )
                        else:
                            print("Неверный выбор метода")
                            continue

                        # Визуализируем результат фильтрации
                        visualize_outlier_filter(df, col, filtered_data, outlier_mask, bounds, stats,
                                                 save_folder=filter_folder)

                        # Спрашиваем, применить ли фильтр
                        apply_filter = input("\nПрименить этот фильтр и обновить границы? (да/нет): ").strip().lower()
                        if apply_filter in ['да', 'yes', 'y', 'д']:
                            if bounds is not None:
                                config['lower'] = bounds[0]
                                config['upper'] = bounds[1]
                                print(f"Границы обновлены: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
                            else:
                                filtered_clean = filtered_data.dropna()
                                if len(filtered_clean) > 0:
                                    new_mean = filtered_clean.mean()
                                    new_std = filtered_clean.std()
                                    suggested_lower = new_mean - 3 * new_std
                                    suggested_upper = new_mean + 3 * new_std
                                    print(f"\nРекомендуемые границы: [{suggested_lower:.4f}, {suggested_upper:.4f}]")
                                    use_suggested = input("Использовать эти границы? (да/нет): ").strip().lower()
                                    if use_suggested in ['да', 'yes', 'y', 'д']:
                                        config['lower'] = suggested_lower
                                        config['upper'] = suggested_upper
                                        config['mean'] = new_mean
                                        print("Границы обновлены")

                            df[col] = filtered_data
                            print("Данные обновлены. Выбросы заменены на NaN.")

                            print("\nПоказываю обновленный график с новыми границами...")
                            plot_single_column(df, col, config['data_type'],
                                               config['lower'], config['upper'], config['mean'])
                        else:
                            print("Фильтр не применен.")

                    except Exception as e:
                        print(f"Ошибка при применении фильтра: {e}")

                elif action == '5':
                    continue

                else:
                    print("Неверный выбор")

            else:
                print("Неверный номер столбца")

        except ValueError:
            print("Ошибка: введите номер столбца")
        except Exception as e:
            print(f"Ошибка: {e}")

    return bounds_config
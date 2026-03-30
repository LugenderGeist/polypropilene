import os
import pandas as pd
from visualization import plot_single_column, plot_all_columns


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

    # Показываем начальные графики
    print("\n" + "=" * 80)
    print("НАЧАЛЬНЫЕ ГРАФИКИ С ГРАНИЦАМИ ±50% ОТ СРЕДНЕГО")
    print("=" * 80)
    plot_all_columns(df, bounds_config, input_columns, output_columns, adjusted_folder)

    # Интерактивный цикл изменения границ
    while True:
        print("\n" + "=" * 80)
        print("НАСТРОЙКА ГРАНИЦ")
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

        print(f"\n0. Завершить настройку и показать финальные графики")

        # Выбор столбца
        try:
            choice = input("\nВыберите номер столбца для настройки (0 для выхода): ").strip()
            if choice == '0':
                break

            idx = int(choice) - 1
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
                print("4. Сохранить текущий график этого столбца")
                print("5. Показать все графики с текущими настройками")
                print("6. Вернуться к выбору столбца")

                action = input("Ваш выбор (1-6): ").strip()

                if action == '1':
                    try:
                        new_lower = float(
                            input(f"Введите новое значение нижней границы (текущая: {config['lower']:.4f}): "))
                        config['lower'] = new_lower
                        print(f"Нижняя граница изменена на {new_lower:.4f}")

                        # Показываем обновленный график
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

                        # Показываем обновленный график
                        print("\nПоказываю обновленный график...")
                        plot_single_column(df, col, config['data_type'],
                                           config['lower'], config['upper'], config['mean'])

                    except ValueError:
                        print("Ошибка: введите корректное число")

                elif action == '3':
                    # Показываем график этого столбца
                    plot_single_column(df, col, config['data_type'],
                                       config['lower'], config['upper'], config['mean'])

                elif action == '4':
                    # Сохраняем текущий график
                    safe_name = col.replace('/', '_').replace('\\', '_').replace(':', '_')
                    save_path = os.path.join(adjusted_folder, f'{safe_name}_adjusted.png')
                    plot_single_column(df, col, config['data_type'],
                                       config['lower'], config['upper'], config['mean'],
                                       save_path=save_path)
                    print(f"График сохранен: {save_path}")

                elif action == '5':
                    # Показываем все графики с текущими настройками
                    print("\nПоказываю все графики с текущими границами...")
                    plot_all_columns(df, bounds_config, input_columns, output_columns, adjusted_folder)

                elif action == '6':
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
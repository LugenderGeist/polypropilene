import os
import chardet
import pandas as pd
from datetime import datetime
from config import ENCODINGS_TO_TRY  # Добавляем импорт из config

# Функция для определения кодировки файла
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']


# Функция для преобразования всех столбцов в числовой формат
def convert_to_numeric(df):
    """Преобразует все столбцы DataFrame в числовой формат где возможно"""
    df_numeric = df.copy()
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    return df_numeric


# Функция для загрузки данных с определением кодировки
def load_data(file_path):
    """Загружает данные из CSV файла с автоматическим определением кодировки"""
    try:
        encoding = detect_encoding(file_path)
        print(f"Определенная кодировка файла: {encoding}")
        df = pd.read_csv(file_path, encoding=encoding)

        # Преобразуем все столбцы в числовой формат
        df = convert_to_numeric(df)

        print("Файл успешно загружен и преобразован в числовой формат!")
        return df, encoding
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        # Пробуем альтернативные кодировки из config
        for enc in ENCODINGS_TO_TRY:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                # Преобразуем все столбцы в числовой формат
                df = convert_to_numeric(df)
                print(f"Файл успешно загружен с кодировкой: {enc}")
                print("Данные преобразованы в числовой формат")
                return df, enc
            except:
                continue
        print("Не удалось загрузить файл")
        return None, None


# Функция для создания папки для графиков
def create_plots_folder():
    """Создает папку для сохранения графиков"""
    folder_name = f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


# Функция для сохранения конфигурации границ
def save_bounds_config(bounds_config, save_folder):
    """Сохраняет конфигурацию границ в текстовый файл"""
    config_file = os.path.join(save_folder, 'bounds_config.txt')
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("Конфигурация границ для столбцов\n")
        f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        for col, config in bounds_config.items():
            f.write(f"{col} ({config['data_type']}):\n")
            f.write(f"  Среднее: {config['mean']:.6f}\n")
            f.write(f"  Нижняя граница: {config['lower']:.6f}\n")
            f.write(f"  Верхняя граница: {config['upper']:.6f}\n\n")
    return config_file


# Функция для настройки входных и выходных столбцов
def setup_columns(df):
    total_cols = df.shape[1]

    while True:
        try:
            input_cols = int(input("\nВведите количество столбцов с начала файла, которые являются входными данными: "))
            output_cols = int(input("Введите количество столбцов с конца файла, которые являются выходными данными: "))

            if input_cols + output_cols > total_cols:
                print(
                    f"Ошибка: Сумма входных ({input_cols}) и выходных ({output_cols}) столбцов превышает общее количество столбцов ({total_cols})")
            elif input_cols < 0 or output_cols < 0:
                print("Ошибка: Количество столбцов не может быть отрицательным")
            else:
                break
        except ValueError:
            print("Ошибка: Пожалуйста, введите целое число")

    input_columns = df.columns[:input_cols].tolist()
    output_columns = df.columns[-output_cols:].tolist() if output_cols > 0 else []

    return input_columns, output_columns


# Функция для удаления выбросов из данных
def remove_outliers(df, bounds_config, all_columns):
    """
    Удаляет строки, в которых хотя бы один столбец имеет значение за пределами границ
    """
    df_cleaned = df.copy()
    removed_mask = pd.Series(False, index=df.index)

    removal_report = {}

    print("\n" + "=" * 80)
    print("ПРОВЕРКА ВЫБРОСОВ ДЛЯ УДАЛЕНИЯ")
    print("=" * 80)

    for col in all_columns:
        if col in bounds_config:
            try:
                data = pd.to_numeric(df[col], errors='coerce')
                config = bounds_config[col]

                outlier_mask = (data < config['lower']) | (data > config['upper'])
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    removal_report[col] = {
                        'outlier_count': outlier_count,
                        'outlier_percent': (outlier_count / len(data)) * 100,
                        'indices': df.index[outlier_mask].tolist()
                    }
                    removed_mask = removed_mask | outlier_mask

                    print(f"\n{col} ({config['data_type']}):")
                    print(f"  Выбросов: {outlier_count} ({outlier_count / len(data) * 100:.2f}%)")
                    print(f"  Диапазон выбросов: [{data[outlier_mask].min():.4f}, {data[outlier_mask].max():.4f}]")

            except Exception as e:
                print(f"\n{col}: Ошибка при проверке - {str(e)}")

    total_removed_rows = removed_mask.sum()

    if total_removed_rows > 0:
        removed_indices = df.index[removed_mask].tolist()
        df_cleaned = df_cleaned[~removed_mask]

        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТ УДАЛЕНИЯ")
        print("=" * 80)
        print(f"Всего удалено строк: {total_removed_rows} из {len(df)} ({total_removed_rows / len(df) * 100:.2f}%)")
        print(f"Осталось строк: {len(df_cleaned)}")

        return df_cleaned, removed_indices, removal_report
    else:
        print("\nВыбросов не обнаружено. Удаление не требуется.")
        return df_cleaned, [], removal_report


# Функция для сохранения очищенных данных
def save_cleaned_data(df_cleaned, original_filename, save_folder):
    """Сохраняет очищенные данные в новый CSV файл"""
    base_name = os.path.splitext(original_filename)[0]
    cleaned_filename = f"{base_name}_cleaned.csv"
    cleaned_path = os.path.join(save_folder, cleaned_filename)

    df_cleaned.to_csv(cleaned_path, index=False, encoding='utf-8-sig')
    print(f"\nОчищенные данные сохранены в: {cleaned_path}")

    info_filename = f"{base_name}_removal_info.txt"
    info_path = os.path.join(save_folder, info_filename)

    return cleaned_path, info_path
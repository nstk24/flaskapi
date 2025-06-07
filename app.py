# Подключение библиотек
import numpy as np
import os
import random
import flask
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy import sparse
from matplotlib.figure import Figure
from time import perf_counter
from flask import Flask, request, jsonify, render_template


from scipy.signal import find_peaks


# Функции удаления базовой линии
app = Flask(__name__)
# -----------------------------
# Вспомогательные функции валидации

def ensure_keys(data, keys):
    if data is None:
        raise ValueError("Отсутствуют данные запроса")
    for key in keys:
        if key not in data:
            raise ValueError(f"Отсутствует параметр {key}")


def validate_array(value, name):
    if not isinstance(value, list) or not all(isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{name} должен быть списком чисел")


def validate_array_list(value, name):
    if not isinstance(value, list) or not all(
        isinstance(arr, list) and all(isinstance(v, (int, float)) for v in arr)
        for arr in value
    ):
        raise ValueError(f"{name} должен быть списком списков чисел")


def to_float(value, name):
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} должен быть числом")


def to_int(value, name):
    try:
        return int(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} должен быть целым числом")

# -----------------------------

# Функция нахождения базовой линии
def baseline_als(amplitudes, lam, p, niter=10):
    L = len(amplitudes)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + float(lam) * D.dot(D.transpose())
        z = spsolve(Z, w * amplitudes)
        w = p * (amplitudes > z) + (1 - p) * (amplitudes < z)
    return z

# Flask маршрут для обработки данных
@app.route('/baseline', methods=['POST'])
def process_baseline():
    try:
        # Получение данных из POST-запроса
        data = request.json
        ensure_keys(data, ['amplitudes'])
        validate_array(data['amplitudes'], 'amplitudes')
        amplitudes = np.array(data['amplitudes'])  # Входной массив
        lam = to_float(data.get('lam', 1000), 'lam')
        p = to_float(data.get('p', 0.001), 'p')
        niter = to_int(data.get('niter', 10), 'niter')

        # Обработка данных
        baseline = baseline_als(amplitudes, lam, p, niter)

        # Возвращаем результат в виде JSON
        return jsonify({'baseline': baseline.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
@app.route('/delete_baseline', methods=['POST'])
def delete_baseline():
    try:
        # Получение данных из POST-запроса
        data = request.json
        ensure_keys(data, ['amplitudes_list'])
        validate_array_list(data['amplitudes_list'], 'amplitudes_list')
        amplitudes_list = [np.array(amplitude) for amplitude in data['amplitudes_list']]  # Список амплитуд
        lam = to_float(data.get('lam', 1000), 'lam')
        p = to_float(data.get('p', 0.001), 'p')

        # Применение функции для удаления базовой линии
        amplitudesBL_list = []
        for amplitudes in amplitudes_list:
            baseline = baseline_als(amplitudes, lam, p)
            cleaned_spectrum = amplitudes - baseline
            amplitudesBL_list.append(cleaned_spectrum.tolist())

        # Возвращаем результат в JSON
        return jsonify({'amplitudesBL_list': amplitudesBL_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/average_spectrum', methods=['POST'])
def average_spectrum():
    try:
        # Получение данных из POST-запроса
        data = request.json
        ensure_keys(data, ['averaged'])
        validate_array_list(data['averaged'], 'averaged')
        averaged = [np.array(spectrum) for spectrum in data['averaged']]  # Список спектров

        # Вычисление средней спектрограммы
        averaged_result = np.mean(np.array(averaged), axis=0).tolist()

        # Возвращаем результат в JSON
        return jsonify({'average_spectrum': averaged_result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/select_frequency_range', methods=['POST'])
def select_frequency_range():
    try:
        # Получение данных из POST-запроса
        data = request.json
        ensure_keys(data, ['freq_list', 'ampl_list'])
        validate_array_list(data['freq_list'], 'freq_list')
        validate_array_list(data['ampl_list'], 'ampl_list')
        freq_list = [np.array(freq) for freq in data['freq_list']]  # Список частот
        ampl_list = [np.array(ampl) for ampl in data['ampl_list']]  # Список амплитуд
        min_freq = to_float(data.get('min_freq', 0), 'min_freq')
        max_freq = to_float(data.get('max_freq', 10000), 'max_freq')

        # Обработка данных
        freq_list2 = []
        ampl_list2 = []
        for freq, ampl in zip(freq_list, ampl_list):
            mask = (freq >= min_freq) & (freq <= max_freq)
            if np.any(mask):
                freq_list2.append(freq[mask].tolist())
                ampl_list2.append(ampl[mask].tolist())

        # Возвращаем результат
        return jsonify({
            'filtered_freq_list': freq_list2,
            'filtered_ampl_list': ampl_list2
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

from scipy.signal import savgol_filter

@app.route('/smooth_signal', methods=['POST'])
def smooth_signal():
    try:
        # Получение данных из POST-запроса
        data = request.json
        ensure_keys(data, ['spectrum_list'])
        validate_array_list(data['spectrum_list'], 'spectrum_list')
        spectrum_list = [np.array(spectrum) for spectrum in data['spectrum_list']]  # Список спектров
        window_length = to_int(data.get('window_length', 25), 'window_length')
        polyorder = to_int(data.get('polyorder', 2), 'polyorder')

        # Проверка на корректность параметров
        if window_length % 2 == 0 or window_length <= 0:
            return jsonify({'error': 'window_length должен быть положительным и нечетным'}), 400
        if polyorder >= window_length:
            return jsonify({'error': 'polyorder должен быть меньше window_length'}), 400

        # Применение сглаживания
        smoothed_spectra = []
        for spectrum in spectrum_list:
            smoothed_spectrum = savgol_filter(spectrum, window_length, polyorder)
            smoothed_spectra.append(smoothed_spectrum.tolist())

        # Возвращаем результат
        return jsonify({'smoothed_spectra': smoothed_spectra})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/normalize_spectrum_snv', methods=['POST'])
def normalize_spectrum_snv():
    try:
        # Получение данных из POST-запроса
        data = request.json
        ensure_keys(data, ['spectrum_list'])
        validate_array_list(data['spectrum_list'], 'spectrum_list')
        spectrum_list = [np.array(spectrum) for spectrum in data['spectrum_list']]  # Список спектров

        # Применение нормализации
        normalized_spectrum = []
        for spectrum in spectrum_list:
            mean_spectrum = np.mean(spectrum)
            std_spectrum = np.std(spectrum)
            normalized = (spectrum - mean_spectrum) / std_spectrum
            normalized_spectrum.append(normalized.tolist())

        # Возвращаем результат
        return jsonify({'normalized_spectrum': normalized_spectrum})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/normalize_by_max', methods=['POST'])
def normalize_by_max():
    try:
        # Получение данных из POST-запроса
        data = request.json
        ensure_keys(data, ['spectrum_list'])
        validate_array_list(data['spectrum_list'], 'spectrum_list')
        spectrum_list = [np.array(spectrum) for spectrum in data['spectrum_list']]  # Список спектров

        # Нормализация каждого спектра относительно его максимального значения
        normalized_spectrum_list = []
        for spectrum in spectrum_list:
            max_value = np.max(spectrum)
            normalized_spectrum = spectrum / max_value
            normalized_spectrum_list.append(normalized_spectrum.tolist())

        # Возвращаем результат
        return jsonify({'normalized_spectrum': normalized_spectrum_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import io
import base64

@app.route('/plot_graph', methods=['POST'])
def plot_graph():
    try:
        data = request.json
        ensure_keys(data, ["frequencies_list", "amplitudes_list"])
        validate_array_list(data["frequencies_list"], "frequencies_list")
        validate_array_list(data["amplitudes_list"], "amplitudes_list")
        frequencies_list = [np.array(freq) for freq in data["frequencies_list"]]
        amplitudes_list = [np.array(ampl) for ampl in data["amplitudes_list"]]
        find_flag = bool(data.get("find_flag", False))
        peak_params = data.get("peak_params", {})
        width = to_float(peak_params.get("width", 1), "width")
        prominence = to_float(peak_params.get("prominence", 1), "prominence")

        # Создание графика
        fig, ax = plt.subplots(figsize=(11.5, 7.9))
        for i in range(len(amplitudes_list)):
            ax.plot(frequencies_list[i], amplitudes_list[i], alpha=0.5)
            # Поиск пиков, если включен флаг
            if find_flag:
                peaks, _ = find_peaks(amplitudes_list[i], width=width, prominence=prominence)
                ax.plot(frequencies_list[i][peaks], amplitudes_list[i][peaks], 'ro')
                for j in range(len(peaks)):
                    ax.text(
                        frequencies_list[i][peaks[j]],
                        amplitudes_list[i][peaks[j]],
                        f'({frequencies_list[i][peaks[j]]:.2f},\n{amplitudes_list[i][peaks[j]]:.2f})',
                        fontsize=8
                    )

        ax.set_xlabel('Рамановский сдвиг, см^-1')
        ax.set_ylabel('Интенсивность')

        # Сохранение графика в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)

        # Возвращаем график в формате Base64
        return jsonify({'plot_image': encoded_image})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/process_and_plot', methods=['POST'])
def process_and_plot():
    try:
        # Получение данных из POST-запроса
        data = request.json
        print("Полученные данные:", data)  # Отладочный вывод

        ensure_keys(data, ['frequencies_list', 'amplitudes_list'])
        validate_array_list(data['frequencies_list'], 'frequencies_list')
        validate_array_list(data['amplitudes_list'], 'amplitudes_list')
        frequencies_list = [np.array(freq) for freq in data['frequencies_list']]  # Частоты
        amplitudes_list = [np.array(ampl) for ampl in data['amplitudes_list']]    # Амплитуды
        print("Частоты:", frequencies_list)
        print("Амплитуды:", amplitudes_list)

        # Флаги обработки
        selection_flag = bool(data.get('selection_flag', False))
        savgol_filter_flag = bool(data.get('savgol_filter_flag', False))
        remove_flag = bool(data.get('remove_flag', False))
        normalize_snv_flag = bool(data.get('normalize_snv_flag', False))
        normalize_flag = bool(data.get('normalize_flag', False))
        average_flag = bool(data.get('average_flag', False))

        # Проверка параметров
        print("Флаги:", {
            "selection_flag": selection_flag,
            "savgol_filter_flag": savgol_filter_flag,
            "remove_flag": remove_flag,
            "normalize_snv_flag": normalize_snv_flag,
            "normalize_flag": normalize_flag,
            "average_flag": average_flag
        })

        # Параметры для каждой операции
        selection_params = data.get('selection_params', {})
        savgol_params = data.get('savgol_params', {})
        baseline_params = data.get('baseline_params', {})
        print("Параметры:", {
            "selection_params": selection_params,
            "savgol_params": savgol_params,
            "baseline_params": baseline_params
        })

        # Применение операций в зависимости от флагов
        if selection_flag:
            frequencies_list, amplitudes_list = select_frequency_range(
                frequencies_list, amplitudes_list, selection_params)

        if savgol_filter_flag:
            amplitudes_list = apply_savgol_filter(amplitudes_list, savgol_params)

        if remove_flag:
            amplitudes_list = delete_baseline(amplitudes_list, baseline_params)

        if normalize_snv_flag:
            amplitudes_list = normalize_spectrum_snv(amplitudes_list)
        elif normalize_flag:
            amplitudes_list = normalize_by_max(amplitudes_list)

        if average_flag:
            amplitudes_list = average_spectrum(amplitudes_list)
            frequencies_list = average_spectrum(frequencies_list)

        # Построение графика
        fig, ax = plt.subplots(figsize=(11.5, 7.9))
        for i in range(len(amplitudes_list)):
            ax.plot(frequencies_list[i], amplitudes_list[i], alpha=0.5)
        ax.set_xlabel('Рамановский сдвиг, см^-1')
        ax.set_ylabel('Интенсивность')

        if normalize_flag:
            amplitudes_list = normalize_by_max(amplitudes_list)


        # Сохранение графика в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)

        # Возвращаем результат
        return jsonify({'plot_image': encoded_image})
    except Exception as e:
        print("Ошибка:", str(e))  # Вывод ошибки в консоль
        return jsonify({'error': str(e)}), 400



# Вспомогательные функции для обработки
def select_frequency_range(freq_list, ampl_list, params):
    min_freq = float(params.get('min_freq', 0))
    max_freq = float(params.get('max_freq', 10000))
    freq_list2 = []
    ampl_list2 = []
    for freq, ampl in zip(freq_list, ampl_list):
        mask = (freq >= min_freq) & (freq <= max_freq)
        if np.any(mask):
            freq_list2.append(freq[mask])
            ampl_list2.append(ampl[mask])
    return freq_list2, ampl_list2

def apply_savgol_filter(amplitudes_list, params):
    window_length = int(params.get('window_length', 25))
    polyorder = int(params.get('polyorder', 2))
    smoothed_list = []
    for amplitudes in amplitudes_list:
        smoothed = savgol_filter(amplitudes, window_length, polyorder)
        smoothed_list.append(smoothed)
    return smoothed_list

def delete_baseline(amplitudes_list, params):
    lam = float(params.get('lam', 1000))
    p = float(params.get('p', 0.001))
    return [
        amplitudes - baseline_als(amplitudes, lam, p) for amplitudes in amplitudes_list
    ]
def normalize_by_max(spectrum_list):
    normalized_spectrum = []
    for spectrum in spectrum_list:
        max_value = np.max(spectrum)
        if max_value == 0:
            raise ValueError("Максимум равен 0, нормализация невозможна.")
        normalized_spectrum.append((spectrum / max_value).tolist())
    return normalized_spectrum




import os
from flask import request
@app.route('/upload_files', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'Файлы не загружены'}), 400

        files = request.files.getlist('files')
        frequencies_list = []
        amplitudes_list = []

        for file in files:
            # Прочитаем содержимое файла
            content = file.read().decode('utf-8')
            print("Содержимое файла:\n", content)

            # Обработаем данные
            try:
                # Заменяем запятые на точки для корректной работы с числами
                content = content.replace(',', '.')

                # Используем табуляцию как разделитель
                data = np.genfromtxt(io.StringIO(content), delimiter='\t')

                frequencies_list.append(data[:, 0])  # Первый столбец - частоты
                amplitudes_list.append(data[:, 1])  # Второй столбец - амплитуды
            except Exception as e:
                print("Ошибка обработки файла:", str(e))
                return jsonify({'error': f'Ошибка обработки файла: {file.filename}', 'details': str(e)}), 400

        return jsonify({
            'frequencies_list': [freq.tolist() for freq in frequencies_list],
            'amplitudes_list': [ampl.tolist() for ampl in amplitudes_list]
        })
    except Exception as e:
        print("Общая ошибка:", str(e))
        return jsonify({'error': str(e)}), 400

@app.route('/set_flags', methods=['POST'])
def set_flags():
    try:
        # Получаем флаги из запроса
        data = request.json
        if data is None:
            return jsonify({'error': 'Данные не переданы'}), 400

        # Читаем флаги
        remove_flag = bool(data.get('remove_flag', False))
        average_flag = bool(data.get('average_flag', False))
        find_flag = bool(data.get('find_flag', False))
        normalize_flag = bool(data.get('normalize_flag', False))
        normalize_snv_flag = bool(data.get('normalize_snv_flag', False))
        savgol_filter_flag = bool(data.get('savgol_filter_flag', False))
        selection_flag = bool(data.get('selection_flag', False))

        # Возвращаем подтверждение
        return jsonify({
            'message': 'Флаги установлены',
            'flags': {
                'remove_flag': remove_flag,
                'average_flag': average_flag,
                'find_flag': find_flag,
                'normalize_flag': normalize_flag,
                'normalize_snv_flag': normalize_snv_flag,
                'savgol_filter_flag': savgol_filter_flag,
                'selection_flag': selection_flag
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/clear_data', methods=['POST'])
def clear_data():
    try:
        # Здесь можно добавить логику очистки временных данных, если используется хранилище
        return jsonify({'message': 'Данные очищены'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def index():
    return render_template('index.html')


import os

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))

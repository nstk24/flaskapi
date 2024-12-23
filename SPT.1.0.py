# Подключение библиотек
import numpy as np
import os
import pywt
import random
import flask
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy import sparse
from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from time import perf_counter
from flask import Flask, request, jsonify


from scipy.signal import find_peaks


# Функции удаления базовой линии
app = Flask(__name__)

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
        amplitudes = np.array(data['amplitudes'])  # Входной массив
        lam = float(data.get('lam', 1000))         # Параметр lam (по умолчанию 1000)
        p = float(data.get('p', 0.001))           # Параметр p (по умолчанию 0.001)
        niter = int(data.get('niter', 10))        # Число итераций (по умолчанию 10)

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
        amplitudes_list = [np.array(amplitude) for amplitude in data['amplitudes_list']]  # Список амплитуд
        lam = float(data.get('lam', 1000))  # Значение lam
        p = float(data.get('p', 0.001))    # Значение p

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
        freq_list = [np.array(freq) for freq in data['freq_list']]  # Список частот
        ampl_list = [np.array(ampl) for ampl in data['ampl_list']]  # Список амплитуд
        min_freq = float(data.get('min_freq', 0))  # Минимальная частота (по умолчанию 0)
        max_freq = float(data.get('max_freq', 10000))  # Максимальная частота (по умолчанию 10000)

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
        spectrum_list = [np.array(spectrum) for spectrum in data['spectrum_list']]  # Список спектров
        window_length = int(data.get('window_length', 25))  # Длина окна (по умолчанию 25)
        polyorder = int(data.get('polyorder', 2))  # Степень полинома (по умолчанию 2)

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
        # Получение данных из POST-запроса
        data = request.json
        frequencies_list = [np.array(freq) for freq in data['frequencies_list']]  # Частоты
        amplitudes_list = [np.array(ampl) for ampl in data['amplitudes_list']]    # Амплитуды
        find_flag = data.get('find_flag', False)                                 # Флаг поиска пиков
        peak_params = data.get('peak_params', {})                                # Параметры поиска пиков
        width = float(peak_params.get('width', 1))                               # Ширина пиков
        prominence = float(peak_params.get('prominence', 1))                     # Значение выделенности

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
        frequencies_list = [np.array(freq) for freq in data['frequencies_list']]  # Частоты
        amplitudes_list = [np.array(ampl) for ampl in data['amplitudes_list']]    # Амплитуды

        # Флаги обработки
        selection_flag = data.get('selection_flag', False)
        savgol_filter_flag = data.get('savgol_filter_flag', False)
        remove_flag = data.get('remove_flag', False)
        normalize_snv_flag = data.get('normalize_snv_flag', False)
        normalize_flag = data.get('normalize_flag', False)
        average_flag = data.get('average_flag', False)

        # Параметры для каждой операции
        selection_params = data.get('selection_params', {})
        savgol_params = data.get('savgol_params', {})
        baseline_params = data.get('baseline_params', {})

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



import os
from flask import request

@app.route('/upload_files', methods=['POST'])
def upload_files():
    try:
        # Проверяем, что файлы загружены
        if 'files' not in request.files:
            return jsonify({'error': 'Файлы не загружены'}), 400

        files = request.files.getlist('files')  # Получаем список файлов
        frequencies_list = []
        amplitudes_list = []

        for file in files:
            # Читаем данные из файла
            data = np.genfromtxt(file, skip_header=1)
            frequencies_list.append(data[:, 0])
            amplitudes_list.append(data[:, 1])
         # Возвращаем данные для использования
        return jsonify({
            'frequencies_list': [freq.tolist() for freq in frequencies_list],
            'amplitudes_list': [ampl.tolist() for ampl in amplitudes_list]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/set_flags', methods=['POST'])
def set_flags():
    try:
        # Получаем флаги из запроса
        data = request.json

        # Читаем флаги
        remove_flag = data.get('remove_flag', False)
        average_flag = data.get('average_flag', False)
        find_flag = data.get('find_flag', False)
        normalize_flag = data.get('normalize_flag', False)
        normalize_snv_flag = data.get('normalize_snv_flag', False)
        savgol_filter_flag = data.get('savgol_filter_flag', False)
        selection_flag = data.get('selection_flag', False)

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

# Создание окна
root = tk.Tk()
root.title("Spectrum")
root.state('zoomed')

# Устанавливаем размеры и положение окна на полный экран
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.attributes("-fullscreen", True)
root.geometry(f"{screen_width}x{screen_height}+0+0")

def show_help():
    message = "Эта программа предназначена для обработки и визуализации спектральных данных.\n\n" \
              "Использование программы:\n\n" \
              "1. Выберите файлы с данными, нажав кнопку 'Файл' -> 'Открыть файлы'.\n" \
              "   Поддерживаемые форматы файлов: ESP, TXT.\n\n" \
              "2. После открытия файлов, программа автоматически построит график спектров.\n\n" \
              "3. Для дальнейшей обработки спектров, воспользуйтесь меню 'Действия':\n\n" \
              "   - Выбор частот и амплитуд в заданном диапазоне: позволяет выбрать диапазон частот и амплитуд, которые будут отображаться на графике.\n\n" \
              "   - Сглаживание: применяет фильтр Савицкого-Голоя для сглаживания спектров.\n\n" \
              "   - Удаление базовой линии: удаляет базовую линию из спектров методом ALS (Alternating Least Squares).\n\n" \
              "   - Нормализация: нормализует спектры методом стандартного отклонения или по максимальному значению.\n\n" \
              "   - Вычисление средней спектрограммы: вычисляет среднюю спектрограмму по всем спектрам.\n\n" \
              "4. После выбора необходимых действий, нажмите кнопку 'run', программа построит обновленный график на основе выбранных действий и данных из выбранных файлов.\n\n" \
              "5. Для завершения работы программы, нажмите кнопку 'Файл' -> 'Выход'."
    messagebox.showinfo("Помощь", message)


def show_programm_info():
    message = "О программе\n\n" \
              "Имя: Программа для обработки спектров\n" \
              "Эта программа предназначена для обработки и визуализации спектров.\n" \
              "Программа позволяет выбирать файлы с данными, выполнять различные действия над спектрами и визуализировать результаты.\n" \
              "В меню 'Действия' доступны следующие функции:\n" \
              "- Выбор частот и амплитуд в заданном диапазоне\n" \
              "- Сглаживание\n" \
              "- Удаление базовой линии\n" \
              "- Нормализация\n" \
              "- Вычисление средней спектрограммы\n\n" \
    
    messagebox.showinfo("О программе", message)



# Панель инструментов
mainmenu = tk.Menu(root)
root.config(menu=mainmenu)
# Открытие файла с
filemenu = tk.Menu(mainmenu, tearoff=0)
filemenu.add_command(label="Открыть файлы", command=open_folder)
filemenu.add_command(label="Сохранить...")
filemenu.add_separator()
filemenu.add_command(label="Выход", command=lambda: root.destroy())
# Помощь
helpmenu = tk.Menu(mainmenu, tearoff=0)
helpmenu.add_command(label="Помощь", command = show_help)
helpmenu.add_command(label="О программе", command= show_programm_info)

building = tk.Menu(mainmenu, tearoff=0)
building.add_command(label="Построить график", command=get_input)
building.add_command(label="Очистить данные о файле", command=clear_data)
# Поле ввода 1

# Создаем метку (Label) с текстом
label = tk.Label(root, text="lam:", font=("Arial", 10))
label.pack()  # Размещаем метку на окне
label.place(x=5, y=140)

# Создаем метку (Label) с текстом
label = tk.Label(root, text="p:", font=("Arial", 10))
label.pack()  # Размещаем метку на окне
label.place(x=5, y=160)

# Создаем метку (Label) с текстом
label = tk.Label(root, text="min:", font=("Arial", 10))
label.pack()  # Размещаем метку на окне
label.place(x=5, y=50)

# Создаем метку (Label) с текстом
label = tk.Label(root, text="max:", font=("Arial", 10))
label.pack()  # Размещаем метку на окне
label.place(x=5, y=70)

# Создаем метку (Label) с текстом
label = tk.Label(root, text="wlen:", font=("Arial", 10))
label.pack()  # Размещаем метку на окне
label.place(x=5, y=283)

# Создаем метку (Label) с текстом
label = tk.Label(root, text="poly:", font=("Arial", 10))
label.pack()  # Размещаем метку на окне
label.place(x=5, y=303)

label = tk.Label(root, text="width:", font=("Arial", 10))
label.pack()  # Размещаем метку на окне
label.place(x=5, y=385)

label = tk.Label(root, text="prom:", font=("Arial", 10))
label.pack()  # Размещаем метку на окне
label.place(x=5, y=405)
# Полоса частотного диапазона
# Поле ввода 1 (min_freq)
entry1 = tk.Entry(root)
entry1.insert(0, "min_freq")
entry1.bind("<FocusIn>", on_entry_click)
entry1.pack()
entry1.place(x=40, y=50)
# Поле ввода 2 (max_freq)
entry2 = tk.Entry(root)
entry2.insert(0, "max_freq")
entry2.bind("<FocusIn>", on_entry_click)
entry2.pack()
entry2.place(x=40, y=70)
# baseline_correction
# Поле ввода 3 (lam)
entry3 = tk.Entry(root)
entry3.insert(0, "1000")
entry3.bind("<FocusIn>", on_entry_click)
entry3.pack()
entry3.place(x=40, y=142)
# Поле ввода 4 (p)
entry4 = tk.Entry(root)
entry4.insert(0, "0.001")
entry4.bind("<FocusIn>", on_entry_click)
entry4.pack()
entry4.place(x=40, y=162)
# savgol_filter
# Поле ввода 4 (window_length)
entry5 = tk.Entry(root)
entry5.insert(0, "25")
entry5.bind("<FocusIn>", on_entry_click)
entry5.pack()
entry5.place(x=40, y=285)
# Поле ввода 4 (polyorder)
entry6 = tk.Entry(root)
entry6.insert(0, "2")
entry6.bind("<FocusIn>", on_entry_click)
entry6.pack()
entry6.place(x=40, y=305)
#find_picks
# Поле ввода 5 (width)
entry7 = tk.Entry(root)
entry7.insert(0, "10")
entry7.bind("<FocusIn>", on_entry_click)
entry7.pack()
entry7.place(x=40, y=385)
# Поле ввода 6 (prominence)
entry8 = tk.Entry(root)
entry8.insert(0, "10")
entry8.bind("<FocusIn>", on_entry_click)
entry8.pack()
entry8.place(x=40, y=405)


# Поле инструментов
mainmenu.add_cascade(label="Файл", menu=filemenu)
mainmenu.add_cascade(label="Справка", menu=helpmenu)

mainmenu.add_cascade(label="Действия над файлом", menu=building)
mainmenu.add_command(label="run", command=get_input)

# Создаем метку (Label) с текстом
label = tk.Label(root, text="Функции и параметры", font=("Arial", 13))
label.pack()  # Размещаем метку на окне
label.place(x=5, y=5)


label = tk.Label(root, text="Удаление БЛ", font=("Arial", 11))
label.pack()  # Размещаем метку на окне
label.place(x=5, y=95)

label = tk.Label(root, text="Нормировка", font=("Arial", 11))
label.pack()  # Размещаем метку на окне
label.place(x=5, y=180)
label = tk.Label(root, text="Сглаживание", font=("Arial", 11))
label.pack()  # Размещаем метку на окне
label.place(x=5, y=242)


checkbox_var = tk.IntVar()
checkbox = tk.Checkbutton(root, text="Метод BL_ALS", font=("Arial", 10), variable=checkbox_var, command=actions1)
checkbox.pack()
checkbox.place(x=5, y=115)

checkbox_var2 = tk.IntVar()
checkbox2 = tk.Checkbutton(root, text="Поиск пиков", variable=checkbox_var2, command=actions3)
checkbox2.pack()
checkbox2.place(x=5, y=360)

checkbox_var4 = tk.IntVar()
checkbox4 = tk.Checkbutton(root, text="Норм. методом SNV", variable=checkbox_var4, command=actions5)
checkbox4.pack()
checkbox4.place(x=5, y=200)

checkbox_var3 = tk.IntVar()
checkbox3 = tk.Checkbutton(root, text="Норм. по макс", variable=checkbox_var3, command=actions4)
checkbox3.pack()
checkbox3.place(x=5, y=220)

checkbox_var6 = tk.IntVar()
checkbox6 = tk.Checkbutton(root, text="Фильт. Савицкого-Голая", variable=checkbox_var6, command=actions6)
checkbox6.pack()
checkbox6.place(x=5, y=260)

# checkbox_var8 = tk.IntVar()
# checkbox8 = tk.Checkbutton(root, text="Метод ср. скользящей", variable=checkbox_var8, command=actions7)
# checkbox8.pack()
# checkbox8.place(x=5, y=322)

checkbox_var7 = tk.IntVar()
checkbox7 = tk.Checkbutton(root, text="Нахожение среднего", variable=checkbox_var7, command=actions2)
checkbox7.pack()
checkbox7.place(x=5, y=342)

checkbox_var8 = tk.IntVar()
checkbox8 = tk.Checkbutton(root, text="Выбор полосы частот", variable=checkbox_var8, command=actions9)
checkbox8.pack()
checkbox8.place(x=5, y=25)

# Создаем фрейм для размещения downbar
bottom_frame = tk.Frame(root, height=30, bg='lightgray')
bottom_frame.pack(side='bottom', fill='x')

# Отображаем версию tkinter на полосе
Program_version = 0.3572
version_label = tk.Label(bottom_frame, text=f"Program version: {Program_version}", bg='lightgray')
version_label.pack(side='right', padx=10)


root.mainloop()
let frequenciesList = [];
let amplitudesList = [];

// Загрузка файлов
async function uploadFiles() {
    const files = document.getElementById('files').files;
    const formData = new FormData();

    for (let file of files) {
        formData.append('files', file);
    }

    const response = await fetch('/upload_files', {
        method: 'POST',
        body: formData,
    });

    const result = await response.json();
    console.log(result);

    if (result.frequencies_list && result.amplitudes_list) {
        frequenciesList = result.frequencies_list; // Сохраняем частоты
        amplitudesList = result.amplitudes_list;  // Сохраняем амплитуды
        alert('Файлы успешно загружены!');
    } else {
        alert('Ошибка при загрузке файлов.');
    }
}

// Обработка данных
async function processData() {
    const params = {
        frequencies_list: frequenciesList,
        amplitudes_list: amplitudesList,
        min_freq: parseFloat(document.getElementById('min_freq').value) || 0,
        max_freq: parseFloat(document.getElementById('max_freq').value) || 10000,
        window_length: parseInt(document.getElementById('window_length').value) || 25,
        polyorder: parseInt(document.getElementById('polyorder').value) || 2,
        remove_flag: document.getElementById('remove_flag').checked,
        normalize_flag: document.getElementById('normalize_flag').checked,
        selection_flag: true  // Фильтрация активна
    };

    const response = await fetch('/process_and_plot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
    });

    const result = await response.json();
    if (result.plot_image) {
        document.getElementById('plot').src = `data:image/png;base64,${result.plot_image}`;
    } else {
        alert(`Ошибка: ${result.error}`);
    }
}


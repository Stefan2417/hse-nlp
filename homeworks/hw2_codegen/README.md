# HW2: генерация кода

## Обучение

Python 3.11, NVIDIA GPU с BF16. На Linux для pass@1 нужен `bubblewrap` (`bwrap` в PATH).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare.py

export WANDB_MODE=offline
python -m experiments.initial_train +runs=baseline
python train.py +runs=recovery
python train.py +runs=epoch1
python train.py +runs=epoch2
python train.py +runs=epochs3to6
python train.py +runs=epochs7to8
```

Команды выполняются из каталога проекта. Данные сохраняются в `data/prepared`,
модели и checkpoint — в `outputs`. Начальный запуск сохраняет checkpoint-2000
с исходным расписанием LR; настройки следующих этапов находятся в `config/runs/`.

Для сборки усреднённых весов:

```bash
python average_checkpoints.py \
  --first outputs/epochs7to8/checkpoints/checkpoint-60000 \
  --second outputs/epochs7to8/checkpoints/checkpoint-92394 \
  --questions /path/to/test_questions.jsonl \
  --output outputs/average
```

Готовый архив модели: `outputs/average/kaggle_model.zip`.

## Инференс в Kaggle

1. Скачать [готовые веса](https://github.com/Stefan2417/hse-nlp/releases/download/hw2-final/hw2_kaggle_average_60000_92394.zip)
   и загрузить ZIP в приватный Kaggle Dataset. Kaggle автоматически его распакует.
2. Импортировать `kaggle_inference.ipynb` и подключить Dataset через **Add Input**.
3. Выбрать **GPU T4 ×2**. Код использует одну карту — `cuda:0`.
4. Оставить `mode = "beam4"` или поменять на `mode = "greedy"`.
5. Запустить **Run All** и скачать из Output `submission_beam4.jsonl`
   или `submission_greedy.jsonl`. Это файл для бота: 1919 строк с полем `code`.

Ноутбук работает с локальными весами из Dataset, без интернета. В конце выводится
время генерации и полное время с загрузкой; замер нужно делать в новой сессии.
Архив модели должен содержать `kaggle_hw2/model/` и `kaggle_hw2/test_questions.jsonl`.

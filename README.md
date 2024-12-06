# Обучение GPT-2 на русском языке

Проект реализует обучение модели GPT-2 на русскоязычных данных с использованием датасета Alpaca.

## Установка
```bash
git clone https://github.com/lenjjiv/Pretrain-GPT.git
cd Pretrain-GPT
```

```bash
python -m venv venv
source venv/bin/activate  # для Linux/MacOS
# или
venv\Scripts\activate     # для Windows
```

```bash
pip install -r requirements.txt
```

## Обучение

Для обучения:

```bash
python scripts/train.py
```

Пайплайн:
1. Подготовка токенизатора
2. Инициализация модели
3. Загрузка и подготовка датасета
4. Обучение модели
5. Сохранение

## Инференс

```python
from src.generator import TextGenerator

generator = TextGenerator(
    model_path="./gpt2_finetuned_final",
    device=0  # используйте -1 для CPU
)

text = generator.generate(
    prompt="some_text",
    max_length=100,
    temperature=0.8
)
print(text)
```

## Параметры обучения

Параметры можно найти в `src/trainer.py`:

- epochs: 5
- batch_size: 16
- learning_rate: 5e-5
- gradient_accumulation: 10
- max_length: 512 токенов

## Технические детали

- base_model: GPT-2
- dataset: Alpaca Russian
- vocab_size: 20,000 токенов (обрезанный словарь токенизатора `ai-forever/sbert_large_nlu_ru`)
- hidden_dim: 384 (768/2)
- num_transformer_blocks: 12
- attention_heads: 12

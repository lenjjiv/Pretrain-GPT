import sys
from pathlib import Path

# Добавляем src в путь для импортов
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from tokenizer import prepare_tokenizer
from dataset import get_dataset, get_data_collator
from model import create_model
from trainer import get_training_args, setup_trainer, train_and_save

def main():
    # Инициализация токенизатора
    print("Подготовка токенизатора...")
    tokenizer = prepare_tokenizer()
    
    # Создание модели
    print("Создание модели...")
    model = create_model(tokenizer.vocab_size)
    
    # Подготовка датасета
    print("Загрузка датасета...")
    dataset = get_dataset(tokenizer)
    data_collator = get_data_collator(tokenizer)
    
    # Настройка обучения
    print("Настройка параметров обучения...")
    training_args = get_training_args()
    trainer = setup_trainer(model, training_args, dataset, data_collator)
    
    # Обучение и сохранение
    print("Начало обучения...")
    output_dir = "./gpt2_finetuned_final"
    train_and_save(trainer, output_dir, tokenizer)
    print(f"Обучение завершено. Модель сохранена в {output_dir}")

if __name__ == "__main__":
    main()
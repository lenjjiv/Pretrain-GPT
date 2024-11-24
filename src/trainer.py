from transformers import TrainingArguments, Trainer

def get_training_args(output_dir="./gpt2_finetuned"):
    """
    Конфигурация параметров обучения.
    
    Args:
        output_dir: Директория для сохранения результатов
        
    Returns:
        TrainingArguments: Настройки обучения
    """
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=10000,                    # максимальное количество шагов
        num_train_epochs=5,                 # количество эпох
        per_device_train_batch_size=16,     # размер батча
        gradient_accumulation_steps=10,      # шаги накопления градиента
        warmup_steps=1000,                  # шаги разогрева
        learning_rate=5e-5,                 # скорость обучения
        fp16=True,                          # использование fp16
        gradient_checkpointing=True,        # чекпоинты градиента
        logging_steps=10,                   # шаги логирования
        save_steps=1000,                    # шаги сохранения
        save_total_limit=2,                 # максимальное количество сохранений
    )

def setup_trainer(model, training_args, dataset, data_collator):
    """
    Настройка тренера.
    
    Args:
        model: Модель для обучения
        training_args: Параметры обучения
        dataset: Датасет
        data_collator: Коллатор данных
        
    Returns:
        Trainer: Настроенный тренер
    """
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

def train_and_save(trainer, output_dir, tokenizer):
    """
    Обучение модели и сохранение результатов.
    
    Args:
        trainer: Тренер
        output_dir: Директория для сохранения
        tokenizer: Токенизатор
    """
    trainer.train()
    
    # Сохраняем модель и токенизатор
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
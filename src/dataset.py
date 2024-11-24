from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

class TextDataset(Dataset):
    """Кастомный датасет для текстовых данных."""
    def __init__(self, texts, tokenizer, max_length=512):
        """
        Args:
            texts: Список текстов
            tokenizer: Токенизатор
            max_length: Максимальная длина последовательности
        """
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def get_dataset(tokenizer, cache_dir="./dataset"):
    """
    Загрузка и подготовка датасета.
    
    Args:
        tokenizer: Токенизатор для обработки текстов
        cache_dir: Директория для кэширования датасета
        
    Returns:
        Dataset: Подготовленный датасет
    """
    dataset = load_dataset(
        "pinzhenchen/alpaca-cleaned-ru",
        cache_dir=cache_dir,
        streaming=True
    )

    def tokenize_function(examples):
        """Токенизация примеров на лету."""
        return tokenizer(
            examples["output"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors=None
        )

    # Применяем токенизацию к потоковому датасету
    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        remove_columns=["output"]
    )

    return tokenized_dataset

def get_data_collator(tokenizer):
    """
    Создание коллатора данных для языкового моделирования.
    
    Args:
        tokenizer: Токенизатор
        
    Returns:
        DataCollator: Коллатор данных
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )
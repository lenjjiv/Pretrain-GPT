from transformers import AutoTokenizer, BertTokenizer

def get_base_tokenizer(model_name='ai-forever/sbert_large_nlu_ru'):
    """Загрузка базового токенизатора из предобученной модели."""
    return AutoTokenizer.from_pretrained(model_name)

def normalize_vocab(tokenizer):
    """
    Нормализация словаря путем удаления дубликатов по регистру.
    
    Args:
        tokenizer: Исходный токенизатор
        
    Returns:
        BertTokenizer: Новый токенизатор с нормализованным словарем
    """
    vocab = tokenizer.get_vocab()
    
    norm_vocab = {}  # нормализованный словарь без дубликатов
    seen_lower = {}  # для отслеживания уже встреченных токенов
    
    # Сортируем по ID чтобы сохранить приоритет токенов
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        token_lower = token.lower()
        
        if token_lower not in seen_lower:
            norm_vocab[token] = idx
            seen_lower[token_lower] = token
            
    # Записываем нормализованный словарь
    with open('vocab.txt', 'w', encoding='utf-8') as f:
        for token in norm_vocab:
            f.write(f"{token}\n")
            
    return BertTokenizer('vocab.txt', do_lower_case=True)

def cut_vocab(tokenizer, max_tokens=20000):
    """
    Уменьшение размера словаря до указанного максимума.
    
    Args:
        tokenizer: Исходный токенизатор
        max_tokens: Максимальное количество токенов в новом словаре
        
    Returns:
        BertTokenizer: Новый токенизатор с уменьшенным словарем
    """
    vocab = tokenizer.get_vocab()
    top_tokens = dict(sorted(vocab.items(), key=lambda x: x[1])[:max_tokens])
    
    # Записываем сокращенный словарь
    with open('vocab.txt', 'w', encoding='utf-8') as f:
        for token in top_tokens:
            f.write(f"{token}\n")
            
    return BertTokenizer('vocab.txt', do_lower_case=tokenizer.do_lower_case)

def prepare_tokenizer():
    """
    Подготовка токенизатора с нормализацией и уменьшением словаря.
    
    Returns:
        BertTokenizer: Готовый токенизатор для обучения
    """
    base_tokenizer = get_base_tokenizer()
    normalized = normalize_vocab(base_tokenizer)
    final_tokenizer = cut_vocab(normalized, max_tokens=20000)
    return final_tokenizer
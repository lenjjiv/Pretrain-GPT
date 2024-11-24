from transformers import GPT2Config, GPT2LMHeadModel

def get_model_config(vocab_size):
    """
    Создание конфигурации модели.
    
    Args:
        vocab_size: Размер словаря
        
    Returns:
        GPT2Config: Конфигурация модели
    """
    return GPT2Config(
        vocab_size=vocab_size,  # размер словаря
        n_positions=1024,       # максимальная длина последовательности
        n_embd=768//2,         # размер эмбеддингов
        n_layer=12,            # количество слоев
        n_head=12,             # количество голов внимания
        activation_function='gelu'
    )

def create_model(vocab_size):
    """
    Создание и инициализация модели.
    
    Args:
        vocab_size: Размер словаря
        
    Returns:
        GPT2LMHeadModel: Инициализированная модель
    """
    config = get_model_config(vocab_size)
    return GPT2LMHeadModel(config)
from transformers import pipeline

class TextGenerator:
    def __init__(self, model_path, device=0):
        """
        Инициализация генератора текста.
        
        Args:
            model_path: Путь к модели
            device: Устройство для инференса (0 - GPU, -1 - CPU)
        """
        self.generator = pipeline(
            "text-generation",
            model=model_path,
            device=device
        )

    def generate(
        self,
        prompt,
        max_length=100,
        num_return_sequences=1,
        temperature=1.0,
        top_p=0.85,
        top_k=2
    ):
        """
        Генерация текста на основе промпта.
        
        Args:
            prompt: Начальный текст
            max_length: Максимальная длина генерации
            num_return_sequences: Количество вариантов генерации
            temperature: Температура (креативность) генерации
            top_p: Порог вероятности для nucleus sampling
            top_k: Количество топ-токенов для выборки
            
        Returns:
            list: Список сгенерированных текстов
        """
        outputs = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            no_repeat_ngram_size=2  # предотвращение повторений n-грамм
        )
        
        return outputs
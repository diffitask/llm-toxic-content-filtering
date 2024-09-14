# Структура репозитория

## Эксперименты с данными:
- **`multiclass_classification_script.ipynb`** – ноутбук для мультиклассовой разметки данных с использованием API GPT-4.
- **`wildjailbreaks_translation_script.py`** – скрипт для перевода датасета jailbreak prompts на русский язык.

## Эксперименты с моделями:
- **`jailbreak_clf_llama_instruct.ipynb`** – ноутбук для дообучения модели Saiga Llama на задачу бинарной классификации, с генерацией токенов и последующей интерпретацией классов из вывода модели.
- **`jailbreaks_clf_baseline.ipynb`** – ноутбук с базовой моделью для классификации jailbreak prompts.
- **`jailbreaks_clf_llama_multiclass.ipynb`** – ноутбук для дообучения модели Saiga Llama на задачу мультиклассовой классификации с генерацией токенов и последующим выделением классов.
- **`jailbreaks_clf_mistral_binary.ipynb`** – ноутбук для дообучения модели Mistral Instruct на задачу бинарной классификации в формате sequence classification.
- **`jailbreaks_clf_mistral_instruct.ipynb`** – ноутбук для дообучения модели Mistral Instruct на задачу бинарной классификации с генерацией токенов и выделением классов из вывода модели.

## Директория с классификаторами:
- **`TF-IDF`** – классификатор на основе базовой модели с использованием метода TF-IDF.
- **`llama_instruct`** – бинарный классификатор на базе модели Saiga Llama.
- **`llama_multiclass_instruct`** – мультиклассовый классификатор на основе модели Saiga Llama.
- **`mistral_instruct`** – бинарный классификатор на базе модели Mistral Instruct.

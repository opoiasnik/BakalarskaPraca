import os
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from mistralai import Mistral

# Настройка логирования
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка API ключа для Mistral
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("API ключ не найден. Убедитесь, что переменная MISTRAL_API_KEY установлена.")

client = Mistral(api_key=api_key)

# Инициализация модели Sentence-BERT
logger.info("Загрузка модели SentenceTransformer...")
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Подключение к Elasticsearch
logger.info("Подключение к Elasticsearch...")
es = Elasticsearch("http://localhost:9200")

def process_query_with_mistral(query, k=10):
    # Генерация вектора запроса с помощью SBERT
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)

    # Поиск в Elasticsearch с использованием подсветки
    response = es.search(
        index='drug_docs',
        body={
            'query': {
                'match': {'text': query}
            },
            'highlight': {
                'fields': {
                    'text': {}
                },
                'fragment_size': 200,  # Уменьшаем размер фрагмента до 200 символов
                'number_of_fragments': 1  # Один фрагмент на документ
            },
            'size': k
        }
    )

    hits = response['hits']['hits']
    if not hits:
        return "Ничего не найдено", []

    # Извлекаем релевантные фрагменты и ограничиваем размер текста
    top_documents = hits[:3]  # Сохраняем 3 документа
    documents = []
    links = []
    for hit in top_documents:
        if 'highlight' in hit:
            doc_text = hit['highlight']['text'][0]
        else:
            doc_text = hit['_source']['text'][:200]  # Обрезаем текст до 200 символов
        documents.append(doc_text)
        links.append(hit['_source'].get('link', '-'))

    # Формирование структурированного запроса для Mistral
    structured_prompt = (
        f"Na základe otázky: '{query}' a nasledujúcich informácií: {documents}, "
        "poskytnite odpoveď, ktorá obsahuje tri lieky alebo riešenia s krátkym vysvetlením pre každý z nich. "
        "Poskytnite stručné a štruktúrované vysvetlenie vhodné pre otázku, zamerané na kľúčové body. "
        "Odpoveď by mala byť poskytnutá v slovenčine."
    )

    # Проверяем количество токенов в промпте
    token_count = len(structured_prompt.split())
    print(f"Количество токенов в промпте: {token_count}")

    # Если промпт все еще слишком большой, информируем пользователя
    if token_count > 32000:
        return "К сожалению, запрос слишком большой для обработки моделью. Попробуйте уточнить вопрос или сократить объем данных.", links

    # Запрос к модели Mistral
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{
            "content": structured_prompt,
            "role": "user",
        }]
    )

    summary = response.choices[0].message.content if response else "Ответ не был сгенерирован"
    return summary, links


def main():
    print("Добро пожаловать! Введите ваш запрос или 'exit' для выхода.")
    while True:
        query = input("Ваш вопрос: ")
        if query.lower() == 'exit':
            break
        summary, links = process_query_with_mistral(query)
        print("\nОтвет:")
        print(summary)
        print("\nСсылки:")
        for link in links:
            print(f"- {link}")
        print("\n")

if __name__ == "__main__":
    main()

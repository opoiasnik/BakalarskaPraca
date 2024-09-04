import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util
import spacy
from datasets import load_dataset

# Загрузка моделей и токенизаторов
qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

# Модель для извлечения эмбеддингов предложений
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Подключение к Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Загрузка реального датасета из Hugging Face
dataset = load_dataset("mtarasovic/ner-rent-sk-dataset")

# Функция для индексирования документов в Elasticsearch
def index_documents(dataset):
    for i, item in enumerate(dataset['train']):
        # Пример использования поля 'text' для индексации
        doc = item['text']
        es.index(index='legal_docs', id=i, body={'text': doc})
        print(f"Indexed document {i}")

print("Indexing all documents from dataset...")
index_documents(dataset)

# Функция поиска релевантных документов с использованием Sentence Transformers
def search_documents(query, k=10):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    response = es.search(
        index='legal_docs',
        body={
            'query': {
                'match_all': {}  # Запрос, который вернет все документы
            },
            'size': k
        }
    )
    documents = [hit['_source']['text'] for hit in response['hits']['hits']]
    
    # Проверка количества найденных документов
    if not documents:
        return []  # Возвращаем пустой список, если документы не найдены
    
    # Получение эмбеддингов для всех документов
    document_embeddings = sbert_model.encode(documents, convert_to_tensor=True)
    
    # Нахождение наиболее релевантных документов
    scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    
    # Используем min(k, len(documents)) для topk
    top_k = min(k, len(documents))
    top_k_indices = scores.topk(top_k).indices
    top_documents = [documents[idx] for idx in top_k_indices]
    return top_documents

# Функция для извлечения ответов с помощью модели вопрос-ответ (Question-Answering)
def get_answer(question, context):
    inputs = qa_tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
    input_ids = inputs["input_ids"].tolist()[0]

    answer_start_scores, answer_end_scores = qa_model(**inputs).values()
    answer_start = answer_start_scores.argmax()
    answer_end = answer_end_scores.argmax() + 1

    answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer

# Основная функция для обработки запросов
def process_query(query):
    # Поиск релевантных документов
    relevant_docs = search_documents(query)
    
    # Извлечение ответов из релевантных документов
    answers = []
    for doc in relevant_docs:
        answer = get_answer(query, doc)
        answers.append((doc, answer))
    
    return answers

# Пример использования
if __name__ == "__main__":
    queries = [
        "Aké sú práva podľa zákona o ochrane súkromia?",
        "Aké sú pravidlá podľa zákona o ochrane údajov?",
        "Čo umožňuje zákon o práve na zabudnutie?",
        "Čo je zmluva o prenájme bytu?",
        "Aké sú podmienky prenájmu kancelárskych priestorov?"
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        answers = process_query(query)
        for doc, answer in answers:
            print(f"\nDokument: {doc}\nOdpoveď: {answer}")

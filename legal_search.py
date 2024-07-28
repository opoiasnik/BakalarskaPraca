import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util
import nltk
import spacy

# Загрузка моделей и токенизаторов
qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

nlp = spacy.load("en_core_web_sm")

# Подключение к Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Функция для индексирования документов
def index_documents(documents):
    for i, doc in enumerate(documents):
        es.index(index='legal_docs', id=i, body={'text': doc})
        print(f"Indexed document {i}")

# Пример документов
documents = [
    "Prvý zákon o ochrane údajov. Tento zákon sa zameriava na ochranu osobných údajov a upravuje pravidlá ich spracovania.",
    "Druhý zákon o ochrane súkromia. Tento zákon stanovuje práva a povinnosti týkajúce sa súkromia jednotlivcov.",
    "Tretí zákon o práve na zabudnutie. Tento zákon umožňuje jednotlivcom žiadať o vymazanie ich osobných údajov z verejných záznamov."
]

print("Indexing documents...")
index_documents(documents)

# Создание фиктивного датасета договоров аренды
rental_documents = [
    "Zmluva o prenájme bytu. Prenajímateľ poskytuje nájomcovi byt do užívania na dobu určitú.",
    "Zmluva o prenájme domu. Prenajímateľ sa zaväzuje poskytnúť nájomcovi dom na bývanie za mesačné nájomné.",
    "Zmluva o prenájme kancelárskych priestorov. Prenajímateľ poskytuje nájomcovi kancelárske priestory na podnikanie."
]

print("Indexing rental agreements...")
index_documents(rental_documents)

# Функция поиска релевантных документов
def search_documents(query, k=10):  # Увеличиваем количество возвращаемых документов
    response = es.search(
        index='legal_docs',
        body={
            'query': {
                'match': {
                    'text': {
                        'query': query,
                        'fuzziness': 'AUTO'  # Использование размытого поиска
                    }
                }
            },
            'size': k
        }
    )
    hits = [hit['_source']['text'] for hit in response['hits']['hits']]
    return hits

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


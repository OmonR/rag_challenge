

### Pipeline 


1.  **Ingestion:** Сканирование `pdfs/`
2.  **Indexing:** Разбиение на чанки (1000 токенов) + OpenAI Embeddings.
3.  **Storage:** Локальное хранилище FAISS.
4.  **Retrieval:** Hybrid Search (**BM25** для текста + **FAISS** для семантики).
5.  **Reasoning:** Генерация ответов через модель **OpenAI o1** на основе системного промпта с контекстом.
6.  **Output:** Сохранение прогресса в JSON с указанием источников и страниц.

---

### Стек

1.  LLM: OpenAI o1
2.  Embeddings: text-embedding-3-small
3.  Orchestration: LangChain
4.  Vector Store: FAISS
5.  Parser: PyMuPDF (fitz)

### Установка

**1. Подготовка окружения**
```bash
pip install -r requirements.txt
```

**2. Создайте файл .env с openai api key**
```bash
OPENAI_API_KEY=sk-proj-ваша-комбинация-символов
```

**3. Запустите main.py**
```bash
python main.py
```

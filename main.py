import os
import json
import glob
import hashlib
import logging
import re

from dotenv import load_dotenv

logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

from typing import List, Dict, Any
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PDF_FOLDER = "pdfs"
INDEX_FOLDER = "faiss_index"
LOG_FILE = "processed_log.json"
MODEL_NAME = "o1"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

SUBMISSION_FILENAME = "submission_Pribytkob_v10.json"
EMAIL = "test@rag-tat.com"
SUBMISSION_NAME = "Pribytkob_v10"

api_key = os.getenv("OPENAI_API_KEY")

class RAGSystem:
    def __init__(self):
        print("Инициализация...")
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        self.llm = ChatOpenAI(model=MODEL_NAME)
        self.vector_store = None
        self.processed_hashes = set()
        
        if os.path.exists(INDEX_FOLDER) and os.path.exists(f"{INDEX_FOLDER}/index.faiss"):
            try:
                self.vector_store = FAISS.load_local(
                    INDEX_FOLDER, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("FAISS индекс загружен.")
            except Exception as e:
                print(f"Ошибка: {e}")

        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                self.processed_hashes = set(json.load(f))

    def save_state(self):
        if self.vector_store:
            self.vector_store.save_local(INDEX_FOLDER)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(list(self.processed_hashes), f)

    def calculate_sha1(self, file_path: str) -> str:
        sha1 = hashlib.sha1()
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(65536)
                if not data: break
                sha1.update(data)
        return sha1.hexdigest()

    def ingest_documents(self, pdf_files: List[str]):
        files_to_process = []
        file_hashes_map = {}

        for file_path in pdf_files:
            try:
                f_hash = self.calculate_sha1(file_path)
                file_hashes_map[file_path] = f_hash
                if f_hash not in self.processed_hashes:
                    files_to_process.append(file_path)
            except Exception:
                pass 
        
        if files_to_process:
            print(f"Обработка {len(files_to_process)} новых файлов...")
            documents = []
            for file_path in files_to_process:
                try:
                    loader = PyMuPDFLoader(file_path)
                    pages = loader.load()
                    f_hash = file_hashes_map[file_path]
                    
                    for page in pages:
                        page.metadata["pdf_sha1"] = f_hash
                        page.page_content = page.page_content.replace('\n', ' ')
                    
                    documents.extend(pages)
                    print(f"Прочитан: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Ошибка чтения {file_path}: {e}")

            if documents:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(documents)
                
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                else:
                    self.vector_store.add_documents(chunks)
                
                self.vector_store.save_local(INDEX_FOLDER)
                
                for f_path in files_to_process:
                    self.processed_hashes.add(file_hashes_map[f_path])
                self.save_state()

        print("Индексация завершена.")

    def get_retriever(self):
        if not self.vector_store:
            return None
            
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        
        try:
            from langchain_community.retrievers import BM25Retriever
            if hasattr(self.vector_store, "docstore") and hasattr(self.vector_store.docstore, "_dict"):
                all_docs = list(self.vector_store.docstore._dict.values())
                bm25_retriever = BM25Retriever.from_documents(all_docs)
                bm25_retriever.k = 10
                
                if EnsembleRetriever:
                    return EnsembleRetriever(
                        retrievers=[bm25_retriever, faiss_retriever],
                        weights=[0.4, 0.6]
                    )
        except Exception as e:
            print(f"BM25/Ensemble не удалось создать: {e}. Используем FAISS.")
        
        return faiss_retriever

    def generate_answer(self, question: Dict[str, Any]) -> Dict[str, Any]:
        query_text = question["text"]
        q_kind = question.get("kind", "text")
        
        retriever = self.get_retriever()
        if not retriever:
            return {"question_text": query_text, "value": "N/A", "references": []}
            
        docs = retriever.invoke(query_text)
        
        context_text = "\n\n".join([
            f"[Source: {d.metadata.get('pdf_sha1', 'unknown')} | Page: {d.metadata.get('page', 0)}]\n{d.page_content}" 
            for d in docs
        ])
        
        system_instr = """You are a meticulous analyst. 
        Answer based ONLY on the context.
        Process:
        1. Analyze the Question to understand data needed.
        2. Scan Context.
        3. Output FINAL value inside <answer> tags.
        
        Output Rules:
        - If not found: <answer>N/A</answer>
        - Boolean: <answer>true</answer> / <answer>false</answer>
        - Number: <answer>1234.56</answer> (No currency symbols)
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instr),
            ("user", f"Context:\n{context_text}\n\nQuestion: {query_text}")
        ])
        
        try:
            chain = prompt | self.llm
            res = chain.invoke({})
            raw_content = res.content.strip()
            
            match = re.search(r"<answer>(.*?)</answer>", raw_content, re.DOTALL)
            ans = match.group(1).strip() if match else raw_content
            ans = ans.replace("'", "").replace('"', "")
            
        except Exception as e:
            print(f"Ошибка LLM: {e}")
            ans = "N/A"

        final_value = ans
        is_negative_answer = False
        
        if ans == "N/A":
            is_negative_answer = True
            final_value = "N/A"
        elif q_kind == "boolean":
            if "true" in ans.lower(): final_value = True
            else: 
                final_value = False
                is_negative_answer = True 
        elif q_kind == "number":
            if ans == "N/A":
                is_negative_answer = True
            else:
                clean_str = re.sub(r'[^\d.-]', '', ans)
                try:
                    val_float = float(clean_str)
                    final_value = int(val_float) if val_float.is_integer() else val_float
                except ValueError:
                    final_value = "N/A"
                    is_negative_answer = True

        references = []
        if not is_negative_answer:
            seen_refs = set()
            for d in docs[:3]: 
                sha1 = d.metadata.get("pdf_sha1")
                page_idx = int(d.metadata.get("page", 0)) + 1
                if sha1:
                    ref_key = (sha1, page_idx)
                    if ref_key not in seen_refs:
                        references.append({"pdf_sha1": sha1, "page_index": page_idx})
                        seen_refs.add(ref_key)
        
        return {
            "question_text": query_text,
            "value": final_value,
            "references": references
        }

def main():
    search_path = os.path.join(PDF_FOLDER, "*.pdf")
    pdf_files = glob.glob(search_path)
    if not pdf_files:
        print(f"ОШИБКА: PDF файлы не найдены в {PDF_FOLDER}")
        return

    try:
        with open("questions.json", "r", encoding="utf-8") as f:
            questions = json.load(f)
    except FileNotFoundError:
        print("Ошибка: questions.json не найден.")
        return

    rag = RAGSystem()
    rag.ingest_documents(pdf_files)
    if not rag.vector_store:
        print("Индекс пуст. Завершение.")
        return

    if os.path.exists(SUBMISSION_FILENAME):
        print(f"Найден файл {SUBMISSION_FILENAME}, загружаем прогресс...")
        try:
            with open(SUBMISSION_FILENAME, "r", encoding="utf-8") as f:
                submission = json.load(f)
                if "answers" not in submission:
                    submission["answers"] = []
        except json.JSONDecodeError:
            print("Ошибка чтения JSON. Начинаем заново.")
            submission = {
                "team_email": EMAIL,
                "submission_name": SUBMISSION_NAME,
                "answers": []
            }
    else:
        submission = {
            "team_email": EMAIL,
            "submission_name": SUBMISSION_NAME,
            "answers": []
        }

    answered_questions_texts = {a["question_text"] for a in submission["answers"]}

    print("\nГенерация ответов...")
    
    for i, q in enumerate(questions):
        q_text = q["text"]
        
        if q_text in answered_questions_texts:
            continue

        print(f"Вопрос {i+1}/{len(questions)}: Обработка...")
        
        try:
            answer = rag.generate_answer(q)
            
            submission["answers"].append(answer)
            
            with open(SUBMISSION_FILENAME, "w", encoding="utf-8") as f:
                json.dump(submission, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"!!! Ошибка при обработке вопроса {i+1}: {e}")

    print(f"\n{SUBMISSION_FILENAME} сохранён")

if __name__ == "__main__":
    main()
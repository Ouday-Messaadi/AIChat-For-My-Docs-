from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever , EnsembleRetriever
from langchain_community.retrievers import TFIDFRetriever
import os
import shutil
from typing import List
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from gensim.downloader import load as gensim_load

class RAGSystem:
    def __init__(self):
        # client
        load_dotenv()                                 
        api_key = os.getenv("DEEPSEEK_API_KEY")       
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not found in .env")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )


        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2", # better than MiniLM
            model_kwargs={'device':'cpu'}
        )

        # 2. Alternative: Multilingual support
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        # )

        self.chroma_path = "chroma"
        self.data_path = "data/"
        self.chunks = []

        # i am using gensim to get similar words so that i can expand my query
        print("Loading Word2Vec model... (only once)")
        self.word2vec = gensim_load("word2vec-google-news-300") # will download only the first time

    def create_data_base(self, force_rebuild=False):
        if not force_rebuild and os.path.exists(self.chroma_path):
            print(f"Loading existing Chroma DB from {self.chroma_path}")
            db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)
            results = db.get()
            documents = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(results['documents'], results['metadatas'])
            ]
            self.chunks = documents
            return
        
        print("Creating or rebuilding Chroma DB...")
        documents = self.load_documents()
        chunks = self.text_splitter(documents)
        self.save_to_chroma(chunks)
        self.chunks = chunks

    def load_documents(self):
        loader = DirectoryLoader(self.data_path, glob="*.md")
        documents = loader.load()
        return documents
    
    def text_splitter(self, documents : List[Document]):
        """Advanced text splitter with overlap and semantic boundaries"""
        # better with semantic awareness
        text_splitter_f = RecursiveCharacterTextSplitter(
            chunk_size=500, # larger chunks for better context
            chunk_overlap=150,
            length_function=len,
            add_start_index=True,
            separators=["\n\n","\n",".","!","?",".","",""]
        )

        chunks = text_splitter_f.split_documents(documents)

        # add metadata enhancement
        for chunk in chunks:
            # add chunk length info
            chunk.metadata['chunk_length'] = len(chunk.page_content)
            # add simple keyword extraction
            words = chunk.page_content.lower().split()
            keywords = list(set([w for w in words if len(w) > 4]))[:10]
            chunk.metadata['keywords'] = ", ".join(keywords) 


        print(f'Split {len(documents)} documents into {len(chunks)} chunks')
        return chunks
    
    def save_to_chroma(self, chunks: List[Document]):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

        db = Chroma.from_documents(
            chunks, self.embeddings, persist_directory=self.chroma_path
        )
        db.persist()
        print(f'Saved {len(chunks)} chunks to {self.chroma_path}')

    def hybrid_retriever(self):
        """this retriever combines semantic and keyword search"""
        # 1. Vector/Semantic retriever
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)
        vector_retriever = db.as_retriever(search_kwargs={"k":5})

        # 2. BM25/Vector retriever
        bm25_retriever = BM25Retriever.from_documents(self.chunks)
        bm25_retriever.k = 5

        # 3. Ensemble retriever (combines both)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7,0.3] # 70% semantic , 30% keyword
        )

        return ensemble_retriever
        
    def make_query(self, query:str, method="hybrid"):
        """multiple query strategies"""
        if method=="hybrid":
            return self.hybrid_search(query)
        elif method=="semantic_with_rerank":
            return self.semantic_with_rerank(query)
        elif method=="query_expansion":
            return self.query_expansion_search(query)
        else:
            return self.basic_search(query)
        
    def hybrid_search(self, query:str):
        """combines semantic and keyword search"""
        retriever = self.hybrid_retriever()
        results = retriever.get_relevant_documents(query)

        print(f'Hybrid search found {len(results)} results')
        return results[:3]
    
    def semantic_with_rerank(self, query:str):
        """get more results then rerank by relevance"""
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)

        # get more candidates
        results = db.similarity_search_with_relevance_scores(query, k=10)

        # simple rerank
        query_terms = set(query.lower().split())

        scored_results = []
        for doc, similarity_score in results:
            doc_terms = set(doc.page_content.lower().split())
            term_overlap = len(query_terms.intersection(doc_terms))/len(query_terms)

            # combined score : semantic similarity + term overlap
            combined_score = similarity_score * 0.7 + term_overlap * 0.3
            scored_results.append((doc,combined_score))

        # sort by combined score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        print(f'Reranked search found {len(scored_results)} results')
        return [doc for doc , _ in scored_results[:3]]
    
    def get_similar_words(self, word:str, topn=3):
        """Get top n similar words from Word2Vec"""
        try:
            similar = self.word2vec.most_similar(word, topn=topn)
            return [word] + [w for w, _ in similar]
        except:
            return [word]

    def query_expansion_search(self, query:str):
        """expand query with synonyms and related terms"""
        # simple query expansion ( njm nestaamel wordNet , word2Vec for better results)
        words = query.lower().split()

        expanded_terms=[]
        for word in words:
            expanded_terms += self.get_similar_words(word)

        expanded_query = " ".join(expanded_terms)
        print(f"Expanded query:  '{query}' -> '{expanded_query}'")

        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)
        results = db.similarity_search(expanded_query, k=3)
        print(f'Found {len(results)} results')

        return results

    def basic_search(self, query:str):
        """basic semantic search"""
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)
        results = db.similarity_search(query, k=3)

        return results

    def generate_response(self, query:str, documents:List[Document]) -> str:
        if not documents:
            return "No relevant information found !"
        
        # create rich context 
        context_parts = []
        for i, doc in enumerate(documents):
            entities = doc.metadata.get('entities', {})
            sentiment = doc.metadata.get('sentiment', 'neutral')

            context_part = f"--- Context {i+1} (Sentiment: {sentiment}) ---\n"
            context_part += doc.page_content

            if entities:
                context_part += f"\n[key entities: {entities}]"

            context_parts.append(context_part)

        context = "\n\n".join(context_parts)

        prompt = f"""Based on the following contextual information with semantic analysis, provide a comprehensive and accurate answer to the question.

Context with semantic metadata:
{context}

Question: {query}

Instructions:
1. Use the semantic metadata (entities, sentiment) to provide richer context
2. Reference specific characters, locations, or objects mentioned
3. If multiple contexts are provided, synthesize information from all relevant sources
4. Be specific and detailed in your response

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek/deepseek-r1-0528:free",
                messages=[{"role":"user","content":prompt}],
                temperature=0.3 # lower temperature for more focused  responses
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response : {e}"
        

if __name__ == "__main__":
    # init the RAG
    rag_system = RAGSystem()

    print("Creating semantic database...")
    rag_system.create_data_base(force_rebuild=False)

    # Test queries
    """test_queries = [
        "Is the main character a shepherd ?",
        "Where was the treasure found ?",
        "Why did he choose to travel ?",
        "Did he meet a fortune teller or a witch ?"
    ]
    """

    print("\n🧠 Ask anything! (type 'exit' to quit)\n")
    while True:
        _input = input("📝 Your question: ").split()
        if len(_input)==1 and ( _input[0].lower()=="exit" or _input[0].lower()=="quit"):
            print("👋 Exiting. Goodbye!")
            break
        
        query=""
        for w in _input:
            query += w
            query +=" "
    
        documents = rag_system.make_query(query=query, method="query_expansion")
        for i, doc in enumerate(documents, 1):  
            print(f"--- Result {i} ---")
            #print(f"Metadata: {doc.metadata}")
            snippet = doc.page_content.strip().replace("\n", " ")
            print(f"Content Snippet: {snippet[:300]}{'...' if len(snippet) > 300 else ''}")
            print()
        response = rag_system.generate_response(query=query,documents=documents)
        print(response)
        print()

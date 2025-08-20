from typing import List, Iterator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain_groq.chat_models import ChatGroq
import os
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import shutil

# from langchain.retrievers import BM25Retriever, ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(
        self,
        groq_api_key: str,
        persist_directory: str = None
    ):
        """
        Initialize the RAG Engine with ChromaDB.
        
        Args:
            groq_api_key (str): Groq API key for LLM
            persist_directory (str): Directory to persist ChromaDB
        """
        self.groq_api_key = groq_api_key
        
        # Use /tmp for Cloud Run compatibility, or provided directory
        if persist_directory is None:
            persist_directory = "/tmp/chroma_db"
        
        self.persist_directory = persist_directory
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Improved embeddings model for better semantic understanding
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize vector store with metadata filtering capability
        try:
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            # Ensure directory exists and try again
            os.makedirs(persist_directory, exist_ok=True)
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
        
        # More sophisticated text splitter with better chunk management
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks for more precise retrieval
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
            keep_separator=True,
            add_start_index=True,
        )
        
        self.llm = ChatGroq(
            temperature=0.2,  # Reduced temperature for more focused responses
            groq_api_key=groq_api_key,
            model_name="llama3-70b-8192",
            max_tokens=2048
        )

    def load_documents(self, file_path: str) -> List:
        """
        Load documents from a file.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            List: List of loaded documents
        """
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents

    def process_documents(self, documents: List) -> None:
        """
        Process documents with improved chunking and metadata.
        Clears existing database before processing new documents.
        """
        try:
            # Use the stored persist directory path
            persist_dir = self.persist_directory
            
            # Safely delete the persist directory if it exists
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
            
            # Ensure the directory is created
            os.makedirs(persist_dir, exist_ok=True)
            
            # Now re-instantiate Chroma with proper error handling
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            # Fallback: ensure directory exists and try again
            persist_dir = self.persist_directory
            os.makedirs(persist_dir, exist_ok=True)
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"}
            )
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Add metadata to chunks for better context
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'document_id': f"doc_{i//10}",  # Group every 10 chunks
                'total_chunks': len(chunks)
            })
        
        print(f"Created {len(chunks)} chunks")
        self.vector_store.add_documents(chunks)
        print(f"Vector store now contains {self.vector_store._collection.count()} documents")

    def query(self, query: str, k: int = 4) -> str:
        """
        Simple query processing with basic retrieval.
        """
        # Create basic retriever
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k}
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        # Get response
        response = qa_chain.invoke(query)
        # Extract the answer from the response dictionary
        return response['result']

    def stream_query(self, query: str, k: int = 4) -> Iterator[str]:
        """
        Stream the response for a query in real-time.
        
        Args:
            query (str): The query to process
            k (int): Number of documents to retrieve
            
        Returns:
            Iterator[str]: An iterator that yields response chunks
        """
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k}
        )

        # Create streaming LLM
        streaming_llm = ChatGroq(
            temperature=0.2,
            groq_api_key=self.groq_api_key,
            model_name="llama3-70b-8192",
            max_tokens=2048,
            streaming=True
        )

        # Create QA chain with streaming
        qa_chain = RetrievalQA.from_chain_type(
            llm=streaming_llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        # Stream the response
        for chunk in qa_chain.stream(query):
            if 'result' in chunk:
                # Split the result into smaller chunks for better streaming
                words = chunk['result'].split()
                if words:
                    yield ' '.join(words) + ' '

# Example usage
if __name__ == "__main__":
    # Initialize RAG engine
    rag = RAGEngine(
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Load and process documents
    documents = rag.load_documents("Paper15082.pdf")
    rag.process_documents(documents)
    
    # Example of streaming query
    print("Streaming response for query:")
    for chunk in rag.stream_query("What is the main topic of the document?"):
        print(chunk, end="", flush=True)
  

   
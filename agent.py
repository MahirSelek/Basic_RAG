from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from Chorma import Retriever

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama3",temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
            maximum and keep the answer concise </s> 
            Question: {question} 
            Context: {context} 
            Answer: 
            """
        )
        self.retriever = Retriever()


    def ingest(self, pdf_file_path: str):
        pdf = self.retriever.get_documents(pdf_file_path)
        self.retriever.save_vectordb_locally(documents=pdf)
        
    def ask(self, query: str):
        content = self.retriever.rag()
       
        self.chain = ({"context": content, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
         
        answer = self.chain.invoke(query)

        return answer

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
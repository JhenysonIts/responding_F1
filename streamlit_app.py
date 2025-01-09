import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Carregar variáveis de ambiente
load_dotenv()

# Configurar embeddings do OpenAI
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Configurar modelo de chat
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# Configurar a interface do Streamlit
st.title("Sistema de Perguntas e Respostas - Fórmula 1 🏎️")

# Upload do arquivo PDF
uploaded_file = st.file_uploader("Faça upload do arquivo PDF", type=["pdf"])

if uploaded_file:
    # Carregar o PDF
    loader = PDFPlumberLoader(uploaded_file)
    documents = loader.load()

    # Validação do texto carregado
    if not documents or all(not doc.page_content for doc in documents):
        st.error("Nenhum texto foi carregado do PDF. Verifique o arquivo.")
    else:
        # Dividir texto em pedaços menores
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Criar armazenamento vetorial
        vectorstore = FAISS.from_documents(texts, embeddings_model)

        # Configurar sistema de perguntas e respostas
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever)

        # Caixa de entrada para perguntas
        question = st.text_input("Digite sua pergunta:")
        
        if st.button("Enviar"):
            if question:
                with st.spinner("Processando..."):
                    try:
                        answer = qa_chain.invoke({"query": question})
                        st.success(f"Resposta: {answer}")
                    except Exception as e:
                        st.error(f"Erro ao processar a pergunta: {e}")
            else:
                st.warning("Por favor, digite uma pergunta antes de enviar.")

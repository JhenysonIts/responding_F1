import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from tempfile import NamedTemporaryFile

# Carregar variáveis de ambiente
load_dotenv()

# Configurar embeddings do OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("A chave da API OpenAI não está configurada. Verifique o arquivo .env ou as configurações de ambiente.")
    st.stop()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=api_key)

# Título do App
st.title("Sistema de Perguntas e Respostas - Fórmula 1 🏎️")

# Seção de upload do PDF
st.subheader("Upload do Documento PDF")
uploaded_file = st.file_uploader("Faça upload de um arquivo PDF:", type=["pdf"])

if uploaded_file is not None:
    # Salvar o arquivo temporariamente para leitura
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_pdf_path = temp_file.name
        st.info("Arquivo carregado com sucesso!")

    try:
        # Carregar o PDF usando o PDFPlumberLoader
        loader = PDFPlumberLoader(temp_pdf_path)
        documents = loader.load()

        # Verificar se o texto foi carregado corretamente
        if not documents or all(not doc.page_content for doc in documents):
            st.error("Nenhum texto válido foi encontrado no PDF. Por favor, envie outro arquivo.")
            st.stop()

        # Dividir texto em pedaços menores
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Mostrar exemplos de texto extraído
        st.subheader("Trechos do Texto Extraído:")
        for i, text in enumerate(texts[:3]):
            st.text_area(f"Texto {i+1}:", text.page_content[:200], height=100)

        # Criar armazenamento vetorial
        vectorstore = FAISS.from_documents(texts, embeddings_model)

        # Configurar sistema de perguntas e respostas
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever)

        # Seção de perguntas e respostas
        st.subheader("Faça Sua Pergunta:")
        question = st.text_input("Digite sua pergunta:")

        if st.button("Enviar"):
            if question:
                with st.spinner("Processando sua pergunta..."):
                    try:
                        answer = qa_chain.invoke({"query": question})
                        formatted_answer = answer.get("result", "Erro: Resposta não encontrada.")
                        st.success(f"Resposta: {formatted_answer}")
                    except Exception as e:
                        st.error(f"Erro ao processar a pergunta: {e}")
            else:
                st.warning("Por favor, insira uma pergunta antes de enviar.")

    except Exception as e:
        st.error(f"Erro ao carregar ou processar o PDF: {e}")

else:
    st.warning("Por favor, faça o upload de um arquivo PDF para continuar.")

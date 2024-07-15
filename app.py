import boto3
import streamlit as st
import os
import uuid
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
boto3.set_stream_logger(name='botocore', level=logging.DEBUG)

from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

BUCKET_NAME = os.getenv("BUCKET_NAME")
BEDROCK_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
folder_path = "/tmp/"

s3_client = boto3.client("s3")
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=BEDROCK_REGION)

def get_unique_id():
    return str(uuid.uuid4())

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(request_id, documents):
    try:
        bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        file_name = f"{request_id}.bin"
        vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

        s3_client.upload_file(Filename=f"{folder_path}/{file_name}.faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
        s3_client.upload_file(Filename=f"{folder_path}/{file_name}.pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

        return True
    except ClientError as e:
        st.error(f"Error creating vector store: {e}")
        logger.error(f"Error creating vector store: {e}")
        return False
    except ValueError as e:
        st.error(f"Error during model invocation: {e}")
        logger.error(f"Error during model invocation: {e}")
        return False

def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

def get_llm():
    try:
        llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample': 512})
        return llm
    except ClientError as e:
        st.error(f"Error initializing LLM: {e}")
        logger.error(f"Error initializing LLM: {e}")
        return None

def get_response(llm, vectorstore, question):
    prompt_template = """
    Human: Please use the given context to provide a concise answer to the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>
    Question: {question}
    Assistant:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": question})
    return answer['result']

def main():
    st.title("Admin and Client Site for Chat with PDF")
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Admin", "Client"])

    if app_mode == "Admin":
        st.header("Admin Site")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            request_id = get_unique_id()
            st.write(f"Request Id: {request_id}")
            saved_file_name = f"{request_id}.pdf"
            with open(saved_file_name, mode="wb") as w:
                w.write(uploaded_file.getvalue())

            loader = PyPDFLoader(saved_file_name)
            pages = loader.load_and_split()
            st.write(f"Total Pages: {len(pages)}")

            splitted_docs = split_text(pages, 1000, 200)
            st.write(f"Splitted Docs length: {len(splitted_docs)}")
            st.write("=================")
            st.write(splitted_docs[0])
            st.write("=================")
            st.write(splitted_docs[1])

            st.write("Creating the Vector Store")
            result = create_vector_store(request_id, splitted_docs)

            if result:
                st.write("Hurray!! PDF Processed Successfully")
            else:
                st.write("ERROR !! Please check logs")

    elif app_mode == "Client":
        st.header("Client Site")
        load_index()

        bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
        
        dir_list = os.listdir(folder_path)
        st.write(f"Files and Directories in {folder_path}")
        st.write(dir_list)

        faiss_index = FAISS.load_local(
            index_name="my_faiss",
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )

        st.write("INDEX IS READY")
        question = st.text_input("Please ask your question")
        if st.button("Ask Question"):
            with st.spinner("Querying...."):
                llm = get_llm()
                if llm:
                    st.write(get_response(llm, faiss_index, question))
                    st.success("Done")

if __name__ == "__main__":
    main()

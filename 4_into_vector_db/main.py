import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain, create_retrieval_chain


load_dotenv()


def create_vector_Store():
    print("Hello from 4-into-vector-db!")

    loader = TextLoader("mediumblog1.txt")
    document = loader.load()

    print("Splitting")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(document)


    # Using text-embedding-3-small with 512 dimensions to match Pinecone index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
    

    PineconeVectorStore.from_documents(text, embeddings, index_name="large-model" )

    print("finish")


def retrieve_vector_data():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
    llm = ChatOpenAI()

    query = "What is Vector databases in machine learning"
    chain = PromptTemplate.from_template(template=query) | llm

    # result = chain.invoke(input={})

    # print(result.content)


    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )


    result = retrival_chain.invoke(input={"input": query})

    print(result)



if __name__ == "__main__":
    retrieve_vector_data()

from langchain import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader

openai_api_key = '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'

loader = TextLoader("jlabs_en.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(documents)

llm = OpenAI(temperature=0, openai_api_key = openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
vectorstore = Chroma.from_documents(documents, embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

template = """
Act as if you were a j-labs employee, so when writing about j-labs don't use form 'they' but 'we'.
Give warm and enthusiastic answers.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer and suggest contacting j-labs.

{context}

Question: {question}
Warm and enthusiastic answer written by j-labs employee:
"""

qa = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={
        "prompt": PromptTemplate(
            template=template,
            input_variables=["question", "context"],
        ),
    }
    )

try:
    while True:
        query = input("> ")
        result = qa({"question": query})
        print('J-chat:', result['answer'])
except KeyboardInterrupt:
    print('\nBye!')
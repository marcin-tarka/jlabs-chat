from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader

openai_api_key = '<your key goes here>'

loader = TextLoader("jlabs_en.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(documents)

llm = OpenAI(temperature=0, openai_api_key = openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
vectorstore = Chroma.from_documents(documents, embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(),
    memory=memory,
    )

try:
    while True:
        query = input("> ")
        result = qa({"question": query})
        print('J-chat:', result['answer'])
except KeyboardInterrupt:
    print('\nBye!')
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.json_loader import JSONLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
import pickle


json_path = '/Users/rahulkumar/Downloads/clean_datahub_mysql_13_oct_12_28.json'
user_api_key = 'openai api key here'

loader = JSONLoader(
    file_path=json_path,
    jq_schema='.[]',
    text_content=False)

data = loader.load()

pickle_file_path = '/Users/rahulkumar/Downloads/embedding_model.pkl'

model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# only run for first time to store pickle file
# vectors = FAISS.from_documents(data, hf)
# with open(pickle_file_path, 'wb') as pickle_file:
#     pickle.dump(vectors, pickle_file)


# from 2nd time load pickle file
with open(pickle_file_path, 'rb') as pickle_file:
    loaded_hf = pickle.load(pickle_file)

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo',
                   openai_api_key=user_api_key),
    retriever=loaded_hf.as_retriever())


def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))

    return result["answer"]


if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about " + 'your data' + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

response_container = st.container()
container = st.container()


with container:
    with st.form(key='my_form', clear_on_submit=True):
        prompt = """
        we have data in format like this:
        {"major_tag": "chart", "aspect": "browsePaths", "version": 0, "metadata": "{\"paths\":[\"/looker\"]}",
         "systemmetadata": "{\"lastObserved\":1696859699125,\"lastRunId\":\"file-2023_10_09-19_24_59\",
         \"runId\":\"file-2023_10_09-19_24_59\"}", "createdon": "2023-10-09 13:54:59.000000",
          "createdby": "urn:li:corpuser:__datahub_system", "createdfor": ""}
        do similarity match on major_tag attribute and if data is not available then do similarity match on 
        aspect and description attribute use available data to answer the following questions .
        """
        user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
        user_query = user_input
        submit_button = st.form_submit_button(label='Send')
        user_input = prompt + user_input

    if submit_button and user_input:
        output = conversational_chat(user_input)

        st.session_state['past'].append(user_query)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

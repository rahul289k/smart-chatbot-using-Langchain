import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.json_loader import JSONLoader
from langchain.vectorstores import FAISS


json_path = '/Users/rahulkumar/Downloads/clean_datahub_mysql_900.json'
user_api_key = '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'


loader = JSONLoader(
    file_path=json_path,
    jq_schema='.[]',
    text_content=False)

data = loader.load()


embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
vectors = FAISS.from_documents(data, embeddings)


chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo',
                   openai_api_key='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'),
    retriever=vectors.as_retriever())


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
        user_input = st.text_input("Query:", placeholder="Talk about your data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
        user_query = user_input

        user_input += "search data related to major_tag only and provide details of all parameter " \
                      "available for that major_tag"

    if submit_button and user_input:
        output = conversational_chat(user_input)

        st.session_state['past'].append(user_query)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

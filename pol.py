import streamlit as st
from llama_index.core import VectorStoreIndex, Document, StorageContext
#from llama_index.llms.openai import OpenAI
from llama_index.llms.openai import OpenAI as lOpenAI
from llama_index.llms.groq import Groq
from llama_index.llms.gemini import Gemini
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit.components.v1 as components
import openai
import os
from dotenv import load_dotenv
import requests
import json
from openai import OpenAI
import hmac
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever
)
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.chat_engine import ContextChatEngine
from pathlib import Path

st.set_page_config(page_title=tabtitle, page_icon="💬", layout="centered", initial_sidebar_state="collapsed", menu_items=None)


# Load the environment variables from .env file
load_dotenv()

mentor = st.query_params.get("name", "").upper()

# Debug print
st.write(f"Query parameter 'name': {mentor}")
st.write(f"All query parameters: {st.query_params.to_dict()}")


# Import the variables
name = os.getenv(f'{mentor}_NAME')
topics = os.getenv(f'{mentor}_TOPICS')
creatorimg = os.getenv(f'{mentor}_CREATORIMG')
creatorabout = os.getenv(f'{mentor}_CREATORABOUT')
resourcelink1 = os.getenv(f'{mentor}_RESOURCE_LINK1')
resourcetitle1 = os.getenv(f'{mentor}_RESOURCE_TITLE1')
resourcedesc1 = os.getenv(f'{mentor}_RESOURCE_DESC1')
resourceimg1 = os.getenv(f'{mentor}_RESOURCE_IMG1')
resourcelink2 = os.getenv(f'{mentor}_RESOURCE_LINK2')
resourcetitle2 = os.getenv(f'{mentor}_RESOURCE_TITLE2')
resourcedesc2 = os.getenv(f'{mentor}_RESOURCE_DESC2')
resourceimg2 = os.getenv(f'{mentor}_RESOURCE_IMG2')
backgroundimageurl = os.getenv(f'{mentor}_BACKGROUND_IMAGE_URL')
border_color_choice = os.getenv(f'{mentor}_BORDER_COLOR_CHOICE')
color_choice = os.getenv(f'{mentor}_COLOR_CHOICE')
mild_color = os.getenv(f'{mentor}_MILD_COLOR')

# Debug prints
st.write(f"Retrieved {mentor}_NAME from .env: {name}")
st.write(f"All environment variables: {os.environ}")

# Handle the case where name is None
if name is None:
    st.error(f"No configuration found for mentor: {mentor}")
    st.stop()

root_dir = Path.cwd()  # Get the current working directory (root)
indices_dir = root_dir / "indices"  # Path to the 'indices' folder

# Debug print
st.write(f"Root directory: {root_dir}")
st.write(f"Indices directory: {indices_dir}")

try:
    # Use Path objects for more robust path handling
    mentor_dir = indices_dir / "".join(name.split())
    st.write(f"Looking for mentor directory: {mentor_dir}")

    if not mentor_dir.exists():
        st.error(f"No directory found for mentor: {name}")
        st.stop()

    folders = [
        item.name for item in mentor_dir.iterdir() if item.is_dir()
    ]

    # Debug print
    st.write(f"Found folders: {folders}")

except Exception as e:
    st.error(f"Error processing mentor directory: {str(e)}")
    st.stop()


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs combined search across multiple vector indexes."""

    def __init__(
        self,
        vector_retrievers: list[VectorIndexRetriever],
        top_n: int = None
    ) -> None:
        """Init params."""

        if not vector_retrievers:
            raise ValueError("At least one vector retriever must be provided.")
        self._vector_retrievers = vector_retrievers
        self._top_n = top_n
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes given query."""

        all_vector_nodes = []
        for retriever in self._vector_retrievers:
            vector_nodes = retriever.retrieve(query_bundle)
            all_vector_nodes.extend(vector_nodes)

        combined_dict = {}
        for node in all_vector_nodes:
            page_id = node.node.metadata['page_id']
            if page_id not in combined_dict or node.score > combined_dict[page_id].score:
                combined_dict[page_id] = node

        retrieve_nodes = list(combined_dict.values())
        
        # Sort the nodes by score in descending order
        sorted_nodes = sorted(retrieve_nodes, key=lambda x: x.score, reverse=True)
        
        # Return top_n nodes if specified
        if self._top_n is not None:
            return sorted_nodes[:self._top_n]
        
        return sorted_nodes

def generate_title(prompt, sourcetext):
    client = OpenAI(api_key="sk-proj-Ka1BCucpu4jQkrjYWdLzT3BlbkFJotPblv8HOMod4UUCAewG")

    constructed_prompt = f"""
    Can you give me a title for this text: "{sourcetext}" in 4-5 words? The text is the source that is being referred to, to answer this question: "{prompt}", so make sure that the title suits the intention of the question and also the text that is being referred to?
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": constructed_prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        # Log the exception details
        st.error(f"Failed to generate title due to: {str(e)}")
        return "Error generating title"

result='null'

Settings.llm = lOpenAI(api_key="sk-proj-QySC3TDzzvpKz8m7Hz2xT3BlbkFJ63iBkWnM9hravKv2oMrC", model = "gpt-4o", temperature=0)
Settings.chunk_size = 256
Settings.chunk_overlap = 10

tabtitle = f"AI {name}"
pageheader = f"{name}'s Digital Brain 💡"
unicultpromotext = "The chatbot is powered by [AI Mentors](https://www.aimentors.me/). Head there to build an AI Mentor with a digital brain like mine."
unicultpromoicon = "🤖"
startermessage = f"Hey there! Ask me a question on anything about {topics} and I'll answer from my digital brain..."

unicultdiscsidebar = """If you want to create an AI Mentor similar to mine,<br><br> Build it <b>now on <a href="https://www.aimentors.me/" target="_blank">AI Mentors</a> :)"""
creatorchatavatar = creatorimg
questioneravatar = """https://unicult.s3.eu-north-1.amazonaws.com/studentpic+(2).png"""

#st.set_page_config(page_title=tabtitle, page_icon="💬", layout="centered", initial_sidebar_state="collapsed", menu_items=None)


# Inject CSS, HTML, and JavaScript into Streamlit
st.markdown(
    f"""
    <style>
    a:hover {{
        color: {color_choice} !important;
        border-color: {color_choice} !important;
    }}
    .st-emotion-cache-1dp5vir {{
        display: none !important;
    }}
    .st-emotion-cache-1eo1tir {{
        max-width: 52rem !important;
    }}
    .st-emotion-cache-qcpnpn {{
        border: 1px solid {border_color_choice} !important;
    }}
    .st-emotion-cache-arzcut {{
        max-width: 52rem !important;
    }}
    .st-emotion-cache-6qob1r {{
        background-color: black !important;
    }}
    a:active {{
      background-color: transparent;
    }}
    .st-emotion-cache-1hd8nr8 {{
        border: 1px solid {border_color_choice} !important;
    }}
    .st-emotion-cache-yfhhig {{
        box-shadow: 0 25px 35px -10px rgba(0, 0, 0, 0.25), 0px 0px 25px {color_choice} !important;
    }}
    footer {{
        visibility: hidden;
    }}
    footer:after {{
        content:'Made with Streamlit';
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
    }}    
    .st-emotion-cache-1pbsqtx:hover {{
        color: {color_choice} !important;
    }}
    .st-emotion-cache-p5msec:hover * {{
      color: {color_choice} !important;
    }}
    .st-emotion-cache-p5msec:hover .st-emotion-cache-1pbsqtx {{
      filter: invert(42%) sepia(99%) saturate(1897%) hue-rotate(181deg) brightness(101%) contrast(101%);
    }}
    .st-emotion-cache-h4xjwg {{
          background-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url({backgroundimageurl}) !important;
    }}
    .st-emotion-cache-bm2z3a {{
      background-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url({backgroundimageurl}) !important;
    }}
    .st-emotion-cache-1dp5vir {{
      background-image: linear-gradient(90deg, rgb(106 160 247), rgb(255 255 255)) !important;
    }}
    .st-emotion-cache-15ecox0 {{
        display: none !important;
    }}
    .st-emotion-cache-vj1c9o {{
      background-color: transparent !important;
    }}
    .st-emotion-cache-1ru4d5d {{
      max-width: 52rem !important;
    }}
    .st-emotion-cache-139wi93 {{
      max-width: 52rem !important;
    }}
    .st-emotion-cache-qcqlej {{
      max-width: 52rem !important;
    }}
    h1 {{
      text-align: center !important;
    }}
    .st-emotion-cache-1avcm0n {{
      background: transparent !important;
    }}
    .st-cm {{
      background-color: black !important;
    }}
    .st-bu {{
      border-bottom-color: #123456 !important;
      box-shadow: 0 25px 35px -10px rgba(0, 0, 0, 0.25), 0px 0px 25px {color_choice} !important;
    }}
    .st-bt {{
      border-top-color: {border_color_choice} !important;
    }}
    .st-bs {{
      border-right-color: {border_color_choice} !important;
    }}
    .st-br {{
      border-left-color: {border_color_choice} !important;
    }}
    .st-cq {{
      box-shadow: 0 25px 35px -10px rgba(0, 0, 0, 0.25), 0px 0px 25px {color_choice} !important;
      border-top-color: {border_color_choice} !important;
      border-right-color: {border_color_choice} !important;
      border-left-color: {border_color_choice} !important;
      border-bottom-color: {border_color_choice} !important;
    }}
    .st-cp {{
      border-top-color: {border_color_choice} !important;
      border-right-color: {border_color_choice} !important;
      border-left-color: {border_color_choice} !important;
      border-bottom-color: {border_color_choice} !important;
    }}
    .st-co {{
      border-right-color: {border_color_choice} !important;
      border-top-color: {border_color_choice} !important;
      border-left-color: {border_color_choice} !important;
      border-bottom-color: {border_color_choice} !important;
    }}
    .st-bu {{
      border-right-color: {border_color_choice} !important;
      border-top-color: {border_color_choice} !important;
      border-left-color: {border_color_choice} !important;
      border-bottom-color: {border_color_choice} !important;
    }}
    .st-cn {{
      border-top-color: {border_color_choice} !important;
      border-right-color: {border_color_choice} !important;
      border-left-color: {border_color_choice} !important;
      border-bottom-color: {border_color_choice} !important;
    }}
    .st-bb {{
      background-color: black !important;
    }}
    .st-emotion-cache-s1k4sy {{
      background-color: black !important;
    }}
    hr {{
      border-bottom: 1px dashed {border_color_choice} !important;
    }}
    .st-emotion-cache-1xw8zd0 {{
      border: 1px solid {border_color_choice} !important;
    }}
    .st-emotion-cache-14kqqgu {{
      border: 1px solid {border_color_choice} !important;
    }}
    .st-emotion-cache-janbn0 {{
      background-color: {mild_color} !important;
    }}
    .st-emotion-cache-14kqqgu:hover {{
      border-color: {border_color_choice} !important;
      color: {color_choice} !important;
    }}
    .st-emotion-cache-ocqkz7 {{
      gap: 1.5rem !important;
      margin-left: -45px !important;
      margin-bottom: 15px !important;
    }}
    h2 {{
      text-align: center !important;
      margin-left: -25px !important;
    }}
    .st-emotion-cache-cymc9z {{
      border: 1px solid {border_color_choice} !important;
    }}
    .st-emotion-cache-1avcm0n {{
      z-index: 0 !important;
    }}
    .st-emotion-cache-16txtl3 {{
      background-color: black !important;
    }}
    .st-emotion-cache-p5msec {{
      border: 1.5px solid {border_color_choice} !important;
    }}
    .st-emotion-cache-1clstc5 {{
      border: 1px solid {border_color_choice} !important;
    }}
    .st-emotion-cache-1dtefog:hover {{
      color: {color_choice} !important;
    }}
    .st-emotion-cache-6q9sum {{
      background-color: black !important;
      box-shadow: 0 25px 35px -10px rgba(0, 0, 0, 0.25), 0px 0px 25px {color_choice} !important;
    }}
    .st-emotion-cache-1f3w014 {{
      fill: {color_choice} !important;
    }}
    .st-emotion-cache-1wi2cd3:hover {{
      background-color: {border_color_choice} !important;
    }}
    p,
    ol,
    ul,
    dl {{
      margin-right: 15px !important;
    }}
    .st-b3 {{
      border: 1.6px dashed {border_color_choice} !important;
    }}

    .st-d3 {{
      background-color: {mild_color} !important;  
    }}
    
    #glow {{
      position: fixed !important;
      pointer-events: none !important;
      width: 35px !important; 
      height: 35px !important; 
      border-radius: 100% !important;
      background: {border_color_choice} !important; 
      box-shadow: 0 0 45px 45px {border_color_choice} !important; 
      z-index: -10000 !important; 
      transform: translate(-50%, -50%) !important; 
      will-change: transform !important;
    }}
    #glow::after {{
      transform: translate(-50%, -50%) scaleY(0) !important;
      opacity: 0 !important;
      transition: transform 0.5s, opacity 0.5s !important;
    }}
    #glow.show-tail::after {{
      transform: translate(-50%, -50%) scaleY(1) !important;
      opacity: 1 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.


@st.cache_resource
def load_model():
    return HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.embed_model = load_model()

@st.cache_resource  # Cache the function output to avoid recomputation
def load_sentence_index(folders):
    # Your code to load or compute the sentence index goes here
    indices = []
    for ns in folders:
        storage_context = StorageContext.from_defaults(persist_dir=f"indices/{"".join(name.split())}/{ns}")
        sentence_index = load_index_from_storage(storage_context)
        indices.append(sentence_index)
        
    return indices

all_retrievers = []

if folders:
    sentence_index = load_sentence_index(folders)
    all_retrievers = [sns.as_retriever(similarity_top_k=3) for sns in sentence_index]

combination_retriever = CustomRetriever(all_retrievers, top_n=4)

# BAAI/bge-reranker-base
# link: https://huggingface.co/BAAI/bge-reranker-base
#rerank = SentenceTransformerRerank(
    #top_n=2, model="BAAI/bge-reranker-base"
#)

#openai.api_key = "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"
st.title(f"{pageheader}")
st.write("")
st.info(f"{unicultpromotext}", icon=f"{unicultpromoicon}")

# Adding a sidebar
st.sidebar.title("Must Have Resources")
st.sidebar.write("")

with st.sidebar.expander("About Me"):
    st.write("")

    st.image(f"{creatorimg}")
    st.write("")

    st.write(f"{creatorabout}")
    st.write("")

st.sidebar.markdown("---")

st.sidebar.markdown(f"""
    <div style="text-align: center;">
        <p><i>
        <a href="{resourcelink1}" target="_blank">{resourcetitle1}</a>
         - {resourcedesc1} <br>  <br> </i> </p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.image(f"{resourceimg1}", use_column_width=True)
st.sidebar.write("")
st.sidebar.markdown("---")

st.sidebar.markdown(f"""
    <div style="text-align: center;">
        <p><i>
        <a href="{resourcelink2}" target="_blank">{resourcetitle2}</a>
         - {resourcedesc2}<br><br> </i> </p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.image(f"{resourceimg2}", use_column_width=True)
st.sidebar.write("")
st.sidebar.markdown("---")

st.sidebar.markdown(f"""
    <div style="text-align: center;">
        <p><i>
        {unicultdiscsidebar}</i> 
        </p>
    </div>
""", unsafe_allow_html=True)
st.sidebar.write("")
st.sidebar.markdown("---")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": f"{startermessage}"}
    ]

if "memory" not in st.session_state.keys():
    st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = ContextChatEngine.from_defaults(
        retriever = combination_retriever,
        context_template=(
            f"I am {name}, and you are my chatbot, that is, {name}'s chatbot trained on the linkedin posts, youtube videos and newsletters of mine, and you'll be able to have normal interactions, as well as talk about {topics}, etc.. You must always use multiple analogies and references in detail from the documents you refer to when answering.\n"
            f"Here are the relevant context from the newsletters, youtube videos and linkedin posts of mine that you can use to answer the queries:\n"
            "{{context_str}}"
            "\nInstruction: Make sure to give as much actionable insights as possible, to interact and help the user in a first person language. Always try to bring up an analogy to something you find on the documents you refer while answering the question but do not mention them directly.\nYou MUST use the same language and tone in the context text given to you when answering a question."
        ),
        verbose=True,
        chat_mode="context",
        skip_condense=True,
        system_prompt=f"You are {name}'s chatbot trained on all his newsletters, youtube videos and linkedin posts and you have to assist the user who asks questions to you as {name} himself or herself in first person with their queries. You MUST use the same language and tone in the context text given to you when answering a question."
    )

if prompt := st.chat_input("Your question..."):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    query = prompt

for message in st.session_state.messages:  # Display the prior chat messages
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar=f"{creatorchatavatar}"):
            st.markdown(message["content"], unsafe_allow_html=True)
    else:
        with st.chat_message(message["role"], avatar=f"{questioneravatar}"):
            st.markdown(message["content"], unsafe_allow_html=True)

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar=f"{creatorchatavatar}"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat_engine.chat(prompt)
                score_1 = response.source_nodes[0].score
                score_2 = response.source_nodes[1].score
                print("Score 1 is:", score_1)
                print("Score 2 is:", score_2)
                sourcetext1 = response.source_nodes[0].text
                sourcetext2 = response.source_nodes[1].text

                firsttitle = generate_title(prompt, sourcetext1)
                print(firsttitle)
                secondtitle = generate_title(prompt, sourcetext2)
                print(secondtitle)

                st.session_state.chat_engine.reset()

                if (response.source_nodes[0].score < 0.7) and (response.source_nodes[1].score < 0.7):
                    print("SCORES<0.7 - Sorry, I cannot answer that. Can you please ask questions with respect to learning from something I know? In case you already are, please ask again by being more specific and detailed. Thanks.")
                    st.markdown("Sorry, I cannot answer that. Can you please ask questions with respect to learning from something I know? In case you already are, please ask again by being more specific and detailed. Thanks.", unsafe_allow_html=True)
                    message = {"role": "assistant", "content": "SCORES<0.7, Sorry, I cannot answer that. Can you please ask questions with respect to learning from something I know? In case you already are, please ask again by being more specific and detailed. Thanks.", "avatar": f"{creatorchatavatar}"}
                    st.session_state.messages.append(message)  # Add response to message history
                    st.divider()

                else:
                    try:
                        sourcetext1 = response.source_nodes[0].text
                        sourcetext2 = response.source_nodes[1].text

                        firsttitle = generate_title(prompt, sourcetext1)
                        firsttitle = firsttitle.strip('"')
                        print(firsttitle)
                        secondtitle = generate_title(prompt, sourcetext2)
                        secondtitle = secondtitle.strip('"')
                        print(secondtitle)

                        print(f"Response 1 is {response.source_nodes[0]}")
                        print(f"Response 2 is {response.source_nodes[1]}")

                        print(response)

                        p_id_1 = response.source_nodes[0].metadata['page_id']
                        print("Page ID 1 is:", p_id_1)
                        p_id_2 = response.source_nodes[1].metadata['page_id']
                        print("Page ID 2 is:", p_id_2)

                        st.markdown(response.response, unsafe_allow_html=True)
                        message = {"role": "assistant", "content": response.response, "avatar": f"{creatorchatavatar}"}
                        st.session_state.messages.append(message)  # Add response to message history
                        st.divider()
                        st.header('''Live Answers - Related To Your Question 📖''')
                        st.markdown('', unsafe_allow_html=True)

                        if (0):
                            with st.container(border=True):
                                st.image(cover_url_1)
                                st.link_button(f'{name_1}', f'https://karlkaufman.unicult.club/all-chapters/chapters/{doc_to_link[dict_to_doc[p_id_1]]}', help='Go to lesson', use_container_width=True)

                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                with st.container(border=True):
                                    st.link_button(f"{firsttitle}", f'{p_id_1}', help='Go to lesson', use_container_width=True)

                            with col2:
                                with st.container(border=True):
                                    st.link_button(f"{secondtitle}", f'{p_id_2}', help='Go to lesson', use_container_width=True)

                    except Exception as e:
                        st.error(f"Failed to generate answer due to: {str(e)}")
                        print("Blah, Sorry, I cannot answer that. Can you please ask questions with respect to learning from something I know? In case you already are, please ask again by being more specific and detailed. Thanks.")
                        st.markdown("Blah, Sorry, I cannot answer that. Can you please ask questions with respect to learning from something I know? In case you already are, please ask again by being more specific and detailed. Thanks.", unsafe_allow_html=True)
                        message = {"role": "assistant", "content": "Blah, Sorry, I cannot answer that. Can you please ask questions with respect to learning from something I know? In case you already are, please ask again by being more specific and detailed. Thanks.", "avatar": f"{creatorchatavatar}"}
                        st.session_state.messages.append(message)  # Add response to message history
                        st.divider()

            except Exception as e:
                st.error(f"Failed to generate answer due to: {str(e)}")
                print("Booh, Sorry, I cannot answer that. Can you please ask questions with respect to learning from something I know? In case you already are, please ask again by being more specific and detailed. Thanks.")
                st.markdown("Booh, Sorry, I cannot answer that. Can you please ask questions with respect to learning from something I know? In case you already are, please ask again by being more specific and detailed. Thanks.", unsafe_allow_html=True)
                message = {"role": "assistant", "content": "Booh,Sorry, I cannot answer that. Can you please ask questions with respect to learning from something I know? In case you already are, please ask again by being more specific and detailed. Thanks.", "avatar": f"{creatorchatavatar}"}
                st.session_state.messages.append(message)  # Add response to message history
                st.divider()

import streamlit as st
import os
import hmac
import logging
import json
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import openai
from openai import OpenAI
import pyrebase
import gspread
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from llama_index.core import (
    VectorStoreIndex, Document, StorageContext, SimpleDirectoryReader,
    Settings, QueryBundle
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.llms.openai import OpenAI as lOpenAI
import streamlit.components.v1 as components

# Load the environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="AI Mentor", page_icon="💬", layout="centered", initial_sidebar_state="collapsed", menu_items=None)

# Firebase configuration
firebaseConfig = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "databaseURL": ""  # Add if you're using Firebase Realtime Database
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Initialize session state for 'user'
if 'user' not in st.session_state:
    st.session_state['user'] = None

# Initialize session state for login/signup toggle
if 'login_or_signup' not in st.session_state:
    st.session_state['login_or_signup'] = 'Login'

def login(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state['user'] = user
        st.success(f"Logged in as {email}")
    except Exception as e:
        st.error("Invalid email or password")

def logout():
    st.session_state.pop('user', None)
    st.success("Logged out successfully")

def signup(email, password):
    try:
        auth.create_user_with_email_and_password(email, password)
        st.success("User created successfully. Please log in.")
        st.session_state['login_or_signup'] = 'Login'  # Switch back to login after successful sign up
    except Exception as e:
        st.error(f"Error creating account: {e}")

def login_ui():
    st.title("Welcome to AI Mentor")
    if st.session_state['login_or_signup'] == 'Login':
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            login(email, password)
        st.info("Don't have an account? Sign up below.")
        if st.button("Go to Sign Up"):
            st.session_state['login_or_signup'] = 'Sign Up'
    else:
        st.subheader("Sign Up")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.button("Sign Up"):
            if password == confirm_password:
                signup(email, password)
            else:
                st.error("Passwords do not match.")
        st.info("Already have an account? Log in below.")
        if st.button("Go to Login"):
            st.session_state['login_or_signup'] = 'Login'

def check_password():
    """Returns `True` if the user had the correct password or if no password is required."""
    
    # Get the mentor name from query parameters
    mentor = st.experimental_get_query_params().get("name", [""])[0].lower()
    
    if not mentor:
        st.error("No mentor specified in the URL. Please use '?name=mentorname' in the URL.")
        return False

    # Check if the mentor has a password in environment variables
    password_key = f"STREAMLIT_{mentor.upper()}_PASSWORD"
    mentor_password = os.environ.get(password_key)

    if not mentor_password:
        # No password required for this mentor
        return True

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if "password" in st.session_state and st.session_state["password"]:
            if hmac.compare_digest(st.session_state["password"], mentor_password):
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store the password.
            else:
                st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Enter password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

# Main application logic
if st.session_state['user'] is None:
    login_ui()
else:
    # Show the logout button
    if st.button("Logout"):
        logout()

    # Begin your existing Streamlit app code here
    # -------------------- Existing Code Starts --------------------

    # Rest of your code (e.g., imports, functions, classes)
    # Note: Ensure that any variables or configurations used below are initialized properly

    mentor = st.experimental_get_query_params().get("name", [""])[0].upper()

    # Import the variables
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

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
    additional_context = ""

    if name == "Karl Kaufman":
        additional_context = "Never support diversification as a strategy when answering any question as it is not my philosophy of investing because I support concentrated investing."
    if name == "Exiled Medic":
        additional_context = "Also note that the real name is Fabian Kowallik and I am known as Exiled Medic on the internet. So the questioner can address me either with my real name or internet name."
    if name == "Stephen Pope":
        additional_context = ("Also note that I, Stephen Pope, am a previous investor in Skool. "
                              "And, currently I am running a business/own the company called Kontent Engine. "
                              "We offer the only fully automated and ai-enabled content management system, the Kontent Engine Database. "
                              "Plan, create, store, search, produce, distribute and analyze content in one place. Built using no-code tools for maximum flexibility and customization. "
                              "Also return this video link when answering any questions about kontent engine and its process - https://www.youtube.com/watch?v=1qRnccFrjfs&t=4s ")

    # Handle the case where name is None
    if name is None:
        st.write(f"No configuration found for mentor: {mentor}")
        st.stop()

    root_dir = Path.cwd()  # Get the current working directory (root)
    indices_dir = root_dir / "indices"  # Path to the 'indices' folder

    try:
        # Use Path objects for more robust path handling
        mentor_dir = indices_dir / "".join(name.split())

        folders = [
            item.name for item in mentor_dir.iterdir() if item.is_dir()
        ]

    except Exception as e:
        st.write(f"Error processing mentor directory: {str(e)}")
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
        client = OpenAI(api_key=OPENAI_API_KEY)

        constructed_prompt = f"""
        Can you give me a title for this text: "{sourcetext}" in 4-5 words? The text is the source that is being referred to, to answer this question: "{prompt}", so make sure that the title suits the intention of the question and also the text that is being referred to?
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": constructed_prompt},
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            # Log the exception details
            print(f"Failed to generate title due to: {str(e)}")
            return f"Failed to generate title due to: {str(e)}"

    result='null'

    Settings.llm = lOpenAI(api_key=OPENAI_API_KEY, model = "gpt-4o-mini", temperature=0)
    Settings.chunk_size = 256
    Settings.chunk_overlap = 30

    tabtitle = f"AI {name}"
    pageheader = f"{name}'s Digital Brain 💡"
    unicultpromotext = "The chatbot is powered by [AI Mentors](https://www.aimentors.me/). Head there to build an AI Mentor with a digital brain like mine."
    unicultpromoicon = "🤖"
    startermessage = f"Hey there! Ask me a question on anything about {topics} and I'll answer from my digital brain..."

    unicultdiscsidebar = """If you want to create an AI Mentor similar to mine,<br><br> Build it <b>now on <a href="https://www.aimentors.me/" target="_blank">AI Mentors</a> :)"""
    creatorchatavatar = creatorimg
    questioneravatar = """https://unicult.s3.eu-north-1.amazonaws.com/studentpic+(2).png"""

    # Inject CSS, HTML, and JavaScript into Streamlit
    st.markdown(
        f"""
        <style>
        /* ... Your existing CSS styles ... */
        /* Include your CSS here */
        </style>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def load_model():
        return HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    Settings.embed_model = load_model()

    @st.cache_resource  # Cache the function output to avoid recomputation
    def load_sentence_index(folders):
        indices = []
        for ns in folders:
            try:
                persist_dir = f"indices/{(''.join(name.split()))}/{ns}"
                logger.info(f"Attempting to load index from: {persist_dir}")
                
                if not os.path.exists(persist_dir):
                    logger.warning(f"Directory not found: {persist_dir}")
                    continue
                
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                sentence_index = load_index_from_storage(storage_context)
                indices.append(sentence_index)
                logger.info(f"Successfully loaded index from: {persist_dir}")
            except Exception as e:
                logger.error(f"Error loading index for {ns}: {str(e)}")
        
        logger.info(f"Total indices loaded: {len(indices)}")
        return indices

    all_retrievers = []

    if folders:
        logger.info(f"Folders to process: {folders}")
        sentence_index = load_sentence_index(folders)
        all_retrievers = [sns.as_retriever(similarity_top_k=3) for sns in sentence_index]
        logger.info(f"Total retrievers created: {len(all_retrievers)}")

    if all_retrievers:
        combination_retriever = CustomRetriever(all_retrievers, top_n=4)
        logger.info("CustomRetriever created successfully")
    else:
        logger.error("No valid retrievers found. Please check the index files.")
        st.error("No valid retrievers found. Please check the index files.")
        st.stop()

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
            retriever=combination_retriever,
            context_template=(
                f"I am {name}, and you are my chatbot, that is, {name}'s chatbot trained on the linkedin posts, youtube videos and newsletters of mine, and you'll be able to have normal interactions, as well as talk about {topics}, etc.. You must always use almost exact wordings and tone of voice from the documents you refer to when answering.\n"
                f"Here are the relevant context from the newsletters, youtube videos and linkedin posts of mine that you can use to answer the queries:\n"
                "{{context_str}}"
                f"\nInstruction: Make sure to give as much actionable insights as possible, to interact and help the user in a first person language. Always try to use the same language used, tone used and sentence styles used that you find on the documents you refer to while answering the question but do not mention them directly. {additional_context}"
            ),
            verbose=True,
            chat_mode="context",
            skip_condense=True,
            system_prompt=f"You are {name}'s chatbot trained on all his newsletters, youtube videos and linkedin posts and you have to assist the user who asks questions to you as {name} himself or herself in first person with their queries. You MUST use the same language and tone in the context text given to you when answering a question. {additional_context}"
        )

    # Set up Google Sheets API credentials
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = 'aimentors-8ac58aab995e.json'
    ADMIN_EMAIL = 'kamalprasats@unicult.club'

    def get_credentials():
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        delegated_credentials = creds.with_subject(ADMIN_EMAIL)
        return delegated_credentials

    def get_or_create_sheet(mentor_name):
        creds = get_credentials()
        client = gspread.authorize(creds)
        drive_service = build('drive', 'v3', credentials=creds)
        sheets_service = build('sheets', 'v4', credentials=creds)

        try:
            # Try to open an existing sheet
            sheet = client.open(mentor_name).sheet1
            print(f"Opened existing sheet: https://docs.google.com/spreadsheets/d/{sheet.spreadsheet.id}")
        except gspread.SpreadsheetNotFound:
            # If the sheet doesn't exist, create a new one
            spreadsheet = {
                'properties': {
                    'title': mentor_name
                }
            }
            spreadsheet = sheets_service.spreadsheets().create(body=spreadsheet).execute()
            sheet_id = spreadsheet['spreadsheetId']
            sheet = client.open_by_key(sheet_id).sheet1
            
            # Add headers to the new sheet with new columns
            sheet.append_row(["Timestamp", "Question", "Answer", "Source 1", "Source 2", "Excerpt 1", "Excerpt 2", "Confidence Level"])
            
            print(f"Created new sheet: https://docs.google.com/spreadsheets/d/{sheet_id}")

        return sheet

    def record_qa(mentor_name, question, answer, source1, source2, excerpt1, excerpt2, confidence_level):
        try:
            sheet = get_or_create_sheet(mentor_name)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sheet.append_row([timestamp, question, answer, source1, source2, excerpt1, excerpt2, confidence_level])
            print(f"Successfully recorded Q&A in sheet: https://docs.google.com/spreadsheets/d/{sheet.spreadsheet.id}")
        except Exception as e:
            print(f"Error recording Q&A: {str(e)}")

    # Chat interface
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
            disclaimer = "\n\n<p style='font-size: 0.8em; opacity: 0.7;'>Disclaimer: This AI Chatbot can make mistakes. Please verify the information. This chatbot is intended for educational and informational purposes only.</p>"

            try:
                response_placeholder = st.empty()
                full_response = ""
                source_nodes = []

                # Use the spinner only for the initial setup
                with st.spinner("Thinking..."):
                    stream_response = st.session_state.chat_engine.stream_chat(prompt)

                # Stream the response without the spinner
                for token in stream_response.response_gen:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                
                # Get the source nodes after streaming is complete
                source_nodes = stream_response.source_nodes

                # Display the final response
                response_placeholder.markdown(full_response + disclaimer, unsafe_allow_html=True)

                # Add response to message history
                message = {"role": "assistant", "content": full_response + disclaimer, "avatar": f"{creatorchatavatar}"}
                st.session_state.messages.append(message)

                # Record the question and answer in Google Sheets
                if len(source_nodes) >= 2:
                    score_1 = source_nodes[0].score
                    score_2 = source_nodes[1].score
                    sourcetext1 = source_nodes[0].text[:500]  # Limit excerpt to 500 characters
                    sourcetext2 = source_nodes[1].text[:500]  # Limit excerpt to 500 characters
                    
                    source1 = source_nodes[0].metadata.get('page_id', 'N/A')
                    source2 = source_nodes[1].metadata.get('page_id', 'N/A')
                    
                    # Calculate confidence level
                    avg_score = (score_1 + score_2) / 2
                    if avg_score > 0.7:
                        confidence_level = "Confident"
                    elif 0.5 <= avg_score <= 0.7:
                        confidence_level = "Moderately Confident"
                    else:
                        confidence_level = "Less Confident"
                    
                    # Record the question and answer in Google Sheets with new data
                    record_qa(mentor, prompt, full_response, source1, source2, sourcetext1, sourcetext2, confidence_level)
                else:
                    # If there are fewer than 2 source nodes, use placeholder values
                    record_qa(mentor, prompt, full_response, 'N/A', 'N/A', 'N/A', 'N/A', 'Unknown')

                # Reset the chat engine
                st.session_state.chat_engine.reset()

                # Display additional information (scores, titles, etc.)
                if len(source_nodes) >= 2:
                    score_1 = source_nodes[0].score
                    score_2 = source_nodes[1].score
                    print("Score 1 is:", score_1)
                    print("Score 2 is:", score_2)
                    sourcetext1 = source_nodes[0].text
                    sourcetext2 = source_nodes[1].text

                    if 'page' in source_nodes[0].metadata:
                        firsttitle = f"{name}'s Book - Page {source_nodes[0].metadata['page']}"
                    else:
                        firsttitle = generate_title(prompt, sourcetext1)

                    if 'page' in source_nodes[1].metadata:
                        secondtitle = f"{name}'s Book - Page {source_nodes[1].metadata['page']}"
                    else:
                        secondtitle = generate_title(prompt, sourcetext2)

                    if score_1 >= 0.7 and score_2 >= 0.7:
                        st.divider()
                        st.header('Live Answers - Related To Your Question 📖')
                        st.markdown('', unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            with st.container():
                                st.write(f"[{firsttitle.strip('\"')}]({source_nodes[0].metadata['page_id']})")

                        with col2:
                            with st.container():
                                st.write(f"[{secondtitle.strip('\"')}]({source_nodes[1].metadata['page_id']})")

            except Exception as e:
                st.error(f"Failed to generate answer due to: {str(e)}")
                error_message = "An error occurred while processing your request. Please try again."
                st.markdown(error_message + disclaimer, unsafe_allow_html=True)
                message = {"role": "assistant", "content": error_message + disclaimer, "avatar": f"{creatorchatavatar}"}
                st.session_state.messages.append(message)

    # -------------------- Existing Code Ends --------------------

    # End of the Streamlit app

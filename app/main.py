from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    QueryBundle,
    Settings
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever
)
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.schema import NodeWithScore
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import gspread
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime
import json
import logging

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DISCLAIMER = "Disclaimer : This AI can make mistakes. Please verify the information. This chatbot is intended for educational and informational purposes only."

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
            ],
            timeout=30  # Add a timeout of 30 seconds
        )
        return response.choices[0].message.content
    except Exception as e:
        # Log the exception details
        print(f"Failed to generate title due to: {str(e)}")
        return f"Failed to generate title due to: {str(e)}"

app = FastAPI()
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up your OpenAI API key
if not OPENAI_API_KEY:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

# Initialize the LLM settings
Settings.llm = LlamaOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0,
)

embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embedding_model
Settings.chunk_size = 256
Settings.chunk_overlap = 30

# Define the request model
class ChatRequest(BaseModel):
    name: str
    prompt: str

# Custom Retriever
class CustomRetriever(BaseRetriever):
    """Custom retriever that performs combined search across multiple vector indexes."""

    def __init__(self, vector_retrievers: List[VectorIndexRetriever], top_n: int = None) -> None:
        if not vector_retrievers:
            raise ValueError("At least one vector retriever must be provided.")
        self._vector_retrievers = vector_retrievers
        self._top_n = top_n
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        all_vector_nodes = []
        for retriever in self._vector_retrievers:
            vector_nodes = retriever.retrieve(query_bundle)
            all_vector_nodes.extend(vector_nodes)

        combined_dict = {}
        for node in all_vector_nodes:
            page_id = node.node.metadata.get('page_id')
            if page_id not in combined_dict or node.score > combined_dict[page_id].score:
                combined_dict[page_id] = node

        retrieve_nodes = list(combined_dict.values())

        # Sort the nodes by score in descending order
        sorted_nodes = sorted(retrieve_nodes, key=lambda x: x.score, reverse=True)

        # Return top_n nodes if specified
        if self._top_n is not None:
            return sorted_nodes[:self._top_n]

        return sorted_nodes

# Helper function to load mentor configuration
def load_mentor_config(mentor_name):
    mentor_key = mentor_name.upper()

    name = os.getenv(f'{mentor_key}_NAME')
    topics = os.getenv(f'{mentor_key}_TOPICS')
    additional_context = ""

    if not name:
        raise ValueError(f"No configuration found for mentor: {mentor_name}")

    if name == "Karl Kaufman":
        additional_context = "Never support diversification as a strategy when answering any question as it is not my philosophy of investing because I support concentrated investing."

    config = {
        "name": name,
        "topics": topics,
        "additional_context": additional_context
    }
    return config

# Helper function to load indices
def load_indices(mentor_name):
    name = "".join(mentor_name.split())
    root_dir = Path.cwd()
    indices_dir = root_dir / "indices" / name

    if not indices_dir.exists():
        raise FileNotFoundError(f"Indices directory not found for mentor: {mentor_name}")

    folders = [item.name for item in indices_dir.iterdir() if item.is_dir()]
    if not folders:
        raise ValueError(f"No indices found for mentor: {mentor_name}")

    indices = []
    for ns in folders:
        try:
            persist_dir = indices_dir / ns
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            index = load_index_from_storage(storage_context)
            indices.append(index)
            logger.info(f"Loaded index from: {persist_dir}")
        except Exception as e:
            logger.error(f"Error loading index for {ns}: {str(e)}")
    return indices

SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'aimentors-8ac58aab995e.json'
ADMIN_EMAIL = 'kamalprasats@unicult.club'

def get_credentials():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    delegated_credentials = creds.with_subject(ADMIN_EMAIL)
    return delegated_credentials

def get_or_create_sheet(mentor_key):
    creds = get_credentials()
    client = gspread.authorize(creds)
    drive_service = build('drive', 'v3', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)

    try:
        # Try to open an existing sheet
        sheet = client.open(mentor_key).sheet1
        logger.info(f"Opened existing sheet: https://docs.google.com/spreadsheets/d/{sheet.spreadsheet.id}")
    except gspread.SpreadsheetNotFound:
        # If the sheet doesn't exist, create a new one
        spreadsheet = {
            'properties': {
                'title': mentor_key
            }
        }
        spreadsheet = sheets_service.spreadsheets().create(body=spreadsheet).execute()
        sheet_id = spreadsheet['spreadsheetId']
        sheet = client.open_by_key(sheet_id).sheet1
        
        # Add headers to the new sheet with new columns
        sheet.append_row(["Timestamp", "Question", "Answer", "Source 1", "Source 2", "Excerpt 1", "Excerpt 2", "Confidence Level"])
        
        logger.info(f"Created new sheet: https://docs.google.com/spreadsheets/d/{sheet_id}")

    return sheet

def record_qa(mentor_key, question, answer, source1, source2, excerpt1, excerpt2, confidence_level):
    try:
        sheet = get_or_create_sheet(mentor_key)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, question, answer, source1, source2, excerpt1, excerpt2, confidence_level])
        logger.info(f"Successfully recorded Q&A in sheet: https://docs.google.com/spreadsheets/d/{sheet.spreadsheet.id}")
    except Exception as e:
        logger.error(f"Error recording Q&A: {str(e)}")

# Endpoint for chat
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Extract data from the request
        name = request.name
        prompt = request.prompt

        # Load mentor configuration
        config = load_mentor_config(name)
        mentor_name = config["name"]
        mentor_key = name.upper()  # This is the key we'll use for the sheet name
        topics = config["topics"]
        additional_context = config["additional_context"]

        # Initialize LLM settings
        llm = LlamaOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0,
        )

        # Load indices and retrievers
        indices = load_indices(mentor_name)
        retrievers = [index.as_retriever(similarity_top_k=3) for index in indices]
        if not retrievers:
            raise ValueError("No retrievers found for the mentor.")

        combination_retriever = CustomRetriever(retrievers, top_n=4)

        # Initialize chat engine
        chat_engine = ContextChatEngine.from_defaults(
            retriever=combination_retriever,
            context_template=(
                f"I am {mentor_name}, and you are my chatbot, that is, {mentor_name}'s chatbot trained on the linkedin posts, "
                f"youtube videos, and newsletters of mine, and you'll be able to have normal interactions, as well as talk "
                f"about {topics}, etc. You must always use almost exact wordings and tone of voice from the documents you "
                f"refer to when answering.\n"
                f"Here are the relevant contexts from the newsletters, youtube videos, and linkedin posts of mine that you "
                f"can use to answer the queries:\n"
                "{{context_str}}"
                f"\nInstruction: Make sure to give as much actionable insights as possible, to interact and help the user in "
                f"a first person language. Always try to use the same language used, tone used, and sentence styles used "
                f"that you find on the documents you refer to while answering the question but do not mention them directly. "
                f"{additional_context}"
            ),
            chat_mode="context",
            system_prompt=(
                f"You are {mentor_name}'s chatbot trained on all his newsletters, youtube videos, and linkedin posts and you "
                f"have to assist the user who asks questions to you as {mentor_name} himself or herself in first person with "
                f"their queries. You MUST use the same language and tone in the context text given to you when answering a "
                f"question. {additional_context}"
            ),
            llm=llm,
            embed_model=embedding_model,
        )

        # Generate streaming response
        stream_response = chat_engine.stream_chat(prompt)
        
        async def response_generator():
            full_response = ""
            for chunk in stream_response.response_gen:
                full_response += chunk
                yield chunk.encode('utf-8')
            
            # Add disclaimer
            yield f"\n\n{DISCLAIMER}".encode('utf-8')
            
            # Get source nodes
            source_nodes = stream_response.source_nodes

            # Process source nodes and generate source information
            if len(source_nodes) >= 2:
                source1 = source_nodes[0].node.metadata.get('page_id', 'N/A')
                source2 = source_nodes[1].node.metadata.get('page_id', 'N/A')
                title1 = generate_title(prompt, source_nodes[0].node.text[:500])
                title2 = generate_title(prompt, source_nodes[1].node.text[:500])
                
                # Yield source information as JSON with a clear delimiter
                source_info = {
                    "source1": {"url": source1, "title": title1},
                    "source2": {"url": source2, "title": title2}
                }
                yield f"\n{{\"source1\": {json.dumps(source_info['source1'])}, \"source2\": {json.dumps(source_info['source2'])}}}".encode('utf-8')

        return StreamingResponse(response_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Allow all origins (for development). In production, specify allowed origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with a list of allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from PyPDF2 import PdfReader
import docx
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversations in memory
conversations = {}

class QuestionRequest(BaseModel):
    question: str
    conversation_id: str

@app.post("/process")
async def process_input(
    files: Optional[List[UploadFile]] = File(None),
    blog_url: str = Form(None),
    youtube_url: str = Form(None),
    openai_api_key: str = Form(...)
):
    try:
        text = ""
        
        # Process files if uploaded
        if files:
            for file in files:
                content = await file.read()
                file_extension = os.path.splitext(file.filename)[1].lower()
                
                if file_extension == ".pdf":
                    pdf_file = io.BytesIO(content)
                    pdf_reader = PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                elif file_extension == ".docx":
                    docx_file = io.BytesIO(content)
                    doc = docx.Document(docx_file)
                    text += " ".join([para.text for para in doc.paragraphs])
        
        # Process blog URL
        elif blog_url:
            response = requests.get(blog_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = " ".join([para.get_text() for para in paragraphs])
        
        # Process YouTube URL
        elif youtube_url:
            video_id = youtube_url.split('v=')[-1]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([item['text'] for item in transcript])
        
        if not text:
            raise HTTPException(status_code=400, detail="No content could be extracted from the provided input")

        # Process the text
        text_chunks = get_text_chunks(text)
        vectorstore = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vectorstore, openai_api_key)

        # Generate conversation ID
        import uuid
        conversation_id = str(uuid.uuid4())
        conversations[conversation_id] = conversation_chain

        return {"conversation_id": conversation_id, "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if request.conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        conversation_chain = conversations[request.conversation_id]
        response = conversation_chain({'question': request.question})
        
        chat_history = []
        for i, msg in enumerate(response['chat_history']):
            chat_history.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": msg.content
            })
        
        return {
            "answer": response['answer'],
            "chat_history": chat_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embeddings)

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
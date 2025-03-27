from fastapi import FastAPI, File, UploadFile, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from fastapi.responses import FileResponse, HTMLResponse
from constants import UPLOAD_DIR, ALLOWED_EXTENSIONS, STATIC_DIR
from data_processing.search_embed import search_faiss
from data_processing.embed_store import embed_and_store

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Retrieval Based especially extractive QA


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "Success"
    }


@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File()):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    file_name = file.filename
    ext = file_name.split('.')[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid file type: '{ext}'. Allowed extensions: {ALLOWED_EXTENSIONS}")

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error saving file: {str(e)}")
    try:
        embed_and_store(file_path, ext)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error saving into vector db: {str(e)}")

    return {
        "filename": file.filename,
        "message": "File uploaded and embed stored successfully",
        "status": "success"
    }


@app.get("/search", status_code=status.HTTP_200_OK)
async def search_query(query: str, top_k: int = 3):
    response = search_faiss(query, top_k)
    return {'query': query, "answer": response}

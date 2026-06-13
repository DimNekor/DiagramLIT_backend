import os

from fastapi import FastAPI, UploadFile, File
import uvicorn
from dotenv import load_dotenv

load_dotenv()

UVICORN_PORT = os.getenv("UVICORN_PORT", 8000)

app = FastAPI()


# Твоя функция инференса (заглушка)
def run_my_pipeline(image_bytes: bytes) -> dict:
    # Здесь ты передаешь байты картинки в YOLO / OCR / VLM
    # И формируешь нужный JSON
    return {
        "diagram_type": "BPMN Process Diagram",
        "detected_elements": ["task: Оплата"],
        "message": "Привет с домашнего компа!",
    }


@app.post("/infer")
async def predict_diagram(file: UploadFile = File(...)):
    # Читаем байты картинки, которая прилетела с VPS
    image_bytes = await file.read()

    # Отдаем в пайплайн
    result = run_my_pipeline(image_bytes)

    # FastAPI сам превратит этот словарь в правильный JSON-ответ
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=UVICORN_PORT)

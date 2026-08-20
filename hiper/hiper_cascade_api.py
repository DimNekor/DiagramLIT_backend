import cv2
import numpy as np
import os
import math
import httpx
import json
import re
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from paddleocr import PaddleOCR

app = FastAPI()

# ==========================================
# 1. ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
# ==========================================
print("[Hiper Cascade] Загрузка кастомной YOLOv8...")
custom_model_path = os.path.expanduser("~/models/best.pt")
yolo_model = YOLO(custom_model_path)
yolo_model.to("cuda")

print("[Hiper Cascade] Загрузка PaddleOCR (CPU mode)...")
# Снимаем лимит сжатия, включаем чтение углов и настраиваем фильтры
ocr = PaddleOCR(
    lang="ru", 
    use_angle_cls=True, 
    use_gpu=False,
    det_limit_side_len=2048,
    det_db_thresh=0.3,
    det_db_box_thresh=0.5
)

LLM_API_URL = "http://127.0.0.1:8000/v1/chat/completions"

# ==========================================
# 2. ВСПОМОГАТЕЛЬНАЯ ГЕОМЕТРИЯ
# ==========================================
def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def is_point_in_box(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def build_bpmn_graph(elements):
    nodes = []
    edges = []
    
    for i, el in enumerate(elements):
        el["id"] = f"node_{i}"
        if "arrow" in el["element"].lower():
            edges.append(el)
        else:
            nodes.append(el)
            
    graph_links = []
    
    for edge in edges:
        edge_center = get_center(edge["box"])
        closest_start = None
        closest_end = None
        min_dist_start = float('inf')
        min_dist_end = float('inf')
        
        for node in nodes:
            node_center = get_center(node["box"])
            dist = distance(edge_center, node_center)
            
            if node_center[0] < edge_center[0] or node_center[1] < edge_center[1]:
                if dist < min_dist_start:
                    min_dist_start = dist
                    closest_start = node
            else:
                if dist < min_dist_end:
                    min_dist_end = dist
                    closest_end = node
                    
        if closest_start and closest_end:
            graph_links.append({
                "source": f"{closest_start['element']} ({closest_start.get('text', 'Без текста')})",
                "target": f"{closest_end['element']} ({closest_end.get('text', 'Без текста')})"
            })
            
    return {"nodes": nodes, "links": graph_links}

# ==========================================
# 3. ИНТЕГРАЦИЯ С ЛОКАЛЬНОЙ LLM (QWEN)
# ==========================================
async def analyze_with_llm(graph_data):
    prompt = f"""
    Ты эксперт по анализу бизнес-процессов (BPMN) и информационной безопасности.
    Ниже представлен разобранный граф бизнес-процесса в формате JSON:
    
    {graph_data}
    
    Твоя задача:
    1. Описать этот бизнес-процесс пошагово.
    2. Выявить потенциальные архитектурные риски информационной безопасности.
    
    ОТВЕТ ДОЛЖЕН БЫТЬ СТРОГО В ФОРМАТЕ JSON. Никаких пояснений до или после. Только сырой объект:
    {{
        "step_by_step_description": [
            "Шаг 1: ...",
            "Шаг 2: ..."
        ],
        "security_issues": [
            "Риск 1: ...",
            "Риск 2: ..."
        ]
    }}
    """
    
    payload = {
        "messages": [
            {"role": "system", "content": "Ты ИТ-аналитик. Выдаешь ответы только в формате JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1, 
        "max_tokens": 1500
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(LLM_API_URL, json=payload)
            response.raise_for_status()
            
            raw_text = response.json()["choices"][0]["message"]["content"]
            
            # Очистка текста с использованием hex-кодов (\x60), чтобы не ломать парсеры
            clean_text = raw_text.strip()
            clean_text = re.sub(r"^\x60{3}(?:json)?\s*", "", clean_text)
            clean_text = re.sub(r"\s*\x60{3}$", "", clean_text)
            
            parsed_data = json.loads(clean_text)
            return parsed_data
            
    except json.JSONDecodeError:
        print("[Hiper Cascade] Ошибка парсинга JSON от LLM. Возвращаем сырой текст.")
        return {
            "step_by_step_description": ["Ошибка форматирования ответа LLM. Сырой текст:", raw_text],
            "security_issues": []
        }
    except Exception as e:
        print(f"[Hiper Cascade] Ошибка генерации LLM: {e}")
        return {
            "step_by_step_description": [f"Ошибка API: {str(e)}"],
            "security_issues": []
        }

# ==========================================
# 4. ОСНОВНОЙ ENDPOINT
# ==========================================
@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    print(f"\n[Hiper Cascade] Получен файл: {file.filename}")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    
    # --- БОРЬБА С ПРОЗРАЧНОСТЬЮ (АЛЬФА-КАНАЛ) ---
    cv_img_bgra = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if cv_img_bgra is not None and cv_img_bgra.shape[2] == 4:
        # Заливаем прозрачный фон белым цветом
        alpha_channel = cv_img_bgra[:, :, 3]
        rgb_channels = cv_img_bgra[:, :, :3]
        white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
        cv_img = (rgb_channels * alpha_factor + white_background * (1 - alpha_factor)).astype(np.uint8)
    else:
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # --- ПРЕПРОЦЕССИНГ СПЕЦИАЛЬНО ДЛЯ PADDLE OCR ---
    # 1. Увеличиваем в 2 раза для четкости мелкого текста
    ocr_img = cv2.resize(cv_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # 2. Переводим в ЧБ, чтобы убрать цветной шум
    ocr_gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)
    ocr_ready = cv2.cvtColor(ocr_gray, cv2.COLOR_GRAY2BGR)

    # ШАГ 1: Глобальный OCR
    print("[Hiper Cascade] Выполнение глобального OCR...")
    ocr_results = ocr.ocr(ocr_ready, cls=True) 
    
    all_text_blocks = []
    if ocr_results and ocr_results[0]:
        for line in ocr_results[0]:
            coords = line[0]
            confidence = line[1][1]
            text_str = line[1][0]
            
            # Жесткий фильтр: отсекаем галлюцинации нейросети (порог 60%)
            if confidence > 0.60:
                # ВОЗВРАЩАЕМ КООРДИНАТЫ В НОРМУ (делим на 2.0)
                xs = [p[0] / 2.0 for p in coords]
                ys = [p[1] / 2.0 for p in coords]
                box = [min(xs), min(ys), max(xs), max(ys)]
                all_text_blocks.append({"box": box, "text": text_str})

    # ШАГ 2: YOLO детекция объектов
    print("[Hiper Cascade] Выполнение детекции YOLO...")
    yolo_results = yolo_model(cv_img, conf=0.05, iou=0.5)[0]
    
    detected_elements = []
    for box in yolo_results.boxes:
        coords = box.xyxy[0].tolist() 
        cls_id = int(box.cls[0])
        cls_name = yolo_model.names.get(cls_id, "unknown")
        confidence = float(box.conf[0])

        current_threshold = 0.05 if cls_name == "arrow_end" else 0.3
        if confidence <= current_threshold:
            continue

        obj_text = []
        for block in all_text_blocks:
            text_center = get_center(block["box"])
            if is_point_in_box(text_center, coords):
                obj_text.append(block["text"])
        
        detected_elements.append({
            "element": cls_name,
            "box": coords,
            "text": " ".join(obj_text)
        })

    # ШАГ 3: Граф
    print("[Hiper Cascade] Построение логического графа...")
    graph_data = build_bpmn_graph(detected_elements)

    # ШАГ 4: LLM
    print("[Hiper Cascade] Отправка графа в Qwen2.5-7B на анализ...")
    llm_analysis = await analyze_with_llm(graph_data)

    print("[Hiper Cascade] Форматирование ответа под стандарт Gemini...")
    img_h, img_w = cv_img.shape[:2]
    formatted_elements = [f"{el['element']}: {el['text']}".strip() for el in detected_elements]
    formatted_links = [f"{link['source']} → {link['target']}" for link in graph_data["links"]]
    
    formatted_bboxes = []
    for el in detected_elements:
        x1, y1, x2, y2 = el["box"]
        formatted_bboxes.append({
            "class": el["element"],
            "label": el["text"],
            "x": x1 / img_w,
            "y": y1 / img_h,
            "w": (x2 - x1) / img_w,
            "h": (y2 - y1) / img_h
        })

    print("[Hiper Cascade] Обработка успешно завершена!")
    return {
        "diagram_type": "BPMN Diagram (Локальный Hiper Каскад)",
        "detected_elements": formatted_elements,
        "relationships": formatted_links,
        "bounding_boxes": formatted_bboxes,
        "step_by_step_description": llm_analysis.get("step_by_step_description", []),
        "security_issues": llm_analysis.get("security_issues", [])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8223)

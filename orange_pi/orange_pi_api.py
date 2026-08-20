import os
import cv2
import math
import json
import time
import uuid
import asyncio
import threading
import urllib.request
import numpy as np
import onnxruntime as ort
import pytesseract
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

load_dotenv()
UVICORN_PORT = int(os.getenv("UVICORN_PORT", 8222))

app = FastAPI()

# ============================================================
#  YOLOv8 (детекция элементов диаграммы)
# ============================================================
MODEL_PATH = "best.onnx"
INPUT_SIZE = 640

CLASSES = [
    "arrow_end", "data_base", "data_object", "exclusive_gateway",
    "finish_event", "inclusive_gateway", "intermediate_event",
    "parallel_gateway", "role", "start_event", "task"
]

print(f"Загрузка модели детекции {MODEL_PATH}...")
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("Модель детекции успешно загружена!")
except Exception as e:
    print(f"ВНИМАНИЕ: Ошибка загрузки модели детекции: {e}")
    session = None

# ============================================================
#  LLM Client
# ============================================================
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8223")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 1200))

def call_llm(prompt: str, max_new_tokens: int | None = None) -> str:
    payload = {"prompt": prompt}
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{LLM_SERVER_URL}/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("text", "")
    except Exception as e:
        print(f"LLM-сервер недоступен или ошибка генерации: {e}")
        return ""

def build_llm_prompt(bounding_boxes, relationships, role_names) -> str:
    """Собирает данные диаграммы для LLM с привязкой ролей к каждому действию."""
    nodes = [b for b in bounding_boxes if b['class'] not in ('arrow_end', 'role')]

    lines = []
    if role_names:
        lines.append("Роли (участники процесса):")
        lines += [f"- {r}" for r in role_names]
        lines.append("")

    lines.append("Действия (тип — подпись — роль):")
    if nodes:
        for n in nodes:
            txt = n.get('extracted_text') or '(без подписи)'
            role = n.get('assigned_role') or 'роль не определена'
            line = f"- {n['class']} — {txt} — Роль: {role}"
            surrounding = n.get('surrounding_text')
            if surrounding:
                line += f" — рядом: {surrounding}"
            lines.append(line)
    else:
        lines.append("- (не распознано)")

    lines.append("")
    lines.append("Поток (переходы между действиями):")
    if relationships:
        lines += [f"- {rel}" for rel in relationships]
    else:
        lines.append("- (связи не построены)")

    return "\n".join(lines)


# ============================================================
#  CV-пайплайн
# ============================================================
def process_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1))
    img_tensor = np.expand_dims(img_transposed, axis=0).astype(np.float32) / 255.0
    return img, img_tensor

def postprocess(raw_predictions, iou_threshold=0.5):
    preds = np.squeeze(raw_predictions)
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    boxes = []
    scores = []
    class_ids = []

    for pred in preds:
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]
        class_name = CLASSES[class_id]

        current_threshold = 0.05 if class_name == "arrow_end" else 0.3

        if score > current_threshold:
            cx, cy, w, h = pred[0:4]
            x_min, y_min = cx - (w / 2), cy - (h / 2)
            boxes.append([float(x_min), float(y_min), float(w), float(h)])
            scores.append(float(score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x_min, y_min, w, h = boxes[i]

            norm_x = max(0.0, min(1.0, x_min / INPUT_SIZE))
            norm_y = max(0.0, min(1.0, y_min / INPUT_SIZE))
            norm_w = max(0.0, min(1.0, w / INPUT_SIZE))
            norm_h = max(0.0, min(1.0, h / INPUT_SIZE))

            results.append({
                "class": CLASSES[class_ids[i]],
                "confidence": scores[i],
                "x": norm_x,
                "y": norm_y,
                "w": norm_w,
                "h": norm_h
            })
    return results

def _ocr_prep(crop):
    """Препроцессинг кропа под Tesseract: апскейл, ч/б, бинаризация с учётом фона."""
    # Апскейл — мелкий текст BPMN читается заметно лучше
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Лёгкое шумоподавление без размытия букв
    gray = cv2.bilateralFilter(gray, 5, 50, 50)

    # Otsu-бинаризация
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Tesseract хочет чёрный текст на белом фоне.
    # Если фон тёмный (цветная заливка) — большинство пикселей чёрные → инвертируем
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Небольшая белая рамка вокруг — Tesseract не любит текст впритык к краю
    binary = cv2.copyMakeBorder(binary, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)

    return binary

def _run_tesseract(img, psm):
    import subprocess
    import tempfile
    import os
    
    cfg = f"--psm {psm} -c preserve_interword_spaces=1"
    
    # Сохраняем изображение во временный файл
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        tmp_path = f.name
    
    try:
        # Запускаем tesseract напрямую
        cmd = ["tesseract", tmp_path, "stdout", "-l", "rus+eng"] + cfg.split()
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[WARN] Tesseract return code: {result.returncode}")
            print(f"[WARN] Tesseract stderr: {result.stderr}")
            # Пробуем только английский
            cmd_eng = ["tesseract", tmp_path, "stdout", "-l", "eng"] + cfg.split()
            result_eng = subprocess.run(cmd_eng, capture_output=True, text=True)
            if result_eng.returncode == 0:
                return " ".join(result_eng.stdout.strip().split())
            return ""
        
        text = result.stdout.strip()
        return " ".join(text.split())
    finally:
        # Удаляем временный файл
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def extract_surrounding_text(image_bgr, target_box, all_boxes, img_w, img_h,
                              margin_frac=0.06):
    """
    Читает текст в окрестности бокса (сверху/снизу/слева/справа),
    предварительно замазав белым все ОСТАЛЬНЫЕ боксы, чтобы не
    захватить их подписи. Возвращает строку или "".
    """
    # Пиксельные координаты целевого бокса
    tx1 = int(target_box['x'] * img_w)
    ty1 = int(target_box['y'] * img_h)
    tx2 = int((target_box['x'] + target_box['w']) * img_w)
    ty2 = int((target_box['y'] + target_box['h']) * img_h)

    # Зона поиска = бокс + поля вокруг
    mx = int(margin_frac * img_w)
    my = int(margin_frac * img_h)
    rx1, ry1 = max(0, tx1 - mx), max(0, ty1 - my)
    rx2, ry2 = min(img_w, tx2 + mx), min(img_h, ty2 + my)

    if rx2 - rx1 < 10 or ry2 - ry1 < 10:
        return ""

    region = image_bgr[ry1:ry2, rx1:rx2].copy()

    # Замазываем белым все боксы (включая сам целевой и arrow_end),
    # чтобы остался только "ничейный" текст: подписи стрелок, аннотации
    for b in all_boxes:
        bx1 = int(b['x'] * img_w) - rx1
        by1 = int(b['y'] * img_h) - ry1
        bx2 = int((b['x'] + b['w']) * img_w) - rx1
        by2 = int((b['y'] + b['h']) * img_h) - ry1
        # пересечение с region
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(region.shape[1], bx2), min(region.shape[0], by2)
        if bx2 > bx1 and by2 > by1:
            cv2.rectangle(region, (bx1, by1), (bx2, by2), (255, 255, 255), -1)

    if region.shape[0] < 10 or region.shape[1] < 10:
        return ""

    binary = _ocr_prep(region)
    # psm 11 — разреженный текст, разбросанный по области
    text = _run_tesseract(binary, 11)

    # Фильтр мусора: одиночные символы и слишком короткие огрызки выкидываем
    cleaned = [w for w in text.split() if len(w) >= 2]
    return " ".join(cleaned)

def extract_text_from_box(image_bgr, box, img_w, img_h):
    if box['class'] in ['arrow_end']:
        return ""

    x1 = int(box['x'] * img_w)
    y1 = int(box['y'] * img_h)
    x2 = int((box['x'] + box['w']) * img_w)
    y2 = int((box['y'] + box['h']) * img_h)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    # Паддинг вокруг бокса: YOLO часто режет рамку впритык к тексту
    pad = 4
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return ""

    # Подписи ролей повёрнуты вертикально — выпрямляем
    if box['class'] == 'role':
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    binary = _ocr_prep(crop)

    # PSM под тип элемента:
    #   role  → вертикальный заголовок, по сути одна строка → psm 7
    #   узлы  → короткая подпись в 1–3 строки               → psm 6, фолбэк 7
    if box['class'] == 'role':
        primary, fallback = 7, 6
    else:
        primary, fallback = 6, 7

    text = _run_tesseract(binary, primary)

    # Если основной режим дал пусто — пробуем альтернативный
    if not text:
        text = _run_tesseract(binary, fallback)

    return text

def _y_overlap_ratio(a, b):
    a1, a2 = a['y'], a['y'] + a['h']
    b1, b2 = b['y'], b['y'] + b['h']
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    smaller = min(a['h'], b['h'])
    return inter / smaller if smaller > 0 else 0.0

def is_title_role(candidate, roles, nodes, y_thresh=0.5):
    cand_right = candidate['x'] + candidate['w']
    for other in roles:
        if other is candidate:
            continue
        if other['x'] <= candidate['x']:
            continue
        if _y_overlap_ratio(candidate, other) < y_thresh:
            continue
        yb1 = max(candidate['y'], other['y'])
        yb2 = min(candidate['y'] + candidate['h'], other['y'] + other['h'])
        gap_left, gap_right = cand_right, other['x']
        has_between = any(
            gap_left <= (n['x'] + n['w'] / 2) <= gap_right and
            yb1 <= (n['y'] + n['h'] / 2) <= yb2
            for n in nodes
        )
        if not has_between:
            return True
    return False

def assign_role_to_node(node, roles):
    cy = node['y'] + node['h'] / 2
    candidates = [r for r in roles if r['y'] <= cy <= r['y'] + r['h']]
    if not candidates:
        return "General"
    best = max(candidates, key=lambda r: r['x'])
    return best.get('extracted_text') or "Unknown Role"

def build_graph_with_tracing(image_bgr, bounding_boxes, image_width, image_height):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    arrow_ends = [b for b in bounding_boxes if b['class'] == 'arrow_end']
    nodes = [b for b in bounding_boxes if b['class'] not in ['arrow_end', 'role']]

    def get_pixel_coords(box):
        return (int(box['x'] * image_width), int(box['y'] * image_height),
                int((box['x'] + box['w']) * image_width), int((box['y'] + box['h']) * image_height))

    def get_center(box):
        x1, y1, x2, y2 = get_pixel_coords(box)
        return (x1 + x2) // 2, (y1 + y2) // 2

    lines_only = binary.copy()
    for node in nodes:
        x1, y1, x2, y2 = get_pixel_coords(node)
        cv2.rectangle(lines_only, (x1, y1), (x2, y2), 0, -1)

    kernel = np.ones((11, 11), np.uint8)
    lines_only = cv2.morphologyEx(lines_only, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(lines_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    relationships = []
    search_radius = 30

    for arrow in arrow_ends:
        ax1, ay1, ax2, ay2 = get_pixel_coords(arrow)
        cx, cy = get_center(arrow)

        target_node = None
        min_dist_target = float('inf')
        for node in nodes:
            ncx, ncy = get_center(node)
            dist = math.hypot(cx - ncx, cy - ncy)
            if dist < min_dist_target:
                min_dist_target = dist
                target_node = node

        source_node = None
        matched_contour = None

        arrow_mask = np.zeros_like(lines_only)
        cv2.rectangle(arrow_mask, (ax1 - search_radius, ay1 - search_radius),
                                  (ax2 + search_radius, ay2 + search_radius), 255, -1)

        for contour in contours:
            contour_mask = np.zeros_like(lines_only)
            cv2.drawContours(contour_mask, [contour], -1, 255, 3)
            if cv2.countNonZero(cv2.bitwise_and(arrow_mask, contour_mask)) > 0:
                matched_contour = contour
                break

        if matched_contour is not None and target_node is not None:
            contour_mask = np.zeros_like(lines_only)
            cv2.drawContours(contour_mask, [matched_contour], -1, 255, 5)
            for node in nodes:
                if node == target_node:
                    continue
                nx1, ny1, nx2, ny2 = get_pixel_coords(node)
                node_mask = np.zeros_like(lines_only)
                cv2.rectangle(node_mask, (nx1 - search_radius, ny1 - search_radius),
                                         (nx2 + search_radius, ny2 + search_radius), 255, -1)
                if cv2.countNonZero(cv2.bitwise_and(contour_mask, node_mask)) > 0:
                    source_node = node
                    break

        if source_node is None and target_node is not None:
            min_dist_source = float('inf')
            for node in nodes:
                if node == target_node:
                    continue
                ncx, ncy = get_center(node)
                dist = math.hypot(cx - ncx, cy - ncy)
                if dist < min_dist_source:
                    min_dist_source = dist
                    source_node = node

        if source_node and target_node:
            src_name = source_node.get('extracted_text') or source_node['class']
            tgt_name = target_node.get('extracted_text') or target_node['class']
            relationships.append(f"[{source_node['class']}] {src_name}  →  [{target_node['class']}] {tgt_name}")

    return list(set(relationships))

def check_static_security(bounding_boxes: list, relationships: list) -> list:
    """
    Гибридный статический анализатор уязвимостей в бизнес-процессах BPMN.
    Объединяет динамический анализ сущностей (SoD, Data Leakage) 
    и строгий паттерн-матчинг потоков управления (Control Flow).
    """
    issues = []
    
    # Словари ключевых слов для универсального (динамического) анализа
    creation_keywords = ["созда", "заполн", "отправ", "запрос", "ввод", "create", "submit", "request"]
    approval_keywords = ["провер", "согласов", "утвержд", "подтвержд", "approve", "verify", "check"]
    critical_keywords = ["выдач", "предоставл", "оплат", "удален", "создать", "grant", "execute", "pay"]
    sensitive_keywords = ["парол", "password", "token", "токен", "ключ", "key", "персональн", "пдн", "карт", "secret"]
    untrusted_roles = ["клиент", "client", "гость", "guest", "пользователь", "user", "general"]
    
    # -------------------------------------------------------------------------
    # ЧАСТЬ 1: УНИВЕРСАЛЬНЫЙ АНАЛИЗ СУЩНОСТЕЙ (по Bounding Boxes)
    # -------------------------------------------------------------------------
    creator_roles = set()
    approver_roles = set()
    
    for box in bounding_boxes:
        role = (box.get('assigned_role') or "General").lower()
        text = (box.get('extracted_text') or "").lower()
        cls_name = box.get('class')
        
        # 1.1 Динамический поиск нарушений разделения обязанностей (SoD)
        if cls_name == 'task':
            if any(kw in text for kw in creation_keywords):
                creator_roles.add(box.get('assigned_role'))
            if any(kw in text for kw in approval_keywords):
                approver_roles.add(box.get('assigned_role'))
                
        # 1.2 Универсальный поиск утечек данных (Data Leakage)
        if cls_name in ['data_object', 'data_base']:
            if any(kw in text for kw in sensitive_keywords):
                if any(ur in role for ur in untrusted_roles):
                    issues.append(
                        f"СРЕДНЯЯ: Потенциальная утечка данных. Конфиденциальный объект "
                        f"'{box.get('extracted_text') or cls_name}' обрабатывается в зоне "
                        f"ответственности внешней или недоверенной роли '{box.get('assigned_role')}'."
                    )
                    
    # Проверяем пересечение ролей для SoD (исключая дефолтные пулы)
    sod_intersection = creator_roles.intersection(approver_roles) - {"General", "Unknown Role", "General"}
    for role in sod_intersection:
        issues.append(
            f"КРИТИЧЕСКАЯ: Нарушение разделения обязанностей (SoD). Роль '{role}' "
            f"одновременно инициирует создание заявки/данных и сама же её утверждает/проверяет."
        )

    # -------------------------------------------------------------------------
    # ЧАСТЬ 2: АНАЛИЗ ПОТОКОВ УПРАВЛЕНИЯ (по Реальным Связям из диаграммы)
    # -------------------------------------------------------------------------
    for rel in relationships:
        if "→" not in rel:
            continue
            
        # Разделяем связь на Источник (src) и Цель (tgt)
        parts = rel.split("→")
        src = parts[0].strip().lower()
        tgt = parts[1].strip().lower()
        
        # ПРАВИЛО А (Специфичное): Конечное событие связано с БД напрямую
        # Пример: [finish_event] © → [data_base] Г)
        if "[finish_event]" in src and "[data_base]" in tgt:
            issues.append(
                f"КРИТИЧЕСКАЯ: Небезопасная связь с хранилищем данных. Обнаружен переход '{rel}'. "
                f"Конечное событие не может напрямую изменять или закрывать базу данных. "
                f"Все операции с БД должны строго инкапсулироваться внутри защищенных задач [task] "
                f"с обязательным аудитом и логированием."
            )
            
        # ПРАВИЛО Б (Специфичное + Расширенное): Прямой переход от согласования к выдаче прав
        # Пример: [task] Согласовать уровень доступа → [task] Создать учётную запись...
        if "согласовать" in src and ("создать" in tgt or any(kw in tgt for kw in critical_keywords)):
            issues.append(
                f"ВЫСОКАЯ: Риск обхода авторизационного контроля (Bypassing Controls). Связь '{rel}' "
                f"идет напрямую. Между шагом согласования и критическим действием по выдаче прав "
                f"отсутствует разветвляющий шлюз [exclusive_gateway], обрабатывающий ветку отказа."
            )
            
        # ПРАВИЛО В (Специфичное): Инверсия логического шлюза (Dead-end шлюз)
        # Пример: [task] Согласовать уровень доступа → [exclusive_gateway] >
        if "согласовать уровень доступа" in src and "[exclusive_gateway]" in tgt:
            issues.append(
                f"ВЫСОКАЯ: Нарушение направления доверенных потоков (Control Flow Inversion) в '{rel}'. "
                f"Обнаружен некорректный шлюз проверки. Логический шлюз должен стоять ДО задачи "
                f"согласования и направлять на неё поток, а не служить неконтролируемым стоком."
            )
            
        # ПРАВИЛО Г (Специфичное): Обход шага повторного ввода при доработке
        # Пример: [task] Доработать заявку XX → [task] Проверить заявку
        if "доработать заявку" in src and "проверить заявку" in tgt:
            issues.append(
                f"СРЕДНЯЯ: Уязвимость повторной отправки формы (Re-submission Bypass). "
                f"Связь '{rel}' позволяет миновать шаг обязательного переоформления данных. "
                f"Пользователь может отправить старые невалидные данные на проверку в обход их обновления."
            )

        # ПРАВИЛО Д (Универсальное): Общий контроль защищенных тасков (переход к ним в обход проверок)
        if any(kw in tgt for kw in critical_keywords) and not any(kw in tgt for kw in approval_keywords):
            # Если к критическому действию идут в обход шлюзов или проверок
            if not any(kw in src for kw in approval_keywords) and "[exclusive_gateway]" not in src:
                # Проверим, чтобы не дублировать уже пойманное ПРАВИЛО Б
                if "согласовать" not in src:
                    issues.append(
                        f"ВЫСОКАЯ: Потенциальный обход проверок безопасности. Прямой поток '{rel}' "
                        f"ведет к критической операции в обход контролирующих шлюзов или задач аудита."
                    )
            
    return issues

def run_cv_pipeline(image_bytes: bytes) -> dict:
    if session is None:
        return {"diagram_type": "Ошибка", "error_message": "Модель детекции не загружена"}

    original_img, input_tensor = process_image(image_bytes)
    image_height, image_width = original_img.shape[:2]

    outputs = session.run([output_name], {input_name: input_tensor})
    raw_predictions = outputs[0]

    bounding_boxes = postprocess(raw_predictions)

    roles = [b for b in bounding_boxes if b['class'] == 'role']
    nodes = [b for b in bounding_boxes if b['class'] != 'role' and b['class'] != 'arrow_end']

    for box in bounding_boxes:
        box['extracted_text'] = extract_text_from_box(
            original_img, box, image_width, image_height
        )

    for box in bounding_boxes:
        if box['class'] in ('arrow_end', 'role'):
            box['surrounding_text'] = ""
            continue
        box['surrounding_text'] = extract_surrounding_text(
            original_img, box, bounding_boxes, image_width, image_height
        )

    real_roles = [r for r in roles if not is_title_role(r, roles, nodes)]

    for box in bounding_boxes:
        if box['class'] not in ['role', 'arrow_end']:
            box['assigned_role'] = assign_role_to_node(box, real_roles)

        text = box['extracted_text']
        label_text = text[:15] + "..." if len(text) > 15 else text
        box['label'] = f"{box['class']} ({label_text})" if label_text else box['class']

    relationships = build_graph_with_tracing(original_img, bounding_boxes, image_width, image_height)
    detected_elements = [b['label'] for b in bounding_boxes if b['class'] != 'arrow_end']
    role_names = [r.get('extracted_text') for r in real_roles if r.get('extracted_text')]

    step_by_step_description = [
        f"Распознано ролей: {len(real_roles)}.",
        "Элементы распределены по ролям (дорожкам).",
        f"Построено связей: {len(relationships)}.",
    ]

    static_security_issues = check_static_security(bounding_boxes, relationships)

    return {
        "diagram_type": "BPMN Process Diagram",
        "detected_elements": detected_elements,
        "relationships": relationships,
        "bounding_boxes": bounding_boxes,
        "role_names": role_names,
        "step_by_step_description": step_by_step_description,
        "security_issues": static_security_issues
        }

def build_steps_with_llm(base_steps, llm_description) -> list:
    steps = list(base_steps)
    if llm_description:
        steps.append("Текстовое описание процесса (LLM):")
        for line in llm_description.splitlines():
            line = line.strip()
            if line:
                steps.append(line)
    else:
        steps.append("LLM-описание не сформировано.")
    return steps

# ============================================================
#  Фоновые задачи
# ============================================================
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
JOB_TTL = int(os.getenv("JOB_TTL", 1800))

def _cleanup_jobs():
    now = time.time()
    with jobs_lock:
        stale = [k for k, v in jobs.items() if now - v["created"] > JOB_TTL]
        for k in stale:
            jobs.pop(k, None)

async def _run_llm_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return
    cv = job["cv"]
    # ИСПРАВЛЕНО: Передаем bounding_boxes
    prompt = build_llm_prompt(cv["bounding_boxes"], cv["relationships"], cv["role_names"])
    try:
        text = await asyncio.to_thread(call_llm, prompt)
        with jobs_lock:
            j = jobs.get(job_id)
            if j is not None:
                j["llm_description"] = text
                j["status"] = "done"
    except Exception as e:
        print(f"[job {job_id}] ошибка LLM: {e}")
        with jobs_lock:
            j = jobs.get(job_id)
            if j is not None:
                j["status"] = "error"
                j["error"] = str(e)

# ============================================================
#  Endpoints
# ============================================================
@app.post("/infer")
async def submit_diagram(file: UploadFile = File(...)):
    image_bytes = await file.read()
    cv = await asyncio.to_thread(run_cv_pipeline, image_bytes)

    if cv.get("diagram_type") == "Ошибка":
        return {"status": "error", **cv}

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "created": time.time(),
            "cv": cv,
            "llm_description": "",
            "error": "",
        }

    asyncio.create_task(_run_llm_job(job_id))
    _cleanup_jobs()
    return {"job_id": job_id, "status": "cv_done", **cv}

@app.get("/infer/{job_id}")
async def poll_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"status": "not_found"})
        status = job["status"]
        llm = job["llm_description"]
        error = job["error"]
        base_steps = job["cv"]["step_by_step_description"]

    if status in ("done", "error"):
        steps = build_steps_with_llm(base_steps, llm)
    else:
        steps = list(base_steps)

    return {
        "job_id": job_id,
        "status": status,
        "llm_description": llm,
        "step_by_step_description": steps,
        "error_message": error,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=UVICORN_PORT)

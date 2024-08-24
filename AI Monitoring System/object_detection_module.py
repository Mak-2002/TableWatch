import logging
import math

import cv2
import cvzone
from utils import suppress_stdout_stderr
from config import CLASS_COLORS


def process_video_frame(img, object_detection_model, display_detection_logs, assign_table_identifier,
                        detect_and_log_faces, known_face_encodings, waiter_names, cursor):
    """Process each video frame to detect objects and handle various checks."""
    if not display_detection_logs:
        with suppress_stdout_stderr():
            results = object_detection_model(img)
    else:
        results = object_detection_model(img)

    tables, people, waiters, money, food_items, fights = [], [], [], [], [], []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
            confidence = math.ceil((box.conf[0] * 100)) / 100  # Confidence level
            class_id = int(box.cls[0])
            class_name = object_detection_model.names[class_id]
            # print(class_name)

            if display_detection_logs:
                # Debugging: Print detected class names
                print(f"Detected {class_name} with confidence {confidence}")

            if class_name in CLASS_COLORS:
                color = CLASS_COLORS.get(class_name, (255, 255, 255))  # Default to white if color not defined
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cvzone.putTextRect(img, f'{class_name} {confidence}', (max(0, x1), max(35, y1 - 10)),
                                   scale=0.9, thickness=1, colorR=color)

                # Categorize detected objects
                if class_name == "dining table":
                    table_id = assign_table_identifier((x1, y1, x2, y2))
                    tables.append((x1, y1, x2, y2, table_id))
                    cvzone.putTextRect(img, f'Table {table_id}', (x1, y1 - 40), scale=0.6, thickness=1, colorR=color)
                elif class_name == "person":
                    people.append((x1, y1, x2, y2))
                elif class_name == "waiter":
                    waiters.append((x1, y1, x2, y2))
                elif class_name == "money":
                    money.append((x1, y1, x2, y2))
                elif class_name == "food":
                    food_items.append((x1, y1, x2, y2))
                elif class_name == "fight":
                    fights.append((x1, y1, x2, y2))

    detect_and_log_faces(img, known_face_encodings, waiter_names, cursor, display_detection_logs)
    return tables, people, waiters, money, food_items, fights

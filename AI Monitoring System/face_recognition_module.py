import os

import cv2
import face_recognition
import numpy as np


def load_waiter_images(waiter_images_path):
    """Load images of waiters and extract their names."""
    images = []
    waiter_names = []

    for filename in os.listdir(waiter_images_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(waiter_images_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                waiter_names.append(os.path.splitext(filename)[0])

    return images, waiter_names


def encode_waiter_faces(images):
    """Encode the faces from the loaded images for recognition."""
    encodings = []

    for img in images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_img)
        if face_encodings:
            encodings.append(face_encodings[0])

    return encodings


def detect_and_log_faces(img, known_face_encodings, waiter_names, cursor, display_detection_logs):
    """Detect and recognize faces in the image, and log if a known waiter is detected."""
    small_img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_img)
    face_encodings = face_recognition.face_encodings(rgb_small_img, face_locations)

    for encode_face, face_loc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, encode_face)
        face_distances = face_recognition.face_distance(known_face_encodings, encode_face)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            waiter_id = waiter_names[best_match_index]

            cursor.execute("SELECT name, email FROM waiters WHERE id = %s", (waiter_id,))
            result = cursor.fetchone()

            name_display = f"{result[0]}\n{result[1]}" if result else "Unknown"
        else:
            name_display = "Unknown"

        if display_detection_logs:
            print(name_display)
        y1, x2, y2, x1 = [coord * 4 for coord in face_loc]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for faces
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, name_display, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

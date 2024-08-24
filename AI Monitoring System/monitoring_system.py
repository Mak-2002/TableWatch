from datetime import datetime, timedelta, time

import cv2
import cvzone
from ultralytics import YOLO

from config import VIDEO_FILES, CURRENT_VIDEO_INDEX, WAITER_IMAGES_PATH, PLAYBACK_SPEED, CLASS_COLORS
from database import connect_to_database, log_detected_issue
from face_recognition_module import load_waiter_images, encode_waiter_faces, detect_and_log_faces
from object_detection_module import process_video_frame
from utils import are_tables_similar


class RestaurantMonitoringSystem:
    def __init__(self, playback_speed=PLAYBACK_SPEED, display_detection_logs=True):
        self.database_connection = connect_to_database()
        self.database_cursor = self.database_connection.cursor()

        self.video_capture = cv2.VideoCapture(VIDEO_FILES[CURRENT_VIDEO_INDEX])
        self.object_detection_model = YOLO("../YOLOv8/Yolo-Weights/yolov8x.pt")

        self.playback_speed = playback_speed
        self.display_detection_logs = display_detection_logs

        self.waiter_images, self.waiter_names = load_waiter_images(WAITER_IMAGES_PATH)
        self.known_face_encodings = encode_waiter_faces(self.waiter_images)

        self.table_counter = 0
        self.table_identifiers = {}

    def assign_table_identifier(self, coords):
        """Assign a unique number to each detected table."""
        for existing_coords, table_number in self.table_identifiers.items():
            if are_tables_similar(existing_coords, coords):
                return table_number

        self.table_counter += 1
        self.table_identifiers[coords] = self.table_counter
        return self.table_counter

    def monitor_table_and_food_status(self, img, tables, people, food_items):
        """Monitor each detected table and handle food delivery issues."""
        for coords in tables:
            x1, y1, x2, y2, table_number = coords
            table_occupied = False

            for (px1, py1, px2, py2) in people:
                if px1 < x2 and px2 > x1 and py1 < y2 and py2 > y1:
                    table_occupied = True
                    break

            status = f"Occupied (Table {table_number})" if table_occupied else f"Vacant (Table {table_number})"
            cvzone.putTextRect(img, status, (x1, y1 - 40), scale=0.6, thickness=1, colorR=CLASS_COLORS['dining table'])

            if table_occupied and not any(
                    px1 < x2 and px2 > x1 and py1 < y2 and py2 > y1 for (px1, py1, px2, py2) in food_items):
                issue_name = "Delayed Food Delivery"
                issue_description = f"Table {table_number} at [{x1}, {y1}, {x2}, {y2}] has been occupied for a long time without food delivery."
                video_clip_path = f"clips/delayed_food_delivery_{table_number}.mp4"
                out = cv2.VideoWriter(video_clip_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0,
                                      (img.shape[1], img.shape[0]))
                self.save_video_clip(out, img)
                out.release()

                log_detected_issue(self.database_cursor, issue_name, issue_description, video_clip_path, table_number,
                                   self.display_detection_logs)

            self.check_for_expired_reservations(table_number, coords, img)

    def save_video_clip(self, video_writer, frame, clip_duration=5):
        """Save a video clip of the specified duration."""
        for _ in range(clip_duration * 30):
            video_writer.write(frame)

    def check_for_expired_reservations(self, table_number, table_coords, img):
        """Check if a reservation has expired and log it if so."""
        now = datetime.now()

        query = """
            SELECT reservations.date, reservations.timeStart, reservations.timeEnd, tables.number AS table_number 
            FROM reservations 
            INNER JOIN tables ON reservations.tableID = tables.id 
            WHERE tables.number = %s
        """
        self.database_cursor.execute(query, (table_number,))
        reservation = self.database_cursor.fetchone()

        if reservation:
            # Convert date_start from string to date object
            date_start = datetime.strptime(reservation[0], "%Y-%m-%d").date()

            # Check if time_start is a datetime.time or timedelta
            if isinstance(reservation[1], time):
                time_start = reservation[1]  # If it's already a time object, use it directly
            elif isinstance(reservation[1], timedelta):
                # If it's a timedelta, convert to time assuming it represents the time of day
                total_seconds = reservation[1].seconds
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                time_start = time(hours, minutes, seconds)
            else:
                raise TypeError("Expected reservation[1] to be either a datetime.time or timedelta")

            # Combine date and time, then add a timedelta
            expiry_time = datetime.combine(date_start, time_start) + timedelta(minutes=15)

            if now > expiry_time:
                video_clip_path = f"clips/reservation_expired_{table_number}.mp4"
                out = cv2.VideoWriter(video_clip_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0,
                                      (img.shape[1], img.shape[0]))
                self.save_video_clip(out, img)
                out.release()

                log_detected_issue(self.database_cursor, "Reservation Expired",
                                   f"Table {table_number} reserved at {time_start} is not occupied after 15 minutes.",
                                   video_clip_path, table_number, self.display_detection_logs)

    def monitor_fights(self, img, fights):
        """Monitor and log detected fights."""
        if fights:
            for (fx1, fy1, fx2, fy2) in fights:
                issue_name = "Fight Detected"
                issue_description = f"Fight detected at [{fx1}, {fy1}, {fx2}, {fy2}]."
                video_clip_path = f"clips/fight_detected.mp4"
                out = cv2.VideoWriter(video_clip_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0,
                                      (img.shape[1], img.shape[0]))
                self.save_video_clip(out, img)
                out.release()

                log_detected_issue(self.database_cursor, issue_name, issue_description, video_clip_path, None,
                                   self.display_detection_logs)

    def run(self):
        """Main loop to process video frames."""
        paused = False
        delay = int(1000 / 30 / self.playback_speed)

        while True:
            if not paused:
                success, img = self.video_capture.read()
                if not success:
                    break

                img = self.resize_video_frame(img)
                tables, people, waiters, money, food_items, fights = process_video_frame(img,
                                                                                         self.object_detection_model,
                                                                                         self.display_detection_logs,
                                                                                         self.assign_table_identifier,
                                                                                         detect_and_log_faces,
                                                                                         self.known_face_encodings,
                                                                                         self.waiter_names,
                                                                                         self.database_cursor)

                self.monitor_table_and_food_status(img, tables, people, food_items)
                self.monitor_fights(img, fights)

                cv2.imshow("Restaurant Monitoring", img)

            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q'):
                break

            if key == ord('p'):
                paused = not paused
                if paused:
                    print("Paused. Press 'p' to continue.")

        self.video_capture.release()
        cv2.destroyAllWindows()
        self.database_connection.close()

    def resize_video_frame(self, frame, target_height=720):
        """Resize the video frame to a specific height while maintaining the aspect ratio."""
        height, width = frame.shape[:2]
        scaling_factor = target_height / height
        new_width = int(width * scaling_factor)
        return cv2.resize(frame, (new_width, target_height))

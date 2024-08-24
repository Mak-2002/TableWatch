import mysql.connector


def connect_to_database():
    """Connect to the MySQL database."""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="RestaurantAI"
    )


def log_detected_issue(cursor, issue_name, issue_description, video_clip_path, table_number, display_detection_logs):
    """Log a detected issue into the database."""
    if table_number is not None:
        query = """
            INSERT INTO problems (problem_name, problem_description, video_clip_path, table_number)
            VALUES (%s, %s, %s, %s)
        """
        values = (issue_name, issue_description, video_clip_path, table_number)
        if display_detection_logs:
            print(f"Logging issue: {issue_name} at table {table_number}")
        cursor.execute(query, values)
    else:
        if display_detection_logs:
            print("Skipping issue logging due to invalid table number.")

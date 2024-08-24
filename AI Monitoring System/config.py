VIDEO_FILES = [
    "video_cases/fight and the first client come and leave without pay.MOV",
    "video_cases/money And Order.MOV",
    "video_cases/second client leave without pay.MOV",
    "video_cases/third client leave.MOV",
    "video_cases/empty restaurant.MOV",
    "video_cases/Final video with love.MOV",
    "video_cases/last client leave.MOV",
]

CURRENT_VIDEO_INDEX = 1

# Class colors for bounding boxes
CLASS_COLORS = {
    "dining table": (255, 0, 0),  # Blue for tables
    "person": (0, 255, 0),  # Green for people
    "waiter": (0, 0, 255),  # Red for waiters
    "money": (255, 255, 0),  # Cyan for money
    "food": (255, 0, 255),  # Magenta for food items
    "fight": (0, 255, 255),  # Yellow for fights
}

# Playback speed for video processing
PLAYBACK_SPEED = 2.0

# Waiter images path
WAITER_IMAGES_PATH = '/Backend\\public\\Image\\waiterPhoto'

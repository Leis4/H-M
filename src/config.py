import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'sql', 'hm_market.db')
IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'images_256_256')
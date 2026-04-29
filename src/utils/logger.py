import logging, os
from datetime import datetime

LOG_DIR  = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("hierdr")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s"))
logger.addHandler(console)

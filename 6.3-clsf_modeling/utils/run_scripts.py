import subprocess
from my_logger import logger

# Список скриптов для выполнения
scripts = [
    # "data.py", # сначала генерируем данные
    "tree_clsf.py",
    "logregr.py",
    "knn_clsf.py",
    "forest_clsf.py"
]

for script in scripts:
    logger.info(f"Запуск {script}...")
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при выполнении {script}: {e}")
        break

logger.info("Все скрипты выполнены.")

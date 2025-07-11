import logging
from pathlib import Path
from datetime import datetime

# 项目根目录
project_root = Path(__file__).parent
log_dir = project_root / "Logs"
log_dir.mkdir(exist_ok=True)

log_filename = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 自定义最简格式（只要时间+消息）
class SimpleFormatter(logging.Formatter):
    def format(self, record):
        return f"{datetime.now().strftime('%H:%M')} {record.getMessage()}"

# 清除可能存在的现有处理器
logging.getLogger().handlers = []

# 创建并配置handler
file_handler = logging.FileHandler(str(log_filename), mode='a')
file_handler.setFormatter(SimpleFormatter())

# 配置logger（不再需要设置名称）
logger = logging.getLogger()  # 获取根logger
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# 防止日志重复输出
logger.propagate = False
# log_config.py
import logging
from pathlib import Path
from datetime import datetime

# 项目根目录（假设log_config.py就在项目根目录）
project_root = Path(__file__).parent

log_dir = project_root / "Logs"
log_dir.mkdir(exist_ok=True)  # 确保Logs目录存在

log_filename = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 日志基础配置，只初始化一次
logging.basicConfig(
    filename=str(log_filename),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode='a',  # 追加模式，防止覆盖
)

# 获取全局logger实例，模块中直接导入使用
logger = logging.getLogger("OnlineLearningFramework")
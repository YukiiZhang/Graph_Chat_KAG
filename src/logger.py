import logging
import colorlog
import os

# 创建logger实例
logger = colorlog.getLogger(__name__)

# 如果logger还没有处理器，则添加处理器
if not logger.handlers:
    # 配置日志处理器
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'yellow',
            'INFO': 'green',
            'WARNING': 'cyan',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    
    # 设置日志级别
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    # 防止日志向上传播到根logger
    logger.propagate = False

# 记录不同级别的日志
# logger.debug('这是一条 DEBUG 级别的日志')
# logger.info('这是一条 INFO 级别的日志')
# logger.error('这是一条 ERROR 级别的日志')
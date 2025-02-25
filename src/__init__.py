import inspect
import logging
import transformers
from loguru import logger
from transformers.utils.logging import _get_library_root_logger
from accelerate.state import PartialState

transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if not PartialState().is_main_process:
            return
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(
    level="INFO",
    handlers=[InterceptHandler()],
    force=True,
)
logging.getLogger().handlers = [InterceptHandler()]
_get_library_root_logger().handlers = [InterceptHandler()]

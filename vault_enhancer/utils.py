import logging
import uuid
import sys

_LOGGER = None
_CORRELATION_ID = None

def get_logger():
    global _LOGGER
    if _LOGGER is None:
        # We need a custom filter to handle the 'correlation_id' extra field if it's missing in some logs
        class CorrelationFilter(logging.Filter):
            def filter(self, record):
                if not hasattr(record, 'correlation_id'):
                    record.correlation_id = _CORRELATION_ID or 'N/A'
                return True

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] [%(correlation_id)s] %(message)s'))
        
        _LOGGER = logging.getLogger("vault_enhancer")
        _LOGGER.setLevel(logging.INFO)
        _LOGGER.addHandler(handler)
        _LOGGER.addFilter(CorrelationFilter())
        _LOGGER.propagate = False # Avoid double logging if root logger is also configured
    return _LOGGER

def set_correlation_id(cid=None):
    global _CORRELATION_ID
    _CORRELATION_ID = cid or str(uuid.uuid4())

def log_info(msg):
    get_logger().info(msg)

def log_error(msg):
    get_logger().error(msg)

def log_warning(msg):
    get_logger().warning(msg)


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"

def print_progress(current, total, prefix="", width=30, same_line=True):
    ratio = current / total if total > 0 else 1
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    line = f"{prefix} [{bar}] {current}/{total}"
    if same_line:
        end = "\n" if current >= total else ""
        print(f"\r{line}", end=end, flush=True)
    else:
        print(line, flush=True)

def write_srt(output_path, segments, texts):
    with open(output_path, "w", encoding="utf-8") as output_file:
        for segment, text in zip(segments, texts):
            start_time = format_time(segment.start)
            end_time = format_time(segment.end)
            segment_id = segment.id
            line_out = f"{segment_id}\n{start_time} --> {end_time}\n{text.lstrip()}\n\n"
            output_file.write(line_out)

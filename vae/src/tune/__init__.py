import logging
from rich.logging import RichHandler
import warnings

# Ignore depreciation warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Setup package logger
logging.basicConfig(
    level='INFO',
    format='%(message)s',
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, show_time=True)]
)

log = logging.getLogger(__name__)

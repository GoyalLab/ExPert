import logging
from rich.logging import RichHandler


logging.basicConfig(
    level='INFO',
    format='%(message)s',
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, show_time=True)]
)

log = logging.getLogger(__name__)

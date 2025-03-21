from typing import NamedTuple


class _DATA_SHEET_NT(NamedTuple):
    URL: str = 'download link'
    INDEX: str = 'index'
    P_INDEX: str = 'publication index'
    D_INDEX: str = 'dataset index'
    CANCER: str = 'cancer'

DATA_SHEET_KEYS = _DATA_SHEET_NT()
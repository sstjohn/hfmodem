"""Optional receive decision telemetry shared by VARA modulation families."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SymbolFrame:
    """Optional decision-input telemetry, relative to decode_burst's audio.

    A failed CRC leaves the selected level a hypothesis. Arrays contain payload
    cells only, in coded fill order, before hard slicing. The receiver's existing
    gain normalization is preserved. Consumers must not block a live decoder.
    """
    level: int
    frame_index: int
    crc_ok: bool
    start_sample: int
    end_sample: int
    sample_indices: np.ndarray
    carrier_hz: np.ndarray
    points: np.ndarray
    ideal: np.ndarray
    stage: str
    snr_db: float
    offset_hz: float

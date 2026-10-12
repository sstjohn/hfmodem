"""AI4Y's recorded P2 controls pin the global CS alphabet."""
import pytest
from hfmodem.shrike import p2rx
from hfmodem.tests.shrike.test_robust_answer_0923 import _k4mck

FS = 48000
SLOT = 60000


@pytest.mark.parametrize('slot,entry,swapped', [(1, 0, False), (2, 6, True)])
def test_recorded_p2_controls(slot, entry, swapped):
    x = _k4mck()[slot*SLOT:(slot+1)*SLOT]
    options = dict(entries=(0, 6)) if entry == 6 else {}
    assert p2rx.control_signal_at(x, round(.971*FS), swapped=swapped,
                                 **options) == (entry, 0)

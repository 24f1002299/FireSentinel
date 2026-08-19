import cv2

from scripts.verify_runtime import EXPECTED_SMOKE_HASHES, runtime_report, smoke_hashes


def test_opencv_5_is_installed() -> None:
    assert cv2.__version__.split(".", maxsplit=1)[0] == "5"


def test_smoke_hashes_match_the_clean_install_baseline() -> None:
    assert smoke_hashes() == EXPECTED_SMOKE_HASHES


def test_runtime_report_is_self_consistent() -> None:
    report = runtime_report()
    assert report["opencv"]["version"] == cv2.__version__
    assert report["smoke_hashes"] == EXPECTED_SMOKE_HASHES

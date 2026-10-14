from pathlib import Path

from sign_language.data import discover_video_paths, validate_metadata_rows


def test_metadata_validation_detects_duplicate_and_missing_signer() -> None:
    report = validate_metadata_rows(
        [
            {"sample_id": "sample-1", "signer_id": "S001", "class_name": "HELLO", "video_path": "a.mp4"},
            {"sample_id": "sample-1", "signer_id": "", "class_name": "HELLO", "video_path": "a.mp4"},
        ]
    )

    assert report.duplicate_sample_ids == ("sample-1",)
    assert report.duplicate_video_paths == ("a.mp4",)
    assert report.missing_signer_ids == ("1",)
    assert not report.is_valid


def test_video_discovery_filters_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "HELLO").mkdir()
    (tmp_path / "HELLO" / "sample.mp4").touch()
    (tmp_path / "HELLO" / "notes.txt").touch()

    assert discover_video_paths(tmp_path) == [tmp_path / "HELLO" / "sample.mp4"]

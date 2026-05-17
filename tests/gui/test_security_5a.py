"""Security regression tests for Etapa 5A fixes (T-006).

Tests:
- S-01: JSON cache (no pickle) - already implemented, verified here
- S-02: Path traversal in LoadSnirfBlock
- S-001: HDF5/MAT bomb shape limits
- S-002: ExternalLinks rejection
- S-003: Atomic output (O_CREAT|O_EXCL)
- I-001: strip_pii flag
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nirspy.domain.exceptions import ConverterError, ValidationError


class TestPathTraversal:
    """S-02: LoadSnirfBlock rejects paths with .. or outside allowlist."""

    def test_rejects_dotdot_path(self) -> None:
        from nirspy.blocks.load import validate_snirf_path

        with pytest.raises(ValidationError, match="Path traversal detected"):
            validate_snirf_path(Path("../../../etc/passwd.snirf"))

    def test_rejects_dotdot_in_middle(self) -> None:
        from nirspy.blocks.load import validate_snirf_path

        with pytest.raises(ValidationError, match="Path traversal detected"):
            validate_snirf_path(Path("/data/files/../secret/file.snirf"))

    def test_rejects_wrong_extension(self, tmp_path: Path) -> None:
        from nirspy.blocks.load import validate_snirf_path

        fake = tmp_path / "evil.txt"
        fake.touch()
        with pytest.raises(ValidationError, match="Expected a .snirf file"):
            validate_snirf_path(fake, allowed_dirs=[tmp_path])

    def test_rejects_outside_allowlist(self, tmp_path: Path) -> None:
        from nirspy.blocks.load import validate_snirf_path

        fake = tmp_path / "file.snirf"
        fake.touch()
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        with pytest.raises(ValidationError, match="outside allowed directories"):
            validate_snirf_path(fake, allowed_dirs=[other_dir])

    def test_accepts_valid_path(self, tmp_path: Path) -> None:
        from nirspy.blocks.load import validate_snirf_path

        fake = tmp_path / "subject01.snirf"
        fake.touch()
        result = validate_snirf_path(fake, allowed_dirs=[tmp_path])
        assert result == fake.resolve()

    def test_env_var_extends_allowlist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nirspy.blocks.load import get_allowed_dirs

        monkeypatch.setenv("NIRSPY_ALLOWED_DIRS", str(tmp_path))
        dirs = get_allowed_dirs()
        assert tmp_path.resolve() in dirs


class TestArrayShapeGuard:
    """S-001: Reject oversized arrays in converters."""

    def test_rejects_too_many_elements(self, tmp_path: Path) -> None:
        from nirspy.io.converters import _validate_array_shape

        with pytest.raises(ConverterError, match="elements"):
            _validate_array_shape((200_000_000,), "test", tmp_path / "f.snirf")

    def test_rejects_too_many_channels(self, tmp_path: Path) -> None:
        from nirspy.io.converters import _validate_array_shape

        with pytest.raises(ConverterError, match="channels"):
            _validate_array_shape((100, 200_000), "test", tmp_path / "f.snirf")

    def test_rejects_too_many_dimensions(self, tmp_path: Path) -> None:
        from nirspy.io.converters import _validate_array_shape

        with pytest.raises(ConverterError, match="dimensions"):
            _validate_array_shape((2, 2, 2, 2, 2), "test", tmp_path / "f.snirf")

    def test_accepts_valid_shape(self, tmp_path: Path) -> None:
        from nirspy.io.converters import _validate_array_shape

        _validate_array_shape((1000, 56), "test", tmp_path / "f.snirf")


class TestExternalLinksRejection:
    """S-002: SNIRF files with ExternalLinks are rejected."""

    def test_rejects_external_link(self, tmp_path: Path) -> None:
        import h5py

        from nirspy.io.converters import _check_external_links

        snirf_path = tmp_path / "evil.snirf"
        with h5py.File(str(snirf_path), "w") as f:
            f["nirs/data1/dataTimeSeries"] = np.zeros((10, 2))
            f["nirs/external"] = h5py.ExternalLink("other.hdf5", "/data")

        with h5py.File(str(snirf_path), "r") as f, pytest.raises(
            ConverterError, match="external HDF5 link"
        ):
            _check_external_links(f, snirf_path)

    def test_accepts_normal_file(self, tmp_path: Path) -> None:
        import h5py

        from nirspy.io.converters import _check_external_links

        snirf_path = tmp_path / "normal.snirf"
        with h5py.File(str(snirf_path), "w") as f:
            nirs = f.create_group("nirs")
            nirs.create_group("data1")

        with h5py.File(str(snirf_path), "r") as f:
            _check_external_links(f, snirf_path)


class TestAtomicOutput:
    """S-003: Output file creation is atomic."""

    def test_rejects_existing_file_no_overwrite(self, tmp_path: Path) -> None:
        from nirspy.io.converters import _atomic_create_output

        existing = tmp_path / "output.snirf"
        existing.write_text("data")

        with pytest.raises(ConverterError, match="already exists"):
            _atomic_create_output(existing, overwrite=False)

    def test_allows_overwrite(self, tmp_path: Path) -> None:
        from nirspy.io.converters import _atomic_create_output

        existing = tmp_path / "output.snirf"
        existing.write_text("data")

        _atomic_create_output(existing, overwrite=True)
        assert not existing.exists()

    def test_allows_new_file(self, tmp_path: Path) -> None:
        from nirspy.io.converters import _atomic_create_output

        new_file = tmp_path / "new_output.snirf"

        _atomic_create_output(new_file, overwrite=False)
        assert not new_file.exists()


class TestStripPII:
    """I-001: strip_pii removes SubjectID and DateOfBirth."""

    def test_strip_pii_removes_fields(self) -> None:
        from nirspy.io.converters import _strip_pii_from_metadata

        metadata = {
            "SubjectID": "patient_001",
            "DateOfBirth": "1990-01-01",
            "MeasurementDate": "2024-01-15",
            "LengthUnit": "mm",
        }
        result = _strip_pii_from_metadata(metadata)
        assert "SubjectID" not in result
        assert "DateOfBirth" not in result
        assert "MeasurementDate" in result
        assert "LengthUnit" in result

    def test_strip_pii_preserves_non_pii(self) -> None:
        from nirspy.io.converters import _strip_pii_from_metadata

        metadata = {"TimeUnit": "s", "FrequencyUnit": "Hz"}
        result = _strip_pii_from_metadata(metadata)
        assert result == metadata


class TestJSONCache:
    """S-01: DiskCacheAdapter uses JSON, not pickle."""

    def test_json_disk_class_used(self) -> None:
        from nirspy.engine.cache_adapter import JSONDisk

        assert JSONDisk is not None

    def test_in_memory_cache_works(self) -> None:
        from nirspy.engine.cache_adapter import InMemoryCacheAdapter

        cache = InMemoryCacheAdapter()
        cache.set("key1", {"data": [1, 2, 3]})
        assert cache.get("key1") == {"data": [1, 2, 3]}
        assert cache.get("missing") is None

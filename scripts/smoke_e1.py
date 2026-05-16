"""Smoke manual T-001-E1: pipeline → YAML → exec, com SNIRF do dataset MNE-NIRS sample."""

from __future__ import annotations

from pathlib import Path

import mne
from mne_nirs.io.snirf import write_raw_snirf

from nirspy.blocks import LoadSnirfBlock, LoadSnirfParams, registry
from nirspy.domain.execution import ExecutionContext, run_pipeline_sync
from nirspy.domain.pipeline import Pipeline
from nirspy.io.yaml_serializer import dump_pipeline, load_pipeline


def get_snirf() -> Path:
    """Return SNIRF path. Converte NIRx do fnirs_motor sample se necessário."""
    cache = Path.home() / "mne_data" / "nirspy_smoke" / "fnirs_motor.snirf"
    if cache.exists():
        return cache
    data_dir = Path(mne.datasets.fnirs_motor.data_path(verbose=False))
    existing = list(data_dir.rglob("*.snirf"))
    if existing:
        return existing[0]
    nirx_dirs = [p for p in data_dir.rglob("Participant-*") if p.is_dir()]
    if not nirx_dirs:
        raise RuntimeError("fnirs_motor sample não encontrado")
    raw = mne.io.read_raw_nirx(nirx_dirs[0], preload=True, verbose=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_raw_snirf(raw, cache)
    return cache


def main() -> None:
    snirf = get_snirf()
    print(f"SNIRF: {snirf}")

    pipe = Pipeline(
        name="smoke-e1",
        description="smoke manual T-001-E1",
        steps=[LoadSnirfBlock(LoadSnirfParams(path=str(snirf)))],
    )

    out1 = Path("pipeline_smoke.yml")
    out2 = Path("pipeline_smoke_2.yml")

    dump_pipeline(pipe, out1, overwrite=True)
    print("\n--- YAML emitido ---")
    print(out1.read_text(encoding="utf-8"))

    pipe2 = load_pipeline(out1, registry)
    dump_pipeline(pipe2, out2, overwrite=True)
    assert out1.read_bytes() == out2.read_bytes(), "round-trip não-idempotente"
    print("round-trip YAML: OK (idempotente)")

    results = run_pipeline_sync(pipe2, ExecutionContext())
    r0 = results[0]
    assert r0.block_id == "load_snirf"
    assert r0.metadata["n_channels"] > 0
    assert r0.metadata["sfreq"] > 0
    print(f"exec OK — block={r0.block_id} n_channels={r0.metadata['n_channels']} "
          f"sfreq={r0.metadata['sfreq']} raw={type(r0.data).__name__}")
    print("\nSMOKE E1: PASSED")


if __name__ == "__main__":
    main()

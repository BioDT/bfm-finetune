"""Reproducible acquisition of every EUMon benchmark input, plus consolidated provenance.

``SOURCES`` is the single manifest: URL, on-disk path, SHA-256, licence, DOI and retrieval
timestamp per input. ``fetch`` and ``verify`` are idempotent against it — a file already on
disk with a matching SHA-256 is never re-downloaded. The UKBMS host regenerates parts of
its zips per request, so a fresh re-download can legitimately fail the hash check without
the payload CSV inside having changed; the per-source notes record the stable identities.
"""

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import requests

from .common.runner import artefacts_root as default_artefacts_root
from .common.runner import (atomic_path, code_provenance, project_root, sha256_file, utcnow,
                            write_json)

DOWNLOAD_CHUNK = 1 << 20
REQUEST_TIMEOUT_S = 60
USER_AGENT = "eumon-download/1.0 (+reproducible benchmark acquisition)"

PANEL_ARTEFACTS = ("panel_A.parquet", "panel_B.parquet", "panel_C.parquet")

SOURCES: dict[str, dict[str, Any]] = {
    "task_a": {
        "key": "task_a",
        "role": "task_a",
        "url": "https://www.gbif.se/ipt/archive.do?r=lu_sft_std",
        "local_path": "data/raw/task_a/dwca-lu_sft_std-v1.16.zip",
        "sha256": "c4776d64329e4625e2941413e418159e79aaced610980c84e707ca675e17d500",
        "bytes": 11631528,
        "licence": "CC0-1.0",
        "doi": None,
        "retrieved": "2026-08-09T17:05:45Z",
        "note": "Swedish Bird Survey, fixed routes (Standardrutterna), Lund University GBIF IPT "
                "Darwin Core Archive. No dataset DOI is published on the IPT resource page or in "
                "eml.xml; the eml only cites Davey et al. 2013 (doi:10.1111/1365-2656.12035) as the "
                "source paper, which is not a dataset identifier.",
    },
    "task_b": {
        "key": "task_b",
        "role": "task_b",
        "url": "https://www.gbif.se/ipt/archive.do?r=forestinventory-event",
        "local_path": "data/raw/task_b/dwca-forestinventory-event-v1.2.zip",
        "sha256": "e4100218f03e594a97922ba5edbcc04f643723de92cd5fe6c7da41fd67a7b4db",
        "bytes": 692224402,
        "licence": "CC0-1.0",
        "doi": None,
        "retrieved": "2026-08-09T17:06:14Z",
        "note": "Swedish National Forest Inventory (Riksskogstaxeringen), SLU GBIF IPT Darwin Core "
                "Archive, event core. No dataset DOI is published; eml.xml only cites Fridman et al. "
                "2014 (doi:10.14214/sf.1095) as the source paper.",
    },
    "task_c_indices": {
        "key": "task_c_indices",
        "role": "task_c_indices",
        "url": "https://data-package.ceh.ac.uk/data/04857889-1b09-40ff-a87e-71eb6ac2e998.zip",
        "local_path": "data/raw/ukbms/2023_data_package.zip",
        "sha256": "81e9d223a1dc6106b55c38540282a791b6876f1e71b75b6254755075364de888",
        "bytes": 3780125,
        "licence": "OGL",
        "doi": "10.5285/04857889-1b09-40ff-a87e-71eb6ac2e998",
        "retrieved": "2026-08-09T17:05:44Z",
        "note": "UKBMS site indices 2023 (UKCEH/EIDC). Each annual edition is cumulative back to "
                "1973, so this single edition supplies the full time series and no earlier edition "
                "was downloaded. The bare data-package.ceh.ac.uk/data/<uuid> URL serves an HTML "
                "redirector page, not the file -- the .zip suffix above is required. The extracted "
                "CSV's own sha256 (d34cbd2c89c02505a08f5a378b2fe90fa4106e17a66e2b5fbd0de089f157670a, "
                "per ukbms_survey.json) is the stable identity; the zip container's sha256 can "
                "drift on re-download because the host regenerates "
                "readme.html/ro-crate-metadata.json per request.",
    },
    "task_c_locations": {
        "key": "task_c_locations",
        "role": "task_c_locations",
        "url": "https://data-package.ceh.ac.uk/data/d7256b49-f2e3-4ae7-907d-af729610c768.zip",
        "local_path": "data/raw/ukbms/sitelocs/siteloc_2024_data_package.zip",
        "sha256": "4719e835f02ed97f0f715e629b84d754988b7381e5e4b52e0079820f34127a1e",
        "bytes": 171582,
        "licence": "OGL",
        "doi": "10.5285/d7256b49-f2e3-4ae7-907d-af729610c768",
        "retrieved": None,
        "note": "UKBMS site location data 2024 (UKCEH/EIDC), keyed to site indices by "
                "Site_Number == SITE_CODE. Sensitive sites are withheld from this file by the "
                "publisher. No retrieval timestamp was recorded in site_location_search.json (only "
                "a survey date); the SHA-256 above was not recorded there either and was computed "
                "directly from the on-disk file for this manifest, not carried over from a prior "
                "record. Same data-package.ceh.ac.uk .zip-suffix quirk and per-request "
                "readme.html/ro-crate-metadata.json regeneration as task_c_indices.",
    },
    "task_c_collated": {
        "key": "task_c_collated",
        "role": "task_c_collated",
        "url": "https://data-package.ceh.ac.uk/data/6177caa4-e68e-433c-8135-029ddfd3ba72.zip",
        "local_path": "data/raw/ukbms/collated_2021_data_package.zip",
        "sha256": "05f8da205e83920efea93d69c1c080f2ba8701e487a26127d7a4c3910e27fc44",
        "bytes": 100679,
        "licence": "OGL",
        "doi": "10.5285/6177caa4-e68e-433c-8135-029ddfd3ba72",
        "retrieved": "2026-08-09T17:08:05Z",
        "note": "UKBMS collated indices 2021 (UKCEH/EIDC), UK/country-level species indices "
                "(TIME_PERIOD 1976-2021), a different product from site indices. Same "
                "data-package.ceh.ac.uk .zip-suffix quirk as task_c_indices. Verified by direct "
                "inspection (2026-08-09) that this endpoint regenerates readme.html and "
                "ro-crate-metadata.json with the request timestamp on every call, so a re-download "
                "of this ~100 KB zip will not reproduce the manifest sha256 even though "
                "data/ukbms_collatedindices2021.csv inside it (sha256 "
                "81b41f35867ab9601caa28f1cf33c6d8edb7c3909514bf2e6e8432804cbee26e per "
                "ukbms_survey.json) is unchanged.",
    },
    "task_a_published_index": {
        "key": "task_a_published_index",
        "role": "task_a_published_index",
        "url": "https://www.fageltaxering.lu.se/sites/fageltaxering.lu.se/files/2026-03/populationsindex.xlsx",
        "local_path": "data/raw/task_a/published_trends/populationsindex.xlsx",
        "sha256": "a317ebe063e88a68d60a9c4b89ff44d34f7c4b681aa8c6b6eafaedda04235ee4",
        "bytes": 278709,
        "licence": None,
        "doi": None,
        "retrieved": None,
        "note": "Svensk Fageltaxering (Lund University) official per-species annual TRIM index "
                "workbook, Standardrutter sheet, used to cross-check the reconstructed Task A "
                "panel. No licence is stated on the results page or in the workbook itself, and no "
                "dataset DOI exists -- left as None rather than guessed. No retrieval timestamp was "
                "recorded in trends_search.json (only a search date); the SHA-256 above was not "
                "recorded there either and was computed directly from the on-disk file.",
    },
    "weights": {
        "key": "weights",
        "role": "weights",
        "url": "https://huggingface.co/BioDT/bfm-pretrained/resolve/main/bfm-pretrain-large.safetensors",
        "local_path": "weights/bfm-pretrain-large.safetensors",
        "sha256": "8bbe0db77575fe2a2f84ab78120a6c945da9842a75538fe5d868c4a4c5a8e466",
        "bytes": 2840877880,
        "licence": "MIT",
        "doi": None,
        "retrieved": "2026-08-09T17:07:13Z",
        "note": "BioAnalyst/BFM pretrained-large weights, HF repo BioDT/bfm-pretrained, ungated. "
                "Matches the LFS-reported SHA-256 in the repo's file listing. Only pretrained "
                "weights are published on HF -- no K=6/K=12 rollout-finetuned checkpoint exists "
                "there as of this retrieval; HF repos carry no DOI.",
    },
}


def _local_path(key: str) -> Path:
    return project_root() / SOURCES[key]["local_path"]


def fetch(key: str, force: bool = False) -> dict[str, Any]:
    """Download ``key`` to its manifest ``local_path`` unless it is already present and correct."""
    if key not in SOURCES:
        raise KeyError(f"unknown source {key!r}; choose from {sorted(SOURCES)}")
    src = SOURCES[key]
    path = _local_path(key)

    if not force and path.exists() and src["sha256"] is not None and sha256_file(path) == src["sha256"]:
        return {"key": key, "path": str(path), "sha256": src["sha256"], "bytes": path.stat().st_size,
                "skipped": True, "url": src["url"], "retrieved": src["retrieved"]}

    headers = {"User-Agent": USER_AGENT}
    with requests.get(src["url"], stream=True, timeout=REQUEST_TIMEOUT_S, headers=headers) as resp:
        resp.raise_for_status()
        with atomic_path(path) as tmp:
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK):
                    if chunk:
                        fh.write(chunk)

    got = sha256_file(path)
    if src["sha256"] is not None and got != src["sha256"]:
        path.unlink(missing_ok=True)
        raise ValueError(f"sha256 mismatch for {key!r}: expected {src['sha256']}, got {got}")

    return {"key": key, "path": str(path), "sha256": got, "bytes": path.stat().st_size,
            "skipped": False, "url": src["url"], "retrieved": utcnow()}


def fetch_all(keys: Iterable[str] | None = None, force: bool = False) -> dict[str, Any]:
    """Fetch every source, or the given subset of keys."""
    return {k: fetch(k, force=force) for k in (keys if keys is not None else SOURCES)}


def verify(key: str | None = None) -> dict[str, Any]:
    """Check on-disk SHA-256 against the manifest without downloading anything."""
    report: dict[str, Any] = {}
    for k in ([key] if key is not None else list(SOURCES)):
        src = SOURCES[k]
        path = _local_path(k)
        if not path.exists():
            report[k] = {"status": "missing", "path": str(path)}
            continue
        got = sha256_file(path)
        if src["sha256"] is None:
            report[k] = {"status": "ok", "path": str(path), "sha256": got,
                        "note": "no manifest sha256 recorded to compare against"}
        elif got == src["sha256"]:
            report[k] = {"status": "ok", "path": str(path), "sha256": got}
        else:
            report[k] = {"status": "mismatch", "path": str(path),
                        "expected": src["sha256"], "got": got}
    return report


def write_provenance(path: str | Path | None = None,
                     artefacts_root: str | Path | None = None) -> Path:
    """Write the consolidated provenance deliverable: sources, derived artefacts, code.

    ``artefacts_root`` selects which directory is described and where the manifest lands,
    so a redirected campaign documents its own outputs.
    """
    root = project_root()
    arte = Path(artefacts_root) if artefacts_root is not None else default_artefacts_root()
    if not arte.is_absolute():
        arte = root / arte
    out_path = Path(path) if path is not None else arte / "provenance.json"
    if not out_path.is_absolute():
        out_path = root / out_path

    sources = {}
    for k, src in SOURCES.items():
        p = _local_path(k)
        entry = dict(src)
        entry["on_disk"] = p.exists()
        entry["sha256_on_disk"] = sha256_file(p) if p.exists() else None
        sources[k] = entry

    derived = {}
    for name in PANEL_ARTEFACTS:
        p = arte / name
        if p.exists():
            derived[name] = {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}

    payload = {
        "generated": utcnow(),
        "sources": sources,
        "derived": derived,
        "code": code_provenance(),
    }
    write_json(out_path, payload)
    return out_path


def _cli(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "verify"

    if cmd == "verify":
        report = verify(argv[2] if len(argv) > 2 else None)
        for k, rec in report.items():
            print(f"{rec['status']:8s} {k}")
        return 0 if all(rec["status"] == "ok" for rec in report.values()) else 1

    if cmd == "fetch":
        if len(argv) < 3:
            print("usage: download.py fetch <key>", file=sys.stderr)
            return 2
        print(json.dumps(fetch(argv[2]), indent=2))
        return 0

    if cmd == "fetch-all":
        results = fetch_all()
        for k, rec in results.items():
            print(f"{'skip' if rec['skipped'] else 'fetched':8s} {k}")
        return 0

    if cmd == "provenance":
        print(str(write_provenance()))
        return 0

    print(f"unknown command {cmd!r}; choose from verify|fetch|fetch-all|provenance", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))

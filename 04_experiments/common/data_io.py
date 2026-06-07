"""Per-case data loaders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from . import paths


@dataclass
class CaseData:
    case_id: str
    ingestion: dict
    design: pd.DataFrame
    responses: pd.DataFrame
    raw_dir: Path
    proc_dir: Path


def load_case(case_id: str) -> CaseData:
    proc = paths.DATA_PROC / case_id
    raw  = paths.DATA_RAW  / case_id

    with open(proc / f"{case_id.split('_')[0]}_ingestion.yaml", "r", encoding="utf-8") as fh:
        ingestion = yaml.safe_load(fh)

    if case_id == "case1b_akhtar_bilayer":
        # Composite case: TAM + FIN
        design = pd.concat([
            pd.read_csv(proc / "case1b_tam_design.csv").assign(layer="TAM"),
            pd.read_csv(proc / "case1b_fin_design.csv").assign(layer="FIN"),
        ], ignore_index=True)
        responses = pd.concat([
            pd.read_csv(proc / "case1b_tam_responses.csv").assign(layer="TAM"),
            pd.read_csv(proc / "case1b_fin_responses.csv").assign(layer="FIN"),
        ], ignore_index=True)
    else:
        slug = case_id.split("_")[0]
        design = pd.read_csv(proc / f"{slug}_design.csv")
        responses = pd.read_csv(proc / f"{slug}_responses.csv")

    return CaseData(case_id=case_id, ingestion=ingestion,
                    design=design, responses=responses,
                    raw_dir=raw, proc_dir=proc)

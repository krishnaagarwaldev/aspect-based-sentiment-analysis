"""
Dataset parsers for ABSA datasets:
- SemEval-2014 (Laptop, Restaurant)
- EduRABSA (College)
- HOSSemEval-EB23 (Hospitality)
"""

import ast
import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RawSample:
    review_id: str
    text: str
    aspect: str
    polarity: str
    domain: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


def parse_semeval(csv_path: Path, domain: str) -> List[RawSample]:
    """
    Parses SemEval-2014 CSV file (Laptop_Train_v2.csv or Restaurants_Train_v2.csv).
    Columns: id, Sentence, Aspect Term, polarity, from, to
    """
    samples = []
    if not csv_path.exists():
        logger.warning(f"File not found: {csv_path}")
        return samples

    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            text = row.get("Sentence", "").strip()
            aspect = row.get("Aspect Term", "").strip()
            polarity = row.get("polarity", "").strip().lower()
            rid = row.get("id", f"{domain}_{idx}").strip()
            from_idx = row.get("from")
            to_idx = row.get("to")

            try:
                start_c = int(from_idx) if from_idx is not None and from_idx != "" else None
                end_c = int(to_idx) if to_idx is not None and to_idx != "" else None
            except ValueError:
                start_c, end_c = None, None

            if not text or not aspect:
                continue

            samples.append(
                RawSample(
                    review_id=f"{domain}_{rid}",
                    text=text,
                    aspect=aspect,
                    polarity=polarity,
                    domain=domain,
                    start_char=start_c,
                    end_char=end_c,
                    metadata={"source": "semeval_2014", "original_id": rid},
                )
            )
    return samples


def parse_edurabsa(jsonl_path: Path, domain: str = "college") -> List[RawSample]:
    """
    Parses EduRABSA ASQE JSONL file.
    Structure: {"text": "...", "labels": [{"aspect": "...", "opinion": "...", "polarity": "...", "category": "..."}]}
    """
    samples = []
    if not jsonl_path.exists():
        logger.warning(f"File not found: {jsonl_path}")
        return samples

    with open(jsonl_path, mode="r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = data.get("text", "").strip()
            labels = data.get("labels", [])
            if not text or not labels:
                continue

            review_id = f"{domain}_{idx}"

            # Group multiple opinions for the same aspect within this review
            aspect_polarities: Dict[str, List[str]] = {}
            aspect_metadata: Dict[str, List[Dict]] = {}

            for lbl in labels:
                aspect = lbl.get("aspect", "").strip()
                pol = lbl.get("polarity", "").strip().lower()

                # Handle null/implicit aspect terms
                if not aspect or aspect.lower() == "null":
                    continue

                if aspect not in aspect_polarities:
                    aspect_polarities[aspect] = []
                    aspect_metadata[aspect] = []
                aspect_polarities[aspect].append(pol)
                aspect_metadata[aspect].append(
                    {"opinion": lbl.get("opinion", ""), "category": lbl.get("category", "")}
                )

            for aspect, pol_list in aspect_polarities.items():
                if not pol_list:
                    continue
                # Majority vote for polarities
                pol_counts = Counter(pol_list)
                most_common = pol_counts.most_common()
                if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                    # Tie-breaking
                    final_pol = "neutral"
                else:
                    final_pol = most_common[0][0]

                # Find aspect char positions if present
                start_c = text.lower().find(aspect.lower())
                end_c = start_c + len(aspect) if start_c != -1 else None
                if start_c == -1:
                    start_c = None

                samples.append(
                    RawSample(
                        review_id=review_id,
                        text=text,
                        aspect=aspect,
                        polarity=final_pol,
                        domain=domain,
                        start_char=start_c,
                        end_char=end_c,
                        metadata={
                            "source": "edurabsa",
                            "raw_opinions": aspect_metadata[aspect],
                            "vote_counts": dict(pol_counts),
                        },
                    )
                )
    return samples


def parse_hos_semeval(txt_path: Path, domain: str = "hospitality") -> List[RawSample]:
    """
    Parses HOSSemEval-EB23 train.txt / dev.txt.
    Format: <id>,<text> ####<list of [aspect_span, category, polarity]>
    """
    samples = []
    if not txt_path.exists():
        logger.warning(f"File not found: {txt_path}")
        return samples

    with open(txt_path, mode="r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or "####" not in line:
                continue

            header_text, annotations_str = line.split("####", 1)
            header_text = header_text.strip()
            annotations_str = annotations_str.strip()

            # header is typically "<id>,<text>"
            parts = header_text.split(",", 1)
            if len(parts) == 2:
                raw_id, text = parts[0].strip(), parts[1].strip()
            else:
                raw_id, text = f"{domain}_{line_num}", header_text

            review_id = f"{domain}_{raw_id}"

            try:
                quads = ast.literal_eval(annotations_str)
            except Exception:
                continue

            if not isinstance(quads, list):
                continue

            for quad in quads:
                if not isinstance(quad, (list, tuple)) or len(quad) < 3:
                    continue
                aspect = str(quad[0]).strip()
                category = str(quad[1]).strip()
                pol = str(quad[2]).strip().lower()

                if not aspect or not pol:
                    continue

                start_c = text.lower().find(aspect.lower())
                end_c = start_c + len(aspect) if start_c != -1 else None
                if start_c == -1:
                    start_c = None

                samples.append(
                    RawSample(
                        review_id=review_id,
                        text=text,
                        aspect=aspect,
                        polarity=pol,
                        domain=domain,
                        start_char=start_c,
                        end_char=end_c,
                        metadata={"source": "hos_semeval", "category": category},
                    )
                )
    return samples

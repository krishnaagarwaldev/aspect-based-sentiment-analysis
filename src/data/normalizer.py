"""
Data normalization pipeline for ABSA.
Normalizes heterogeneous datasets into a unified schema:
text, aspect, sentiment, domain, review_id
Strictly enforces label integrity (never silently converts conflict to neutral).
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .parsers import RawSample, parse_edurabsa, parse_hos_semeval, parse_semeval

logger = logging.getLogger(__name__)

STANDARD_SENTIMENTS = {"positive", "negative", "neutral"}
SENTIMENT2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2SENTIMENT = {0: "negative", 1: "neutral", 2: "positive"}


@dataclass
class NormalizedSample:
    id: str
    review_id: str
    text: str
    aspect: str
    sentiment: str
    sentiment_id: int
    domain: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    raw_polarity: Optional[str] = None
    metadata: Optional[Dict] = None


class ABSADataPipeline:
    def __init__(
        self,
        dataset_dir: str = "Dataset",
        allowed_sentiments: Optional[set] = None,
        drop_conflict: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.allowed_sentiments = allowed_sentiments or STANDARD_SENTIMENTS
        self.drop_conflict = drop_conflict
        self.filtered_stats: Dict[str, Dict[str, int]] = {}

    def load_all_domains(self) -> List[NormalizedSample]:
        """Loads and normalizes all 4 domains: laptop, restaurant, college, hospitality."""
        raw_samples: List[RawSample] = []

        # 1. SemEval Laptop
        laptop_path = self.dataset_dir / "SemEval-2014" / "Laptop_Train_v2.csv"
        raw_laptop = parse_semeval(laptop_path, domain="laptop")
        logger.info(f"Loaded {len(raw_laptop)} raw samples from Laptop (SemEval-2014)")
        raw_samples.extend(raw_laptop)

        # 2. SemEval Restaurant
        rest_path = self.dataset_dir / "SemEval-2014" / "Restaurants_Train_v2.csv"
        raw_rest = parse_semeval(rest_path, domain="restaurant")
        logger.info(f"Loaded {len(raw_rest)} raw samples from Restaurant (SemEval-2014)")
        raw_samples.extend(raw_rest)

        # 3. EduRABSA College
        edu_train_path = self.dataset_dir / "EduRASA_Datasets_ASQE" / "Train_validation_Merged_ASQE_N4000.jsonl"
        edu_test_path = self.dataset_dir / "EduRASA_Datasets_ASQE" / "DEV_TEST_Merged_ASQE_N2500.jsonl"
        raw_edu = parse_edurabsa(edu_train_path, domain="college") + parse_edurabsa(edu_test_path, domain="college")
        logger.info(f"Loaded {len(raw_edu)} raw samples from College (EduRABSA)")
        raw_samples.extend(raw_edu)

        # 4. HOSSemEval Hospitality
        hos_train_path = self.dataset_dir / "HOSSemEval-EB23" / "train.txt"
        hos_dev_path = self.dataset_dir / "HOSSemEval-EB23" / "dev.txt"
        raw_hos = parse_hos_semeval(hos_train_path, domain="hospitality") + parse_hos_semeval(hos_dev_path, domain="hospitality")
        logger.info(f"Loaded {len(raw_hos)} raw samples from Hospitality (HOSSemEval-EB23)")
        raw_samples.extend(raw_hos)

        return self.normalize(raw_samples)

    def normalize(self, raw_samples: List[RawSample]) -> List[NormalizedSample]:
        """
        Normalizes samples to common schema.
        Handles incompatible labels (e.g. conflict) transparently without silent re-mapping.
        """
        normalized: List[NormalizedSample] = []
        self.filtered_stats = {}

        for idx, sample in enumerate(raw_samples):
            domain = sample.domain
            if domain not in self.filtered_stats:
                self.filtered_stats[domain] = {"retained": 0, "dropped_conflict": 0, "dropped_other": 0}

            pol = sample.polarity.strip().lower()

            # Strict policy: NEVER silently convert 'conflict' to 'neutral'
            if pol == "conflict":
                if self.drop_conflict:
                    self.filtered_stats[domain]["dropped_conflict"] += 1
                    continue
                else:
                    # In 4-class mode if explicitly enabled, keep as 'conflict'
                    sentiment = "conflict"
            elif pol in self.allowed_sentiments:
                sentiment = pol
            else:
                self.filtered_stats[domain]["dropped_other"] += 1
                continue

            sentiment_id = SENTIMENT2ID.get(sentiment, -1)
            sample_id = f"{domain}_{idx:06d}"

            normalized.append(
                NormalizedSample(
                    id=sample_id,
                    review_id=sample.review_id,
                    text=sample.text,
                    aspect=sample.aspect,
                    sentiment=sentiment,
                    sentiment_id=sentiment_id,
                    domain=domain,
                    start_char=sample.start_char,
                    end_char=sample.end_char,
                    raw_polarity=sample.polarity,
                    metadata=sample.metadata,
                )
            )
            self.filtered_stats[domain]["retained"] += 1

        logger.info("Normalization & Filtering Summary:")
        for dom, stats in self.filtered_stats.items():
            logger.info(
                f"  [{dom.upper()}] Retained: {stats['retained']} | "
                f"Dropped 'conflict': {stats['dropped_conflict']} | "
                f"Dropped other: {stats['dropped_other']}"
            )

        return normalized

    def to_dataframe(self, samples: List[NormalizedSample]) -> pd.DataFrame:
        """Converts normalized samples to a pandas DataFrame with common schema."""
        data = [
            {
                "id": s.id,
                "review_id": s.review_id,
                "text": s.text,
                "aspect": s.aspect,
                "sentiment": s.sentiment,
                "sentiment_id": s.sentiment_id,
                "domain": s.domain,
                "start_char": s.start_char,
                "end_char": s.end_char,
            }
            for s in samples
        ]
        return pd.DataFrame(data)

    def compute_class_weights(self, samples: List[NormalizedSample]) -> Dict[str, float]:
        """
        Computes balanced inverse-frequency class weights:
        w_c = Total / (Num_Classes * Count_c)
        """
        counts = {s: 0 for s in self.allowed_sentiments}
        for s in samples:
            if s.sentiment in counts:
                counts[s.sentiment] += 1

        total = sum(counts.values())
        num_classes = len(counts)
        weights = {}
        for s, count in counts.items():
            weights[s] = (total / (num_classes * count)) if count > 0 else 1.0

        return weights

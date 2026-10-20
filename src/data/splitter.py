"""
Leakage-free dataset splitting logic.
Groups samples by review_id so that multiple aspects from the same sentence/review
never cross the boundary between train, validation, and test splits.
"""

import logging
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .normalizer import NormalizedSample

logger = logging.getLogger(__name__)


class ReviewLevelSplitter:
    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(
        self, samples: List[NormalizedSample]
    ) -> Tuple[List[NormalizedSample], List[NormalizedSample], List[NormalizedSample]]:
        """
        Splits samples into Train, Val, Test without review-level leakage.
        Groups by review_id, then allocates entire review groups to splits.
        """
        rng = random.Random(self.seed)

        # 1. Group samples by review_id
        review_to_samples: Dict[str, List[NormalizedSample]] = defaultdict(list)
        review_domain: Dict[str, str] = {}
        for s in samples:
            review_to_samples[s.review_id].append(s)
            review_domain[s.review_id] = s.domain

        # 2. Stratify reviews by domain to ensure balanced domain representation
        domain_to_reviews: Dict[str, List[str]] = defaultdict(list)
        for rid, dom in review_domain.items():
            domain_to_reviews[dom].append(rid)

        train_reviews, val_reviews, test_reviews = set(), set(), set()

        for dom, rids in domain_to_reviews.items():
            rng.shuffle(rids)
            n_total = len(rids)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)

            dom_train = rids[:n_train]
            dom_val = rids[n_train : n_train + n_val]
            dom_test = rids[n_train + n_val :]

            train_reviews.update(dom_train)
            val_reviews.update(dom_val)
            test_reviews.update(dom_test)

        train_samples = [s for s in samples if s.review_id in train_reviews]
        val_samples = [s for s in samples if s.review_id in val_reviews]
        test_samples = [s for s in samples if s.review_id in test_reviews]

        # Verify zero intersection of review_ids
        train_ids = set(s.review_id for s in train_samples)
        val_ids = set(s.review_id for s in val_samples)
        test_ids = set(s.review_id for s in test_samples)

        assert train_ids.isdisjoint(val_ids), "Data leakage detected: train and val share review IDs!"
        assert train_ids.isdisjoint(test_ids), "Data leakage detected: train and test share review IDs!"
        assert val_ids.isdisjoint(test_ids), "Data leakage detected: val and test share review IDs!"

        logger.info(
            f"Leakage-free split complete: "
            f"Train={len(train_samples)} aspects ({len(train_reviews)} reviews), "
            f"Val={len(val_samples)} aspects ({len(val_reviews)} reviews), "
            f"Test={len(test_samples)} aspects ({len(test_reviews)} reviews)."
        )

        return train_samples, val_samples, test_samples

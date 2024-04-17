from typing import Set

import pandas as pd

from practice_prioritization.constants import DEPRECATED_SUBCHARS, PRACTICES_COLUMN_NAME, QUALITY_ATTRIBUTE_COLUMN_NAME



def read_merged_scores(path: str, remove_deprecated_subchars: bool = True) -> pd.DataFrame:
    raw_scores = pd.read_csv(path, sep=",").set_index([QUALITY_ATTRIBUTE_COLUMN_NAME, PRACTICES_COLUMN_NAME])

    if remove_deprecated_subchars:
        raw_scores = ignore_deprecated_subchars(raw_scores=raw_scores, deprecated_subchars=DEPRECATED_SUBCHARS)

    return raw_scores


def ignore_deprecated_subchars(raw_scores: pd.DataFrame, deprecated_subchars: Set[str]) -> pd.DataFrame:
    """
    Removes certain sub-characteristics from the scores.
    """
    return raw_scores.iloc[~raw_scores.index.get_level_values(0).isin(deprecated_subchars)]

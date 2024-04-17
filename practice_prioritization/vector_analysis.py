from typing import List

import numpy as np
import pandas as pd


def compute_mean_weights(merged_vectors: pd.DataFrame) -> pd.Series:
    merged_vectors["mean_weights"] = merged_vectors[list(merged_vectors.columns)].mean(axis=1)
    return merged_vectors[["mean_weights"]].squeeze()


def compute_mean_influence(mean_weights: pd.Series) -> pd.Series:
    mean_influence = mean_weights.groupby("Practice").mean().sort_values(ascending=False)

    return mean_influence


def compute_n_practices_influenced(mean_weights: pd.Series) -> pd.Series:
    has_influence = mean_weights.apply(lambda x: 1 if x > 0 else 0).copy(deep=True)

    n_practices_influenced = has_influence.groupby("Practice").sum().sort_values(ascending=False)

    return n_practices_influenced


def compute_largest_influence_per_subchar(
    mean_weights: pd.Series,
    quality_subchar: str,
    topN: int,
    influence_threshold: float,
) -> pd.Series:
    """
    Returns the topN practices with the largest influence above the influence_threshold, per quality sub-characteristic
    """
    largest_influence_per_subchar = (
        mean_weights[mean_weights.index.get_level_values("quality sub-characteristic") == quality_subchar]
        .sort_values(ascending=False)
        .head(topN)
    )

    return largest_influence_per_subchar[largest_influence_per_subchar > influence_threshold]


def compute_largest_influence_per_practice(
    mean_weights: pd.Series, practice: str, topN: int, influence_threshold: float
) -> pd.Series:
    """
    Returns the topN quality sub-characteristics with the largest influence above the influence_threshold, per practice
    """
    largest_influence_per_practice = (
        mean_weights[mean_weights.index.get_level_values("Practice") == practice]
        .sort_values(ascending=False)
        .head(topN)
    )

    return largest_influence_per_practice[largest_influence_per_practice > influence_threshold]


def create_practice_recommendations(mean_weights: pd.Series, influence_threshold: float) -> pd.DataFrame:
    unique_subchars = np.unique(mean_weights.index.get_level_values("quality sub-characteristic").values)

    practice_recommendations = pd.DataFrame()
    for subchar in unique_subchars:
        top_5_practices = compute_largest_influence_per_subchar(
            mean_weights=mean_weights,
            quality_subchar=subchar,
            topN=10,
            influence_threshold=influence_threshold,
        )

        top_5_practices = top_5_practices.to_frame().reset_index()
        practice_recommendations = pd.concat([practice_recommendations, top_5_practices], axis=0)

    return practice_recommendations


def determine_quality_aspects_with_gaps(
    mean_weights: pd.Series,
    unique_subchars: np.array,
    gap_threshold: float,
) -> pd.Series:
    quality_aspects_with_gaps = pd.Series(dtype=float)

    for subchar in unique_subchars:
        top_practice = compute_largest_influence_per_subchar(
            mean_weights=mean_weights,
            quality_subchar=subchar,
            topN=1,
            influence_threshold=0.0,
        )

        if top_practice.values[0] < gap_threshold:
            if len(quality_aspects_with_gaps.index) == 0:
                quality_aspects_with_gaps = top_practice
            else:
                quality_aspects_with_gaps = quality_aspects_with_gaps.append(top_practice)

    return quality_aspects_with_gaps.sort_values(ascending=True)


def display_quality_aspects_with_gaps(quality_aspects_with_gaps: pd.Series) -> None:
    print("The quality sub-characteristics with gaps in the associated practices are:")
    print("\n")
    print("\n")
    for index, value in quality_aspects_with_gaps.iteritems():
        print(f' "{index[0]}" for which the most influential practice is "{index[1]}" with mean influence: {value}')
        print("\n")


def compute_agreement_between_practitioners(
    weights: pd.DataFrame,
    prefix: str,
    agreement_func: callable,
) -> pd.DataFrame:
    data = {}
    cols = sorted(col for col in weights.columns if col.startswith(prefix))
    for col1 in cols:
        name1 = col1[len(prefix) :]
        pers = {}
        for col2 in cols:
            if col1 == col2:
                continue
            name2 = col2[len(prefix) :]
            pers[name2] = agreement_func(weights[col1], weights[col2])
        pers["mean"] = sum(pers.values()) / len(pers)
        data[name1] = pers
    return pd.DataFrame(data)


def compute_variance_per_quality_subchar(mean_weights: pd.Series) -> pd.Series:
    """
    Computes the variance of the mean weights per quality sub-characteristic.
    """
    return mean_weights.groupby(level=0).var().sort_values(ascending=False)


def get_top_N_variance_per_pair(dataframe_w_scores: pd.DataFrame, topN: int = 10) -> pd.Series:
    """
    Computes the variance per quality aspect - practice pair, and returns the pairs with the N pairs with the
    highest variances.
    """
    return dataframe_w_scores.var(axis=1).sort_values(ascending=False).head(topN)


def index_within_bounds(index: int, list: List):
    if index + 1 < len(list):
        return True

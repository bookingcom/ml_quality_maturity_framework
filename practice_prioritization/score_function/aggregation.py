from typing import Callable, Dict, Optional, Protocol, Set, Union

import numpy as np
import pandas as pd

from practice_prioritization.constants import QUALITY_ASSESSMENT_PRACTICES, AssessmentPractices


def step_score_rescaler(score: Union[float, int]):

    if score > 4 or score < 0:
        raise ValueError(f"Not supported domain of {score=}, x needs to be in [0; 4]")
    if 0 <= score <= 2:
        return score
    if 2 <= score <= 3:
        return score * 4 - 6
    if score >= 3:
        return score * 18 - 48


def rescale_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.applymap(step_score_rescaler)


class ScoringFunction(Protocol):
    def score(self, sub_char: str, practice: str) -> float:
        ...  # pragma: no cover

    @property
    def sub_characteristics(self) -> Set[str]:
        ...  # pragma: no cover

    @property
    def practices(self) -> Set[str]:
        ...  # pragma: no cover

    @property
    def covered_score(self) -> float:
        ...  # pragma: no cover


class AggregatedScores:
    def __init__(
        self,
        raw_scores: pd.DataFrame,
        aggregation_function: Callable[[np.ndarray], float],
        sub_characteristics: Optional[set] = None,
        weights: Optional[Dict[str, float]] = None,
        assessment_practices: AssessmentPractices = AssessmentPractices.all,
        already_applied_practices: Optional[Set[str]] = None,
        coverage_score: Optional[int] = 24,
    ) -> None:
        """
        raw_scores: pandas dataframe with raw non-aggregated scores
        aggregation_function: takes values np.nanmean, np.nanmedian. This is to ignore nan's in the computations
        sub_characteristics: set of quality sub-characteristics to optimize for. If omitted, all of them are included.
        weights: dictionary with importance weights per sub-characteristic. The keys of the dictionary should be
                the same as the sub_characteristics set.
        assessment_practices: if True only the assessment related practices are considered,
                    if False only the non-assessment practices are considered,
                    if None all the practices are considered
        already_applied_practices: List with practices already applied to be excluded.
        coverage_score: The score after which a sub-characteristic is covered.
        """

        self.assessment_practices = assessment_practices
        self.already_applied_practices = already_applied_practices
        self.coverage_score = coverage_score
        self.aggregated = rescale_scores(raw_scores).apply(aggregation_function, axis=1, raw=True)
        aggregated_max = self.aggregated.max()

        if sub_characteristics is not None:
            self._sub_characteristics = sub_characteristics
        else:
            self._sub_characteristics = set(self.aggregated.index.get_level_values(0))

        if weights is not None:
            if set(weights.keys()) != self._sub_characteristics:
                raise IndexError("The sub characteristics provided are not the same as in the weights.")
            if 0.0 in weights.values():
                raise ValueError("A weight should not be 0! Please exclude the sub-characteristic!")
            self._weights = weights
        else:
            self._weights = {sub_char: 1.0 for sub_char in self._sub_characteristics}

        if aggregated_max > self.coverage_score:
            raise ValueError(f"Invalid max value {aggregated_max}, expected no more than {self.coverage_score}")

    def score(self, sub_char: str, practice: str) -> float:
        return self.aggregated.loc[(sub_char, practice)] * self._weights[sub_char]

    @property
    def sub_characteristics(self) -> Set[str]:
        return self._sub_characteristics

    @property
    def weights(self) -> Dict[str, float]:
        return self._weights

    @property
    def practices(self) -> Set[str]:

        if self.already_applied_practices is not None:
            already_applied_practices = self.already_applied_practices
        else:
            already_applied_practices = set()

        if self.assessment_practices == AssessmentPractices.all:
            practices = set(self.aggregated.index.get_level_values(1))
        elif self.assessment_practices == AssessmentPractices.assessment:
            practices = set(QUALITY_ASSESSMENT_PRACTICES).intersection(self.aggregated.index.get_level_values(1))
        else:
            practices = set(self.aggregated.index.get_level_values(1)) - set(QUALITY_ASSESSMENT_PRACTICES)

        return practices - already_applied_practices

    @property
    def covered_score(self) -> float:
        return self.coverage_score


class AggregatedScoresSelectedPractices:
    def __init__(
        self,
        raw_scores: pd.DataFrame,
        aggregation_function: Callable[[np.ndarray], float],
        practices_to_rank: Set[str],
        sub_characteristics: Set[str],
        weights: Optional[Dict[str, float]] = None,
        coverage_score: Optional[int] = 24,
    ) -> None:
        """
        raw_scores: pandas dataframe with raw non-aggregated scores
        aggregation_function: takes values np.nanmean, np.nanmedian. This is to ignore nan's in the computations
        sub_characteristics: set of quality sub-characteristics to optimize for. If omitted, all of them are included.
        weights: dictionary with importance weights per sub-characteristic. The keys of the dictionary should be
                the same as the sub_characteristics set.
        practices_to_rank: The set of practices to rank. The practices need to exist in the scores.
        coverage_score: The score after which a sub-characteristic is covered.
        """

        self.practices_to_rank = practices_to_rank
        self.coverage_score = coverage_score
        self.aggregated = rescale_scores(raw_scores).apply(aggregation_function, axis=1, raw=True)
        aggregated_max = self.aggregated.max()

        self._sub_characteristics = sub_characteristics

        if weights is not None:
            if set(weights.keys()) != self._sub_characteristics:
                raise IndexError("The sub characteristics provided are not the same as in the weights.")
            if 0.0 in weights.values():
                raise ValueError("A weight should not be 0! Please exclude the sub-characteristic!")
            self._weights = weights
        else:
            self._weights = {sub_char: 1.0 for sub_char in self._sub_characteristics}

        if aggregated_max > self.coverage_score:
            raise ValueError(f"Invalid max value {aggregated_max}, expected no more than {self.coverage_score}")

    def score(self, sub_char: str, practice: str) -> float:
        return self.aggregated.loc[(sub_char, practice)] * self._weights[sub_char]

    @property
    def sub_characteristics(self) -> Set[str]:
        return self._sub_characteristics

    @property
    def weights(self) -> Dict[str, float]:
        return self._weights

    @property
    def practices(self) -> Set[str]:
        return self.practices_to_rank

    @property
    def covered_score(self) -> float:
        return self.coverage_score

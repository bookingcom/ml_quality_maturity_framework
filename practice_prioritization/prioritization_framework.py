import dataclasses
import pathlib
from operator import itemgetter
from typing import Dict, List, Mapping, Optional, Protocol, Set, Tuple, Union

import numpy as np
import pandas as pd

from practice_prioritization.constants import AssessmentPractices
from practice_prioritization.score_collection import merge_scores
from practice_prioritization.score_function.aggregation import (
    AggregatedScores,
    AggregatedScoresSelectedPractices,
    ScoringFunction,
)


@dataclasses.dataclass(frozen=True)
class ScoredPractice:
    practice: str
    points_covered: float

    @classmethod
    def from_map(cls, map: Mapping[str, float]) -> Set["ScoredPractice"]:
        result = set()
        for practice, points_covered in map.items():
            result.add(ScoredPractice(practice, points_covered))
        return result

    def debug_str(self) -> str:
        return f"{self.practice}\t{self.points_covered:.3f} points"

    def simple_str(self) -> str:
        return self.practice

    def __str__(self) -> str:
        return self.simple_str()


class PrioritisationFramework(Protocol):
    def prioritise(self, num_practices: int) -> Set[ScoredPractice]:
        ...


class PrioritisationFrameworkGreedy:
    def __init__(self, scoring_function: ScoringFunction):
        self.scoring_function = scoring_function

    def prioritise(self, num_practices: int) -> Set[ScoredPractice]:
        """
        :param num_practices:
            Budget of practices we can implement
        :return:
            A set of practices. Caution: the resulting set is not necessarily a feasible solution covering
            all sub-characteristics. If a feasible solution does not exist, the next best is returned.
        """
        covered: Set[str] = set()
        selected_practices: Set[str] = set()
        sub_chars = self.scoring_function.sub_characteristics
        left_to_cover = {sub_char: self.scoring_function.covered_score for sub_char in sub_chars}
        points_covered = {}
        num_rounds = 0

        practices_available = self.scoring_function.practices.copy()

        if num_practices > len(self.scoring_function.practices):
            raise IndexError(
                f"The number of practices requested ({num_practices}) is larger than the total number of practices ({len(self.scoring_function.practices)})."
            )

        while covered != sub_chars and num_rounds < num_practices:
            num_rounds += 1

            scored_practices = []

            for practice in practices_available:
                scaled_score = 0
                for sub_char in sub_chars.difference(covered):
                    scaled_score += self.scoring_function.score(sub_char=sub_char, practice=practice)
                scored_practices.append((practice, scaled_score)),

            # greedy selection by the most r points a practice achieves on all sub characteristics
            selected_practice, points = sorted(scored_practices, key=lambda x: -x[1])[0]
            points_covered[selected_practice] = points
            # Apply the selected practice to each sub-characteristic
            for sub_char in list(left_to_cover.keys()):
                left_to_cover[sub_char] -= self.scoring_function.score(sub_char, selected_practice)

                # Once a sub characteristic is covered, add it to the set of covered once,
                # we do not need to improve it further
                if left_to_cover[sub_char] <= 0:
                    covered.add(sub_char)
                    del left_to_cover[sub_char]

            practices_available.remove(selected_practice)
            selected_practices.add(selected_practice)

        return ScoredPractice.from_map(points_covered)


def prioritize_practices(
    sub_characteristics: Optional[Set[str]],
    weights: Optional[Dict[str, float]],
    num_practices: int,
    debug: bool,
    assessment_practices: AssessmentPractices = AssessmentPractices.all,
    already_applied_practices: Optional[List[str]] = None,
    coverage_score: Optional[int] = 24,
) -> Set[ScoredPractice]:
    scores_path = str(pathlib.Path(__file__).parent / "score_collection" / "merged" / "merged_scores.csv")

    scoring_function = AggregatedScores(
        raw_scores=merge_scores.read_merged_scores(path=scores_path, remove_deprecated_subchars=True),
        aggregation_function=np.nanmean,
        sub_characteristics=sub_characteristics,
        weights=weights,
        assessment_practices=assessment_practices,
        already_applied_practices=already_applied_practices,
        coverage_score=coverage_score,
    )

    prioritization_instance = PrioritisationFrameworkGreedy(scoring_function=scoring_function)

    selected_practices = prioritization_instance.prioritise(num_practices=num_practices)

    if debug:
        debug_metadata = DebugMetadata.build_debug_metadata(scoring_function, selected_practices)
        print(debug_metadata)

    return selected_practices


def prioritize_selected_practices(
    sub_characteristics: Optional[Set[str]],
    weights: Optional[Dict[str, float]],
    practices_to_rank: Set[str],
    coverage_score: Optional[int] = 24,
) -> List[Tuple[str, float]]:
    scores_path = str(pathlib.Path(__file__).parent / "score_collection" / "merged" / "merged_scores.csv")

    scoring_function = AggregatedScoresSelectedPractices(
        raw_scores=merge_scores.read_merged_scores(path=scores_path, remove_deprecated_subchars=True),
        aggregation_function=np.nanmean,
        sub_characteristics=sub_characteristics,
        weights=weights,
        practices_to_rank=practices_to_rank,
        coverage_score=coverage_score,
    )

    prioritization_instance = PrioritisationFrameworkGreedy(scoring_function=scoring_function)

    selected_practices = prioritization_instance.prioritise(num_practices=len(practices_to_rank))

    practice_list = [(elem.practice, round(elem.points_covered, 2)) for elem in selected_practices]

    # if a practice is missing, add it later with score 0. We need to rank all the practices
    practices_included = [elem[0] for elem in practice_list]
    missing_practices = [practice for practice in practices_to_rank if practice not in practices_included]
    if len(missing_practices) > 0:
        for missing_practice in missing_practices:
            practice_list.append((missing_practice, 0.0))
    return sorted(practice_list, key=itemgetter(1), reverse=True)


@dataclasses.dataclass
class DebugMetadata:
    results_frame: pd.DataFrame
    score_matrix: pd.DataFrame

    @classmethod
    def build_debug_metadata(
        cls,
        scoring_function: ScoringFunction,
        selected_practices: Set[ScoredPractice],
    ) -> "DebugMetadata":
        results = []
        selected_practice_names = [sp.practice for sp in selected_practices]

        sub_characteristics = scoring_function.sub_characteristics

        for sub_characteristic in sub_characteristics:
            for practice in selected_practice_names:
                score = scoring_function.score(sub_characteristic, practice)
                results.append(
                    {
                        "score": score,
                        "practice": practice,
                        "sub_characteristic": sub_characteristic,
                    }
                )
        results_frame = pd.DataFrame(results).sort_values(by=["sub_characteristic", "practice"])
        score_matrix = results_frame.pivot_table(
            index="practice",
            columns="sub_characteristic",
            values="score",
            aggfunc="sum",
        )

        return DebugMetadata(results_frame=results_frame, score_matrix=score_matrix)

    def __str__(self) -> str:
        with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
            return f"results, long format:\n{self.results_frame}\nscore matrix:\n{self.score_matrix}"

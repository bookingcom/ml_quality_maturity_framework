from typing import Optional, Set

import click

from practice_prioritization.constants import AssessmentPractices
from practice_prioritization.prioritization_framework import ScoredPractice, prioritize_practices

"""
Run as:

practice_prioritization --sub_characteristics accuracy,maintainability --num_practices 3

"""


@click.command()
@click.option(
    "--subchars",
    type=str,
    multiple=False,
    help="Sub-characteristics to improve",
    default=None,
)
@click.option(
    "--weights",
    type=str,
    multiple=False,
    help="Importance weights per sub-characteristic",
    default=None,
)
@click.option("--num_practices", type=int, help="Number of practices")
@click.option(
    "--assessment_practices",
    type=click.Choice(AssessmentPractices),
    help="Set of practices to consider. One of: assessment, non-assessment, all",
    default=None,
)
@click.option(
    "--already_applied_practices",
    type=str,
    multiple=False,
    help="Practices that are already applied, to be excluded from the recommendations.",
    default=None,
)
@click.option("--coverage_score", type=int, help="The score after which a sub-characteristic is covered", default=24)
@click.option("--debug", type=bool, is_flag=True, help="Debug mode", default=False)
def cli(
    subchars: Optional[str],
    num_practices: int,
    weights: Optional[str],
    assessment_practices: Optional[AssessmentPractices],
    already_applied_practices: Optional[str],
    coverage_score: Optional[int],
    debug: bool,
) -> None:
    if subchars is not None:
        # Note: We do not make a set here, because it changes the order of the subchars in the following weights dictionary
        subchars = subchars.split(",")
        subchars_set = set(subchars)
    else:
        subchars_set = None

    if weights is not None:
        weights = weights.split(",")
        weights = [float(weight) for weight in weights]
        weights = {sub_char: weight for sub_char, weight in zip(subchars, weights)}
    if already_applied_practices is not None:
        already_applied_practices = set(already_applied_practices.split(","))

    if assessment_practices is None:
        assessment_practices = AssessmentPractices.all

    selected_practices = prioritize_practices(
        sub_characteristics=subchars_set,
        weights=weights,
        num_practices=num_practices,
        debug=debug,
        assessment_practices=assessment_practices,
        already_applied_practices=already_applied_practices,
        coverage_score=coverage_score,
    )

    _print_selected(selected_practices, debug)


def _print_selected(scored_practices: Set[ScoredPractice], debug: bool) -> None:
    n = 1
    for practice in sorted(scored_practices, key=lambda sp: -sp.points_covered):
        echo_str = practice.debug_str() if debug else practice.simple_str()
        click.echo(f"{n} {echo_str}")
        n = n + 1

import argparse
import itertools
from typing import List, Tuple, Union

import pandas as pd

from practice_prioritization import constants


def store_csv_with_pairs(output_path: str) -> None:

    pairs = create_pairs_given_practices_and_attributes(
        practices=constants.PRACTICES, quality_attributes=constants.QUALITY_ATTRIBUTES
    )
    table = convert_tuples_to_table(pairs=pairs)
    store_table_as_csv(table=table, path=output_path)


def create_pairs_given_practices_and_attributes(practices: List[str], quality_attributes: List[str]) -> List[Tuple]:

    pairs = []

    for pair in itertools.product(practices, quality_attributes):
        pairs.append(pair)

    return pairs


def convert_tuples_to_table(pairs: List[Tuple]) -> pd.DataFrame:
    table_columns = [
        constants.PRACTICES_COLUMN_NAME.lower(),
        constants.QUALITY_ATTRIBUTE_COLUMN_NAME,
        constants.WEIGHTS_COLUMN_NAME,
    ]

    table = pd.DataFrame(columns=table_columns)

    table[table_columns[0]] = list(zip(*pairs))[0]
    table[table_columns[1]] = list(zip(*pairs))[1]
    table[constants.WEIGHTS_COLUMN_NAME] = ""

    return table


def store_table_as_csv(table: pd.DataFrame, path=Union[str, None]) -> None:
    table.to_csv(path_or_buf=path, sep=",", header=True, encoding="utf-8", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a table with pairs of practices and quality attributes, in "
        "order for ML experts to add effect sizes on each pair."
    )
    parser.add_argument("--output_path", dest="output_path", type=str, help="Output path of the csv.")

    args = parser.parse_args()
    output_path = args.output_path
    store_csv_with_pairs(output_path=output_path)

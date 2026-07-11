import argparse
import json

from datasets import Dataset


def read_jsonl(input_path: str) -> list[dict]:
    """
    Read JSONL records from disk into a list of dictionaries.
    """

    records = []

    with open(input_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            records.append(json.loads(line))

    return records


def convert_jsonl_to_parquet(input_path: str, output_path: str) -> None:
    """
    Convert a JSONL dataset into Parquet using Hugging Face Datasets.
    """

    records = read_jsonl(input_path)
    dataset = Dataset.from_list(records)
    dataset.to_parquet(output_path)


def main() -> None:
    """
    Parse CLI arguments and convert one JSONL file into Parquet.
    """

    parser = argparse.ArgumentParser(
        description="Convert a JSONL dataset file into Parquet."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    convert_jsonl_to_parquet(args.input, args.output)
    print(f"Wrote Parquet file to {args.output}")


if __name__ == "__main__":
    main()

from datasets import Dataset, DatasetDict, ClassLabel
import numpy as np
import collections
from typing import Optional


def test_train_split(
    dataset: Dataset,
    test_size: float,
    seed: int = 42,
    stratify_by_column: Optional[str] = None,
    validation_size: Optional[float] = None,
):
    assert isinstance(dataset, Dataset), (
        "dataset must be a Hugging Face Dataset instance"
    )
    assert isinstance(test_size, float) and 0 < test_size < 1, (
        "test_size must be a float between 0 and 1"
    )
    assert isinstance(seed, int), "seed must be an integer"
    assert stratify_by_column is None or isinstance(stratify_by_column, str), (
        "stratify_by_column must be None or a string"
    )
    assert validation_size is None or (
        isinstance(validation_size, float) and 0 < validation_size < 1
    ), "validation_size must be None or a float between 0 and 1"
    if validation_size is not None:
        assert test_size + validation_size < 1, (
            "test_size + validation_size must be less than 1"
        )
    # Ensure the stratification column is a ClassLabel
    if stratify_by_column is not None:
        if stratify_by_column not in dataset.column_names:
            raise ValueError(f"Column '{stratify_by_column}' not found in dataset.")

        if not isinstance(dataset.features[stratify_by_column], ClassLabel):
            print(
                f"Note: Converting column '{stratify_by_column}' to ClassLabel to allow stratification."
            )
            dataset = dataset.class_encode_column(stratify_by_column)
    if stratify_by_column:
        assert stratify_by_column in dataset.column_names, (
            "stratify_by_column must exist in dataset"
        )
    if stratify_by_column:
        assert isinstance(dataset.features[stratify_by_column], ClassLabel), (
            "stratify_by_column must be ClassLabel after processing"
        )

    iterations = 0
    assert iterations >= 0, "iterations must be non-negative"
    while True:
        current_seed = seed + iterations
        assert isinstance(current_seed, int), "current_seed must be an integer"

        # If no validation size is provided (or is 0), perform standard split
        if not validation_size:
            train_test_split = dataset.train_test_split(
                test_size=test_size,
                seed=current_seed,
                stratify_by_column=stratify_by_column,
            )
            assert isinstance(train_test_split, DatasetDict), (
                "train_test_split must be a DatasetDict"
            )
            assert "train" in train_test_split and "test" in train_test_split, (
                "train_test_split must have 'train' and 'test' keys"
            )
            assert len(train_test_split["train"]) + len(
                train_test_split["test"]
            ) == len(dataset), (
                "train and test splits must sum to original dataset length"
            )
        else:
            # 1. Create stratified split with size of test_size + validation_size
            combined_split_size = test_size + validation_size
            assert 0 < combined_split_size < 1, (
                "combined_split_size must be between 0 and 1"
            )

            temp_split = dataset.train_test_split(
                test_size=combined_split_size,
                seed=current_seed,
                stratify_by_column=stratify_by_column,
            )
            assert len(temp_split["train"]) + len(temp_split["test"]) == len(dataset), (
                "temp_split must preserve dataset length"
            )

            # 2. Calculate the proportion of validation data relative to the combined (test + val) set
            val_share = validation_size / combined_split_size
            assert 0 < val_share < 1, "val_share must be between 0 and 1"

            # 3. Split the temporary test set into final Test and Validation sets
            # The 'train' split here becomes our final 'test' set
            # The 'test' split here becomes our 'validation' set
            test_val_split = temp_split["test"].train_test_split(
                test_size=val_share,
                seed=current_seed,
                stratify_by_column=stratify_by_column,
            )
            assert len(test_val_split["train"]) + len(test_val_split["test"]) == len(
                temp_split["test"]
            ), "test_val_split must preserve temp test set length"

            # 4. Construct the final DatasetDict
            train_test_split = DatasetDict(
                {
                    "train": temp_split["train"],
                    "test": test_val_split["train"],
                    "validation": test_val_split["test"],
                }
            )
            assert "validation" in train_test_split, (
                "train_test_split must include 'validation' when validation_size is provided"
            )
            assert len(train_test_split["train"]) + len(train_test_split["test"]) + len(
                train_test_split["validation"]
            ) == len(dataset), "all splits must sum to original dataset length"

        # --- Score Normalization & Printing ---
        print(f"\n--- Iteration {iterations} (Seed: {current_seed}) ---")

        def print_label_counts(split_name, ds_split):
            # Use the stratified column if provided, otherwise default to "labels" if it exists
            target_col = stratify_by_column if stratify_by_column else "labels"

            if target_col in ds_split.column_names:
                labels = np.array(ds_split[target_col])
                assert len(labels) == len(ds_split), (
                    "labels array must match dataset split length"
                )
                label_counts = collections.Counter(labels)
                print(f"Count of each label in the {split_name} set:")
                for label, count in sorted(label_counts.items()):
                    print(f"  Label {label}: {count} occurrences")
            else:
                print(f"  (Column '{target_col}' not found, skipping count)")

        print_label_counts("training", train_test_split["train"])
        print_label_counts("test", train_test_split["test"])

        if "validation" in train_test_split:
            print_label_counts("validation", train_test_split["validation"])

        result = input("Continue? (Y/n/x) ")

        if result.lower() == "y":
            break
        elif result.lower() == "x":
            raise ValueError("Splitting cancelled")
        iterations += 1

    assert isinstance(train_test_split, DatasetDict), (
        "return value must be a DatasetDict"
    )
    assert "train" in train_test_split and "test" in train_test_split, (
        "return value must have 'train' and 'test' keys"
    )
    if validation_size is not None:
        assert "validation" in train_test_split, (
            "return value must have 'validation' key when validation_size provided"
        )
    return train_test_split

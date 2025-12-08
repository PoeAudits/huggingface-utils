from typing import overload, Literal
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


@overload
def _load_dataset(
    path: str, *, split: str, streaming: Literal[False], token: str
) -> Dataset: ...
@overload
def _load_dataset(
    path: str, *, split: None, streaming: Literal[False], token: str
) -> DatasetDict: ...
@overload
def _load_dataset(
    path: str, *, split: str, streaming: Literal[True], token: str
) -> IterableDataset: ...
@overload
def _load_dataset(
    path: str, *, split: None, streaming: Literal[True], token: str
) -> IterableDatasetDict: ...


def _load_dataset(
    path: str,
    *,
    split: str | None,
    token: str,
    streaming: bool = False,
):
    return load_dataset(path=path, split=split, streaming=streaming, token=token)


def load_dataset_split(path: str, *, split: str = "train", token: str) -> Dataset:
    return _load_dataset(path=path, split=split, streaming=False, token=token)


def load_dataset_all(path: str, *, token: str) -> DatasetDict:
    return _load_dataset(path=path, split=None, streaming=False, token=token)


def load_iterable_split(
    path: str, *, split: str = "train", token: str
) -> IterableDataset:
    return _load_dataset(path=path, split=split, streaming=True, token=token)


def load_iterable(path: str, *, token: str) -> IterableDatasetDict:
    return _load_dataset(path=path, split=None, streaming=True, token=token)

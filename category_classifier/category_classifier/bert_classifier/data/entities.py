import typing as T
from abc import ABC, abstractclassmethod, abstractmethod, abstractstaticmethod

from transformers import PreTrainedTokenizerFast


class CategoryClassificationEntity(ABC):
    """
    Abstraction of a Category Classification row.
    """

    @abstractmethod
    def is_correct(self) -> bool:
        ...

    @abstractmethod
    def get_transformed_entity(
        self, tokenizer: PreTrainedTokenizerFast, category_mapping: T.Dict[str, int]
    ) -> T.Dict[str, T.Union[T.List[int], int]]:
        ...

    @abstractclassmethod
    def from_csv_entity(cls, csv_entities: T.Iterable[str]) -> "CategoryClassificationEntity":
        ...

    @abstractstaticmethod
    def _entity_is_in_training_format(csv_entity: T.Iterable["CategoryClassificationEntity"]) -> bool:
        ...

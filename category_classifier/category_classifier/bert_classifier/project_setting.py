import typing as T

from attr import attrib, attrs

from category_classifier.bert_classifier.config.constants import AnchorBertDatasetConstants, AnchorBertTokenizerConstants
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerFast

from category_classifier.bert_classifier.data.entities import CategoryClassificationEntity


class ProjectSettings:
    entity: CategoryClassificationEntity
    dataset: Dataset
    category_column_name: str
    expected_columns: T.List[str]
    features_field_name: str
    label_field_name: str
    index_field_name: str
    predictions_output_file_name: str
    predictions_file_header: str


class AnchorEntityTransformation:
    @staticmethod
    def _get_transformed_entity(
        text: str,
        category_id: str,
        true_category_id: str,
        offer_id: str,
        tokenizer: PreTrainedTokenizerFast,
        category_mapping: T.Dict[str, int],
    ) -> T.Dict[str, T.Union[T.List[int], int]]:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            padding=AnchorBertTokenizerConstants.PADDING,
            max_length=AnchorBertTokenizerConstants.MAX_LENGTH,
        )
        label = category_mapping.get(category_id, -1)
        true_label = category_mapping.get(true_category_id, -1)
        return {
            AnchorBertDatasetConstants.OFFER_ID_COL_NAME: offer_id,
            AnchorBertDatasetConstants.FEATURES_FIELD_NAME: token_ids,
            AnchorBertDatasetConstants.LABEL_FIELD_NAME: label,
            AnchorBertDatasetConstants.TRUE_LABEL_FIELD_NAME: true_label,
        }


@attrs
class AnchorTrainingEntity(CategoryClassificationEntity, AnchorEntityTransformation):
    text = attrib(converter=str)
    category_id = attrib(converter=str)
    true_category_id = attrib(converter=str)
    offer_id = attrib(converter=str, default="unknown")

    def is_correct(self) -> bool:
        return len(self.text) > 0

    def get_transformed_entity(
        self, tokenizer: PreTrainedTokenizerFast, category_mapping: T.Dict[str, int]
    ) -> T.Dict[str, T.Union[T.List[int], int]]:
        return self._get_transformed_entity(
            self.text, self.category_id, self.true_category_id, self.offer_id, tokenizer, category_mapping
        )

    @classmethod
    def from_csv_entity(cls, csv_entities: T.Iterable[str]) -> CategoryClassificationEntity:
        if AnchorBertDatasetConstants.TRUE_CATEGORY_ID_COL_NAME in csv_entities.keys():
            return cls(
                text=csv_entities[AnchorBertDatasetConstants.TEXT_COL_NAME],
                category_id=csv_entities[AnchorBertDatasetConstants.CATEGORY_ID_COL_NAME],
                true_category_id=csv_entities[AnchorBertDatasetConstants.TRUE_CATEGORY_ID_COL_NAME],
            )
        else:
            return cls(
                text=csv_entities[AnchorBertDatasetConstants.TEXT_COL_NAME],
                category_id=csv_entities[AnchorBertDatasetConstants.CATEGORY_ID_COL_NAME],
                true_category_id=csv_entities[AnchorBertDatasetConstants.CATEGORY_ID_COL_NAME],
            )

    @staticmethod
    def _entity_is_in_training_format(csv_entity: T.Iterable["CategoryClassificationEntity"]) -> bool:
        return True

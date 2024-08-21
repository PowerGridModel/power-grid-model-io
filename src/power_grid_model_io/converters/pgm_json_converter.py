# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Power Grid Model 'Converter': Load and store power grid model data in the native PGM JSON format.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from power_grid_model import initialize_array
from power_grid_model._utils import is_nan
from power_grid_model.data_types import ComponentList, Dataset, SingleDataset, SinglePythonDataset
from power_grid_model.utils import json_deserialize, json_serialize

from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.data_stores.json_file_store import JsonFileStore
from power_grid_model_io.data_types import ExtraInfo, StructuredData
from power_grid_model_io.utils.dict import merge_dicts


class PgmJsonConverter(BaseConverter[StructuredData]):
    """
    A 'converter' class to load and store power grid model data in the native PGM JSON format. The methods are simmilar
    to the utils in power_grid_model, with the addition of storing and loading 'extra info'. Extra info is the set of
    attributes that don't match the power grid model's internal structure, but are important to keep close to the data.
    The most common example is the original object ID, if the original IDs are not numeric, or not unique over all
    components.

    Args:

    Returns:

    """

    def __init__(
        self,
        source_file: Optional[Union[Path, str]] = None,
        destination_file: Optional[Union[Path, str]] = None,
        log_level: int = logging.INFO,
    ):
        source = JsonFileStore(file_path=Path(source_file)) if source_file else None
        destination = JsonFileStore(file_path=Path(destination_file)) if destination_file else None
        super().__init__(source=source, destination=destination, log_level=log_level)

    def _parse_data(self, data: StructuredData, data_type: str, extra_info: Optional[ExtraInfo]) -> Dataset:
        """This function expects Structured data, which can either be a dictionary (single dataset) or a list of
        dictionaries (batch dataset). The structured dataset consists of components + attributes that exist within
        power-grid-model, but can also contain other data. If this data should be saved for later usage an extra_info
        dictionary can be provided when calling this function

        Args:
          data: Structured data, which can either be a dictionary or a list of dictionaries
          data_type: the data type of the dataset, i.e. "input", "update", "sym_output" or "asym_output"
          extra_info: an optional dictionary where extra component info (that can't be specified in
        power-grid-model data) can be specified
          data: StructuredData:
          data_type: str:
          extra_info: Optional[ExtraInfo]:

        Returns:
          a dictionary containing the components as keys and their corresponding numpy arrays as values: a
          power-grid-model "input" or "update" dataset

        """
        self._log.debug(f"Loading PGM {data_type} data")

        result = json_deserialize(
            json.dumps(
                {
                    "attributes": {},
                    "data": data,
                    "is_batch": isinstance(data, list),
                    "type": data_type,
                    "version": "1.0",
                }
            )
        )

        if extra_info is not None:
            self._extract_extra_info(original_data=data, deserialized_data=result, extra_info=extra_info)

        return result

    def _parse_dataset(
        self, data: SinglePythonDataset, data_type: str, extra_info: Optional[ExtraInfo]
    ) -> SingleDataset:
        """This function parses a single Python dataset and returns a power-grid-model input or update dictionary

        Args:
          data: a single Python dataset
          data_type: the data type of the dataset, i.e. "input" or "update"
          extra_info: an optional dictionary where extra component info (that can't be specified in
        power-grid-model data) can be specified
          data: SinglePythonDataset:
          data_type: str:
          extra_info: Optional[ExtraInfo]:  (Default value = None)

        Returns:
          a dictionary containing the components as keys and their corresponding numpy arrays as values: a
          power-grid-model "input" or "update" dataset

        """
        return {
            component: self._parse_component(
                objects=objects, component=component, data_type=data_type, extra_info=extra_info
            )
            for component, objects in data.items()
        }

    @staticmethod
    def _parse_component(
        objects: ComponentList, component: str, data_type: str, extra_info: Optional[ExtraInfo]
    ) -> np.ndarray:
        """This function generates a structured numpy array (power-grid-model native) from a structured dataset

        Args:
          objects: a list with dictionaries, where each dictionary contains all attributes of a component
          component: the type of component, eg. node, line, etc. Note: it should be a valid power-grid-model
        component
          data_type: a string specifying the data type: input/update
          extra_info: an optional dictionary where extra component info (that can't be specified in
        power-grid-model data) can be specified
          objects: ComponentList:
          component: str:
          data_type: str:
          extra_info: Optional[ExtraInfo]:  (Default value = None)

        Returns:
          a numpy structured array for a power-grid-model component

        """
        # We'll initialize an 1d-array with NaN values for all the objects of this component type
        array = initialize_array(data_type, component, len(objects))

        for i, obj in enumerate(objects):
            # As each object is a separate dictionary, and the attributes may differ per object, we need to check
            # all attributes. Non-existing attributes are stored as extra_info, or ignored.
            for attribute, value in obj.items():
                if attribute in array.dtype.names:
                    # Assign the value or raise an error if the value cannot be stored in the specific numpy array
                    # data format for this attribute.
                    try:
                        array[i][attribute] = value
                    except ValueError as ex:
                        raise ValueError(f"Invalid '{attribute}' value for {component} {data_type} data: {ex}") from ex

                # If an attribute doesn't exist, it is added to the extra_info lookup table
                elif extra_info is not None:
                    obj_id = obj["id"]
                    if not isinstance(obj_id, int):
                        raise ValueError(f"Invalid 'id' value for {component} {data_type} data")
                    if obj_id not in extra_info:
                        extra_info[obj_id] = {}
                    extra_info[obj_id][attribute] = value
        return array

    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfo]) -> StructuredData:
        """This function converts a power-grid-model dataset to a structured dataset. First, the function checks if the
        dataset is a single dataset or batch dataset. If it is a batch, the batch data is converted to a list of
        batches, then each batch is converted individually.

        Args:
          data: a power-grid-model dataset
          extra_info: an optional dictionary with extra information. If supplied, the extra info is added to the
        structured dataset. The keys in this dictionary should match with id's of components in the power-grid-model
        dataset. Note, extra info can only be supplied for single datasets
          data: Dataset:
          extra_info: Optional[ExtraInfo]:  (Default value = None)

        Returns:
          the function returns a structured dataset

        """
        result = json.loads(json_serialize(data))["data"]

        if extra_info is not None:
            if self._is_batch(data=data):
                self._log.warning("Extra info is not supported for batch data export")
            else:
                for component_data in result.values():
                    for component in component_data:
                        component.update(extra_info.get(component["id"], {}))

        return result

    @staticmethod
    def _is_batch(data: Dataset) -> bool:
        """
        This function checks if a dataset is single or batch. The function loops through all components in the dataset
        and checks for each component if the corresponding dataset is single or batch. All components should have the
        same array type (single or batch). If this is not the case a ValueError will be raised. An array is a batch
        dataset if it is 2D array or has an indptr/data structure

        Args:
          data: a power-grid-model dataset which is either single or a batch
          data: Dataset:

        Returns:
          returns True if the dataset is a batch dataset, False if it is a single dataset

        """
        is_batch: Optional[bool] = None
        for component, array in data.items():
            is_dense_batch = isinstance(array, np.ndarray) and array.ndim == 2
            is_sparse_batch = isinstance(array, dict) and "indptr" in array and "data" in array
            if is_batch is not None and is_batch != (is_dense_batch or is_sparse_batch):
                raise ValueError(
                    f"Mixed {'' if is_batch else 'non-'}batch data "
                    f"with {'non-' if is_batch else ''}batch data ({component})."
                )
            is_batch = is_dense_batch or is_sparse_batch
        return bool(is_batch)

    @staticmethod
    def _serialize_dataset(data: SingleDataset, extra_info: Optional[ExtraInfo] = None) -> SinglePythonDataset:
        """This function converts a single power-grid-model dataset to a structured dataset

        Args:
          data: a power-grid-model (single) dataset
          extra_info: an optional dictionary with extra information. If supplied, the extra info is added to the
        structured dataset. The keys in this dictionary should match with id's of components in the power-grid-model
        dataset
          data: SingleDataset:
          extra_info: Optional[ExtraInfo]:  (Default value = None)

        Returns:
          the function returns a structured dataset

        """

        # This should be a single data set
        for component, array in data.items():
            if not isinstance(array, np.ndarray) or array.ndim != 1:
                raise ValueError("Invalid data format")

        if extra_info is None:
            extra_info = {}
        # Convert each numpy array to a list of objects, which contains only the non-NaN attributes:
        # For example: {"node": [{"id": 0, ...}, {"id": 1, ...}], "line": [{"id": 2, ...}]}
        return {
            component: [
                merge_dicts(
                    {
                        attribute: obj[attribute].tolist()
                        for attribute in objects.dtype.names
                        if not is_nan(obj[attribute])
                    },
                    extra_info.get(obj["id"], {}),
                )
                for obj in objects
            ]
            for component, objects in data.items()
        }

    def _extract_extra_info(
        self, original_data: StructuredData, deserialized_data: SingleDataset, extra_info: ExtraInfo
    ) -> None:
        if not isinstance(original_data, dict):
            warnings.warn("Extracting extra info is not supported for batch data.")
            return

        reserialized_data = self._serialize_data(data=deserialized_data, extra_info=extra_info)
        if len(original_data) != len(reserialized_data) or not isinstance(reserialized_data, dict):
            warnings.warn("The extra info cannot be determined.")
            return

        for component, component_data in original_data.items():
            for entry in component_data:
                self._extract_extra_component_info(component, entry, reserialized_data, extra_info)

    def _extract_extra_component_info(
        self, component: str, attributes: Dict[str, Any], reserialized_data: SinglePythonDataset, extra_info: ExtraInfo
    ):
        entry_id = attributes["id"]
        reserialized_entry = self._get_first_by(reserialized_data[component], "id", entry_id)
        if reserialized_entry is None:
            warnings.warn(f"The extra info cannot be determined for component '{component}' with ID {entry_id}")
            return

        for attribute, value in attributes.items():
            if attribute not in reserialized_entry:
                if entry_id not in extra_info:
                    extra_info[entry_id] = {}

                extra_info[entry_id][attribute] = value

    @staticmethod
    def _get_first_by(data: List[Dict[str, Any]], field: str, value: Any) -> Optional[Dict[str, Any]]:
        for entry in data:
            if entry[field] == value:
                return entry

        return None

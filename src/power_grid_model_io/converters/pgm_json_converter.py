# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Power Grid Model 'Converter': Load and store power grid model data in the native PGM JSON format.
"""

from pathlib import Path
from typing import Optional, Union, cast

import numpy as np
from power_grid_model.data_types import BatchDataset, ComponentList, Dataset, SingleDataset, SinglePythonDataset
from power_grid_model.utils import (
    convert_batch_dataset_to_batch_list,
    convert_list_to_batch_data,
    initialize_array,
    is_nan,
)

from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.data_stores.json_file_store import JsonFileStore
from power_grid_model_io.data_types import ExtraInfoLookup, StructuredData


class PgmJsonConverter(BaseConverter[StructuredData]):
    """
    A 'converter' class to load and store power grid model data in the native PGM JSON format. The methods are simmilar
    to the utils in power_grid_model, with the addition of storing and loading 'extra info'. Extra info is the set of
    attributes that don't match the power grid model's internal structure, but are important to keep close to the data.
    The most common example is the original object ID, if the original IDs are not numeric, or not unique oover all
    components.
    """

    def __init__(
        self, source_file: Optional[Union[Path, str]] = None, destination_file: Optional[Union[Path, str]] = None
    ):
        source = JsonFileStore(file_path=Path(source_file)) if source_file else None
        destination = JsonFileStore(file_path=Path(destination_file)) if destination_file else None
        super().__init__(source=source, destination=destination)

    def _parse_data(
        self, data: StructuredData, data_type: str, extra_info: Optional[ExtraInfoLookup] = None
    ) -> Dataset:
        self._log.debug(f"Loading PGM {data_type} data")
        if isinstance(data, list):
            parsed_data = [
                self._parse_dataset(data=dataset, data_type=data_type, extra_info=extra_info) for dataset in data
            ]
            return convert_list_to_batch_data(parsed_data)
        if not isinstance(data, dict):
            raise TypeError("Raw data should be either a list or a dictionary!")
        return self._parse_dataset(data=data, data_type=data_type, extra_info=extra_info)

    def _parse_dataset(
        self, data: SinglePythonDataset, data_type: str, extra_info: Optional[ExtraInfoLookup] = None
    ) -> SingleDataset:
        return {
            component: self._parse_component(
                objects=objects, component=component, data_type=data_type, extra_info=extra_info
            )
            for component, objects in data.items()
        }

    def _parse_component(
        self, objects: ComponentList, component: str, data_type: str, extra_info: Optional[ExtraInfoLookup] = None
    ) -> np.ndarray:
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
                    if obj["id"] not in extra_info:
                        extra_info[obj["id"]] = {}
                    extra_info[obj["id"]][attribute] = value
        return array

    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfoLookup] = None) -> StructuredData:
        # Check if the dataset is a single dataset or batch dataset
        # It is batch dataset if it is 2D array or a indptr/data structure

        # If it is a batch, convert the batch data to a list of batches, then convert each batch individually.
        if self._is_batch(data=data):
            if extra_info is not None:
                self._log.warning("Extra info is not supported for batch data export")
            # We have established that this is batch data, so let's tell the type checker that this is a BatchDataset
            data = cast(BatchDataset, data)
            list_data = convert_batch_dataset_to_batch_list(data)
            return [self._serialize_dataset(data=x) for x in list_data]

        # We have established that this is not batch data, so let's tell the type checker that this is a BatchDataset
        data = cast(SingleDataset, data)
        return self._serialize_dataset(data=data, extra_info=extra_info)

    def _is_batch(self, data: Dataset) -> bool:
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

    def _serialize_dataset(
        self, data: SingleDataset, extra_info: Optional[ExtraInfoLookup] = None
    ) -> SinglePythonDataset:

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
                {attribute: obj[attribute].tolist() for attribute in objects.dtype.names if not is_nan(obj[attribute])}
                | extra_info.get(obj["id"], {})
                for obj in objects
            ]
            for component, objects in data.items()
        }

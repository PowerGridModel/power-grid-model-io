# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Custom exceptions for power-grid-model-io."""


class PowerGridModelIoError(Exception):
    """Base exception for all power-grid-model-io errors."""


class ComponentAlreadyExistsError(ValueError, PowerGridModelIoError):
    """Raised when a component already exists in the dataset."""

    def __init__(self, component: str, dataset: str):
        super().__init__(f"'{component}' component already exists in {dataset}")


class ComponentNotFoundError(KeyError, PowerGridModelIoError):
    """Raised when a required component is missing."""


class InvalidDatasetTypeError(ValueError, PowerGridModelIoError):
    """Raised when an invalid dataset type (e.g. other than input, output, or update) is encountered."""


class InvalidComponentTypeError(ValueError, PowerGridModelIoError):
    """Raised when an invalid component type is encountered."""


class MappingNotFoundError(ComponentNotFoundError):
    """Raised when a specific mapping for a component or attribute is missing."""


class AttributeNotFoundError(KeyError, PowerGridModelIoError):
    """Raised when a required component attribute is missing."""


class InvalidDataFormatError(ValueError, PowerGridModelIoError):
    """Raised when data format parsing or validation fails."""


class IndexAlreadyExistsError(ValueError, PowerGridModelIoError):
    """Raised when an index already exists."""


class IndexToComponentNotFoundError(KeyError, PowerGridModelIoError):
    """Raised when a pgm_id can not be found in any of the indexes."""

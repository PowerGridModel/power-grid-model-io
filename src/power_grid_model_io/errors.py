# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Custom exceptions for power-grid-model-io."""


class PowerGridModelIoError(Exception):
    """Base exception for all power-grid-model-io errors."""


class ComponentAlreadyExistsError(ValueError, PowerGridModelIoError):
    """Raised when a component already exists in the dataset."""


class ComponentNotFoundError(KeyError, PowerGridModelIoError):
    """Raised when a required component or attribute is missing."""


class InvalidDatasetTypeError(ValueError, PowerGridModelIoError):
    """Raised when an invalid dataset type (e.g. input/output) is encountered."""


class InvalidComponentTypeError(ValueError, PowerGridModelIoError):
    """Raised when an invalid component type is encountered."""


class MappingNotFoundError(ComponentNotFoundError):
    """Raised when a specific mapping for a component or attribute is missing."""


class AttributeNotFoundError(ComponentNotFoundError):
    """Raised when an attribute is missing from a component/dataset."""


class InvalidDataFormatError(ValueError, PowerGridModelIoError):
    """Raised when data format parsing or validation fails."""

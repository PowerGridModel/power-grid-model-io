# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Command Line Interface to convert files from one type to another
"""
# pylint: disable=duplicate-code

import logging
from pathlib import Path

import structlog
from power_grid_model.validation import errors_to_string, validate_input_data

from power_grid_model_io.converters.pgm_json_converter import PgmJsonConverter
from power_grid_model_io.data_stores.json_file_store import JsonFileStore
from power_grid_model_io.utils.modules import import_optional_module

typer = import_optional_module("typer", extra="cli")
app = typer.Typer(pretty_exceptions_show_locals=False)


def _validate(input_data, symmetric, log):
    """
    Helper function to validate data and show the issues in a log
    """
    log.debug("Validating Power Grid Model data")
    errors = validate_input_data(input_data, symmetric=symmetric)
    if not errors:
        log.info("Validation OK")
    else:
        log.error(errors_to_string(errors, name="input data", details=True))


@app.command()
def info():
    """
    Dummy function, we need at least two commands for the typer app
    TODO: Remove this info command.
    """
    log = structlog.get_logger("info")
    log.info("This client can be used to validate data, without storing the result.")


@app.command()
def pgm_json(pgm_json_file: Path, symmetric: bool = True, verbose: bool = False) -> None:
    """
    Validate a Power Grid Model JSON file
    """
    log = structlog.get_logger("pgm")

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG if verbose else logging.INFO),
    )

    pgm_converter = PgmJsonConverter()
    input_file = JsonFileStore(pgm_json_file)
    input_data, _extra_info = pgm_converter.load_input_data(input_file.load())
    _validate(input_data, symmetric, log)

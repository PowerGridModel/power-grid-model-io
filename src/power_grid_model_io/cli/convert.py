# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Command Line Interface to convert files from one type to another
"""

# pylint: disable=duplicate-code

import logging
from pathlib import Path
from typing import Optional

import structlog
from power_grid_model.validation import errors_to_string, validate_input_data

from power_grid_model_io.converters.gaia_excel_converter import GaiaExcelConverter
from power_grid_model_io.converters.pgm_json_converter import PgmJsonConverter
from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter
from power_grid_model_io.utils.modules import import_optional_module

typer = import_optional_module("typer", extra="cli")
app = typer.Typer(pretty_exceptions_show_locals=False)


def _validate(input_data, extra_info, symmetric, log):
    """
    Helper function to validate data and show the issues in a log
    """
    log.debug("Validating Power Grid Model data")
    errors = validate_input_data(input_data, symmetric=symmetric)
    if not errors:
        log.info("Validation OK")
    else:
        log.error(errors_to_string(errors))
        debug_str = "Error(s):\n"
        for error in errors:
            debug_str += f"{type(error).__name__}: {error}\n"
            for obj_id in error.ids:
                obj_info = extra_info.get(obj_id, {"component": error.component, "field": error.field, "id": error.ids})
                info_str = ", ".join(f"{key}={val}" for key, val in obj_info.items())
                debug_str += f"{obj_id:>6}. {info_str}\n"
        log.debug(debug_str)


@app.command()
# pylint: disable=too-many-arguments
def vision2pgm(
    excel_file: Path,
    pgm_json_file: Optional[Path] = None,
    symmetric: bool = True,
    validate: bool = False,
    verbose: bool = False,
) -> None:
    """
    Convert a Vision Excel export to a Power Grid Model JSON file
    """

    log_level = logging.DEBUG if verbose else logging.INFO
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(log_level))

    if pgm_json_file is None:
        pgm_json_file = excel_file.with_suffix(".json")

    vision_converter = VisionExcelConverter(source_file=excel_file)
    input_data, extra_info = vision_converter.load_input_data()

    pgm_converter = PgmJsonConverter(destination_file=pgm_json_file)
    pgm_converter.save(data=input_data, extra_info=extra_info)

    if validate:
        log = structlog.get_logger("vision2pgm")
        _validate(input_data=input_data, extra_info=extra_info, symmetric=symmetric, log=log)


@app.command()
# pylint: disable=too-many-arguments
def gaia2pgm(
    excel_file: Path,
    types_file: Optional[Path] = None,
    pgm_json_file: Optional[Path] = None,
    symmetric: bool = True,
    validate: bool = False,
    verbose: bool = False,
) -> None:
    """
    Convert a Gaia Excel export to a Power Grid Model JSON file
    """

    log_level = logging.DEBUG if verbose else logging.INFO
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(log_level))

    if pgm_json_file is None:
        pgm_json_file = excel_file.with_suffix(".json")

    gaia_converter = GaiaExcelConverter(source_file=excel_file, types_file=types_file)
    input_data, extra_info = gaia_converter.load_input_data()

    pgm_converter = PgmJsonConverter(destination_file=pgm_json_file)
    pgm_converter.save(data=input_data, extra_info=extra_info)

    if validate:
        log = structlog.get_logger("gaia2pgm")
        _validate(input_data=input_data, extra_info=extra_info, symmetric=symmetric, log=log)

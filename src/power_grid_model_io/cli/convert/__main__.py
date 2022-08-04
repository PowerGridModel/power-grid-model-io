# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import logging
from pathlib import Path
from typing import Optional

import structlog
import typer
from power_grid_model.validation import errors_to_string, validate_input_data

from power_grid_model_io import ExcelConverter, PgmConverter

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def info():
    log = structlog.getLogger("info")
    log.info("This client can be used to convert one filetype into another.")


@app.command()
def excel2pgm(
    excel_file: Path, mapping_file: Path, pgm_json_file: Optional[Path] = None, verbose: bool = False
) -> None:
    log = structlog.getLogger("excel2pgm")

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG if verbose else logging.INFO),
    )

    excel_converter = ExcelConverter()
    excel_converter.set_mapping_file(mapping_file)
    input_data, extra_info = excel_converter.load_input_file(excel_file)

    if pgm_json_file is None:
        pgm_json_file = excel_file.with_suffix(".json")

    pgm_converter = PgmConverter()
    pgm_converter.save_data(input_data, extra_info, pgm_json_file)

    # Validate data
    log.debug("Validating Power Grid Model data")
    errors = validate_input_data(input_data, symmetric=False)
    if not errors:
        log.info("Conversion OK")
    else:
        log.error(errors_to_string(errors))
        debug_str = "Error(s):\n"
        for error in errors:
            debug_str += f"{type(error).__name__}: {error}\n"
            for obj_id in error.ids:
                sheet = extra_info[obj_id].pop("sheet")
                info = ", ".join(f"{key}={val}" for key, val in extra_info[obj_id].items())
                debug_str += f"{obj_id:>6}. {sheet}: {info}\n"
        log.debug(debug_str)


if __name__ == "__main__":
    app()

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
def excel2pgm(excel_file: Path, mapping_file: Path, pgm_json_file: Optional[Path] = None, verbose: bool = True) -> None:
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


if __name__ == "__main__":
    app()

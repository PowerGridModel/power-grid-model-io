<!--
SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->
# Power Grid Model Input/Output

## Documentation:
  * Converters
    * [Tabular Converter](docs/converters/tabular_converter.md)


## Client tools

Installation:
```bash
pip install -e .[cli]
```

Usage:
```bash
pgm_validate DATA_FORMAT [ARGS]
pgm_convert CONVERSION [ARGS]
```

### Native Power Grid Model JSON format

```bash
pgm_validate pgm_json input_data.json
```

### Vision Excel format

```bash
pgm_convert vision2pgm myfile.xlsx --validate
```

### Gaia Excel format

```bash
pgm_convert gaia2pgm myfile.xlsx types.xlsx --validate
```

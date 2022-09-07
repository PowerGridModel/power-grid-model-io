<!--
SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->
# Power Grid Model Input/Output

## Client tools

Installation:
```bash
pip install -e .[cli,excel]
```

Usage:
```bash
validate DATA_FORMAT [ARGS]
convert CONVERSION [ARGS]
```

### Native Power Grid Model JSON format

```bash
validate pgm_json input_data.json
```

### Vision Excel format

```bash
convert vision2pgm myfile.xlsx mapping.yaml --validate
```

### Gaia Excel format

```bash
convert vision2pgm myfile.xlsx mapping.yaml --validate
```

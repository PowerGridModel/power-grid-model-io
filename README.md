<!--
SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->
# Power Grid Model Input/Output

## Conversion client tools

Installation:
```bash
pip install -e .[cli,excel]
```

Usage:
```bash
convert COMMAND [ARGS]
```
Currently, the only available command is `excel2pgm`.

### Excel conversion (work in progress)

Usage:
```bash
convert excel2pgm myfile.xlsx mapping.yaml
```

By default, the result is stored in a file with the same name as the .xslx file, but with an .json extension.
You can use the `--pgm-json-file` option to supply an output location.
```bash
convert excel2pgm myfile.xlsx mappin.yaml --pgm-json-file input_data.json
```

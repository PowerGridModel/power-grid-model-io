<!--
SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->

[![PyPI version](https://badge.fury.io/py/power-grid-model-io.svg)](https://badge.fury.io/py/power-grid-model-io)
[![License: MIT](https://img.shields.io/badge/License-MPL2.0-informational.svg)](https://github.com/alliander-opensource/power-grid-model-io/blob/main/LICENSE)
[![Build and Test Python](https://github.com/alliander-opensource/power-grid-model-io/actions/workflows/build-test-and-sonar.yml/badge.svg)](https://github.com/alliander-opensource/power-grid-model-io/actions/workflows/build-test-and-sonar.yml)
[![Check Code Quality](https://github.com/alliander-opensource/power-grid-model-io/actions/workflows/check-code-quality.yml/badge.svg)](https://github.com/alliander-opensource/power-grid-model-io/actions/workflows/check-code-quality.yml)
[![REUSE Compliance Check](https://github.com/alliander-opensource/power-grid-model-io/actions/workflows/reuse-compliance.yml/badge.svg)](https://github.com/alliander-opensource/power-grid-model-io/actions/workflows/reuse-compliance.yml)
[![docs](https://readthedocs.org/projects/power-grid-model-io/badge/)](https://power-grid-model-io.readthedocs.io/en/stable/)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_power-grid-model-io&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=alliander-opensource_power-grid-model-io)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_power-grid-model-io&metric=coverage)](https://sonarcloud.io/summary/new_code?id=alliander-opensource_power-grid-model-io)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_power-grid-model-io&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=alliander-opensource_power-grid-model-io)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_power-grid-model-io&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=alliander-opensource_power-grid-model-io)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_power-grid-model-io&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=alliander-opensource_power-grid-model-io)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_power-grid-model-io&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=alliander-opensource_power-grid-model-io)

<img src="docs\images\pgm-logo-color.svg" alt="Power Grid Model logo" width="100"/>

# Power Grid Model Input/Output

`power-grid-model-io` can be used for various conversions to the [power-grid-model](https://github.com/alliander-opensource/power-grid-model).

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
# License
This project is licensed under the Mozilla Public License, version 2.0 - see [LICENSE](LICENSE) for details.

# Contributing
Please read [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) and [CONTRIBUTING](CONTRIBUTING.md) for details on the process 
for submitting pull requests to us.

# Contact
Please read [SUPPORT](SUPPORT.md) for how to connect and get into contact with the Power Gird Model IO project.

<!--
SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->

# Power Grid Model IO

```{image} https://github.com/PowerGridModel/.github/raw/main/artwork/svg/color.svg
:alt: pgm_logo
:width: 300px
:align: right
```

Power Grid Model IO is a tool to convert grid data to and from the native data format of [power-grid-model](https://github.com/PowerGridModel/power-grid-model).
Currently, conversions from Vision excel exports and Pandapower are possible.

Some formats may be similar to the format used by [power-grid-model](https://github.com/PowerGridModel/power-grid-model), but use a different underlying data structure.
While we do not formally support those formats, you may find examples in this documentation for some of them.

## Install from PyPI

You can directly install the package from PyPI.

```
pip install power-grid-model-io
```

## Install from Conda

If you are using `conda`, you can directly install the package from `conda-forge` channel.

```
conda install -c conda-forge power-grid-model-io
```

## Citations

If you are using Power Grid Model IO in your research work, please consider citing our library using the references in [Citation](release_and_support/CITATION.md).

## Contents

Detailed contents of the documentation are structured as follows.

```{toctree}
:caption: "Installation"
:maxdepth: 2
quickstart.md
```

```{toctree}
:caption: "Converters"
:maxdepth: 2
converters/converter.md
converters/tabular_converter.md
converters/vision_converter.md
converters/pandapower_converter.md
```

```{toctree}
:caption: "Examples"
:maxdepth: 2
examples/pgm_json_example.ipynb
examples/vision_example.ipynb
examples/pandapower_example.ipynb
examples/arrow_example.ipynb
```

```{toctree}
:caption: "API Documentation"
:maxdepth: 4
power_grid_model_io
```

```{toctree}
:caption: "Contribution"
:maxdepth: 2
contribution/CODE_OF_CONDUCT.md
contribution/CONTRIBUTING.md
contribution/GOVERNANCE.md
```

```{toctree}
:caption: "Release and Support"
:maxdepth: 2
release_and_support/RELEASE.md
release_and_support/SUPPORT.md
release_and_support/SECURITY.md
release_and_support/CITATION.md
```

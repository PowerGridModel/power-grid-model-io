<!--
SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->

```{warning}
The documentation is under heavy development
```

# Power Grid Model IO

```{image} images/pgm-logo-color.svg
:alt: pgm_logo
:width: 150px
:align: right
```

Power Grid Model IO is a tool to convert grid data to and from the native data format of [power-grid-model](https://github.com/alliander-opensource/power-grid-model).
Currently, conversions from Vision excel exports is possible. Pandapower conversions are under development.


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
```

```{toctree}
:caption: "Examples"
:maxdepth: 2
examples/pgm_json_example.ipynb
examples/vision_example.ipynb
```



```{toctree}
:caption: "API Documentation"
:maxdepth: 4
power_grid_model_io
```


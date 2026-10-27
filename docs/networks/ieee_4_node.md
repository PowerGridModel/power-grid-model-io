<!--
SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# IEEE Four Node Networks

The IEEE Four Node Test Feeder provides a simple system for testing three-phase two-winding transformer connections. The newtork consists of four nodes, 2 line segments and a three-phase transformer bank connected between the two segments.
The variants of this network comprise transformer step-up/ step-down configuration, balanced/ un-balanced loading and transformer winding connections/ vector groups.
Following transformer connection variant are available in step-up and step-down configuration, balanced and unbalanced loading.

- Grounded Wye - Grounded Wye
- Grounded Wye - Delta
- Ungrounded Wye - Delta
- Delta - Grounded Wye
- Delta - Delta

```{note}
Open Winding transformers are currently not supported by `power-grid-model`, therefore, these variants are currently not available for loading.
```

All available variants are included in the `IeeeFourNodeVariant` enum. See [API Documentation](../power_grid_model_io) for more details.

```{warning}
As `power-grid-model` does not directly support delta-connected load, an additional ideal `dy` transformer is added to model the delta-connected loads. Accordingly, the additional `node` with id 10 and `transformer` with id 11, included in `ynd`, `yd`, `dd` transformers are virtual node and transformer, and are not relevant in the context of original IEEE Four Node Networks.
```

## Data Source

The network data, wire data and standard results are all obtained from the publically available datasets of IEEE at [IEEE PES Test Feeders](https://cmte.ieee.org/pes-testfeeders/).
The converted network models are loaded to `power-grid-model` and power flow results are validated against the results published by the IEEE.

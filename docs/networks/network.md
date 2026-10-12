<!--
SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# Networks

The `networks` module provides access to pre-built, standardized test networks from public power system benchmarks (such as IEEE test feeders) in `power-grid-model` format. These pre-built networks can be loaded as `Dataset`.

---

## Available Networks

Following pre-built test/ standard networks are available for loading:

- **IEEE 4 Node Test Feeder:** Refer to the [IEEE PES Test Feeders](https://cmte.ieee.org/pes-testfeeders/)

---

## Example Usage

```python
from power_grid_model import CalculationMethod, PowerGridModel
from power_grid_model_io.networks import IeeeFourNodeVariant, ieee_four_node

# Load network input data for a target variant
input_data = ieee_four_node(variant=IeeeFourNodeVariant.step_down_dyn_balanced_loading)

model = PowerGridModel(input_data)

asym_output = model.calculate_power_flow(symmetric=False, calculation_method=CalculationMethod.newton_raphson)
```

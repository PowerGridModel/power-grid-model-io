<!--
SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
SPDX-License-Identifier: MPL-2.0
-->

# Vision converter

The vision converter converts the excel exports of vision to PGM data. As mentioned in [Converters](converters/converter.md), vision converter is an implementation of the tabular converter.
The mapping of all attributes is stored in the `vision_en.yaml` and `vision_nl.yaml` files in [config](https://github.com/alliander-opensource/power-grid-model-io/tree/main/src/power_grid_model_io/config) directory.

## Load rate of elements 

Certain `elements` in vision, ie. appliances like transformer loads and induction motor have a result parameter of load rate.
In vision the load rate is calculated without considering the simultaneity factor of the connected node.
So we may observe a variation in power inflow/outflow result (ie. P,Q and S) due to different simultaneity factors. But the load rate always corresponds to `simultaneity of loads=1`.

When we make conversion to PGM, the input data attributes of PGM for loads like `p_specified` and `q_specified` are modified as per simultaneity. The resulting loading then takes simultaneity into account. 
**Hence, the loading of such elements may not correspond to the load rate obtained in vision**

## Voltage angle of buses in symmetric power-flow

Note that vision does not include clock angles of transformer for symmetrical calculations in the result of voltage angles. power-grid-model however does consider them so a direct comparison of results cannot be made in such case.

## Modelling differences or unsupported attributes

Some components are yet to be modelled for conversions because they might not have a straightforward mapping in power-grid-model. Those are listed here.

- power-grid-model currently does not support PV(Active Power-Voltage) bus and related corresponding features. 
- Currently, the efficiency type of PVs(Photovoltaics) element is also unsupported for all types except the `100%` type.
- The conversions for load behaviors of `industry`, `residential`, `business` are not yet modelled
- The source bus in PGM is mapped with a source impedance. `Sk"nom`, `R/X` and `Z0/Z1` are the attributes used in modelling source impedance. In vision, these attributes are used only for short circuit calculations
- A minor difference in results is expected since Vision uses a power mismatch in p.u. as convergence criteria whereas power-grid-model uses voltage mismatch.

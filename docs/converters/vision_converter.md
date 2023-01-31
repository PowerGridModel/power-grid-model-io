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

## Transformer load modelling

power-grid-model-io converts the transformer load into a individual transformer and a load for usage in power-grid-model. 
In vision, the modelling of a transformer load seems to be different from an individual transformer and load.
There is a minor difference in both in the reactive power consumed/generated. 
This can correspond to a minor voltage deviation too in the results.

```{tip}
It is recommended to split the transformer load into a individual components in vision beforehand to avoid this issue.
This can be done by first selecting the transformer loads: (Start | Select | Object -> Element -> Check Transformer load, Ok)
Then split it into individual components: (Start | Edit | Topological | Split)
```

## Voltage angle of buses in symmetric power-flow

Note that vision does not include clock angles of transformer for symmetrical calculations in the result of voltage angles. power-grid-model however does consider them so a direct comparison of angle results needs to be done with this knowledge.

## Modelling differences or unsupported attributes

Some components are yet to be modelled for conversions because they might not have a straightforward mapping in power-grid-model. Those are listed here.

- power-grid-model currently does not support PV(Active Power-Voltage) bus and related corresponding features. 
- Currently, the efficiency type of PVs(Photovoltaics) element is also unsupported for all types except the `100%` type.
- The conversions for load behaviors of `industry`, `residential`, `business` are not yet modelled. The load behaviors usually do not create a significant difference in power-flow results for most grids when the voltage at bus is close to 1 p.u. Hence, the conversion of the mentioned load behaviors is approximated to be of `Constant Power` type for now. 
- The source bus in PGM is mapped with a source impedance. `Sk"nom`, `R/X` and `Z0/Z1` are the attributes used in modelling source impedance. In vision, these attributes are used only for short circuit calculations
- A minor difference in results is expected since Vision uses a power mismatch in p.u. as convergence criteria whereas power-grid-model uses voltage mismatch.

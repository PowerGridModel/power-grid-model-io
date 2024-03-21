<!--
SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
SPDX-License-Identifier: MPL-2.0
-->

# Vision converter

The Vision Excel converter converts the Excel exports from Vision to PGM data format. As mentioned in [Converters](converters/converter.md), Vision Excel converter is an implementation of the tabular converter.
The default mapping of all attributes is stored in the `vision_en.yaml` and `vision_nl.yaml` files in [config](https://github.com/PowerGridModel/power-grid-model-io/tree/main/src/power_grid_model_io/config) directory.
Custom mapping files are supported via passing the file directory to the constructor of the converter.

## Load rate of elements 

Certain `elements` in Vision, i.e., appliances like transformer loads and induction motor have a result parameter of load rate.
In Vision, load rates are calculated without considering the simultaneity factors of connected nodes.
So we may observe a variation in power inflow/outflow result (i.e., P, Q and S) due to different simultaneity factors. But the load rate always corresponds to `simultaneity of loads=1`.

When we make conversion to PGM, the input data attributes of PGM for loads like `p_specified` and `q_specified` are modified as per simultaneity. The resulting loading then takes simultaneity into account. 
**Hence, the loading of such elements may not correspond to the load rate obtained in Vision**

## Transformer load modeling

power-grid-model-io converts the transformer load into a individual transformer and a load for usage in power-grid-model. 
To the best of our knowledge, Vision modeles a transformer load differently than an individual transformer plus load.
There is a minor difference in both the reactive power consumed and generated. 
This might correspond to a minor voltage deviation in the results.

```{tip}
To avoid this issue, it is recommended to split the transformer load into individual components in Vision beforehand.
This can be done by first selecting the transformer loads: [Start | Select | Object -> Element -> Check Transformer load -> Ok], then split it into individual components: [Start | Edit | Topological | Split]
```

## Voltage angle of buses in symmetric power-flow

Note that in symmetrical calculations, Vision does not include clock angles of transformers in the result of voltage angles. power-grid-model, however, does consider them. Therefore when doing a direct comparison of angle results, this needs to be taken into consideration.

## Modeling differences or unsupported attributes

Some components are yet to be modeled for conversions because they might not have a straightforward mapping in power-grid-model. Those are listed here.

- power-grid-model currently does not support PV(Active Power-Voltage) bus and related corresponding features. 
- Currently, the efficiency type of PVs(Photovoltaics) element is also unsupported for all types except the `100%` type. For the efficiency type: `0, 1 pu: 93 %; 1 pu: 97 %`, the generation power is multiplied by 97% as a closest approximation.
- The conversions for load behaviors of `industry`, `residential` and `business` are not yet modeled. The load behaviors usually do not create a significant difference in power-flow results for most grids when the voltage at bus is close to `1 pu`. Hence, the conversion of the mentioned load behaviors is approximated to be of `Constant Power` type for the time being. 
- The source bus in power-grid-model is mapped with a source impedance. `Sk"nom`, `R/X` and `Z0/Z1` are the attributes used in modeling source impedance. In Vision, these attributes are used only for short circuit calculations
- The load rate for transformer is calculated in Vision by current i.e., `load_rate = max(u1 * I1, u2 * I2) * sqrt(3) / Snom * 100`. Whereas in power-grid-model, loading is calculated by power, i.e., `loading = max(s1,s2)/sn`. (Note: The attribute names are as per relevant notation in Vision and PGM respectively). This gives a slight difference in load rate of transformer.
- A minor difference in results is expected since Vision uses a power mismatch in p.u. as convergence criteria whereas power-grid-model uses voltage mismatch.
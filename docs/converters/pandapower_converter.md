<!--
SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# Pandapower converter

The pandapower converter can convert the pandapower `net` to power-grid-model input data.
It also converts the power flow output of power-grid-model into the `res_*` Dataframes in the pandapower `net`.
The converter can be used in a similar way as described in [Converters](converter.md).

## Cross referencing

Due to differences between the power-grid-model and pandapower, some components in one model are modeled as multiple
components in the other or vise versa.
An example on how to cross-reference can be found in the
[pandapower conversion example](../examples/pandapower_example.ipynb#cross-referencing-objects).

## Defaults

If any of the essential tap attributes: `tap_pos`, `tap_nom`, `tap_side` are not available then the tap feature for
transformers are disabled by setting `tap_nom=tap_pos=0` via conversion.
If a `vector_group` is not available for transformer then a default is set in conversion as `YNyn` for even clocks and
`DYn` for odd clocks.
Similarly for three winding transformer, `YNynyn` is set for even clocks of `shift_mv_degree` and `shift_lv_degree`.
If the clocks are odd, then the vector group is converted as `YNynd`, `YNdyn` or `YNdd`.

## Modeling differences

The user must be aware of following unsupported features or differences in conversion.

* Currently, the conversions only support powerflow calculations and their relevant attributes.
* Any feature involving a PV bus, i.e., generator, DC line are unsupported as of now.

In addition, the following component-specific modeling differences exist.

### Load

Delta type of loads, `type="delta"` are not supported in power-grid-model.

### Switch

The features regarding `in_ka` and `z_ohm` attributes are currently unsupported.

### External grid

The external grid is modeled with a source impedance all sequence components in power-grid-model.
Whereas it is only present in negative and zero sequence networks in pandapower.
Hence, this impedance value has to be mentioned in the power-flow calculation when using the power-grid-model power flow
calculation.

### Transformer

Custom zero sequence parameters `vk0_percent` and `vkr0_percent` in transformers are not supported.
The same ones as positive sequence admittance is used in power-grid-model.
However, `mag0_percent` and `mag0_rx` are fetched from `pandapower`. In case `vk0_percent` is different from
`vk_percent` all zero sequence parameters are ignored.

**`mag0_percent` is treated as a ratio not as a percent in line with definition of `pandapower`**

The changing of impedance by the means of characteristics in `net.characteristic` is not directly supported.
The phase shift in angle by transforermers by the `shift_degree` attribute only supports clock values in PGM.
The `tap_phase_shifter` transformer and shifting of angles with tap via `tap_step_degree` are features not added in
power-grid-model yet.
The default for transformer model to be used in pandapower is `t` model but power-grid-model supports only `pi` model of
transformer hence this may create a minor difference calculation.

### Three  winding transformer

Custom zero sequence parameters in three winding transformers are not supported.
The differences defined in [Transformer](#transformer) are applicable here as well.
Additionally, tap connection at star point, i.e., `tap_at_star_point` is not supported.

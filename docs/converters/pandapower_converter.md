<!--
SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->

# Pandapower converter

Pandapower converter can convert the pandapower `net` to power-grid-model input data. 
It also converts the power flow output of power-grid-model into the `res_*` Dataframes in the pandapower `net`.
The converter can be used in a similar way as described in [Converters](converter.md).

## Modelling differences

The user must be aware of following unsupported features or differences in conversion. 
Currently, the conversions only support powerflow calculations and their relevant attributes.
Any feature involving a PV bus, ie. generator, DC line are unsupported as of now.

### Load

Delta type of loads, `type="delta"` are not supported in power-grid-model.

### Switch

The features regarding `in_ka` and `z_ohm` attributes are currently unsupported.

### External grid

The external grid is modelled with a source impedance all sequence components in power-grid-model.
Whereas it is only present in negative and zero sequence networks in pandapower.
Hence, this impedance value has to be mentioned in the power-flow calculation when using the power-grid-model power flow calculation.

### Transformer

Custom zero sequence parameters in transformers are not supported.
The same ones as positive sequence admittance is used in power-grid-model.
The changing of impedance by the means of characteristics in `net.characteristic` is not directly supported.
The phase shift in angle by transforermers by the `shift_degree` attribute only supports clock values in PGM.
The `tap_phase_shifter` transformer and shifting of angles with tap via `tap_step_degree` are features not added in power-grid-model yet.
The default for transformer model to be used in pandapower is `t` model but power-grid-model supports only `pi` model of transformer hence this may create a minor difference calculation.

### Three  winding transformer

The differences defined in [Transformer](#transformer) are applicable here as well.
Additionally, tap connection at star point, ie. `tap_at_star_point` is not supported.
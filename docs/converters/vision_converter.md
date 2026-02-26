<!--
SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
SPDX-License-Identifier: MPL-2.0
-->

# Vision converter

The Vision Excel converter converts the Excel exports from Vision to PGM data format.
As mentioned in [Converters](converters/converter.md), Vision Excel converter is an implementation of the tabular
converter.
The default mapping of all attributes is stored in the `vision_en.yaml` and `vision_nl.yaml` files in
[config](https://github.com/PowerGridModel/power-grid-model-io/tree/main/src/power_grid_model_io/config) directory.
Custom mapping files are supported via passing the file directory to the constructor of the converter.

## Load rate of elements

Certain `elements` in Vision, i.e., appliances like transformer loads and induction motor have a result parameter of
load rate.
In Vision, load rates are calculated without considering the simultaneity factors of connected nodes.
So we may observe a variation in power inflow/outflow result (i.e., P, Q and S) due to different simultaneity factors.
But the load rate always corresponds to `simultaneity of loads=1`.

When we make conversion to PGM, the input data attributes of PGM for loads like `p_specified` and `q_specified` are
modified as per simultaneity. The resulting loading then takes simultaneity into account.
**Hence, the loading of such elements may not correspond to the load rate obtained in Vision**

## Transformer load modeling

power-grid-model-io converts the transformer load into a individual transformer and a load for usage in
power-grid-model.
To the best of our knowledge, Vision modeles a transformer load differently than an individual transformer plus load.
There is a minor difference in both the reactive power consumed and generated.
This might correspond to a minor voltage deviation in the results.

```{tip}
To avoid this issue, it is recommended to split the transformer load into individual components in Vision beforehand.
This can be done by first selecting the transformer loads: [Start | Select | Object -> Element -> Check Transformer load -> Ok], then split it into individual components: [Start | Edit | Topological | Split]
```

## Voltage angle of buses in symmetric power-flow

Note that in symmetrical calculations, Vision does not include clock angles of transformers in the result of voltage
angles.
power-grid-model, however, does consider them.
Therefore when doing a direct comparison of angle results, this needs to be taken into consideration.

## Modeling differences or unsupported attributes

Some components are yet to be modeled for conversions because they might not have a straightforward mapping in
power-grid-model.
Those are listed here.

- power-grid-model currently does not support PV(Active Power-Voltage) bus and related corresponding features.
- Currently, the efficiency type of PVs(Photovoltaics) element is also unsupported for all types except the `100%` type.
  For the efficiency type: `0, 1 pu: 93 %; 1 pu: 97 %`, the generation power is multiplied by 97% as a closest
  approximation.
- The conversions for load behaviors of `industry`, `residential` and `business` are not yet modeled.
  The load behaviors usually do not create a significant difference in power-flow results for most grids when the
  voltage at bus is close to `1 pu`.
  Hence, the conversion of the mentioned load behaviors is approximated to be of `Constant Power` type for the time
  being.
- The source bus in power-grid-model is mapped with a source impedance. `Sk"nom`, `R/X` and `Z0/Z1` are the attributes
  used in modeling source impedance.
  In Vision, these attributes are used only for short circuit calculations
- The load rate for transformer is calculated in Vision by current i.e.,
  `load_rate = max(u1 * I1, u2 * I2) * sqrt(3) / Snom * 100`.
  Whereas in power-grid-model, loading is calculated by power, i.e., `loading = max(s1,s2)/sn`.
  (Note: The attribute names are as per relevant notation in Vision and PGM respectively).
  This gives a slight difference in load rate of transformer.
- A minor difference in results is expected since Vision uses a power mismatch in p.u. as convergence criteria whereas
  power-grid-model uses voltage mismatch.
- The Voltage Control option in vision is modelled by `transformer_tap_regulator` in PGM.
  Currently, PGM only supports line drop compensation on both loading and generating direction.
  So this is equivalent to unchecking of `Also in backw. direction`.
  Neither `Load Dependent` compensation or a master-slave configuration for control under the `State` section is
  supported.

## Accomdation of UUID-based id system since Vision 9.7

Vision introduced UUID based identifier system since version 9.7.
It is implemented following the Microsoft naming scheme by replacing all the original identifier fields
(i.e., '*Number', '*Subnumber', '*Nummber' and '*Subnummer') to GUID.
This change brings the many benefits of UUIDs in general while on the other hand adds certain work on the conversion
side.
Since computations in PGM alone do not benefit from a string based identifier, we hence made the descision to perform
UUID to integer conversion while maintaining the 'GUID' information in the `extra_info` field.
Note that existing mapping files can still be used without significant changes, apart from adding `GUID` to the `extra`
fields of interest.

An examplery usage can be found in the example notebook as well as in the test cases.

## Optional extra columns

When working with Vision Excel exports, some metadata columns (like `GUID` or `StationID`) may not always be present,
especially in partial exports.
The `optional_extra` feature allows you to specify columns that should be included in `extra_info` if present,
but won't cause conversion failure if missing.

**Syntax:**

```yaml
grid:
  Transformers:
    transformer:
      id:
        auto_id:
          key: Number
      # ... other fields ...
      extra:
        - ID            # Required - fails if missing
        - Name          # Required - fails if missing
        - optional_extra:
            - GUID      # Optional - skipped if missing
            - StationID # Optional - skipped if missing
```

**Behavior:**

- Required columns (listed directly under `extra`) will cause a KeyError if missing
- Optional columns (nested under `optional_extra`) are silently skipped if not found
- If some optional columns are present and others missing, only the present ones are included in `extra_info`
- This feature is particularly useful for handling different Vision export configurations or versions

**Duplicate handling:**
When a column appears in both the regular `extra` list and within `optional_extra`,
the regular `extra` entry takes precedence and duplicates are automatically eliminated from `optional_extra`:

```yaml
extra:
  - ID              # Regular column - always processed
  - Name            # Regular column - always processed  
  - optional_extra:
      - ID          # Duplicate - automatically removed
      - GUID        # Unique optional - processed if present
      - StationID   # Unique optional - processed if present
```

In this example, `ID` will only be processed once (from the regular `extra` list),
while `GUID` and `StationID` are processed as optional columns.
This prevents duplicate data in the resulting `extra_info`
and ensures consistent behavior regardless of column ordering.

## Common/Known issues related to Vision

So far we have the following issue known to us related to Vision exported spread sheets.
We provide a solution from user perspective to the best of our knowledge.

### Duplicated `P` columns

Vision can export sheets with duplicated `P` columns, one of which being unitless additional information.
This field is of no actual purpose within PGM calculation.
In case of duplicate column names, PGM-IO renames them to, e.g., `P`, `P_2` and so on.
However the change in mapping to the duplicate name should be done manually by the user in a custom mapping file.

**Tip:** We advice users to uncheck the `specifics` when exporting from Vision and an extra `P` column would not appear.

### Different names for columns

In different versions of exports, user can sometimes find different names for columns.
For example, the column name `N1` might be represented as `Grounding1` in some exports.
To address this, the `VisionExcelConverter`'s `terms_changed` argument can be used.

### Sheets being ignored

In different versions of exports, user can find different names for the spreadsheets.
All sheets not specified in the mapping file are ignored for conversion.
Hence if the mapping has an incorrect sheet name, they would not be considered in conversion and no error shall be given
out.
The user should use the adequate mapping file for different vision versions.
And if the default mapping is modified, it is advised to verify if all required sheets are being converted.

## Security Considerations

Safe XML parsing is used to process Excel-based inputs, ensuring secure handling of mapping and tabular data.
See security considerations of the [tabular_converter](converters/tabular_converter.md#security-considerations)

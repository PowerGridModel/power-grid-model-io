<!--
SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->
# Tabular converter
Tabular data can come from Excel files, a set of CSV files, GNF files, databases, pandas DataFrames, etc etc.
The similarity between all tabular data is that it contains multiple `tables`,
each with multiple `columns`, possibly with a specific `unit`.
Others may have a categorical value which needs to be mapped (i.e. open: 0, closed: 1); in general we'll call these 
`substitutions`.

## Mapping file
A mapping file is a yaml with three main sections `grid`, `units` and `substitutions`:
```yaml
grid:

  Nodes:
    node:
      id: Number
      u_rated: Unom
      extra: ID

  Cables:
    line:
      id: Number
      from_node: From.Number
      from_status: From.Switch state
      to_node: To.Number
      to_status: To.Switch state

units:
  A:
  F:
    µF: 0.000001

substitutions:
  ".*Switch state":
    "off": 0
    "on": 1
```

## Grid
For each `table`, the target PGM `component` is listed (e.g. Nodes: node, Cables: line).
The for each PGM `column` the source column is supplied (e.g. u_rated: Unom, from_status: From.SwitchStatus).

## Field Definitions
If the `column` definition is a one on one mapping,
the value is simply the name of the source column (e.g.u_rated: Unom),
but in many cases the mapping is a bit more complex.
You can use the following `column` definitions: 

  * Column name `str`
    ```yaml
    from_node: From.Number
    ```
  * First matching column name `str`
    ```yaml
    p_specified: Inverter.Pnom | Inverter.Snom
    ```
  * Reference to a column on another sheet (using `!` notation as in Excel) `str`
    ```yaml
    r1: CableProperties!R[Shortname=Type short]
    ```
    You may also specify the reference more explicitly:
    ```yaml
    r1: CableProperties!R[CableProperties!Shortname=Cables!Type short]
    ```
  * Constant value `int | float`
    ```yaml
    from_status: 1
    tan1: 0.0
    ```
  * Functions `Dict[str, List[Any]`
    ```yaml
    p_specified:
      min:
        - Pnom
        - Inverter.Pnom
    ```
  * Nested definitions:
    ```yaml
    q_specified:
      power_grid_model_io.filters.phase_to_phase.reactive_power:
        - min:
            - Pnom
            - Inverter.Pnom | Inverter.Snom
        - Inverter.cos phi
        - 1.0
    ```
    Is similar to:
    ```python
    from power_grid_model_io.filters.phase_to_phase import reactive_power
    
    q_specified = reactive_power(
      min(
        table["Pnom"],
        table["Inverter.Pnom"] if "Inverter.Pnom" in table else table["Inverter.Snom"]
      ),
      table["Inverter.Snom"],
      1.0
    )
    ```
## Units
Power Grid Model uses SI units (e.g. "W" for Watts),
but source data may be supplied in different units (e.g. "MW" for Mega Watts).
If units are supplied in the tabular data,
the pandas DataFrame storing the data is expected to have `MultiIndexes` for the columns.
For our application, a `MultiIndex` can be interpreted as a tuple; the first element is the column name, the second 
element is the column unit. For example: `("C0", "µF")`.

If a unit is supplied, it should be defined in the units section of the mapping.
Undefined units are not allowed to prevent errors.

```yaml
units:
  A:
  F:
    µF: 0.000001
  ohm/m:
    ohm/km: 0.001
```
The definitions above can be interpreted as:
  * **A** is a valid SI unit
  * **F** is a valid SI unit
    * 1 **µF** = 0.000001 **A**
  * **ohm/m** is a valid SI unit
    * 1 **ohm/km** = 0.001 **ohm/m**

## Substitutions
Some columns may contain categorical values (enums) which should be replaced. The column names can be defined as 
regular expressions. 
```yaml
substitutions:
  ".*Switch state":
    "off": 0
    "in": 1
  N1:
    none: false
    own: true
```
The definitions above can be interpreted as:
  * In all columns that end with `SwitchState` (e.g. `From.Switch State`, `To.Switch State` or just `Switch State`),
    the word "off" should be replaced with the integer 0 and the word "in" should be replaced with the value 1.
  * In all columns called "N1",
    the word "none" should be replaced with the boolean `false`
    and the word "own" should be replaced with the value boolean `true`.

## AutoID
The `id` field is special in the sense that each object should have a unique numerical id in power grid model. 
Therefore, each id definition is mapped to a numerical ID.
Also each field name that ends with `node` is converted into the matching numerical ID.

```python
from power_grid_model_io.utils.auto_id import AutoID

auto_id = AutoID()
a = auto_id("Alpha")   # a = 0
b = auto_id("Bravo")   # b = 1
c = auto_id("Alpha")   # c = 0 (because key "Alpha" already existed)
item = auto_id[1]      # item = "Bravo"
 ```
  
See also {py:class}`power_grid_model_io.utils.AutoID`

### Vision and Gaia
For Vision and Gaia files, an extra trick is applied. Let's assume this mapping:

```yaml
grid:
  Nodes:
    node:
      id: Number
      ...
  Cables:
    line:
      id: Number
      from_node: From.Number
      ...
```

The PGM `node["id"]` will be a number, based on the values in the `Nodes!Number` column.
The PGM `line["from_node"]` (which ends with `node`) will be based on the values in the `Nodes!From.Number`.
Originally this didn't work, due to the hashing function, as the column names differ: `"Number" != "From.Number"`.
Therefore, the Vision and Gaia converters overload the `_id_lookup()` method.
They split the column name on `.` so that the `Number` matches `From.Number`.
<!--
SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>

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
      auto_id:
        key: Number
      u_rated: Unom
      extra: ID

  Cables:
    line:
      id:
        auto_id:
          key: Number
      from_node:
        auto_id:
          table: Nodes
          key:
            Number: From.Number      
      from_status: From.Switch state
      to_node:
        auto_id:
          table: Nodes
          key:
            Number: To.Number      
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
  * First matching column name that exists in the data `str`
    ```yaml
    p_specified: Inverter.Pnom | Inverter.Snom
    ```
  * Automatic IDs `Dict[str, Dict[str, Any]]` with single key `reference`, required attribute `key` and optinal 
    attributes `table` and `name`. More extensive examples are shown in the section [AutoID Mapping](##autois-mapping).
    ```yaml
    id:
      auto_id:
        key: Number
    ```
 
  * Reference to a column on another sheet `Dict[str, Dict[str, Any]]` with single key `reference` and the 
    ```yaml
    r1:
      reference:
        query_column: Shortname
        other_table: Cable Properties
        key_column: Type short
        value_column: R
    ```
  * Constant value `int` or `float`
    ```yaml
    from_status: 1
    tan1: 0.0
    ```
  * Pandas DataFrame functions `Dict[str, List[Any]]`
    (`prod`, `sum`, `min`, `max`, etc and the alias `multiply` which translates to `prod`)
    ```yaml
    p_specified:
      min:
        - Pnom
        - Inverter.Pnom
    ```
  * Custom functions `Dict[str, Dict[str, Any]]`
    ```yaml
      g0:
        power_grid_model_io.functions.complex_inverse_real_part:
          real: R0
          imag: X0
    ```
  * Nested definitions:
    ```yaml
    q_specified:
      power_grid_model_io.functions.phase_to_phase.reactive_power:
        p:
          min:
            - Pnom
            - Inverter.Pnom | Inverter.Snom
        cos_phi: Inverter.cos phi
    ```
    Is similar to something like:
    ```python
    from power_grid_model_io.functions.phase_to_phase import reactive_power
    
    q_specified = reactive_power(
      p=min(
        table["Pnom"],
        table["Inverter.Pnom"] if "Inverter.Pnom" in table else table["Inverter.Snom"]
      ),
      cos_phi=1.0
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

## AutoID Mapping
Let's consider a very common example of the usage of `auto_id` in a mapping file.
(Note that we're focussing on the ids and references, so the other attributes have been disregarded.)
```yaml
  Nodes:
    node:
      id:
        auto_id:
          key: Number
    Cables:
      line:
        id:
          auto_id:
            key: Number
        from_node:
          auto_id:
            table: Nodes
            key:
              Number: From_Number
        to_node:
          auto_id:
            table: Nodes
            key:
              Number: To_Number
```
This basically reads as:
* For each row in the Nodes table, a PGM node instance is created.
  * For each node instance, a numerical id is generated, which is unique for each value in the Number column. This 
    assumes that the Number column is unique in the source table. Let's say tha values of the Number column in that 
    Nodes source table are `[101, 102, 103]`, then the generated IDs will be `[0, 1, 2]`. However, if the source 
    column is not unique, the pgm ids won't be unique as well: `[101, 102, 103, 101] -> [0, 1, 2, 0]`.
  * Under the hood, the table name `Nodes` and the column name `Number` are used to generate these IDs:
    * `{"table": "Nodes", "key" {"Number": 101} -> 0`
    * `{"table": "Nodes", "key" {"Number": 102} -> 1`
    * `{"table": "Nodes", "key" {"Number": 103} -> 2`
* For each row in the Cables table, a PGM line instance is created.
  * For each line instance, a numerical id is generated, just like for the nodes.
    Let's say there are two Cables `[201, 202]` and the corresponding lines will have IDs `[3, 4]`.  
    * `{"table": "Cables", "key" {"Number": 201} -> 3`
    * `{"table": "Cables", "key" {"Number": 202} -> 4`
  * A Cable connects to two Nodes.
    In this example Cable `201` connects Node `101` and `102`, and Cable `201` connects Node `102` and `103`.
    These Node Numbers are stored in the columns `From_Number` and `To_Number`.
    In order to retrieve the right PGM IDs, we have to explicitly state that the table in which the Nodes are 
    defined is called `Nodes` and the original column storing the Node Numbers is called `Number`.
  * On the 'from' side of the cables:
    * `{"table": "Nodes", "key" {"Number": 101} -> 0`
    * `{"table": "Nodes", "key" {"Number": 102} -> 1`
  * On the 'to' side of the cables:
    * `{"table": "Nodes", "key" {"Number": 102} -> 1`
    * `{"table": "Nodes", "key" {"Number": 103} -> 2`


## Advanced AutoID Mapping
In some cases, multiple components have to be created for each row in a source table.
In such cases, the `name` attribute may be necessary to create multiple PGM IDs for a single row. Let's consider
this example:
```yaml
  Transformer loads:
    transformer:
      id:
        auto_id:
          name: transformer
          key:
            - Node_Number
            - Subnumber
      from_node:
        auto_id:
          table: Nodes
          key:
            Number: Node_Number
      to_node:
        auto_id:
          name: internal_node
          key:
            - Node_Number
            - Subnumber
    node:
      id:
        auto_id:
          name: internal_node
          key:
            - Node_Number
            - Subnumber
    sym_load:
      id:
        auto_id:
          name: load
          key:
            - Node_Number
            - Subnumber
      node:
        auto_id:
          name: internal_node
          key:
            - Node_Number
            - Subnumber
```

Let's say we have one Transformer Load connected to the Node Number 103 and it's Subnumber is 1.
Then the following IDs will be generated / retrieved:
* `transformer.id`:
  `{"table": "Transformer loads", "name": "transformer", "key" {"Node_Number": 103, "Subnumber": 1} -> 5`
* `transformer.from_node`:
  `{"table": "Nodes", "key" {"Number": 103} -> 2`
* `transformer.to_node`:
  `{"table": "Transformer loads", "name": "internal_node", "key" {"Node_Number": 103, "Subnumber": 1} -> 6`
* `node.id`:
  `{"table": "Transformer loads", "name": "internal_node", "key" {"Node_Number": 103, "Subnumber": 1} -> 6`
* `sym_load.id`:
  `{"table": "Transformer loads", "name": "load", "key" {"Node_Number": 103, "Subnumber": 1} -> 7`
* `sym_load.node`:
  `{"table": "Transformer loads", "name": "internal_node", "key" {"Node_Number": 103, "Subnumber": 1} -> 6`

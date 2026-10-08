<!--
SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
SPDX-License-Identifier: MPL-2.0
-->

# CGMES2PGM Converter

The [CGMES2PGM Converter](https://github.com/SOPTIM/cgmes2pgm_converter) is an external tool for converting ENTSO-E CIM/CGMES datasets into the Power Grid Model (PGM) input data. The [CGMES2PGM Suite](https://github.com/SOPTIM/cgmes2pgm_suite) provides additional tools for uploading datasets and running state estimation workflows.

The CGMES2PGM Converter and Suite are released under the Apache-2.0 license and are supported by [SOPTIM AG](https://www.soptim.de/en/cgmes2pgm/).

## Requirements

The CGMES2PGM converter requires a graph database, e.g. [Apache Fuseki](https://jena.apache.org/documentation/fuseki2/), into which the user has to upload the CGMES dataset prior to running the conversion. Alternatively, the CGMES2PGM Suite can upload the dataset automatically.

The CGMES2PGM converter can be installed directly via
```bash
pip install cgmes2pgm_converter
```

or indirectly as a dependency of the CGMES2PGM Suite
```bash
pip install cgmes2pgm_suite
```

Install other packages as needed
```bash
pip install power_grid_model pandas
```

## Datasets

Due to licensing restrictions, CGMES datasets cannot be provided here as examples and users must obtain the datasets separately. However, conformity test datasets can be downloaded from the [ENTSO-E website](https://www.entsoe.eu/data/cim/cim-conformity-and-interoperability/).

## Conversion

The CGMES2PGM Converter supports CIM/CGMES in versions 2.x (e.g. 2.14.15) and 3.x (e.g. 3.0). The CIM/CGMES format is extensive. Therefore, only a subset of the classes is converted to the PGM format with the primary focus on components relevant for state estimation and for power flow calculations.

| CGMES                                                                                                 | PGM                                       |
| ------------------------------------------------------------------------------------------------------|-------------------------------------------|
| TopologicalNode                                                                                       | `node`                                    |
| Switch and subclasses                                                                                 | `link`                                    |
| ACLineSegment, EquivalentBranch, SeriesCompensator                                                    | `generic_branch`                          |
| PowerTransformer (2W)                                                                                 | `generic_branch`                          |
| PowerTransformer (3W)                                                                                 | three `generic_branch` with auxiliary `node` |
| AsynchronousMachine, ConformLoad, EnergyConsumer, NonConformLoad, StationSupply                       | `sym_load`                                |
| EnergySource, ExternalNetworkInjection, EquivalentInjection, StaticVarCompensator, SynchronousMachine | `sym_gen`                                 |

The supported branch classes are represented as `generic_branch` components directly using electrical properties such as r, x, g and b. Three-winding transformers are represented as three `generic_branch` components connected through an auxiliary `node`.

RatioTapChangers are converted to `generic_branch` components using the `k` argument to represent the tap ratio. PhaseTapChangers (PST) are converted to `generic_branch` components using the `theta` and `k` arguments to represent the phase shift and tap ratio, respectively.

The converter does not currently map CGMES voltage-regulation controls to PGM `voltage_regulator` components.

## Usage

### Basic example

In the simplest case, the dataset is already uploaded into the default graph of the graph database and the converter can directly read from it, as shown in the example below.

```python
import pandas as pd
from cgmes2pgm_converter import CgmesDataset, CgmesToPgmConverter
from power_grid_model import ComponentType

# cim_namespace = "http://iec.ch/TC57/2013/CIM-schema-cim16#", # Use this for CIM 2.x
cim_namespace = "http://iec.ch/TC57/CIM100#", # Use this for CIM 3.x

# data set with name ExampleGrid must be uploaded to the graph database (change URL if needed)
dataset_url = "http://localhost:3030/ExampleGrid"

dataset = CgmesDataset(base_url=dataset_url, cim_namespace=cim_namespace, split_profiles=False)
converter = CgmesToPgmConverter(datasource=dataset)

# input_data is the dataset in the PGM format,
# extra_info contains additional mapping information.
input_data, extra_info = converter.convert()

print("\nInput data for nodes:")
print(pd.DataFrame(input_data[ComponentType.node]))
print("\nInput data for generic branches:")
print(pd.DataFrame(input_data[ComponentType.generic_branch]))
```

### Example with dataset upload

For more advanced cases, the user might want to store each CIM profile in its own graph within the graph database, rather than using the default graph. Uploading data into separate graphs manually can be cumbersome, therefore the CGMES2PGM Suite provides functionality to automate this process.

```python
import pandas as pd
from cgmes2pgm_converter import CgmesDataset, CgmesToPgmConverter
from cgmes2pgm_suite.rdf_store.xml_dir_import import RdfXmlDirectoryImport
from power_grid_model import ComponentType

# cim_namespace = "http://iec.ch/TC57/2013/CIM-schema-cim16#", # Use this for CIM 2.
cim_namespace = "http://iec.ch/TC57/CIM100#", # Use this for CIM 3.x

# XML or ZIP source files containing the CGMES dataset
input_path = "path/to/ExampleGrid"

# upload CGMES data to this dataset (must exist) in the graph database (change URL if needed)
dataset_url = "http://localhost:3030/ExampleGrid"

# Upload the XML files to the Fuseki graph database if needed.
# Set initially to true, and change to false for repeated runs with the same dataset.
upload_files = True

# Keep everything in the default graph or upload each CIM profile into its own graph
split_profiles = True

dataset = CgmesDataset(base_url=dataset_url, cim_namespace=cim_namespace, split_profiles=split_profiles)

if upload_files:
    importer = RdfXmlDirectoryImport(
        dataset=dataset,
        base_iri=dataset_url,
        split_profiles=dataset.split_profiles,
    )
    full_models = importer.import_directory(input_path)
    print(f"Imported {len(full_models)} full models from XML files.")
else:
    print("Skipping XML file import.")

converter = CgmesToPgmConverter(datasource=dataset)

# input_data is the dataset in the PGM format,
# extra_info contains additional mapping information.
input_data, extra_info = converter.convert()

print("\nInput data for nodes:")
print(pd.DataFrame(input_data[ComponentType.node]))
print("\nInput data for generic branches:")
print(pd.DataFrame(input_data[ComponentType.generic_branch]))
```

### Other use cases

The examples above convert CGMES data into PGM input data, but they do not run a state estimation calculation. State estimation also requires measurement data. Measurement data will be converted if it is present in the CGMES dataset. Otherwise it must be added manually to the `input_data`. For testing purposes, the CGMES2PGM Suite provides functionality to automatically generate synthetic measurement data based on the provided SV Profile in the CGMES dataset.

For more advanced examples and detailed usage instructions, please refer to the official examples of the [CGMES2PGM Suite](https://github.com/SOPTIM/cgmes2pgm_suite/tree/main/example).

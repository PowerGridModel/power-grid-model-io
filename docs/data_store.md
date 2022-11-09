<!--
SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->

# Data Stores

Data stores are part of converters which perform the actions of loading and saving the data in the particular format of the data store.
There are two data stores in each converter for source and destination data.

The inheritance structure of data stores is as follows:

- Data Store
  - Json file store
  - Excel file store
    - Vision-excel file store
      - Gaia-excel file store

Of these, JSON file store, Vision-excel and Gaia-excel file stores are used in their respective converters.
A custom data store can be based on any of these existing stores. It has to convert the data to the specific data type. eg. tabular data.

## JSON file store

This is the JSON format used in power-grid-model. It is used by PGM JSON converter. See `power_grid_model.utils` for more info.

## Excel file store

It reads or saves the data in .xlsx excel files.
This format has one excel file with all components data in different spreadsheets.
Internally, the data is stored in the form of {py:class}`power_grid_model_io.data_types.TabularData` data type.

Also refer {py:class}`power_grid_model_io.data_stores.ExcelFileStore` for specific details.

### Vision-excel file store

The vision excel export specific operations in conversion  are carried out.
Eg. Vision exports has the units on the 2nd row.

#### Gaia file store

The gaia excel export specific operations are in this file store.
Gaia also exports are in multiple excel files.
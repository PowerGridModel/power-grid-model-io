

# Data Stores

File stores perform the actions of loading and saving the data in the particular format of the data store.

There are 2 types of data stores in power-grid-model-io. 

Inheritance structure

- Data Store
  - Json file store
  - Excel file store
    - Vision-excel file store
      - Gaia-excel file store

## JSON file store

This is the JSON format used in power-grid-model.

## Excel file store

It reads or saves the data in .xlsx excel files.
This format has one excel file with all components data in different spreadsheets.
Internally, the data is stored in the form of {py:class}`power_grid_model_io.data_types.TabularData` data type.

Also refer {py:class}`power_grid_model_io.data_stores.ExcelFileStore` for specific details.

### Vision-excel file store

The vision excel export specific operations in conversion  are carried out.
Eg. Vision exports has the units on the 2nd row.

### Gaia file store

The gaia excel export specific operations are in this file store.
Gaia file stores unlike
// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#define PGM_DLL_EXPORTS

#include <../dll_helpers/include/dll_helpers/import_export_helpers.h>
#include <../dll_helpers/include/dll_helpers/load_helpers.h>

#include "../include/c_converter_api.h"
#include "../include/vnf_pgm_converter.h"

#include "power_grid_model_c.h"

extern "C" {

PgmVnfConverter* PGM_VNF_create_converter(PGM_Handle* handle, char* file_buffer) {};

PGM_dataset_const_* PGM_VNF_get_input_data(PGM_Handle* handle, PgmVnfConverter* converter_ptr, PGM_dataset_const_* dataset){};

} // extern "C"

// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#ifndef C_CONVERTER_API_H
#define C_CONVERTER_API_H

#ifndef PGM_DLL_EXPORTS
#define PGM_DLL_EXPORTS
#endif

#include <../dll_helpers\include\dll_helpers\import_export_helpers.h>
#include <../dll_helpers\include\dll_helpers\load_helpers.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PgmVnfConverter PgmVnfConverter;


PGM_VNF_converter_API PgmVnfConverter* PGM_VNF_create_converter(PGM_Handle* handle, char* file_buffer);


PGM_VNF_converter_API PGM_dataset_const_* PGM_VNF_get_input_data(PGM_Handle* handle, PgmVnfConverter* converter_ptr, PGM_dataset_const_* dataset);

};
#endif //C_CONVERTER_API_H
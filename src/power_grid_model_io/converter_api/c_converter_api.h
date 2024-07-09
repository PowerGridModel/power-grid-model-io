// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PgmVFConverter PgmVFConverter;


PGM_F_converter_API PgmVFConverter* PGM_F_create_converter(PGM_Handle* handle, char* file_buffer);


PGM_F_converter_API PGM_dataset_const_* PGM_F_get_input_data(PGM_Handle* handle, PgmFConverter* converter_ptr, PGM_dataset_const_* dataset);

};
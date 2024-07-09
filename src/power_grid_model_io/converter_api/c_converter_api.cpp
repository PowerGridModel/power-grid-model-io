#include "c_converter_api.h"
#include "f_pgm_converter.h"


extern "C" {

PgmFConverter* PGM_F_create_converter(PGM_Handle* handle, char* file_buffer) {
    PgmFConverter* converter_ptr = new PgmFConverter();
    converter_ptr->f_file_buffer = file_buffer; // could be done while creating the object
    converter_ptr->parse_vnf_file();  

    return converter_ptr;
}

PGM_dataset_const_* PGM_F_get_input_data(PGM_Handle* handle, PgmFConverter* converter_ptr, PGM_dataset_const_* dataset){
    converter_ptr->pgm_input_data = dataset;
    converter_ptr->convert_input();

    return converter_ptr->pgm_input_data
}

} // extern "C"

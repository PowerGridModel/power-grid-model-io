<!--
SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->
# Pandapower release 3.2.0

`pandapower` made following breaking changes to its release 3.2.0:

1. `Load` attributes have been changed:
   * `const_i_percent` replaced with `const_i_p_percent` and `const_i_q_percent`
   * `const_z_percent` replaced with `const_z_p_percent` and `const_z_q_percent`
2. loss attributes in 3ph load flow for `res_line_3ph` and `res_trafo_3ph` have been changed:
   * `p_a_l_mw` changed to `pl_a_mw` and same for other phases
   * `q_a_l_mvar` changed to `ql_a_mvar` and same for other phases

In order to maintain backward compatibility, data files compatible with both `pandapower` versions have been moved
to separate folders **v3.1.2** and **v3.2.0**.

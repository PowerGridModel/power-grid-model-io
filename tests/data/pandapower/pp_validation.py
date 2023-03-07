# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from functools import lru_cache

import pandapower as pp


@lru_cache
def pp_net() -> pp.pandapowerNet:
    #  (ext #1)         shunt - [104]  - 3w - [105] - sym_gen
    #   |                                |
    #  [101] ---OO- [102] ---------- [103]
    #   |                                |
    #  -/-                          (load #31)
    #   |
    #  [106]
    net = pp.create_empty_network(f_hz=50)
    pp.create_bus(net, index=101, vn_kv=110)
    pp.create_bus(net, index=102, vn_kv=20)
    pp.create_bus(net, index=103, vn_kv=20)
    pp.create_bus(net, index=104, vn_kv=30.1)
    pp.create_bus(net, index=105, vn_kv=60)
    pp.create_bus(net, index=106, vn_kv=110)
    pp.create_ext_grid(
        net, index=1, in_service=True, bus=101, vm_pu=1, s_sc_max_mva=1e10, rx_max=0, va_degree=0, r0x0_max=0
    )
    pp.create_transformer_from_parameters(
        net,
        index=101,
        hv_bus=101,
        lv_bus=102,
        i0_percent=3.0,
        pfe_kw=11.6,
        vkr_percent=10.22,
        sn_mva=40,
        vn_lv_kv=20.0,
        vn_hv_kv=110.0,
        vk_percent=17.8,
        vector_group="Dyn",
        shift_degree=30,
        tap_side="hv",
        tap_pos=2,
        tap_min=-1,
        tap_max=3,
        tap_step_percent=2,
        tap_neutral=1,
        parallel=2,
    )
    pp.create_line(
        net, index=101, from_bus=103, to_bus=102, length_km=1.23, parallel=2, df=0.2, std_type="NAYY 4x150 SE"
    )
    pp.create_load(
        net, index=101, bus=103, p_mw=2.5, q_mvar=0.24, const_i_percent=26.0, const_z_percent=51.0, cos_phi=2
    )
    pp.create_switch(net, index=101, et="l", bus=103, element=101, closed=True)
    pp.create_switch(net, index=3021, et="b", bus=101, element=106, closed=True)
    pp.create_switch(net, index=321, et="t", bus=101, element=101, closed=True)
    pp.create_shunt(net, index=1201, in_service=True, bus=104, p_mw=0.1, q_mvar=0.55, step=3)
    pp.create_sgen(net, index=31, bus=105, p_mw=1.21, q_mvar=0.81)
    pp.create_asymmetric_sgen(
        net, index=32, bus=105, p_a_mw=0.1, p_b_mw=0.2, p_c_mw=3, q_a_mvar=0.01, q_b_mvar=0.01, q_c_mvar=0.01
    )
    pp.create_asymmetric_load(
        net, index=33, bus=105, p_a_mw=0.1, p_b_mw=0.2, p_c_mw=3, q_a_mvar=0.01, q_b_mvar=0.01, q_c_mvar=0.01
    )
    pp.create_ward(net, index=34, bus=105, ps_mw=0.1, qs_mvar=0.1, pz_mw=0.1, qz_mvar=0.1)
    pp.create_motor(
        net, bus=105, index=12, pn_mech_mw=0.1, cos_phi=0.9, loading_percent=80, efficiency_percent=90, scaling=0.8
    )
    pp.create_transformer3w_from_parameters(
        net,
        index=102,
        hv_bus=103,
        mv_bus=105,
        lv_bus=104,
        in_service=True,
        vn_hv_kv=20.0,
        vn_mv_kv=60.0,
        vn_lv_kv=30.1,
        sn_hv_mva=40,
        sn_mv_mva=100,
        sn_lv_mva=50,
        vk_hv_percent=10,
        vk_mv_percent=11,
        vk_lv_percent=12,
        vkr_hv_percent=1,
        vkr_mv_percent=2,
        vkr_lv_percent=4,
        i0_percent=0.1,
        pfe_kw=10,
        vector_group="Dyny",
        shift_mv_degree=30,
        shift_lv_degree=30,
        tap_side="lv",
        tap_pos=2,
        tap_min=1,
        tap_max=3,
        tap_step_percent=3,
        tap_neutral=2,
    )

    return net


@lru_cache
def pp_net_3ph() -> pp.pandapowerNet:
    """
    Creates a pandapower net used for validating 3 phase calculations

     (ext #1)         shunt - [104], sym_gen - [105], motor, ward, asym_load, asym_gen
      |                                |
     [101] ---OO- [102] ---------- [103]
      |                                |
     -/-                          (load #31)
      |
     [106]

    Returns:

    """

    net = pp.create_empty_network(f_hz=50)
    pp.create_bus(net, index=101, vn_kv=110)
    pp.create_bus(net, index=102, vn_kv=20)
    pp.create_bus(net, index=103, vn_kv=20)
    pp.create_bus(net, index=106, vn_kv=110)
    pp.create_ext_grid(
        net, index=0, in_service=True, bus=101, vm_pu=1, s_sc_max_mva=1e10, rx_max=0, va_degree=0, r0x0_max=0, x0x_max=1
    )
    pp.create_transformer_from_parameters(
        net,
        index=0,
        hv_bus=101,
        lv_bus=102,
        i0_percent=0.3,
        pfe_kw=51.6,
        vkr_percent=4.0,
        sn_mva=40,
        vn_lv_kv=20.0,
        vn_hv_kv=110.0,
        vk_percent=8.9,
        vector_group="YNyn",
        shift_degree=0,
        tap_side="hv",
        tap_pos=1,
        tap_min=-1,
        tap_max=3,
        tap_step_percent=2,
        tap_neutral=1,
        parallel=2,
        vk0_percent=8.9,
        vkr0_percent=4.0,
        mag0_percent=374531.83520599245,
        mag0_rx=0.47628073,
        si0_hv_partial=0.9,
        in_service=True,
    )
    pp.create_line_from_parameters(
        net,
        index=101,
        from_bus=103,
        to_bus=102,
        length_km=1.23,
        parallel=2,
        df=0.2,
        r_ohm_per_km=0.208,
        x_ohm_per_km=0.08,
        c_nf_per_km=261.0,
        r0_ohm_per_km=0.208,
        x0_ohm_per_km=0.08,
        c0_nf_per_km=261.0,
        max_i_ka=0.27,
    )
    pp.create_load(net, index=101, bus=103, p_mw=2.5, q_mvar=0.24, const_i_percent=0.0, const_z_percent=0)
    pp.create_switch(net, index=101, et="l", bus=103, element=101, closed=True)
    pp.create_switch(net, index=3021, et="b", bus=101, element=106, closed=True)
    pp.create_switch(net, index=321, et="t", bus=101, element=0, closed=True)
    pp.create_sgen(net, index=31, bus=103, p_mw=1.21, q_mvar=0.81)
    pp.create_asymmetric_sgen(
        net, index=32, bus=103, p_a_mw=0.1, p_b_mw=0.2, p_c_mw=3, q_a_mvar=0.01, q_b_mvar=0.01, q_c_mvar=0.01
    )
    pp.create_asymmetric_load(
        net, index=33, bus=103, p_a_mw=0.4, p_b_mw=0.5, p_c_mw=1, q_a_mvar=0.01, q_b_mvar=0.01, q_c_mvar=0.01
    )
    # TODO Shunt motor ward are not validated.
    # pp.create_shunt(net, index=1201, in_service=True, bus=103, p_mw=1, q_mvar=1, step=1)
    # pp.create_ward(net, index=34, bus=103, ps_mw=0.1, qs_mvar=0.1, pz_mw=0.1, qz_mvar=0.1)
    # pp.create_motor(
    #     net, bus=103, index=12, pn_mech_mw=0.1, cos_phi=0.9, loading_percent=80, efficiency_percent=90, scaling=0.8
    # )

    return net

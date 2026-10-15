<!--
SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
SPDX-License-Identifier: MPL-2.0
-->

# Sparlectra.jl: CGMES to power-grid-model

[Sparlectra.jl](https://github.com/Welthulk/Sparlectra.jl) (Julia, Apache 2.0) reads ENTSO-E CGMES 2.4.15 deliveries and writes a plain power-grid-model input dataset. Two ways: the Web UI and a Julia program.

Sparlectra.jl is released under the Apache-2.0 license and is sponsored by SOPTIM AG.

## Web UI

1. Start the Web UI (`start_webui.sh` / `start_webui.bat`) and open the **Case** page.
   - Lookup Installation details in the README file.
2. **Import case files**: select the CGMES delivery ZIP, or the EQ, SSH, TP and SV profile files together (they are packed into one ZIP). Alternatively type `cgmes:<alias>` into **Or type case file path**, for example `cgmes:microgrid_be`; this downloads the ENTSO-E test configuration once and packs it with its boundary set. Every upload is checked for completeness; a missing boundary set is named.
3. Select the case and click **Export as plain PGM**. The file `<case>.pgm.json` is written to the case directory and can be fetched with **Download selected case**.

## Julia

### Installation
```julia
import Pkg; Pkg.add("Sparlectra")
```

### Example
```julia
using Sparlectra

# change directory as needed
const OUT_DIR = joinpath(pwd(), "pgm_out")   # target directory
mkpath(OUT_DIR)

# A: ENTSO-E test configuration, downloaded once and cached under data/CGMES
cgmes = Sparlectra.CGMESImporter.fetchCGMESTestSet("microgrid_be"; outdir = OUT_DIR)
# B: own delivery, as a ZIP, a folder, or a vector of both
# cgmes = ["grid_EQ_SSH_TP_SV.zip", "boundary.zip"]

result = importCGMES(path = cgmes)
pgm_file = joinpath(OUT_DIR, "microgrid_be.pgm.json")
exportSCF(result.net; file = pgm_file, strict_pgm = true, source_format = "CGMES")
```

`strict_pgm = true` writes the PGM input dataset only; the export lists in a warning what it dropped. Without the flag the file is a Sparlectra Case Format file: the same PGM dataset plus a namespaced `sparlectra` block that PGM ignores.

## How the conversion works

CGMES is first built into a bus-branch network, then mapped onto the PGM vocabulary in SI units:

| CGMES | PGM |
|---|---|
| `TopologicalNode` (or `ConnectivityNode` plus switch states without a TP profile) | `node` |
| `ACLineSegment`, `SeriesCompensator` | `line` |
| `PowerTransformer`, 2 windings, phase shifters, lines between two nominal voltages | `generic_branch` |
| `PowerTransformer`, 3 windings | three `generic_branch` legs on a star node |
| retained `Switch`, `Breaker` | `link` |
| `SynchronousMachine`, `EquivalentInjection`, `StaticVarCompensator` | `sym_gen`, PV control as `voltage_regulator` |
| `ExternalNetworkInjection`, the reference machine | `source` |
| `EnergyConsumer`, `ConformLoad`, `NonConformLoad` | `sym_load` |
| `LinearShuntCompensator`, `NonlinearShuntCompensator` | `shunt` |

Every transformer is a `generic_branch`: `r1`, `x1`, `g1`, `b1` referenced to the to side, `k` the live off-nominal ratio including the tap position, `theta` the live phase shift in radians, `sn` the rating. The tap position is frozen into `k` and `theta`; the tap grid itself is not part of the dataset.

## What the PGM dataset does not carry

| Lost | Why |
|---|---|
| Tap-changer grid, position, band, neutral ratio | `generic_branch` has no tap model; only the live `k` and `theta` remain |
| Transformer vector group, zero-sequence data (`r0`, `x0`, star-point treatment) | `generic_branch` has no zero-sequence parameters |
| Names and mRIDs | PGM ids are integers |
| Distributed slack, participation factors | the reference becomes one `source` |
| IEC 60909 machine and feeder data beyond `sk`, `rx_ratio` | no PGM attribute |
| Controllers: tap control, remote voltage control, HVDC setpoints | not a PGM concept; HVDC converters arrive as injections |
| Three-winding grouping | the star equivalent stays, the grouping of the three legs is gone |

## Short circuit

The `generic_branch` carries positive-sequence data only, so a PGM dataset converted this way supports the symmetric (three-phase) short-circuit calculation and no asymmetric faults. This is a limitation of the target component, not of CGMES: the CGMES ShortCircuit profile does carry zero-sequence data for lines, transformer ends, machines and network injections. Sparlectra reads that data and keeps it in its own case format, but the plain PGM export has no place for it.

# autofm - 5G NR Multi-Level Dataset Specification

**Infrastructure / platform / RAN dataset for the FM phase**

Team AFM_Ericsson · European AI Hackathon 2026

---

## Table of contents

1. [Overview](#1-overview)
2. [Infrastructure level](#2-infrastructure-level)
3. [Platform level](#3-platform-level)
4. [RAN level](#4-ran-level)
5. [Topology and cross-level coupling](#5-topology-and-cross-level-coupling)
6. [Cadence](#6-cadence)
7. [Dynamics](#7-dynamics)
8. [Fault model](#8-fault-model)
9. [Labels](#9-labels-per-entity-per-rop)
10. [Output format](#10-output-format)
11. [Loader requirements](#11-loader-requirements)
12. [ML pipeline contract](#12-ml-pipeline-contract)

---

## 1. Overview

This spec defines the multi-level dataset the FM phase trains and evaluates on:
its levels, topology, time cadence, dynamics, fault model, labels, and output
format.

The dataset is **multi-level**, spanning the three levels of a virtualized 5G NR
O-RAN deployment, each described in its own section:

- **Infrastructure** (§2) - host (O-Cloud) metrics.
- **Platform** (§3) - container (network-function) metrics.
- **RAN** (§4) - 5G NR performance (PM) and configuration (CM) counters.

The **RAN** level follows 3GPP 5G NR models: the managed-object model (gNB with
the CU-CP / CU-UP / DU split, NR cells, carriers) aligns with the 3GPP 5G NRM
concepts (TS 28.541), and the performance-measurement model and Recording Output
Period (ROP) align with the 3GPP PM framework (TS 28.552). The **infrastructure**
and **platform** levels are O-Cloud host/container metrics referenced by 3GPP via
the O2 interface; 3GPP does not enumerate these metrics counter-by-counter, so
they follow standard host/container telemetry conventions. Specific counter and
object **names are implementation-specific identifiers**, not literal TS strings.

Multi-level telemetry is what lets the model avoid cross-level misdiagnosis: a
RAN symptom (packet loss) can originate at the infrastructure level (host CPU
throttling under a thermal spike), and only the combined view attributes it
correctly. Unlike raw telemetry, the dataset carries **fault labels** (type,
origin, time-to-failure, causal path), which make the FM prediction task
supervised.

**Provenance.** The RAN level mirrors a real 5G NR PM/CM export *schema* (group
names, counters, `fdn` topology, ROP cadence); the infrastructure and platform
levels and the multi-level fault model follow the FALCON approach (host/container
telemetry + cross-level fault injection). All values are produced by the dataset
build - no raw records are used.

All levels share a common file convention (§10): per-level, per-day
`;`-delimited CSVs with 6 metadata columns (`fdn, mecontext, cell,
rdiTimestamp, rdiTimestampTZ, ropCount`), so one loader reads every level.

> This spec defines the dataset in detail; the system design doc refers to it
> for schema, topology, and units.

---

## 2. Infrastructure level

Host / O-Cloud telemetry. **Entity = host**, each serving one or more gNBs
(default 2 gNBs/host). Emitted per ROP (§6).

| Group | Metric families | Entity |
|-------|-----------------|--------|
| **HostMetrics** | `host_cpu_util`, `host_cpu_freq`, `host_mem_used`, `host_disk_io`, `host_net_rx/tx`, `host_net_err`, `host_temp`, `host_power` | host |

- Names are standard host telemetry (node-exporter style), **not 3GPP**; the
  exact metric set is configurable (default ~10-15 metrics).
- All scalar; no distribution-vector counters at this level.
- The host level is the origin of **CPU-stress** faults (§8), which propagate
  down the host → container → RAN chain (§5).

---

## 3. Platform level

Container / network-function telemetry. **Entity = NF container**, one per CU
and one per DU. Emitted per ROP (§6).

| Group | Metric families | Entity |
|-------|-----------------|--------|
| **ContainerMetrics** | `ctr_cpu_util`, `ctr_mem_used`, `ctr_net_rx/tx`, `ctr_fs_util`, `ctr_restart_count` | CU / DU container |

- Names are standard container telemetry (cAdvisor style), **not 3GPP**;
  configurable set (default ~10-15 metrics).
- All scalar.
- The platform level is the origin of **memory-stress** faults (§8), which
  surface as NF instability and RAN symptoms.

---

## 4. RAN level

5G NR performance (PM) and configuration (CM) data, largely following 3GPP 5G NR
models (with the NSA/transport caveats noted in §5).

### 4.1 PM counter groups

Daily-rollup CSVs, `;`-delimited; distribution counters stored as
**comma-separated vectors inside a single field**. Entities are **cells** for
CU/DU/SectorCarrier and **transport links** for S1U/Rp.

| Group | Total cols | Scalar | Vector | Empty/other | Entity | Role |
|-------|-----------:|-------:|-------:|------------:|--------|------|
| **NRCellDU** | 280 | 152 | 95 | 27 | cell | Radio/MAC/PHY - richest group |
| **NRCellCU** | 282 | 242 | 1 | 33 | cell | Mobility / ENDC / handover |
| **NRSectorCarrier** | 28 | 6 | 16 | 0 | cell (`cell` blank; id in `fdn`) | Carrier power + **downtime** |
| **S1ULink** | 11 | 0 | 5 | 0 | transport link (`cell` blank) | Transport (PDCP volumes) |
| **RpUserPlaneLink** | 32 | 0 | 15 | 11 | transport link (`cell` blank) | Transport + **UL packet loss** |

(Columns = 6 metadata + scalar + vector + empty/other. "Empty/other" are
all-null or non-numeric columns in this window. Column counts are from the
reference schema.)

> **`cell` column caveat.** Only NRCellDU/NRCellCU populate the `cell` column.
> For NRSectorCarrier, S1ULink, RpUserPlaneLink (and the infra/platform groups)
> the `cell` field is blank - entity identity must be parsed from `fdn`.

> **Transport is per-link, not per-cell.** On a normal day S1ULink has ~29,000
> rows across ≈17,800 distinct link FDNs (≈2 rows per FDN per day, matching the
> two-timestamp pattern; some links report once) and RpUserPlaneLink ~4,100 rows
> across ≈2,300 distinct link FDNs - IP-pair links under `GNBCUUPFunction`, not
> 309 cells. They do not map 1:1 to cells and require an aggregation rule (roll
> up to the owning gNB / CUUP) before joining to a cell-indexed tensor (§10).

### 4.2 Counter families

The counters use **implementation-specific `pm*` names** over `fdn`-addressed
managed objects, mapping to 3GPP NR measurement families (TS 28.552) rather than
being literal TS strings. The same loaders and the `falcon` baseline parse the
PM/CM CSVs directly. The dataset models a curated, high-signal subset per group
(full fidelity - all **247 populated NRCellDU counters = 152 scalar + 95
vector** - is optional via config).

Counters marked **(confirmed)** are present in the reference schema headers;
**(illustrative)** are representative names to be confirmed before use.

| Group | Modelled counter families (examples) | Type |
|-------|--------------------------------------|------|
| **NRCellDU** | `pmActiveUeDlMax/Sum/Samp` *(confirmed)*, `pmMacHarqDlFail/UlFail` *(confirmed)*, `pmRadioRaCbFailMsg*` *(confirmed)*, `pmRadioPucchFailSr` *(confirmed)*, `pmPagDiscarded` *(confirmed)*, `pmCellDowntimeAuto/Man` *(confirmed)*, MAC/PDCP volume + PRB-utilisation counters *(illustrative)* | scalar + vector |
| **NRCellCU** | `pmHoPrepFail*` *(confirmed)*, `pmHoPrepRejIn*` *(confirmed)*, `pmEndcRelUeAbnormal*` *(confirmed)*, `pmEndcSetupFailReconfigFail` *(confirmed)*, `pmDrbEstabFailIntgProt` *(confirmed)*, RRC-connection-establishment counters *(illustrative)* | scalar (+ 1 vector: `pmCaConfigDlSumEndcDistr`) |
| **NRSectorCarrier** | `pmSectorCarrierDowntime/Auto/Man` *(confirmed)*, `pmMicroTxSleepTime` *(confirmed)*, `pmEmfPwrBackoff*Distr` *(confirmed)* | scalar + vector |
| **S1ULink** | `pmPdcpPktRecDl5qi` *(confirmed)*, `pmPdcpPktRecDlDisc5qi` *(confirmed)*, `pmPdcpVolTransUl5qi` *(confirmed)* | vector (per-5QI) |
| **RpUserPlaneLink** | `pmPdcpPktLossUlQos` *(confirmed)*, `pmPdcpPktLossUlToDiscQos/ToQos` *(confirmed)*, `pmPdcpPktTransDl*Qos` *(confirmed)* | vector (per-QoS) |

**Distribution-vector counters** are emitted as comma-separated bins with a
**fixed per-counter bin count** (e.g. 16- or 40-bin histograms), matching how
3GPP distribution-type measurements are reported. Bins are **driven by the same
latent cell state** as the scalar counters (shifting mass across bins as
load/fault state changes) rather than independent noise, so the histograms move
coherently with the scalars.

Per-counter value ranges are set from the reference schema (aggregate ranges
only). Those schema ranges are **daily-rollup** ranges and are mapped to the
15-min ROP cadence by counter type:

- **Count/volume counters** (events, bytes, UE sums) are **rescaled by
  occupancy** - distributed across the ~96 ROPs of a day rather than divided
  uniformly, since activity concentrates in busy ROPs.
- **Time counters** (downtime in seconds) are **clamped to the ROP length**:
  `pmSectorCarrierDowntime` cannot exceed **900 s per 15-min ROP** (its daily
  `[0, 3600]` range is a per-*day* total, not a per-ROP value).

`pmActiveUeDlMax` is small integers; PDCP volumes are large counts.

### 4.3 CM attributes

33 configuration object types per day. Largest: `NRCellDU` (2360 attrs × 316
rows), `NRCellCU` (2358 × 316), `SectorEquipmentFunction` (818 × 488),
`NRSectorCarrier` (115), `NRFrequency` (83). Plus ~25 small per-cell profile
tables (`DrxProfile`, `Rach`, `AdmissionLimit`, ...). CM lists ~316 configured
cells vs 309 reporting in PM: the 7-cell surplus are configured-but-not-reporting
cells (provisioned, no PM in this window).

### 4.4 Reference-data notes

These describe the **source export** the RAN schema is modelled on (and that the
loader, §11, must tolerate) - not the generated dataset's output, which is clean
and per-ROP.

- **Reference rows/day.** The source daily file holds **two rows per entity per
  day** (one per `rdiTimestamp`: the ROP date and the following midnight), each a
  rollup of ~96 fifteen-minute ROPs (`ropCount` ≈ 96) - e.g. NRCellDU ~618 rows
  (309 cells × 2).
- **10-01 is an outlier - exclude it.** The `2025-10-01` files report ~14,787
  rows, 48 distinct timestamps and `ropCount` 2-4, i.e. a non-standard dump; the
  loader excludes 10-01 by default.
- **Data-quality hazards.** The reference CSVs are messy: a `.txt` duplicate, a
  doubled-up filename (`NRCellCU_attributesMcfbCellProfile...`), a date typo
  (`2025-1-01`), a missing-date file (`NRCellCU_counters.csv`), a mangled
  extension (`..._2025-10-27csv.csv`), some groups missing days (NRCellCU 29/31,
  S1ULink 30/31), and ragged rows (trailing `;`). The loader (§11) is defensive
  about all of these.

---

## 5. Topology and cross-level coupling

The three levels connect in a **host → container → RAN** hierarchy. The dataset
topology is configurable and seeded. **Default: 10 hosts, 20 gNBs (2/host), 3
cells/gNB = 60 cells**; scalable to the full 107 gNBs / 309 cells.

```
host_h              (h = 1..H, infrastructure, HostMetrics)
  └─ gNB_k          (k = 1..G, 2 gNBs/host default)
       ├─ CU container, DU container        (platform, ContainerMetrics)
       ├─ cell_{k,c}  (c = 1..3)   → NRCellDU + NRCellCU + NRSectorCarrier (1:1:1)
       └─ transport   → S1ULink, RpUserPlaneLink  (per gNB CUUP)

RAN addressing (parsed from fdn):
SubNetwork=NET
  └─ SubNetwork=RAN
       └─ SubNetwork={REGION_A, REGION_B, REGION_C, REGION_D}
            └─ gNB  (MeContext / ManagedElement: GNB001, GNB002, ...)
                 ├─ GNBDUFunction      → NRCellDU, NRSectorCarrier
                 ├─ GNBCUCPFunction    → NRCellCU
                 └─ GNBCUUPFunction    → S1ULink, RpUserPlaneLink
```

> All node names (host, gNB, cell, SubNetwork identifiers) are **illustrative
> placeholders**, not real network identifiers. The two blocks show the same
> gNBs: `gNB_k` in the deployment view is `GNB001`, `GNB002`, ... in the RAN
> addressing view.

- **Cross-level mapping.** Each gNB's CU/DU NFs run as containers (platform) on a
  host (infrastructure); each gNB owns RAN cells and transport links. A host maps
  to its gNBs, each gNB to its containers and its cells/transport. This chain
  carries cross-level faults (§8).
- The `fdn` nests SubNetworks (`SubNetwork=NET, SubNetwork=RAN,
  SubNetwork=REGION_B, ...`): the top two are fixed, the regions are children.
- NRCellCU/NRCellDU/NRSectorCarrier for a given cell share the same (gNB,
  cell-id). **107 gNBs**, **309 cells** of each NR type; cells per gNB
  `{1:3, 2:9, 3:94, 6:1}` - the norm is 3 (tri-sector site).
- **Transport interfaces.** `S1ULink` reflects an **NSA / EN-DC** deployment
  (LTE/EPC S1-U user plane; pure 5G SA would be NG-U / F1-U). `RpUserPlaneLink`
  is an **implementation-specific transport managed object**, not a 3GPP NRM
  class. Both are user-plane bearers under the CU-UP.

### 5.1 GAT graph

The GAT operates on a **heterogeneous graph** over all three levels:

- **Node types:** `host` (infra), `container` (CU/DU platform NFs), `gnb`,
  `cell` (CU/DU/SectorCarrier triple), `transport` (S1U/Rp per gNB CUUP),
  `subnetwork`.
- **Edge types:**
  - `cell -- cell` for **sibling cells under the same gNB** (intra-site).
  - `cell -- gnb`, `container -- gnb`, `gnb -- host`, `gnb -- transport`,
    `gnb -- subnetwork` (containment, carrying the host → container → RAN chain).
  - `cell -- cell` **inter-site handover neighbours**: a configurable kNN set
    per cell (default k=4), *not* an all-pairs clique. (No neighbour list exists
    in the schema; neighbours are assigned by a seeded geographic layout - a
    modelling choice.)

Explicit neighbour edges (rather than "all gNBs in a SubNetwork are peers") keep
the graph sparse and avoid a ~5,600-edge clique at the top level.

**Default ML-tensor shape:** 60 cells × 1,344 ROPs × F, where **F** is the
per-cell feature dimension = RAN scalar counters (NRCellDU/NRCellCU/
NRSectorCarrier) + expanded distribution-vector features (§12.4) + broadcast
per-gNB transport counters (§10) + the host (infra) and container (platform)
metrics attached via the host → container → RAN chain. F is on the order of a few
thousand (247 is NRCellDU's own counter count alone); the exact value is produced
by the feature builder and recorded in `meta.json`.

**F is the raw, pre-compression input dimension - not what the classifier sees.**
The MAE (design doc §6) compresses each per-ROP feature vector to a **32-dim
latent**; the GAT/PatchTST/heads operate on that latent, so the operative
dimensionality downstream is ~32, well below the sample count (≈80k cell-ROPs at
smoke scale). Raw F being large is expected and is the reason the pipeline leads
with a learned-compression stage. The GAT also consumes the typed edge index.

---

## 6. Cadence

The reference RAN form is a daily aggregate (`ropCount`≈96 per row). The
**dataset is at the native 15-min ROP** (`ropCount=1`, **96 rows/entity/day**) -
the per-ROP level the daily rollup aggregates, **96× finer** - which is required
for forecasting at ROP resolution. All levels (infra/platform/RAN) are emitted at
this ROP cadence.

- **Default:** 15-min ROP, 14 days → 1,344 steps/entity.
- Horizon `m` (design doc, m=5..120) is measured in **ROPs** (15 min each),
  i.e. 75 min .. 30 h ahead.

---

## 7. Dynamics (observed properties)

Per metric, the series exhibits the following statistical behaviour (knobs in
`config.yaml`):

- **Mean-reverting noise:** metrics fluctuate around an entity-specific baseline
  and revert toward it (Ornstein-Uhlenbeck-like autocorrelation), with volatility
  scaled to each metric's range.
- **Diurnal modulation:** UE / PRB / volume / load metrics follow a busy-hour vs
  night cycle (period = 96 ROPs = 24 h).
- **Cross-level coupling with delay:** disturbances propagate down the host →
  container → RAN chain, e.g. `host CPU/temp → container CPU → DU
  scheduling/HARQ → CU mobility/abnormal-rel → transport loss/discard`, and the
  RAN-internal `SectorCarrier → DU → CU → transport` chain, each hop with a
  per-hop lag (1-3 ROPs).
- **Configuration dependence:** selected CM attributes (DRX, admission limits,
  RACH config) shift RAN baselines, so CM is reflected in PM behaviour.

---

## 8. Fault model

The dataset carries explicit **fault labels** that raw telemetry lacks. Each
fault is aligned to the fault-signal metrics in its origin level, so labels are
grounded in metric behaviour.

| Fault type | Origin level | Primary signature | Propagates to |
|------------|--------------|-------------------|---------------|
| **Normal** | - | baseline behaviour | - |
| **CPU Stress** | Infrastructure (host) | `host_cpu_util` ↑, `host_cpu_freq` ↓, `host_temp` ↑ | container CPU ↑ → DU scheduling/HARQ fail ↑ → RAN latency/volume ↓ |
| **Memory Stress** | Platform (container) | `ctr_mem_used` ↑, `ctr_restart_count` ↑ | NF instability → CU abnormal-release ↑, DU volume ↓ |
| **Cell/Carrier Downtime** | NRSectorCarrier (RAN) | `pmSectorCarrierDowntime*`, `pmCellDowntimeAuto` ↑ | DU volume ↓, CU abnormal-release ↑ |
| **Accessibility (RACH) fail** | NRCellDU (RAN) | `pmRadioRaCbFailMsg*`, `pmRadioPucchFailSr` ↑ | CU RRC-establishment failures ↑ |
| **Mobility / ENDC fail** | NRCellCU (RAN) | `pmHoPrepFail*`, `pmEndcRelUeAbnormal*` ↑ | neighbour-cell HO counters ↑ |
| **Packet Loss (transport)** | RpUserPlaneLink (RAN) | `pmPdcpPktLossUlQos`, `...ToDiscQos` ↑ | DU `pmMacHarq*Fail` ↑, CU abnormal-release ↑ |
| **Cascading (composite)** | host → container → RAN | full cross-level chain | all of the above |

The **CPU Stress** and **Memory Stress** faults are the cross-level cases: they
originate in the infra/platform levels and surface as RAN symptoms - exactly the
misdiagnosis a RAN-only view gets wrong. "Propagates to" effects reference
metrics in the modelled set (§2-§4).

**Label vs feature decoupling (leakage control).** The fault label corresponds to
a **latent severity state** `z(t) ∈ [0,1]` per entity that is **not an observable
metric**. The signature metrics correlate with `z` only **partially and noisily**
(a monotone-but-noisy relationship with per-metric gain `g < 1`, plus the
mean-reverting noise of §7), so **no input feature equals or thresholds the
label**. The hard class boundary sits on the hidden `z`, not on any metric the
model sees - preventing a model from recovering the labeling rule from its own
inputs (which would make accuracy a tautology). `z` is **recorded in `meta` for
audit and is never a feature**.

This applies to **every** fault class, including Packet-Loss and the broadcast
transport counters (§10.1): the broadcast `pmPdcpPktLoss*` feature is itself a
noisy, gain-limited function of the transport `z`, so even though the fault label
is attached to all sibling cells under the gNB (§9), no cell sees a counter that
directly equals or thresholds its label. The broadcast makes transport faults
**gNB-localizable but not trivially separable** - localization is therefore
scored at gNB granularity (§12.5), and classification still relies on the
noisy-`z` signal rather than a clean feature→label shortcut.

**Degradation ramp and horizon alignment.** Before a label flips, `z` rises
across a **ramp**. To keep early prediction a genuine forecasting task at every
evaluated horizon, the **ramp length is kept ≥ the maximum horizon**
(`ramp_rops ≥ max(m)`; default ramp **120-150 ROPs** for `m` up to 120). A faint
precursor therefore exists throughout the evaluated range. Fault onset (ramp
start) follows a slow per-entity hazard, so pre-ramp drift is also present. The
horizon-degradation curve is defined over `m ≤ ramp length`.

**Window vs ramp vs horizon.** The lookback window (§12.3, default 60 ROPs) need
not contain the whole ramp. At horizon `m`, the model sees the window ending `m`
ROPs before the flip, so for `m` approaching the ramp length only the *earliest,
faintest* part of the ramp is visible - which is what makes the
horizon-degradation curve informative. Horizons whose precursor falls outside the
window are reported as the natural accuracy floor, not excluded.

**Fault mix (default):** the dataset has **8 classes (Normal + 7 faults)**.
Default ~51% Normal with the remaining ~49% split across the 7 fault classes
(CPU Stress, Memory Stress, Cell/Carrier Downtime, RACH, Mobility/ENDC, Packet
Loss, Cascading) at ~7% each. The imbalance toward Normal gives the evidential /
conformal components a meaningful calibration setting; exact proportions are set
in `config.yaml`.

---

## 9. Labels (per entity, per ROP)

Six labels, emitted over the **union of node types** (host, container, cell,
carrier, transport link), so a fault at any level can name its true origin.

| Label | Meaning |
|-------|---------|
| `fault_type` | one of the 8 classes above (int, 0 = Normal, 1-7 = fault types) |
| `fault_origin_node` | FDN of the originating node (host / container / cell / carrier / transport link; `none` when Normal) |
| `time_to_failure_rop` | ROPs until label flips (counts down during ramp; sentinel `-1` when Normal / not degrading) |
| `causal_path` | ordered FDN chain, pipe-delimited in the CSV, e.g. `host=HOST01\|container=GNB001-DU\|NRCellDU=GNB001-2\|RpUserPlaneLink=<ip_ip>` |
| `is_degrading` | 1 during the ramp window, else 0 |
| `ramp_progress` | 0.0 → 1.0 across the ramp (0.0 when not degrading) |

In the primary `fm/` CSV files these are plain columns (FDN strings,
pipe-delimited path). In the optional derived Parquet view (§10.1) node FDNs are
mapped to int codes via a codebook in `meta.json`, and `causal_path` becomes a
list column.

A fault whose origin is not a cell (host, container, transport) is attached to
the affected cells (via the host → container → RAN mapping of §5) so it is
learnable from the cell-indexed tensor, while `fault_origin_node` still records
the true origin. Specifically: a **host** fault attaches to all cells under all
gNBs on that host; a **container** fault attaches to the cells of its gNB (CU
container → all the gNB's cells; DU container → the DU's cells); a **transport**
fault attaches to the gNB's cells. The attachment set is recorded alongside the
label.

**TTF masking.** `time_to_failure_rop = -1` is a **mask sentinel, not a
regression target**. The TTF (quantile) head is trained with its loss **masked
to degrading rows** (`is_degrading = 1`); Normal/non-degrading rows contribute no
TTF loss. Omitting this mask would collapse the regressor toward `-1`.

Because TTF counts down over the ramp (§8), it is **bounded in `[0, ramp_len]`**
(default ≤ 150 ROPs); the head cannot - and is not asked to - predict failures
beyond the ramp. Consequently **TTF RMSE (design doc §10) is defined only over
degrading rows within the ramp**, not over the full horizon; a low RMSE should be
read as "accurate once a precursor exists," not "accurate at arbitrary lead."

**Window labeling.** For a windowed sample (§12), the target is the window's
**last timestep** (causal, no look-ahead); a window straddling a ramp-to-flip
transition is labeled by its final step, so pre-flip windows carry the pre-flip
label with `is_degrading`/`ramp_progress` describing the approach.

---

## 10. Output format

The **RAN trees (`pm/`, `cm/`) follow a conventional 5G NR PM/CM export
layout**, so a standard PM/CM CSV loader (and the baseline pipeline) reads those
trees directly. The dataset **extends** that layout with three more trees in the
same CSV convention: `infra/` and `platform/` (host and container metrics) and
`fm/` (fault labels).

```
dataset/
├── infra/<YYYY-MM-DD>/
│   └── HostMetrics_<date>.csv                 # host (O-Cloud) metrics
├── platform/<YYYY-MM-DD>/
│   └── ContainerMetrics_<date>.csv            # CU/DU container metrics
├── pm/<YYYY-MM-DD>/
│   ├── NRCellCU_counters_<date>.csv
│   ├── NRCellDU_counters_<date>.csv
│   ├── NRSectorCarrier_counters_<date>.csv
│   ├── S1ULink_counters_<date>.csv
│   └── RpUserPlaneLink_counters_<date>.csv
├── cm/<YYYY-MM-DD>/
│   ├── <Object>_attributes_<date>.csv         # per CM object type
│   └── ...
├── fm/<YYYY-MM-DD>/
│   └── <Level>_labels_<date>.csv              # one per level, keyed by (fdn, rdiTimestamp)
└── meta.json                                   # seeds, config hash, schema, norm stats
```

- **RAN trees (`pm/`, `cm/`)** use the conventional export format: `;`-delimited,
  fixed column order, the 6 metadata columns (`fdn, mecontext, cell,
  rdiTimestamp, rdiTimestampTZ, ropCount`), distribution counters as
  comma-separated vectors. Parseable by a standard PM/CM CSV loader.
- **Infra / platform trees** use the **same `;`-delimited CSV convention** but a
  reduced metadata set - `fdn, rdiTimestamp, rdiTimestampTZ, ropCount` populated,
  `mecontext`/`cell` left blank (hosts and containers are not RAN managed
  elements). Entity identity is in `fdn`, with a per-level addressing scheme:
  `fdn=host=HOST01` for infrastructure, `fdn=container=GNB001-DU` /
  `=GNB001-CU` for platform. All metrics are scalar (no distribution vectors).
- **`fm/` label files** are emitted **per level** (`HostMetrics_labels_*`,
  `ContainerMetrics_labels_*`, `NRCellDU_labels_*`, ...), each row joined to its
  data row by `(fdn, rdiTimestamp)` and carrying the six labels of §9. Keeping
  labels in a separate tree leaves the data trees in their native format.

  Note the three coexisting delimiters in RAN files: `;` between fields, `,`
  within a distribution-vector field, and `|` within `causal_path` (§9) - loaders
  must handle all three. Infra/platform/`fm` files use only `;` (and `|` for
  `causal_path`).
- **Cadence:** per-day foldering across all trees; one row per entity per ROP
  (§6), so a day holds up to 96 rows per entity (vs the reference RAN export's
  two daily-aggregate rows).

A single loader reads all five trees: it keys every row on `fdn` + `rdiTimestamp`
and tolerates the blank RAN-only columns at the infra/platform levels.

### 10.1 Optional derived ML view

For training, a **derived** Parquet view can be materialised from the CSVs (a
cache, not the primary format):

- `features.parquet` - tensor `(entity, rop, feature)` across all levels; cell
  counters indexed directly, transport counters aggregated per gNB CUUP and
  broadcast to the gNB's cells (§4.1), host/container metrics attached via the
  host → container → RAN chain, with a parallel **mask channel** (0 = present,
  1 = missing) for sensor-dropout and missing days.
- `labels.parquet` - the six labels; `topology.json` - typed nodes + edges for
  the GAT.
- Dtypes: metrics `float32`, mask `uint8`, `fault_type`/`fault_origin` int codes,
  ROP index `int32`.

Everything is seeded from `config.yaml`; `--quick` emits a tiny smoke dataset
(e.g. 2 gNBs × 1 day) in the same layout.

---

## 11. Loader requirements

The loader is defensive by design: it excludes the 10-01 outlier, tolerates
missing days (NRCellCU 29/31, S1ULink 30/31), handles ragged rows and the
misnamed/duplicate files noted in §4.4, parses entity identity from `fdn` when
the `cell` column is blank, and reads all four data trees (infra / platform / pm
/ cm) through one code path.

---

## 12. ML pipeline contract

How the dataset is consumed for training and evaluation. These rules keep
reported metrics trustworthy (no leakage) and reproducible.

### 12.1 Split (leakage-safe)

- **Chronological**: 60% train / 20% val / 20% test by ROP time, with val and
  test strictly **after** train. No shuffling across the time boundary.
- **Self-supervised pretraining uses the train split only.** MAE pretraining is
  unsupervised and easy to leak - it must not see val/test windows, or the
  representation (and the reconstruction-error anomaly score it feeds downstream)
  has observed test-time distribution.
- **Entity policy** (configurable, both reported):
  - *temporal* (default): all entities in all splits, split only on time - tests
    forecasting on known entities.
  - *entity-holdout*: a disjoint set of gNBs reserved for val/test - tests
    generalization to unseen cells. Inter-site kNN handover edges (§5.1) are
    **not drawn across the holdout boundary**.
- A **calibration slice** is held out for Conformal Prediction, never used for
  gradient updates. Taken from the end of train (adjacent to val/test); under
  temporal drift calibration and test are only approximately exchangeable, so the
  coverage guarantee is **approximate, not exact** (§12.5).

### 12.2 Normalization (fit on train only)

- Normalization is **global per metric**, not per entity: one mean/std (and
  `log1p` for heavy-tailed count/volume metrics - PDCP volumes, UE sums) per
  metric column, estimated across all **train** entities and ROPs.
- Statistics computed on the **train split only**, applied unchanged to
  val/test/calibration. Under entity-holdout (§12.1) held-out entities are
  normalized with the same global-per-metric stats, so there is no per-entity
  stat to leak; the **entity-specific baseline (§7) is left in the signal** for
  the model to learn, not normalized away.
- Stored in `meta.json`.

### 12.3 Windowing

- Fixed-length lookback windows for PatchTST (default length **60 ROPs**,
  `patch_size=5` → 12 patches), stride 1 within an entity, **never crossing
  entity boundaries**.
- Causal labeling per §9 (target = window's last step). Windows built **after**
  the split so none spans the train/val/test boundary.
- Training uses stride 1; **val/test use non-overlapping windows**
  (stride = window length) so near-duplicate windows do not inflate the effective
  sample size and metrics stay honest.

### 12.4 Distribution-vector representation

Each RAN distribution-vector counter enters the feature tensor as its bin vector
**normalized to a distribution** (bins sum to 1, separating composition from
scale) *plus* three reduced summaries (mean-bin, entropy, top-bin fraction) and
the total count as a separate standardized scalar. Per-bin values are not
individually standardized. Config can switch to summaries-only to cut input
dimension. The chosen representation and per-group feature dimension are recorded
in `meta.json`.

### 12.5 Imbalance and metrics

- The fault-type head uses **class-balanced loss** (inverse-frequency weights;
  focal loss optional) for **training only**. The conformal calibration slice
  (§12.1) uses the **natural (unweighted) class distribution** - reweighting is
  not applied to calibration scores, so the coverage guarantee is computed on the
  distribution the model actually meets at inference.
- **Headline metric is macro-F1 and per-class recall, not accuracy.** The trivial
  Normal-always baseline achieves ~51% accuracy but near-zero macro-F1 (it scores
  nothing on the 7 fault classes); macro-F1 is therefore the headline so a
  majority-class predictor cannot look strong.
- Localization scored as top-k node accuracy; for non-cell-origin faults, where
  sibling cells share broadcast host/container/transport features (§5, §10),
  localization is scored at **gNB / host granularity** as appropriate.

### 12.6 Scale and per-class counts

- **Default (smoke) scale:** 60 cells × 1,344 ROPs ≈ 80k cell-ROPs; after
  windowing and the ~7% per-fault mix, rarer classes yield only a few hundred
  windowed examples - adequate for a pipeline smoke test, **not** final training.
- **Training scale:** 309 cells × longer span (config), to give each fault class
  ≥ a few thousand windowed examples. Realized per-class counts are reported in
  `meta.json`; targets are set in `config.yaml`.

### 12.7 Determinism

- Independent seeds for topology, metric dynamics, fault onset/injection, and the
  train/val/test split, all derived from one master seed and recorded in
  `meta.json`.
- DataLoader workers are seeded per-worker to avoid cross-worker correlation.
- The **data build** (`build_data`) is byte-reproducible given the same master
  seed and `config.yaml`. **Training** is seed-stable but **not** guaranteed
  bitwise-identical across GPU runs or hardware (CUDA/cuDNN atomics, DDP
  all-reduce ordering); force deterministic algorithms for closest
  reproducibility.

---

Team AFM_Ericsson - European AI Hackathon 2026

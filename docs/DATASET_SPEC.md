# autofm - Dataset Specification

**PM/CM dataset for the FM phase**

Team AFM_Ericsson · European AI Hackathon 2026

---

## Table of contents

1. [Overview](#1-overview)
2. [Schema](#2-schema)
3. [Dataset topology](#3-dataset-topology)
4. [Counter families](#4-counter-families)
5. [Cadence](#5-cadence)
6. [Dynamics](#6-dynamics)
7. [Fault model](#7-fault-model)
8. [Labels](#8-labels-per-entity-per-rop)
9. [Output format](#9-output-format)
10. [Loader requirements](#10-loader-requirements)
11. [ML pipeline contract](#11-ml-pipeline-contract)

---

## 1. Overview

This spec defines the PM/CM dataset the FM phase trains and evaluates on: its
schema, topology, time cadence, dynamics, fault model, labels, and output
format.

The dataset is **3GPP-aligned 5G NR** performance (PM) and configuration (CM)
data over a cell / carrier / transport topology. The managed-object model (gNB
with the CU-CP / CU-UP / DU split, NR cells, carriers) follows the 3GPP 5G NRM
(TS 28.541), and the performance measurements and the Recording Output Period
(ROP) follow the 3GPP performance-measurement framework (TS 28.552). Unlike raw
counter data, it carries **fault labels** (type, origin, time-to-failure, causal
path), which make the FM prediction task supervised.

> This spec defines the dataset in detail; the system design doc refers to it
> for schema, topology, and units.

---

## 2. Schema

### 2.1 PM counter groups (daily-rollup CSVs, `;`-delimited)

Distribution counters are stored as **comma-separated vectors inside a single
field**. Every group carries 6 metadata columns:
`fdn, mecontext, cell, rdiTimestamp, rdiTimestampTZ, ropCount`.

Each normal file holds **one aggregated row per entity per day** (the row is the
rollup of ~96 fifteen-minute ROPs; `ropCount` ≈ 96). Entities are **cells** for
CU/DU/SectorCarrier and **transport links** for S1U/Rp.

| Group | Total cols | Scalar | Vector | Empty/other | Rows/day | Entity | Role |
|-------|-----------:|-------:|-------:|------------:|---------:|--------|------|
| **NRCellDU** | 280 | 152 | 95 | 27 | ~618 (309 cells × 2 ts) | cell | Radio/MAC/PHY - richest group |
| **NRCellCU** | 282 | 242 | 1 | 33 | ~618 | cell | Mobility / ENDC / handover |
| **NRSectorCarrier** | 28 | 6 | 16 | 0 | ~618 | cell (`cell` col blank; id in `fdn`) | Carrier power + **downtime** |
| **S1ULink** | 11 | 0 | 5 | 0 | ~29,000 | transport link (`cell` blank) | Transport (PDCP volumes) |
| **RpUserPlaneLink** | 32 | 0 | 15 | 11 | ~4,100 | transport link (`cell` blank) | Transport + **UL packet loss** |

(Columns = 6 metadata + scalar + vector + empty/other. "Empty/other" are
all-null or non-numeric columns in this window.)

> **`cell` column caveat.** Only NRCellDU/NRCellCU populate the `cell` column.
> For NRSectorCarrier, S1ULink and RpUserPlaneLink the `cell` field is blank -
> entity identity must be parsed from `fdn`.

> **Transport is per-link, not per-cell.** On a normal day S1ULink has ~29,000
> rows (≈17,800 distinct link FDNs) and RpUserPlaneLink ~4,100 rows (≈2,300
> distinct link FDNs) - IP-pair links under `GNBCUUPFunction`, not 309 cells.
> They do not map 1:1 to cells and require an aggregation rule (roll up to the
> owning gNB / CUUP) before joining to a cell-indexed tensor (see §9).

> **10-01 is an outlier - exclude it.** The `2025-10-01` files report ~14,787
> rows, 48 distinct timestamps and `ropCount` 2-4, i.e. a non-standard dump
> rather than the daily rollup seen on every other day. The loader excludes
> 10-01 by default.

### 2.2 Topology (parsed from `fdn`)

```
SubNetwork=CH
  └─ SubNetwork=RAN
       └─ SubNetwork={MRAN11, MRAN19, NRAN11, NRAN19}
            └─ gNB  (MeContext / ManagedElement, 107 distinct: AVUG03, BECO01, ...)
                 ├─ GNBDUFunction
                 │    ├─ NRCellDU         ─┐
                 │    └─ NRSectorCarrier  ─┤  1:1:1 per cell (309 of each)
                 ├─ GNBCUCPFunction        │
                 │    └─ NRCellCU         ─┘
                 └─ GNBCUUPFunction
                      ├─ S1ULink          (transport link)
                      └─ RpUserPlaneLink  (transport link)
```

- The `fdn` nests SubNetworks (`SubNetwork=CH, SubNetwork=RAN, SubNetwork=MRAN19,
  ...`): CH → RAN are top levels, MRAN*/NRAN* are children.
- Each gNB splits into three functions: **GNBDUFunction** (DU + SectorCarrier),
  **GNBCUCPFunction** (CU), **GNBCUUPFunction** (transport). The NRCellCU,
  NRCellDU and NRSectorCarrier for a given cell share the same (gNB, cell-id).
- **107 gNBs**, **309 cells** of each NR type (CU/DU/SectorCarrier id sets are
  identical).
- Cells per gNB: `{1 cell: 3 gNBs, 2 cells: 9, 3 cells: 94, 6 cells: 1}` -
  **the norm is 3 cells per gNB** (tri-sector site).

### 2.3 CM attributes

33 configuration object types per day. Largest: `NRCellDU` (2360 attrs × 316
rows), `NRCellCU` (2358 × 316), `SectorEquipmentFunction` (818 × 488),
`NRSectorCarrier` (115), `NRFrequency` (83). Plus ~25 small per-cell profile
tables (`DrxProfile`, `Rach`, `AdmissionLimit`, ...). CM lists ~316 configured
cells vs 309 reporting in PM: the surplus are configured-but-not-reporting cells
(provisioned, no PM in this window).

### 2.4 Time

- 31 daily folders each for `pm/` and `cm/`, **2025-10-01 .. 2025-10-31**.
- Native ROP = **15 min** (96 ROPs/day). The daily-rollup form carries **one
  aggregated row per entity**, the rollup of those ~96 ROPs: `ropCount` ≈ 93-96
  on a normal day. (The 10-01 outlier reports `ropCount` 2-4; see §2.1.)
- So the delivered granularity is **1 row/entity/day** - too coarse for a
  "predict N steps ahead" story at ROP resolution (see §5).

### 2.5 Data-quality hazards (loader must be defensive)

The daily-rollup CSVs are messy: a `.txt` duplicate, a doubled-up filename
(`NRCellCU_attributesMcfbCellProfile...`), a date typo (`2025-1-01`), a
missing-date file (`NRCellCU_counters.csv`), a mangled extension
(`..._2025-10-27csv.csv`), and some groups missing a few days (NRCellCU 29/31,
S1ULink 30/31). Ragged rows (trailing `;`) are common.

---

## 3. Dataset topology

The topology follows the §2 hierarchy, configurable and seeded. **Default: 1
SubNetwork (a single leaf level, e.g. one MRAN), 20 gNBs, 3 cells/gNB = 60
cells** (small enough to train fast, same structure as §2). Scalable to the full
107 gNBs / 309 cells for the supercomputer run.

```
SubNetwork
  └─ gNB_k            (k = 1..G)
       └─ cell_{k,c}  (c = 1..3)   → NRCellDU + NRCellCU + NRSectorCarrier (1:1:1)
       └─ transport   → S1ULink, RpUserPlaneLink  (per gNB CUUP)
```

The **GAT operates on a heterogeneous graph** derived from this hierarchy:

- **Node types:** `cell` (one per CU/DU/SectorCarrier triple), `gnb`,
  `transport` (S1U/Rp aggregated per gNB CUUP), `subnetwork`.
- **Edge types:**
  - `cell -- cell` for **sibling cells under the same gNB** (intra-site
    neighbours).
  - `cell -- gnb`, `gnb -- transport`, `gnb -- subnetwork` (containment).
  - `cell -- cell` **inter-site handover neighbours**: a configurable
    k-nearest-neighbour set per cell (default k=4), *not* an all-pairs clique.
    (There is no neighbour list in the schema; neighbours are
    assigned by a seeded geographic layout. This adjacency is a modelling
    choice.)

Connecting gNBs through explicit neighbour edges rather than "all gNBs in a
SubNetwork are peers" keeps the graph sparse and avoids a ~5,600-edge clique at
the top level.

**Default ML-tensor shape:** 60 cells × 1,344 ROPs × F, where **F is the
per-cell feature dimension** = scalar counters across NRCellDU/NRCellCU/
NRSectorCarrier **plus** the expanded distribution-vector features (§11.4)
**plus** the broadcast per-gNB transport counters (§9). F is on the order of a
few thousand, not 247 (247 is NRCellDU's own counter count); the exact value is
produced by the feature builder and recorded in `meta.json`. The GAT additionally
consumes the typed edge index.

---

## 4. Counter families

The counters are 3GPP NR performance measurements (TS 28.552), written here with
a `pm*` counter name over an `fdn`-addressed managed object (TS 28.541). The same
loaders and the `falcon` baseline parse the PM/CM CSVs directly. The dataset
models a curated, high-signal subset per group (full fidelity - all **247
populated NRCellDU counters = 152 scalar + 95 vector** - is optional via config).

Counters marked **(confirmed)** are present in the §2 schema headers;
**(illustrative)** are representative names to be confirmed against the headers
before use.

| Group | Modelled counter families (examples) | Type |
|-------|--------------------------------------|------|
| **NRCellDU** | `pmActiveUeDlMax/Sum/Samp` *(confirmed)*, `pmMacHarqDlFail/UlFail` *(confirmed)*, `pmRadioRaCbFailMsg*` *(confirmed)*, `pmRadioPucchFailSr` *(confirmed)*, `pmPagDiscarded` *(confirmed)*, `pmCellDowntimeAuto/Man` *(confirmed)*, MAC/PDCP volume + PRB-utilisation counters *(illustrative)* | scalar + vector |
| **NRCellCU** | `pmHoPrepFail*` *(confirmed)*, `pmHoPrepRejIn*` *(confirmed)*, `pmEndcRelUeAbnormal*` *(confirmed)*, `pmEndcSetupFailReconfigFail` *(confirmed)*, `pmDrbEstabFailIntgProt` *(confirmed)*, RRC-connection-establishment counters *(illustrative)* | scalar |
| **NRSectorCarrier** | `pmSectorCarrierDowntime/Auto/Man` *(confirmed)*, `pmMicroTxSleepTime` *(confirmed)*, `pmEmfPwrBackoff*Distr` *(confirmed)* | scalar + vector |
| **S1ULink** | `pmPdcpPktRecDl5qi` *(confirmed)*, `pmPdcpPktRecDlDisc5qi` *(confirmed)*, `pmPdcpVolTransUl5qi` *(confirmed)* | vector (per-5QI) |
| **RpUserPlaneLink** | `pmPdcpPktLossUlQos` *(confirmed)*, `pmPdcpPktLossUlToDiscQos/ToQos` *(confirmed)*, `pmPdcpPktTransDl*Qos` *(confirmed)* | vector (per-QoS) |

**Distribution-vector counters** are emitted as comma-separated bins with a
**fixed per-counter bin count** (e.g. 16- or 40-bin histograms), matching how
3GPP distribution measurements (TS 28.552) are reported. Bins are
**driven by the same latent cell state** as the scalar counters (shifting mass
across bins as load/fault state changes) rather than independent noise, so the
histograms move coherently with the scalars.

Per-counter value ranges are set from the §2 schema (aggregate ranges only).
Those schema ranges are **daily-rollup** ranges and are mapped to the 15-min ROP
cadence by counter type:

- **Count/volume counters** (events, bytes, UE sums) are **rescaled by
  occupancy** - distributed across the ~96 ROPs of a day rather than divided
  uniformly, since activity concentrates in busy ROPs.
- **Time counters** (downtime in seconds) are **clamped to the ROP length**:
  `pmSectorCarrierDowntime` cannot exceed **900 s per 15-min ROP** (its daily
  `[0, 3600]` range is a per-*day* total, not a per-ROP value).

`pmActiveUeDlMax` is small integers; PDCP volumes are large counts.

---

## 5. Cadence

The daily-rollup form is 1 row/entity/day (`ropCount`≈96). The **dataset is at
the native 15-min ROP** (`ropCount=1`, **96 rows/entity/day**) - the per-ROP
level the daily rollup aggregates, **96× finer** - which is required for
forecasting at ROP resolution.

- **Default:** 15-min ROP, 14 days → 1,344 steps/entity.
- Horizon `m` (design doc, m=5..120) is measured in **ROPs** (15 min each),
  i.e. 75 min .. 30 h ahead.

---

## 6. Dynamics

Per-counter **Ornstein-Uhlenbeck** process (mean-reverting) around a
cell-specific baseline. Indicative defaults (all in `config.yaml`):

- **OU parameters:** mean-reversion `θ` ≈ 0.1-0.3 /ROP, volatility `σ` scaled to
  each counter's schema range. Baseline drawn per cell from the schema range.
- **Diurnal load profile:** busy-hour vs night sinusoid (period = 96 ROPs)
  modulating UE / PRB / volume counters.
- **Cross-level coupling with delay** along the hierarchy
  `SectorCarrier power/downtime → DU radio → CU mobility/abnormal-rel →
  transport loss/discard`, with per-hop lag (default 1-3 ROPs) and a configurable
  coupling gain.
- **Config (CM) influence:** selected CM attributes (DRX, admission limits, RACH
  config) shift PM baselines, linking the CM pillar to PM behaviour.

---

## 7. Fault model

The dataset carries explicit **fault labels** that raw counter data lacks. Each
fault is aligned to the fault-signal counters in the schema, so labels are
grounded in counter behaviour.

| Fault type | Origin level | Primary signature counters | Propagates to |
|------------|--------------|----------------------------|---------------|
| **Normal** | - | baseline OU | - |
| **Cell/Carrier Downtime** | NRSectorCarrier | `pmSectorCarrierDowntime*`, `pmCellDowntimeAuto` ↑ | DU volume ↓, CU abnormal-release ↑ |
| **Accessibility (RACH) fail** | NRCellDU | `pmRadioRaCbFailMsg*`, `pmRadioPucchFailSr` ↑ | CU RRC-establishment failures ↑ |
| **Mobility / ENDC fail** | NRCellCU | `pmHoPrepFail*`, `pmEndcRelUeAbnormal*` ↑ | neighbour-cell HO counters ↑ |
| **Packet Loss (transport)** | RpUserPlaneLink | `pmPdcpPktLossUlQos`, `...ToDiscQos` ↑ | DU `pmMacHarq*Fail` ↑, CU abnormal-release ↑ |
| **Cascading (composite)** | SectorCarrier | downtime → DU → CU → transport, full chain | all of the above |

Every "propagates to" effect references a counter in the §4 modelled set
(volume, `pmMacHarq*Fail`, abnormal-release, RRC-establishment, HO).

**Label vs feature decoupling (leakage control).** The label is driven by a
**latent fault state** `z(t) ∈ [0,1]` per entity, *not* by any observable
counter. The signature counters respond to `z` only **partially and noisily**
(a monotone-but-noisy link with per-counter gain, plus the OU noise of §6), so
no input feature equals or thresholds the label. Concretely: `z` crosses its
flip threshold to set `fault_type`, while counters see `g·f(z) + OU noise` with
`g < 1`. This prevents a model from recovering the labeling rule from its own
inputs, which would make accuracy a tautology. The latent `z` is **recorded in
`meta` for audit but is not a feature**.

**Degradation ramp and horizon alignment.** Each fault has a ramp over which `z`
rises from 0 to the flip threshold. **Ramp length is sampled to be ≥ the maximum
evaluation horizon** (`ramp_rops ≥ max(m)`; default ramp 20-130 ROPs for
`m` up to 120), so a genuine (if faint) precursor exists at every evaluated
horizon and early prediction is a real forecasting task rather than
unpredictable-by-construction. Fault *onset* (ramp start) is itself scheduled by
a slow per-entity hazard process, so the model can pick up pre-ramp drift. The
horizon-degradation curve is defined over `m ≤ max ramp`.

**Window vs ramp vs horizon.** The lookback window (§11.3, default 64 ROPs)
need not contain the whole ramp. At evaluation horizon `m`, the model sees the
window ending `m` ROPs before the flip, so for `m` approaching the ramp length
only the *earliest, faintest* part of the ramp is visible - which is exactly
what makes the horizon-degradation curve informative. Horizons longer than the
window's reach into the ramp are reported as the natural accuracy floor, not
excluded.

**Fault mix (default):** 55% Normal and ~9% each of the five fault classes
(Cell/Carrier Downtime, RACH, Mobility/ENDC, Packet Loss, Cascading), summing to
100%. The imbalance toward Normal gives the evidential / conformal components a
meaningful calibration setting.

---

## 8. Labels (per entity, per ROP)

Matches the six labels in the design doc, in RAN terms. Labels are emitted over
the **union of node types** (cell, carrier, transport link), not cells alone, so
a transport-origin fault can name a transport link as its origin.

| Label | Meaning |
|-------|---------|
| `fault_type` | one of the 6 classes above (int, 0 = Normal) |
| `fault_origin_node` | FDN of the originating cell / carrier / transport link (`none` when Normal) |
| `time_to_failure_rop` | ROPs until label flips (counts down during ramp; sentinel `-1` when Normal / not degrading) |
| `causal_path` | ordered FDN chain, pipe-delimited in the CSV, e.g. `NRSectorCarrier=GCAM2U\|NRCellDU=GCAM2U\|NRCellCU=GCAM2U\|RpUserPlaneLink=<ip_ip>` |
| `is_degrading` | 1 during the ramp window, else 0 |
| `ramp_progress` | 0.0 → 1.0 across the ramp (0.0 when not degrading) |

In the primary `fm/` CSV files these are plain columns (FDN strings, pipe-delimited
path). In the optional derived Parquet view (§9.1) node FDNs are mapped to int
codes via a codebook in `meta.json`, and `causal_path` becomes a list column.

A transport-origin fault is attached to the affected cells (via the per-gNB
transport aggregation of §9) so it is also learnable from the cell-indexed
tensor, while `fault_origin_node` still records the true transport-link FDN.

**TTF masking.** `time_to_failure_rop = -1` is a **mask sentinel, not a
regression target**. The TTF (quantile) head is trained with its loss **masked
to degrading rows** (`is_degrading = 1`); Normal/non-degrading rows contribute
no TTF loss. Omitting this mask would collapse the regressor toward `-1`.

**Window labeling.** For a windowed sample (§11), the target is taken from the
window's **last timestep** (causal, no look-ahead). `fault_type` is that
step's class; a window straddling a ramp-to-flip transition is labeled by its
final step, so pre-flip windows carry the pre-flip label with
`is_degrading`/`ramp_progress` describing the approach.

---

## 9. Output format

The dataset mirrors the **real export layout exactly**, so the same loaders and
the `falcon` baseline read it without modification. Two top-level trees match
the source (`pm/`, `cm/`); a third (`fm/`) adds the fault labels the FM task
needs.

```
dataset/
├── pm/<YYYY-MM-DD>/
│   ├── NRCellCU_counters_<date>.csv
│   ├── NRCellDU_counters_<date>.csv
│   ├── NRSectorCarrier_counters_<date>.csv
│   ├── S1ULink_counters_<date>.csv
│   └── RpUserPlaneLink_counters_<date>.csv
├── cm/<YYYY-MM-DD>/
│   ├── <Object>_attributes_<date>.csv        # same object types as the source
│   └── ...
├── fm/<YYYY-MM-DD>/
│   └── <Group>_labels_<date>.csv             # labels keyed by (fdn, rdiTimestamp)
└── meta.json                                  # seeds, config hash, schema, norm stats
```

- **PM / CM files** use the **same format as the source**: `;`-delimited,
  identical column order, distribution counters as comma-separated vectors, and
  the 6 metadata columns (`fdn, mecontext, cell, rdiTimestamp, rdiTimestampTZ,
  ropCount`). PM files are `<Group>_counters_<date>.csv`; CM files are
  `<Object>_attributes_<date>.csv`. Clean rows only (none of the ragged or
  misnamed-file quirks of §2.5).
- **FM label files** sit in a parallel `fm/` tree, one per PM group per day,
  joined to the counters by `(fdn, rdiTimestamp)`. Each row carries the six
  labels of §8 (`fault_type`, `fault_origin_node`, `time_to_failure_rop`,
  `causal_path`, `is_degrading`, `ramp_progress`). Keeping labels in a separate
  tree leaves the `pm/`/`cm/` files byte-for-byte in the source format.
- **Cadence:** files follow the real per-day foldering; the dataset writes one
  row per entity per ROP (§5), so a day holds up to 96 rows per entity rather
  than the single daily-rollup row of the source.

### 9.1 Optional derived ML view

For model training a **derived** Parquet view can be materialised from the CSVs
(it is a cache, not the primary format):

- `pm.parquet` - tensor `(cell, rop, counter)`; cell counters indexed directly,
  transport counters aggregated per gNB CUUP and broadcast to the gNB's cells
  (§2.1), with a parallel **mask channel** (0 = present, 1 = missing) for
  sensor-dropout and missing days.
- `labels.parquet` - the six labels; `topology.json` - typed nodes + edges for
  the GAT.
- Dtypes: counters `float32`, mask `uint8`, `fault_type`/`fault_origin` int
  codes, ROP index `int32`.

Everything is seeded from `config.yaml`; `--quick` emits a tiny smoke dataset
(e.g. 2 gNBs × 1 day) in the same layout.

---

## 10. Loader requirements

The loader is defensive by design: it excludes the 10-01 outlier, tolerates
missing days (NRCellCU 29/31, S1ULink 30/31), handles ragged rows and the
misnamed/duplicate files noted in §2.5, and parses entity identity from `fdn`
when the `cell` column is blank.

---

## 11. ML pipeline contract

How the dataset is consumed for training and evaluation. These rules exist to
keep reported metrics trustworthy (no leakage) and reproducible.

### 11.1 Split (leakage-safe)

- **Chronological**: 60% train / 20% val / 20% test by ROP time, with val and
  test strictly **after** train. No shuffling across the time boundary.
- **Entity policy** (configurable, both reported):
  - *temporal* (default): all entities in all splits, split only on time -
    tests forecasting on known cells.
  - *entity-holdout*: a disjoint set of gNBs reserved for val/test - tests
    generalization to unseen cells.
- A **calibration slice** (last 10% of train, by time) is held out for
  Conformal Prediction and never used for gradient updates.

### 11.2 Normalization (fit on train only)

- Per-counter standardization; **`log1p` first** for heavy-tailed count/volume
  counters (PDCP volumes, UE sums) before standardizing.
- Statistics are computed on the **train split only** and applied unchanged to
  val/test/calibration. Stats are stored in `meta.json`.

### 11.3 Windowing

- Fixed-length lookback windows for PatchTST (default length 64 ROPs,
  `patch_size=5`), stride 1 within an entity, **never crossing entity
  boundaries**.
- Causal labeling per §8 (target = window's last step). Windows are built
  **after** the split so no window spans the train/val/test boundary.

### 11.4 Distribution-vector representation

Each distribution-vector counter enters the feature tensor as its **fixed bin
vector** (16 or 40 bins) *plus* three reduced summaries (mean-bin, entropy,
top-bin fraction). Config can switch to summaries-only to cut input dimension.
The chosen representation and the resulting per-group feature dimension are
recorded in `meta.json`.

### 11.5 Imbalance and metrics

- Training the fault-type head uses **class-balanced loss** (inverse-frequency
  weights; focal loss optional).
- **Headline metric is macro-F1 and per-class recall, not accuracy** (a
  Normal-always baseline scores ~55% accuracy and must be beaten on macro-F1).
- Localization scored as top-k node accuracy; for transport-origin faults,
  where sibling cells share the broadcast transport features (§9), localization
  is scored at **gNB granularity**.

### 11.6 Scale and per-class counts

- **Default (smoke) scale:** 60 cells × 1,344 ROPs ≈ 80k cell-ROPs; after
  windowing and the ~9% per-fault mix, rarer classes yield only a few hundred
  windowed examples - adequate for a pipeline smoke test, **not** for final
  training.
- **Training scale:** 309 cells × longer span (config), to give each fault class
  ≥ a few thousand windowed examples. Realized per-class window counts are
  reported in `meta.json`; targets are set in `config.yaml`.

### 11.7 Determinism

- Independent seeds for topology, OU dynamics, fault onset/injection, and the
  train/val/test split, all derived from one master seed and recorded in
  `meta.json`.
- DataLoader workers are seeded per-worker to avoid cross-worker correlation.
- `--quick` and full runs are byte-reproducible given the same master seed and
  `config.yaml`.

---

Team AFM_Ericsson - European AI Hackathon 2026

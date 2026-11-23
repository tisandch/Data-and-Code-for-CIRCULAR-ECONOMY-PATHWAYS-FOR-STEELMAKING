# Data and code for: Circular economy pathways for steelmaking

Data and code accompanying the article

> Salashenko, T., Khaustova, V., Kyzym, M., Ilyash, O., Kotliarov, Ye., & Shulga, I. *Circular economy pathways for steelmaking: resource footprints, fossil dependence, and value creation across metallurgical cycles*. Submitted to *Environmental Economics*.

The repository allows all indicators, tables (A1, A2, B1, B2) and figures (2–7) of the article to be reproduced from a single input file.

## Authors

| Author | ORCID | Affiliation |
|---|---|---|
| Tetiana Salashenko | [0000-0002-1822-5836](https://orcid.org/0000-0002-1822-5836) | Research Centre for Industrial Problems of Development of the NAS of Ukraine, Kharkiv |
| Viktoriia Khaustova | [0000-0002-5895-9287](https://orcid.org/0000-0002-5895-9287) | Research Centre for Industrial Problems of Development of the NAS of Ukraine, Kharkiv |
| Mykola Kyzym | [0000-0001-8948-2656](https://orcid.org/0000-0001-8948-2656) | Research Centre for Industrial Problems of Development of the NAS of Ukraine, Kharkiv |
| Olha Ilyash | [0000-0002-7882-3942](https://orcid.org/0000-0002-7882-3942) | Research Centre for Industrial Problems of Development of the NAS of Ukraine, Kharkiv |
| Yevhen Kotliarov | [0000-0002-6366-6729](https://orcid.org/0000-0002-6366-6729) | Research Centre for Industrial Problems of Development of the NAS of Ukraine, Kharkiv |
| Igor Shulga | [0000-0001-9389-2690](https://orcid.org/0000-0001-9389-2690) | Ukrainian State Research Institute for Carbochemistry, Kharkiv |

## Repository structure

```
data/
  mbc_base.csv            Domestic input–output base: 4 countries × 4 years × 13 industries
notebook/
  steelmaking.ipynb       All calculations, tables and figures (with outputs)
  steelmaking.py          The same code as a plain Python script
  build_base.py           Builds data/mbc_base.csv from the full OECD ICIO files (optional)
results/
  figure_2_input_footprints.csv        Figure_2_Input_Footprint.png / .pdf
  figure_3_output_destinations.csv     Figure_3_Output_Footprint.png / .pdf
  figure_4_direct_gva_table.csv        Figure_4_Direct_GVA_Share.png / .pdf
  Table_A1_Output_Multipliers.csv      Figure_5_Output_and_GVA_Multipliers.png / .pdf
  Table_A2_GVA_Multipliers.csv
  Table_B1_Leontief_Diagonals.csv      Figure_6_Internal_Stability_and_External_Dependence.png / .pdf
  Table_B2_External_Dependence.csv
  figure_7_c24a_classification.csv     Figure_7_C24A_Classification.png / .pdf
  sector_mapping.csv                   Aggregation of ICIO codes into 13 industries
  block_audit.csv, balance_audit_by_industry.csv   Accounting-identity checks of the base
  source_metadata.csv                  SHA-256 of the input file and software versions
```

## Data

**Source.** OECD Inter-Country Input–Output (ICIO) tables, 2025 edition, small-matrix files `{YEAR}_SML.csv` for 2015, 2019, 2021 and 2022 (OECD, 2025; https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html). The full ICIO files are not redistributed here because of their size; they can be downloaded free of charge from the OECD.

**Coverage.** Ukraine (UKR), India (IND), the United States (USA) and Saudi Arabia (SAU); reference years 2015, 2019, 2021, 2022. All values are domestic flows in millions of US dollars at current prices.

**Aggregation.** The 50 ICIO industries are aggregated into 13 industries:

| Code in `mbc_base.csv` | Industry | ICIO codes |
|---|---|---|
| COAL | Coal mining | B05 |
| OILGAS | Oil and gas extraction | B06 |
| ORES | Metal ore mining | B07 |
| COKE | Coke and refined petroleum products | C19 |
| METAL | Basic iron and steel (focal industry) | C24A |
| FABMET | Fabricated metal products | C25 |
| ELEC | Electricity, gas and steam | D |
| WASTE | Water supply, sewerage and waste management | E |
| CONSTR | Construction | F |
| TRADE | Wholesale and retail trade | G |
| TRANSP | Transport | H49, H50, H51 |
| WAREH | Warehousing and support activities | H52 |
| OTHERS | Other industries and services | all remaining codes, including C24B (non-ferrous metals) |

**File layout.** Each country–year block of `mbc_base.csv` contains 18 rows: the 13 industries, `TOTAL_IC` (total domestic intermediate consumption), `IMP_IC` (imported intermediate inputs), `TLS` (net taxes on products), `VA` (gross value added) and `OUTPUT` (gross output). Columns contain intermediate uses by the 13 industries, final demand (`HH`, `NPISH`, `GOV`, `GFCF`, `INVNT`, `DPABR`), `OUTPUT`, `EXPORT` and `IMPORT`, plus `row_note`, `block_status` and `block_note` with source and validation notes.

## Indicators

For industry *j* (basic iron and steel, C24A, unless stated otherwise):

| Indicator | Definition |
|---|---|
| Input footprint (Stage 1) | a<sub>ij</sub> = z<sub>ij</sub> / x<sub>j</sub>, summed by component: raw materials (B05, B07), energy (B06, C19, D), logistics (H49–H52), circular (E) |
| Internal cyclicity | a<sub>jj</sub> |
| Output footprint (Stage 2) | b<sub>ji</sub> = y<sub>ji</sub> / x<sub>j</sub>: sales to other domestic industries (production), exports, domestic final demand (consumption) |
| Direct GVA share (Stage 3) | gva<sub>j</sub> = GVA<sub>j</sub> / x<sub>j</sub> |
| Leontief inverse, output multiplier (Stage 4) | L = (I − A)<sup>−1</sup>; mvo<sub>j</sub> = Σ<sub>i</sub> l<sub>ij</sub> |
| GVA multiplier (Stage 5) | mvao<sub>j</sub> = Σ<sub>i</sub> gva<sub>i</sub> l<sub>ij</sub> = gva<sub>j</sub> + gva<sub>j</sub>(l<sub>jj</sub> − 1) + Σ<sub>i≠j</sub> gva<sub>i</sub> l<sub>ij</sub>; identity mvao<sub>j</sub> + Σ<sub>i</sub>(m<sub>i</sub> + t<sub>i</sub>) l<sub>ij</sub> = 1 |
| Internal stability (Stage 6) | l<sub>jj</sub> |
| External dependence (Stage 7) | ES<sub>j</sub> = Σ<sub>i≠j</sub> l<sub>ij</sub> = mvo<sub>j</sub> − l<sub>jj</sub>, split into raw materials, energy, logistics, services and other industries |

**Classification of configurations (Figure 7).** R = gva<sub>j</sub> l<sub>jj</sub> / mvao<sub>j</sub> (value retention); D = mvao<sub>j</sub> (domestic value capture); X = export share of output; E = Σ gva<sub>i</sub> l<sub>ij</sub> over i ∈ {B05, B06, B07, C19}, i ≠ j (GVA in extractive and fuel industries). Thresholds: R and X 0.50; D 0.75; E 0.10.

| Configuration | R | D | X | E | Outcome |
|---|---|---|---|---|---|
| Income multiplier | < 0.50 | ≥ 0.75 | any | < 0.10 | positive |
| Income generator | ≥ 0.50 | ≥ 0.75 | < 0.50 | any | positive |
| Resource multiplier | < 0.50 | ≥ 0.75 | any | ≥ 0.10 | mixed |
| Isolated link | ≥ 0.50 | < 0.75 | < 0.50 | any | neutral |
| Dispersed link | < 0.50 | < 0.75 | any | < 0.10 | neutral |
| Rent buffer | ≥ 0.50 | any | ≥ 0.50 | any | negative |
| Cost transmitter | < 0.50 | < 0.75 | any | ≥ 0.10 | negative |

## How to reproduce

Requirements: Python 3.12 or later, `numpy`, `pandas`, `matplotlib`, `jupyter`.

```bash
git clone https://github.com/tisandch/Data-and-Code-for-CIRCULAR-ECONOMY-PATHWAYS-FOR-STEELMAKING.git
cd Data-and-Code-for-CIRCULAR-ECONOMY-PATHWAYS-FOR-STEELMAKING
pip install numpy pandas matplotlib jupyter
jupyter notebook notebook/steelmaking.ipynb
```

Run all cells in order. Alternatively, run the script from the repository root:

```bash
python notebook/steelmaking.py
```

Both read `data/mbc_base.csv` and write the tables and figures listed above to `results/`. A full run takes about one minute.

To rebuild `data/mbc_base.csv` from the original OECD files, download `{YEAR}_SML.csv` for 2015, 2019, 2021 and 2022, place them in `raw/icio_full/` and run `python notebook/build_base.py` from the repository root. This step is optional: the published base is sufficient to reproduce all results.

## Limitations

The indicators are monetary and refer to national industries, not to individual steelmaking routes. Only domestic flows are analysed, so imported coal, coke and scrap and physical material flows are not captured. The circular footprint covers purchases from water supply and waste management (E) only. The latest year available in the 2025 edition of the OECD ICIO is 2022.

## Licence

Data and results: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code: [MIT](https://opensource.org/licenses/MIT). OECD ICIO source data are subject to the OECD terms and conditions.

## Funding

This research was funded by the state budget allocated by the Ministry of Education and Science of Ukraine to state research institutions for scientific research and scientific and technical (experimental) development based on the results of state attestation, within the research project "Implementation of the Circular Approach in the Sectors of Ukraine's Economy" (State Registration No. 0124U005146).

## Citation

Please cite the archived version via its Zenodo DOI (see the badge or `CITATION.cff`).

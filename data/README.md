# Benchmark datasets

Source tables for the eight benchmark tasks in the paper. Each file is the table as obtained from its source study. Some records were removed during data cleaning, so a file can contain more rows than the record count reported in the paper.

Please cite the source study when you use a dataset.

| File | Task in the paper | Target column | Rows in file | Source study |
| --- | --- | --- | --- | --- |
| `formate_oxidation.xlsx` | Formate oxidation | `Experiment (mW/cm2)`, sheet `Fig. 1d-e` | 306 | Zhang et al., *Nature* **647**, 390–396 (2025) |
| `organic_laser_emitters.csv` | Organic laser emitters | `Emission Gain Cross Section (cm^2)` | 1,129 (360 with a target value) | Strieth-Kalthoff et al., *Science* **384**, eadk9227 (2024) |
| `inorganic_synthesis_oer_stability.csv` | Inorganic synthesis | `Synthesis` | 378 | Bai et al., *Nat. Catal.* **9**, 28–36 (2026) |
| `inorganic_synthesis_oer_stability.csv` | OER stability (task name `ru_dissolution` in finite-pool search) | `ICP-2h (%)` | 378 (209 with a target value) | Bai et al., *Nat. Catal.* **9**, 28–36 (2026) |
| `oer_activity.xlsx` | OER activity | `η10 (mV)` | 462 | Zhou et al., *Nat. Synth.* **5**, 730–739 (2026) |
| `photovoltaics.xlsx` | Photovoltaics | `FF` | 101 | Wu et al., *Science* **386**, 1256–1264 (2024) |
| `polymer_her.xlsx` | Polymer HER | `HER (TEA/MeOH/H2O) / umol g-1 h-1`, sheet `Sulfone screening` | 99 | Bai et al., *J. Am. Chem. Soc.* **141**, 9063–9071 (2019) |
| `battery_performance.csv` | Battery performance | `LCE_from_CE_aver` | 575 | Lin et al., *Sci. Adv.* **11**, eadu4391 (2025) |

Notes on layout:

- Inorganic synthesis and OER stability are two tasks on the same table, so they share one file. The file has two header rows: a group row, then the column names.
- `formate_oxidation.xlsx` has three sheets. The eight-element recipes are in `Fig. 1d-e`, below a one-line note and the header row.
- `polymer_her.xlsx` has three sheets. `Sulfone screening` has a group row above the column names.

# Automated OER campaign

Data from the prospective campaign reported in the paper. Overpotentials are in mV at 10 mA cm⁻².

| File | Content | Rows |
| --- | --- | --- |
| `oer_campaign_initial.csv` | Initial record the model was fitted to. Eighteen records are below 300 mV. | 125 |
| `oer_campaign_recommendations.csv` | The recommended conditions, with the prediction fixed before measurement (`lsv_model_pred`) and the measured value (`lsv`). Eight are below 300 mV; the lowest is 242 mV. | 11 |
| `oer_campaign_repeats_priority.csv` | Records prioritized for re-measurement (two or more diagnostic flags), with the original value (`lsv_prev_expt`) and the repeat (`lsv`). | 7 |
| `oer_campaign_repeats_controls.csv` | Unflagged matched controls, with the original value (`lsv_prev_expt`), the repeat (`lsv`) and the model prediction (`lsv_model_pred`). | 8 |

Columns shared by all four files: deposition temperature (°C), deposition current (mA) and time (s); precursor amounts in mmol for seven transition-metal cations as chloride, nitrate or sulfate salts (nitrate columns are all zero); complexing-agent volume; final precursor pH; and `lsv`, the measured overpotential.

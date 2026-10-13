#!/bin/bash
# Regenerate the tables and figures of the paper and of the supplementary material from the
# frozen result files of this repository (results/, runs_frozen/).
#   bash code/figures/make_all.sh         tables (II, III, S1-S3, S5, S8, S9) and Figs. 1-7
#   bash code/figures/make_all.sh figs    figures only
# Run from any directory; the output goes to build/ at the repository root. The interpreter is
# taken from $PY (default python3).
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
F=code/figures
RS=results/sarcos; RM=results/sim
FS=runs_frozen/frozen_sarcos.json; FM=runs_frozen/frozen_sim.json
mkdir -p build/fig build/tables build/tables_tmp build/tables_s8 build/tables_e13 build/tables_e14
if [ "$1" != "figs" ]; then
  # Table II, the S2 tables, the S3 slice summaries, and the results manifest of E01-E07
  $PY $F/tables.py --sarcos $RS --sim $RM --runs_sarcos runs_frozen/sarcos --runs_sim runs_frozen/sim --frozen_sarcos $FS --frozen_sim $FM --runtime results/runtime/runtime.json --out build/tables_tmp | tail -n 1
  cp build/tables_tmp/table2.tex build/tables_tmp/s2_sarcos.tex build/tables_tmp/s2_sim.tex build/tables_tmp/s3_k_slice_summary.csv build/tables_tmp/s3_m_slice_summary.csv build/tables_tmp/results_manifest.csv build/tables/
  # S1 (spikes, training), S3 (m, k, E07), and S5 (physical recording) tables
  $PY $F/build_supp_tables.py . build/tables
  # Table III and the S8 tables of the attenuation analyses (E09-E12)
  $PY $F/build_s8_tables.py --sarcos $RS --sim $RM --frozen_sarcos $FS --frozen_sim $FM --out build/tables_s8 | tail -n 1
  # S9 tables of the closed-loop analysis (E13, E13c)
  $PY code/summarize_e13.py $RM build/tables_e13 > build/e13_summary.txt
  $PY $F/build_e13_tables.py --sim $RM --out build/tables_e13 | cut -c1-60
fi
# Fig. 1 numbers (panels b-e) from the result files, then the figure
$PY $F/build_fig1_json.py --sarcos $RS --sim $RM --frozen_sarcos $FS --frozen_sim $FM --out build/tables/fig1_results.json | cut -c1-60
$PY $F/fig1.py --results build/tables/fig1_results.json --out build/fig/fig1.pdf | tail -n 1
$PY $F/fig2.py --sarcos $RS --sim $RM --frozen_sarcos $FS --frozen_sim $FM --out build/fig/fig2.pdf | tail -n 1
$PY $F/fig3.py --sim $RM --frozen_sim $FM --sarcos $RS --frozen_sarcos $FS --runtime results/runtime/runtime.json --out build/fig/fig3.pdf | tail -n 1
$PY $F/fig4.py --sim $RM --out build/fig/fig4.pdf | tail -n 1
$PY $F/fig5.py --sarcos $RS --sim $RM --frozen_sarcos $FS --frozen_sim $FM --out build/fig/fig5.pdf | tail -n 1
$PY $F/fig6.py --sim $RM --out build/fig/fig6.pdf | tail -n 1
# S8.6 table and figure of the underfitting control (E14)
$PY $F/build_e14_tables.py --sarcos $RS --sim $RM --frozen_sarcos $FS --frozen_sim $FM --out build/tables_e14 --fig build/fig/fig7.pdf > build/e14_numbers.txt
echo "tables and figures written to build/"

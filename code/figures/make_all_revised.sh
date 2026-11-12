#!/bin/bash
# Regenerate the figures and tables of the revised paper (main-text Figs. 1-6, Tables II-IV, and
# the supplementary figures and tables of E15-E29) from the result files in results/.
#   bash code/figures/make_all_revised.sh
# Run from any directory. Vector PDFs go to build/fig/, LaTeX tables to build/tables/ at the
# repository root (SNN29_ROOT overrides the root; SNN29_PNG=<dir> also writes PNG previews).
# The interpreter is taken from $PY (default python3). Needs numpy, pandas, and matplotlib.
# The tables of the locked protocol (E00-E14) are built by make_all.sh into build/locked/.
# Result files stored as <name>.csv.gz are unpacked next to the archive (gunzip -k) first.
set -e
PY=${PY:-python3}
cd "$(dirname "$0")/../.."
for z in results/*.csv.gz results/*/*.csv.gz; do
  [ -f "$z" ] && [ ! -f "${z%.gz}" ] && gunzip -k "$z"
done
cd code/figures
for s in fig1_overview fig2_plane fig3_width fig4_ladder fig5_path fig6_task \
         figS_position figS_reachable \
         tables_main supp_tables_s3_s7 supp_tables_s5 supp_tables_s8 supp_table_s7_conformal; do
  echo "== $s"
  $PY $s.py
done
echo "figures and tables written to build/fig and build/tables"

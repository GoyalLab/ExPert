Copies of the ExPert-data code behind the 3,096-gene input panel (Supp. Fig. S1 gene panel), taken from
ExPert-data commit 14811b7. The source of truth is ExPert-data; these are for reference.

- gene_fstat.py         per-dataset gene F-statistic from pseudobulks (workflow/scripts/)
- select.py             select_columns_fstat: union of per-dataset top-500 by F-stat (src/)
- gene_selection_sweep.py  held-out decodability vs panel size and strategy -> results/gene_sweep/sweep_holdout.csv
- score_criteria.py     F-stat vs HVG / mean / raw variance at 500 genes -> results/gene_sweep/_criteria/
- build_gene_panel.py, gene_analysis.py, plot_gene_sweep.py   panel build + the original exploratory figures

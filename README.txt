REPLICATION PACKAGE (Journal of the Operational Research Society submission)
Speed or Scrutiny? Task Fit, Trust and Compliance Risk in Credit-Union Staff
Reliance on Internal GenAI Assistants

No primary data were collected. Every input is a published figure or an
assumption stated in the manuscript; sources are listed in parameters.csv,
robustness_inputs.csv, the code and Table 3 of the manuscript.
Figure image files are supplied separately (JORS_Figures_300dpi.zip); the
scripts below regenerate them at 300 dpi.

REQUIREMENTS
Python 3 with NumPy, SciPy and Matplotlib. Runs in about a minute.

HOW TO REPRODUCE
1. python3 model.py        (seed 20261116, 200,000 Monte Carlo draws)
2. python3 robustness.py   (seed 20261123, 200,000 draws per setting)
3. python3 figs.py         (illustrative Figures 1 and 2; no computation)
Run in this order from this folder. Outputs are deterministic: rerunning
overwrites results.json and robust.json with identical values.

WHAT EACH FILE FEEDS IN THE MANUSCRIPT
parameters.csv         Table 3 (parameter values, ranges and provenance)
robustness_inputs.csv  Inputs for Section 5.1 (second source), 6.8 and 6.9
results.json           Tables 4-8; numbers in the abstract, practitioner
                       summary and Sections 5.1, 6.1-6.7
robust.json            Tables 9-11; numbers in Sections 5.1, 6.8, 6.9, 7.2.
                       Table 10 prints correlations 0 and -0.8; the -0.5
                       setting cited in the text is stored here.
Table 1 (positioning), Table 2 (notation) and Table 12 (controls) are not
computed.

FIGURE 1 DATA SOURCES (four published percentages, plotted by figs.py)
Cornerstone Advisors (2026), What's going on in banking 2026 (59%, 49%;
  pages 14-15): https://www.crnrstone.com/gritty-insights/research/whats-going-on-in-banking-2026
  PDF: https://www.crnrstone.com/hubfs/CS-Whats-Going-On-in-Banking_Cornerstone-Advisors.pdf
Wipfli (2026), State of the credit union industry 2026 (66%, 16%):
  https://www.wipfli.com/insights/research/state-of-the-credit-union-industry-2026
  News release: https://www.wipfli.com/news/2025/2026-state-of-banking-and-credit-union-reports
Only the reported values are used; no chart or table is reproduced.

FIGURES WRITTEN BY THE CODE (file -> manuscript figure)
figs.py:        fig3.png -> Figure 1 (sector indicators, published values)
                fig2.png -> Figure 2 (task-governance typology)
                fig1.png -> not used (earlier conceptual diagram)
model.py:       figA.png -> Figure 3   figB.png -> Figure 4
                figE.png -> Figure 5   figD.png -> Figure 7
                figC.png -> not used (earlier version of Figure 6)
robustness.py:  figC2.png -> Figure 6  figF.png -> Figure 8
                figG.png -> Figure 9

DERIVED QUANTITIES
Detection rates for Irons et al. (2026) are not reported in that study and
are recovered from its reported accuracy and acceptance with Equation 10:
D = (Acc - A + q) / (2q), q = 0.30. Checking speeds (kappa) follow from
Equation 4 using the AI-explanation condition, which recommended a decision,
as the low-effort reference. See Section 5.1 and Appendix A.
The middle asset tier in Table 11 is obtained by subtraction from rounded
system totals.

ANONYMITY
The package contains no author names or identifying metadata. A public
repository DOI will be added on acceptance.

INTEGRITY
SHA256SUMS.txt lists checksums of every file in this package.

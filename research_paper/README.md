# Academic Research Paper Package: MULTI_UAV_RL

This directory contains the complete, publication-ready research paper package developed from the `MULTI_UAV_RL` (`research_v2`) repository, formatted strictly in the official IEEE Conference LaTeX template (`IEEEtran` conference mode) and Markdown.

---

## 1. Directory Structure

```text
research_paper/
├── manuscript/
│   ├── research_paper.tex         <- Complete IEEE conference LaTeX manuscript (two-column IEEEtran)
│   └── research_paper.md          <- Full academic Markdown manuscript with KaTeX equations
├── figures/                       <- Publication-grade figures (PNG 300 DPI & vector plots)
│   ├── fig01_training_return.png
│   ├── fig02_training_rates.png
│   ├── fig03_training_energy_safety.png
│   ├── fig04_validation_curves.png
│   ├── fig05_reward_decomposition.png
│   ├── fig06_benchmark_table.png
│   ├── fig07_navigation_benchmark.png
│   ├── fig08_offloading_distribution.png
│   ├── fig09_pareto_front.png
│   ├── fig10_aerodynamic_power.png
│   ├── fig11_heldout_scatter.png
│   ├── fig12_per_seed_variability.png
│   └── fig13_multiseed_ci.png
├── tables/                        <- Raw simulation CSV tables & multi-seed benchmark summaries
│   ├── benchmark_summary.csv
│   ├── offloading_distribution.csv
│   ├── algorithm_workload_stress_comparison.csv
│   └── task_subsystem_stress_test_battery.csv
├── references/
│   └── references.bib             <- Comprehensive BibTeX database with verified academic citations
├── evidence/
│   ├── claim_evidence_matrix.md   <- Traceability matrix mapping claims to code and checkpoints
│   ├── experiment_inventory.md    <- Detailed catalog of the 12 algorithms, seeds, and hardware
│   ├── novelty_assessment.md      <- Systematic prior-art audit across published IEEE literature
│   └── reproducibility_report.md  <- Deterministic reproduction commands and SHA-256 checksums
├── supplementary/
│   ├── mathematical_notation.md   <- Exhaustive mathematical glossary with physical units
│   └── algorithm_pseudocode.md    <- Complete academic pseudocode for all algorithms
└── README.md                      <- This document
```

---

## 2. Compiling the LaTeX Manuscript

The manuscript [`manuscript/research_paper.tex`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/research_paper/manuscript/research_paper.tex) is written in standard IEEE conference mode.

### Option A: Compiling via Overleaf
1. Create a New Project on [Overleaf](https://www.overleaf.com/).
2. Upload the entire `research_paper/` folder (or upload `manuscript/research_paper.tex`, `references/references.bib`, and the `figures/` folder).
3. Set Compiler to **pdfLaTeX** and compile.

### Option B: Compiling Locally (TeX Live / MiKTeX)
```bash
cd research_paper/manuscript
pdflatex research_paper.tex
bibtex research_paper
pdflatex research_paper.tex
pdflatex research_paper.tex
```

---

## 3. Summary of Core Verified Research Findings

1. **Failure of End-to-End Continuous MARL:**
   - Evaluated across 5 held-out deterministic random seeds, pure end-to-end continuous MARL baselines (Pure MAPPO, IPPO, MADDPG, CPPO) achieve **$0.0\%$ fleet mission arrival success** in 3D urban canyons, stalling with goal errors between $23.61\,\text{m}$ and $70.52\,\text{m}$ and up to $208$ safety interventions.
2. **Success of Hierarchical Decoupling:**
   - Decoupling global geometric navigation (3D A* with pure pursuit and an analytical continuous safety shield) from computation scheduling guarantees **$100.0\%$ individual and fleet arrival**, reducing goal positioning error to **$0.41 \pm 0.00\,\text{m}$** with **$0.0$ collisions** and **$0.0$ safety filter interventions**.
3. **Destabilization of Residual Guidance:**
   - Adding continuous residual reinforcement learning actions ($k_{\text{rl}} = 0.5$) onto nominal A* trajectories collapses fleet arrival to $0.0\%$, increases goal error to $15.59\,\text{m}$, and spikes safety interventions to $198.0$, empirically demonstrating the danger of unconstrained residual policies.
4. **The Task Completion Ceiling Effect and Stress Regime:**
   - In baseline simulations ($T_{\text{max}} \in [1.0, 3.0]\,\text{s}$), local CPU execution trivially finishes in $\le 1.0\,\text{s}$, giving untrained models a $100.0\%$ completion rate.
   - Under a literature-calibrated delay-critical stress regime ($f_{\text{local}} = 0.8\,\text{GHz}, T_{\text{max}} \le 1.0\,\text{s}$), local execution collapses to **$0.0\%$**, random offloading achieves only **$18.1\%$**, and our cooperative MAPPO offloading policy achieves **$89.1\%$** on-time completion.

---

## 4. Key Reference Documents

- **Full Academic LaTeX Paper:** [`manuscript/research_paper.tex`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/research_paper/manuscript/research_paper.tex)
- **Full Academic Markdown Paper:** [`manuscript/research_paper.md`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/research_paper/manuscript/research_paper.md)
- **Novelty Assessment & Literature Matrix:** [`evidence/novelty_assessment.md`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/research_paper/evidence/novelty_assessment.md)
- **Claim-to-Evidence Matrix:** [`evidence/claim_evidence_matrix.md`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/research_paper/evidence/claim_evidence_matrix.md)
- **Reproducibility & Provenance Report:** [`evidence/reproducibility_report.md`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/research_paper/evidence/reproducibility_report.md)
- **Mathematical Notation Reference:** [`supplementary/mathematical_notation.md`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/research_paper/supplementary/mathematical_notation.md)
- **Algorithm Pseudocode:** [`supplementary/algorithm_pseudocode.md`](file:///c:/Users/vkart/Music/WORK/SEM/SEM_7/Minor_project/research_project/research_v2/research_paper/supplementary/algorithm_pseudocode.md)


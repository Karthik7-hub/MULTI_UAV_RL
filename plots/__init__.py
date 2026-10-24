"""
Multi-UAV Reinforcement Learning & Mobile Edge Computing Visualization Suite.
Consolidated into two comprehensive, modular, and publication-grade Python modules:
1. plot_curves.py: 2D quantitative performance curves (convergence, Pareto trade-offs, offloading).
2. render_publication_figures.py: 3D and 2D spatial flight trajectories and obstacle clearance views.
"""

from plots.plot_curves import (
    plot_trajectory_diagnostic,
    plot_training_convergence,
    plot_pareto_latency_energy,
    plot_all_marl_training_convergence,
    plot_offload_distribution_stacked
)
from plots.render_publication_figures import (
    generate_figure_a_coordinated_multi_uav,
    generate_figure_b_multimodel_trajectories,
    plot_publication_3d_trajectory_benchmark
)

__all__ = [
    "plot_trajectory_diagnostic",
    "plot_training_convergence",
    "plot_pareto_latency_energy",
    "plot_all_marl_training_convergence",
    "plot_offload_distribution_stacked",
    "generate_figure_a_coordinated_multi_uav",
    "generate_figure_b_multimodel_trajectories",
    "plot_publication_3d_trajectory_benchmark"
]

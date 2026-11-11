# Optimisation_Carbon_Capture
# Carbon Capture, Transport & Storage (CCTS) Network Optimization

Course project – *Optimization of Complex Decisions*.

This project models and solves the long-term deployment of a CO₂ supply
chain over a 25-year horizon. Captured CO₂ is routed from emission sites to
storage sites through a network using pipelines, rail and trucks, while
respecting a linear emissions-reduction scenario at minimum total cost.

**Key features**
- Graph-based mixed-integer linear programming (MILP) formulation
- Year-by-year decisions: capture installation, pipeline construction
  (with fixed capacity and annualized investment cost), and CO₂ flows per
  transport mode
- Mass-balance constraints and transport-related emissions accounted for
- Solver performance analysis and sensitivity study on key parameters
- Pareto analysis of cost vs. emissions, including cost per tonne of CO₂ avoided

**Tech stack:** Python, Jupyter Notebook, [your solver, e.g. Gurobi / PuLP / OR-Tools]

**Contents**
- `notebook.ipynb`: model, code and numerical results
- `report.pdf`: modeling, analysis and discussion

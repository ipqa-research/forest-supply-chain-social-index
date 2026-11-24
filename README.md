# Bi-objective design of forest biomass supply chain with a multidimensional social index

Code and data for the article "Bi-objective design of forest biomass supply chain with a multidimensional social index" (Dussel, Piedra-Jimenez, Santibañez-Aguilar and Rodríguez).

The model designs a multi-period forest biomass supply chain in north-eastern Argentina. It decides where and when to install or expand plants of five conversion technologies, their integration with three existing kraft pulp mills, the forest area to expand and harvest, and the flows of raw materials, intermediate and final products between 25 supply zones, 13 candidate nodes in Misiones, Entre Ríos and Corrientes, and 29 demand sites over 15 yearly periods.

1. The problem is a bi-objective MILP solved with Gurobi through Pyomo. It minimizes the total present cost (TPC) and maximizes a social index (SI).
2. Each plant can be installed at one of five capacity levels and expanded up to three times; the capital invested in expansions receives a discount `pi_exp`.
3. SI adds two normalized terms: the operating jobs of each plant, weighted by the open unemployment rate over the economically active population of its node (JC), and the sales of each node, weighted by its multidimensional poverty index (EV). Jobs are computed from the accumulated capacity of each plant with curves fitted to the NREL JEDI models.
4. The Pareto front is obtained with the epsilon-constraint method (minimum TPC subject to SI ≥ ε).
5. A growing demand scenario is used to analyze the sensitivity of the minimum-cost design to `pi_exp`.

## Repository structure

```
data/                                   input data (see data/README.md)
    base_case/                          constant demand (Pareto front)
    growing_demand/                     growing demand (pi_exp sensitivity)
build_instance.py                       reading of the Excel files and fixed parameters
model.py                                Pyomo model, solver call and export of results
pareto_front.py                         TPC/SI Pareto front (epsilon-constraint)
sensitivity_piexp_growing_demand.py     minimum-TPC designs for pi_exp = 0.0, 0.1, ..., 1.0
generate_growing_demand.py              builds the growing demand table from parameters.xlsx
plot_pareto.py                          Pareto front figure
plot_piexp_sensitivity.py               pi_exp sensitivity figure
```

## Installation

Python 3.14 and Gurobi are required (Gurobi offers free academic licenses). The scripts call Gurobi through the Pyomo `gurobi` interface, which uses the Gurobi command-line tools, so a full Gurobi installation with its executables in the `PATH` is needed besides the `gurobipy` package.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```
python model.py --objective EV_obj
python model.py --objective JC_obj
python model.py --objective TPC
python pareto_front.py
python pareto_front.py --min-cost-only
python plot_pareto.py
python plot_pareto.py --absolute
python sensitivity_piexp_growing_demand.py
python sensitivity_piexp_growing_demand.py --pi-exp 0.0 0.5 1.0
python plot_piexp_sensitivity.py
python generate_growing_demand.py data/base_case/parameters.xlsx 2.0
```

`model.py` solves a single objective (TPC is minimized and EV_obj, JC_obj and SI are maximized, unless `--sense` is given) for the base case or, with `--scenario growing_demand`, for the growing demand scenario. The EV_obj and JC_obj runs give the normalization constants of SI. `--pi-exp`, `--w1` and `--w2` change the expansion discount and the weights of SI. `pareto_front.py` solves the minimum-TPC and maximum-SI anchors and 15 epsilon-constrained points between them; `--min-cost-only` stops after the minimum-TPC point. `plot_pareto.py` reads the latest Pareto summary, and `plot_piexp_sensitivity.py` reads all the sensitivity summaries (when a value of `pi_exp` appears in more than one, the newest is used).

Results are written to `results/`: one Excel file per solution, with a sheet per variable and the sheets `plant_summary`, `jobs` and `jobs_by_plant`, plus the summaries, figures and Gurobi logs.

## Model settings

The parameters are in `model.py` and `build_instance.py`, and the data in `data/`:

- discount rate: 2% per year; economies-of-scale exponent 0.6 on investment costs; annual fixed cost equal to 4% of the investment;
- selling prices: 845 USD/t for kraft pulp, 313 USD/t for methanol and 1408.41 USD/t for HDO-PO;
- capacity levels of 25%, 50%, 100%, 125% and 150% of the reference capacity; at most four capacity events per plant (installation and three expansions); minimum capacity utilization of 70%; BLG and BL-PO-G only at level c3;
- kraft pulp demand is met exactly; methanol and HDO-PO sales lie between 80% and 120% of demand;
- `pi_exp` = 0.2 and `w1` = `w2` = 1 for the Pareto front;
- normalization constants `MAX_EV` = 6104.679405 million USD and `MAX_JC` = 29.761281, the optima of EV_obj and JC_obj (JC_obj is computed with the economically active population in thousands, so `MAX_JC` is 1000 times the JC_max of the article);
- investment costs are handled in million USD inside the model; TPC is reported in USD and the exported EV and JC are converted back to the units of the article;
- solver: relative MIP gap 0%, eight threads, `MIPFocus` = 3, `Cuts` = 2 and `Seed` = 12345; time limit 15000 s for the Pareto anchors, 6200 s for each point of the front (warm-started from the previous point) and 8000 s for each `pi_exp` of the sensitivity analysis.

## Reproducibility

The reported runs used Python 3.14.2, Pyomo 6.10.1 and Gurobi 13.0.3 with eight threads on an AMD Ryzen 7 5825U. With these settings the minimum-TPC point of the base case has TPC = 4.620 billion USD and SI = 1.181 (`python pareto_front.py --min-cost-only`). Points that stop at the time limit are saved with their gap in the summary. Designs with the same TPC and different plant locations or schedules can exist, so the plants chosen, and SI at a given TPC, may change with the Gurobi version, the number of threads and the seed.

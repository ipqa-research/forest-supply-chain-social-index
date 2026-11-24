import argparse
import os
from collections import Counter
from datetime import datetime

import pandas as pd
import pyomo.environ as pyo

from build_instance import build_instance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
BASE_CASE_DIR = os.path.join(DATA_DIR, "base_case")
GROWING_DEMAND_DIR = os.path.join(DATA_DIR, "growing_demand")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

COST_SCALE = 1e6

# EAP is loaded in thousands of people to narrow the range of the coefficient
# matrix: with EAP in people, the factors OUR/EAP of the employment
# constraints are of order 1e-7. This multiplies JC and JC_obj by EAP_SCALE,
# so MAX_JC is expressed in the same scale (JC_max of the article times
# EAP_SCALE), and the exported JC and JC_obj are divided by EAP_SCALE.
EAP_SCALE = 1000.0

MONETARY_VARS = {
    'EV', 'EV_obj',
    'ICexp', 'FCexp', 'ICint', 'FCint', 'FCK', 'Xdisc',
    'ICexp_si', 'ICexp_no', 'ICint_si', 'ICint_no',
    'FCexp_si', 'FCexp_no', 'FCint_si', 'FCint_no',
}


def _rescale_dict(d, factor):
    return {kk: vv / factor for kk, vv in d.items()}


def build_model(w1, w2, max_EV, max_JC, objective, sense, pi_exp=0.2, solver_name='gurobi',
                solve=True, numeric_focus=None, presolve=None, mip_focus=None, cuts=None,
                data_dir=BASE_CASE_DIR, output_dir=None):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "model")

    print(f"Loading data from {data_dir} ...")
    data = build_instance(datafolder=data_dir)
    print("Data loaded.")

    model = pyo.ConcreteModel("ForestSupplyChain")

    model.J = pyo.Set(initialize=data["J"], doc="Forest management units")
    model.K = pyo.Set(initialize=data["K"], doc="Processing nodes")
    model.L = pyo.Set(initialize=data["L"], doc="Markets")
    model.C = pyo.Set(initialize=data["C"], doc="Discrete capacity levels")
    model.M = pyo.Set(initialize=data["M"], doc="Processing technologies")
    model.R = pyo.Set(initialize=data["R"], doc="Raw materials")
    model.B = pyo.Set(initialize=data["B"], doc="Sawmill residues")
    model.S = pyo.Set(initialize=data["S"], doc="Tree species")
    model.I = pyo.Set(initialize=data["I"], doc="Intermediate products")
    model.O = pyo.Set(initialize=data["O"], doc="Final products")
    model.T = pyo.Set(initialize=data["T"], within=pyo.PositiveIntegers, doc="Time periods (years)")
    model.Mfix = pyo.Set(initialize=data["Mfix"], doc="Existing technologies")
    model.Mnew = pyo.Set(initialize=data["Mnew"], doc="Technologies that can be built")

    model.Mint = pyo.Set(initialize=data["Mint"], dimen=2, doc="Valid (new technology, existing technology) integration pairs")
    model.M_RS = pyo.Set(initialize=data["M_RS"], dimen=3, doc="Valid (technology, raw material, species)")
    model.M_SR = pyo.Set(initialize=data["M_SR"], dimen=2, doc="Valid (technology, sawmill residue)")
    model.I_M = pyo.Set(initialize=data["I_M"], dimen=2, doc="Technology m produces intermediate i")
    model.M_I = pyo.Set(initialize=data["M_I"], dimen=2, doc="Technology m consumes intermediate i")
    model.MO_O = pyo.Set(initialize=data["MO_O"], dimen=2, doc="Technology m produces final product o")
    model.O_L = pyo.Set(initialize=data["O_L"], dimen=2, doc="Final product o is sold in market l")
    model.Mfix_K = pyo.Set(initialize=data["Mfix_K"], dimen=2, doc="Location k of existing plants m_fix")
    model.Set_KL = pyo.Set(initialize=data["Set_KL"], dimen=2, doc="Valid routes from node k to market l")
    model.Set_MK = pyo.Set(initialize=data["Set_MK"], dimen=2, doc="Technologies m that can be installed at node k")

    model.scale_factor = pyo.Param(initialize=data['scale_factor'], doc="Economies-of-scale exponent for investment costs")
    model.Ro = pyo.Param(initialize=data['Ro'], doc="Discount rate")
    model.Ro1 = pyo.Param(model.J, model.K, initialize=data["Ro1"], default=0, doc="Distance [km] from forest j to node k")
    model.Ro2 = pyo.Param(model.K, model.K, initialize=data["Ro2"], default=0, doc="Distance [km] from node k to node kk")
    model.Ro3 = pyo.Param(model.K, model.L, initialize=data["Ro3"], default=0, doc="Distance [km] from node k to market l")
    model.Pi1 = pyo.Param(model.Mint, initialize=data["Pi1"], default=0, doc="Investment cost factor for integration")
    model.Pi2 = pyo.Param(model.Mint, initialize=data["Pi2"], default=0, doc="Fixed cost factor for integration")
    model.Epsilo = pyo.Param(model.C, model.M, initialize=data["Epsilo"], default=0, doc="Capacity level c of technology m (fraction of reference capacity)")
    model.Delta = pyo.Param(model.M, model.K, initialize=data["Delta"], default=0, doc="Reference capacity [t] of technology m at node k")
    model.Beta = pyo.Param(model.M, model.K, initialize=_rescale_dict(data["Beta"], COST_SCALE), default=0, doc="Reference investment cost [million USD] of technology m at node k")
    model.Delta_fixed = pyo.Param(model.Mfix, model.K, initialize=data["Delta_fixed"], default=0, doc="Installed capacity [t] of existing plants")
    model.Alpha = pyo.Param(model.R, model.S, model.J, model.T, initialize=data["Alpha"], default=0, doc="Harvest yield [t/ha] of raw material r and species s in forest j")
    model.Sigma1 = pyo.Param(model.R, model.S, model.O, model.M, initialize=data["Sigma1"], default=0, doc="Conversion of raw material (r,s) into final product o")
    model.Sigma2 = pyo.Param(model.B, model.O, model.M, initialize=data["Sigma2"], default=0, doc="Conversion of sawmill residue b into final product o")
    model.Sigma3 = pyo.Param(model.I, model.O, model.M, initialize=data["Sigma3"], default=0, doc="Conversion of intermediate i into final product o")
    model.Sigma4 = pyo.Param(model.R, model.S, model.I, model.M, initialize=data["Sigma4"], default=0, doc="Conversion of raw material (r,s) into intermediate i")
    model.Sigma5 = pyo.Param(model.B, model.I, model.M, initialize=data["Sigma5"], default=0, doc="Conversion of sawmill residue b into intermediate i")
    model.Eta1_low = pyo.Param(model.R, model.S, model.O, model.M, initialize=data["Eta1_low"], default=0, doc="Minimum share of raw material (r,s) in the recipe of product o")
    model.Eta2_low = pyo.Param(model.B, model.O, model.M, initialize=data["Eta2_low"], default=0, doc="Minimum share of residue b in the recipe of product o")
    model.Eta3_low = pyo.Param(model.I, model.O, model.M, initialize=data["Eta3_low"], default=0, doc="Minimum share of intermediate i in the recipe of product o")
    model.Theta0 = pyo.Param(model.B, initialize=data["Theta0"], default=0, doc="Purchase cost of sawmill residue b [USD/t]")
    model.Theta2 = pyo.Param(model.R, model.S, model.M, initialize=data["Theta2"], default=0, doc="Processing cost of raw material (r,s) [USD/t]")
    model.Theta3 = pyo.Param(model.B, model.M, initialize=data["Theta3"], default=0, doc="Processing cost of residue b [USD/t]")
    model.Theta4 = pyo.Param(model.I, model.M, initialize=data["Theta4"], default=0, doc="Processing cost of intermediate i [USD/t]")
    model.Gamma1 = pyo.Param(model.S, model.J, initialize=data["Gamma1"], default=0, doc="Initial forest area [ha] by species s and forest j")
    model.Fi = pyo.Param(model.J, model.B, model.T, initialize=data["Fi"], default=0, doc="Availability of sawmill residue b at j [t]")
    model.Chi = pyo.Param(model.O, model.T, model.L, initialize=data["Chi"], default=0, doc="Demand of final product o in market l [t/year]")
    model.Beta1 = pyo.Param(model.S, initialize=data["Beta1"], default=0, doc="Forest investment cost [USD/ha] by species s")
    model.Tau1 = pyo.Param(model.R, model.S, initialize=data["Tau1"], default=0, doc="Transport cost of raw material (r,s) [USD/t-km]")
    model.Tau2 = pyo.Param(model.B, initialize=data["Tau2"], default=0, doc="Transport cost of residue b [USD/t-km]")
    model.Tau3 = pyo.Param(model.I, initialize=data["Tau3"], default=0, doc="Transport cost of intermediate i [USD/t-km]")
    model.Tau4 = pyo.Param(model.O, initialize=data["Tau4"], default=0, doc="Transport cost of final product o [USD/t-km]")
    model.Thetacost = pyo.Param(model.S, initialize=data["Thetacost"], default=0, doc="Cost per harvested hectare [USD/ha] of species s")

    model.Pi_exp = pyo.Param(initialize=pi_exp, doc="Expansion discount for the same plant (k,m)")

    def beta2_rule(model, m, c, k):
        if (c, m) in model.Epsilo and (m, k) in model.Delta and model.Epsilo[c, m] != 0 and model.Delta[m, k] != 0:
            return (model.Beta[m, k] * (model.Epsilo[c, m])**model.scale_factor) / (model.Epsilo[c, m] * model.Delta[m, k])
        return 0
    model.beta2 = pyo.Param(model.M, model.C, model.K, initialize=beta2_rule)

    def gamma2_rule(model, m, c, k):
        if (c, m) in model.Epsilo and (m, k) in model.Delta and model.Epsilo[c, m] != 0 and model.Delta[m, k] != 0:
            return (0.04 * model.Beta[m, k] * (model.Epsilo[c, m])**model.scale_factor) / (model.Epsilo[c, m] * model.Delta[m, k])
        return 0
    model.gamma2 = pyo.Param(model.M, model.C, model.K, initialize=gamma2_rule)

    def M4_rule(model, m, c, k):
        if m not in model.Mnew or (m, k) not in model.Delta:
            return 0
        return model.Delta[m, k] * model.Epsilo[c, m]
    model.M4 = pyo.Param(model.M, model.C, model.K, initialize=M4_rule, default=0, doc="Capacity Big-M [t]")

    def M5_rule(model, m, c, k):
        if m not in model.Mnew or (m, k) not in model.Delta:
            return 0
        return model.Delta[m, k] * model.Epsilo[c, m] * model.beta2[m, c, k]
    model.M5 = pyo.Param(model.M, model.C, model.K, initialize=M5_rule, default=0, doc="Investment cost Big-M [million USD]")

    def M6_rule(model, m, c, k):
        if m not in model.Mnew or (m, k) not in model.Delta:
            return 0
        return model.Delta[m, k] * model.Epsilo[c, m] * model.gamma2[m, c, k]
    model.M6 = pyo.Param(model.M, model.C, model.K, initialize=M6_rule, default=0, doc="Fixed cost Big-M [million USD]")

    def Mdisc_rule(model, m, k):
        if (m, k) in model.Set_MK:
            return sum(model.M5[m, c, k] for c in model.C)
        return 0
    model.Mdisc = pyo.Param(model.Mnew, model.K, initialize=Mdisc_rule, within=pyo.NonNegativeReals, doc="Expansion discount Big-M [million USD]")

    price_data = _rescale_dict({'Pulpa_kraft': 845, 'Methanol': 313, 'PO-HDO': 1408.41}, COST_SCALE)
    model.price = pyo.Param(model.O, initialize=price_data, default=0, doc='Selling price [million USD/t]')

    social = pd.read_excel(os.path.join(data_dir, 'social_data.xlsx'))
    model.MPI = pyo.Param(model.K, initialize=dict(zip(social['K'], social['MPI'])), doc='Multidimensional poverty index at k')
    param_EAP = {k: v / EAP_SCALE for k, v in zip(social['K'], social['EAP'])}
    model.EAP = pyo.Param(model.K, initialize=param_EAP, doc='Economically active population at k [thousand people]')
    model.OUR = pyo.Param(model.K, initialize=dict(zip(social['K'], social['OUR'])), doc='Unemployment rate at k')

    df_pt = pd.read_excel(os.path.join(data_dir, 'jobs_per_plant.xlsx'), sheet_name='c3', header=None)
    k_values = df_pt.iloc[0, 1:].tolist()
    m_values = df_pt.iloc[1:, 0].tolist()
    jobs_c3 = {}
    for i_m, m in enumerate(m_values):
        for j_k, k in enumerate(k_values):
            v = df_pt.iloc[i_m + 1, j_k + 1]
            if not pd.isna(v):
                jobs_c3[(k, m)] = v

    df_curves = pd.read_excel(os.path.join(data_dir, 'jedi_employment_curves.xlsx'), sheet_name='curves')
    emp_curves = {str(row['Technology']): row for _, row in df_curves.iterrows()}
    param_F_emp, param_b_emp, param_V_emp = {}, {}, {}
    for m_c, row in emp_curves.items():
        param_F_emp[m_c] = float(row['F'])
        param_b_emp[m_c] = float(row['exponent'])
    for (k_c, m_c), pt_val in jobs_c3.items():
        if m_c in emp_curves:
            V_sheet = emp_curves[m_c]['V']
            param_V_emp[(k_c, m_c)] = float(pt_val) if pd.isna(V_sheet) else float(V_sheet)
            pt_curve_c3 = param_F_emp[m_c] + param_V_emp[(k_c, m_c)]
            if abs(pt_curve_c3 - float(pt_val)) > 0.5:
                print(f"  ! Employment warning: curve jobs at c3 = {pt_curve_c3:.2f} vs table = {float(pt_val):.2f} at ({k_c},{m_c})")
    model.F_emp = pyo.Param(model.Mnew, initialize=param_F_emp, default=0, doc='Fixed jobs per plant (JEDI)')
    model.b_emp = pyo.Param(model.Mnew, initialize=param_b_emp, default=1, doc='Scale exponent of variable jobs (JEDI)')
    model.V_emp = pyo.Param(model.K, model.Mnew, initialize=param_V_emp, default=0, doc='Variable jobs of a plant of size c3 (JEDI)')

    model.N_max = pyo.Param(initialize=4, mutable=True, doc="Maximum number of build events per plant: installation plus up to 3 expansions")
    model.Lambda_util = pyo.Param(initialize=0.7, mutable=True, doc="Minimum capacity utilization")

    print("\nDefining variables...")
    model.y = pyo.Var(model.K, model.Mnew, model.C, model.T, within=pyo.Binary, doc="1 if plant (k,m) builds capacity level c in t")
    model.z_int = pyo.Var(model.K, model.Mnew, model.C, within=pyo.Binary, doc="1 if plant (k,m,c) is integrated")
    model.t_int = pyo.Var(model.K, model.Mnew, model.C, model.Mfix, within=pyo.Binary, doc="1 if plant (k,m,c) is integrated with existing plant mm")
    model.x_fix = pyo.Var(model.K, model.Mfix, within=pyo.Binary, doc="1 if existing plant (k,m) operates")

    model.E = pyo.Var(model.K, model.Mnew, model.T, within=pyo.Binary, doc="1 if plant (k,m) built before t")
    model.Xdisc = pyo.Var(model.K, model.Mnew, model.T, within=pyo.NonNegativeReals, doc="Capital eligible for the expansion discount [million USD]")

    model.EV = pyo.Var(model.K, within=pyo.NonNegativeReals, doc='Economic value at k [million USD]')
    model.JC = pyo.Var(model.K, model.Mnew, model.T, within=pyo.NonNegativeReals, doc='Employment indicator of plant (k,m) in t')
    model.SI = pyo.Var(within=pyo.NonNegativeReals, doc='Social index')
    model.EV_obj = pyo.Var(within=pyo.NonNegativeReals, doc='Total economic value [million USD]')
    model.JC_obj = pyo.Var(within=pyo.NonNegativeReals, doc='Total weighted employment')

    model.Landexp = pyo.Var(model.J, model.S, model.T, within=pyo.NonNegativeReals, doc="Expanded forest area [ha]")
    model.Landav = pyo.Var(model.J, model.S, model.T, within=pyo.NonNegativeReals, doc="Available forest area [ha]")
    model.Landharv = pyo.Var(model.J, model.S, model.T, within=pyo.NonNegativeReals, doc="Harvested forest area [ha]")
    model.Q = pyo.Var(model.J, model.R, model.S, model.T, within=pyo.NonNegativeReals, doc="Available raw material [t]")
    model.AC = pyo.Var(model.K, model.M, model.T, within=pyo.NonNegativeReals, doc="Installed annual capacity [t/year]")
    model.ACexp = pyo.Var(model.K, model.Mnew, model.C, model.T, within=pyo.NonNegativeReals, doc="Capacity added in t [t/year]")
    model.ICexp = pyo.Var(model.K, model.Mnew, model.C, model.T, within=pyo.NonNegativeReals, doc="Investment cost of plant expansion [million USD]")
    model.FCexp = pyo.Var(model.K, model.Mnew, model.C, model.T, within=pyo.NonNegativeReals, doc="Fixed cost of plant expansion [million USD]")
    model.FCK = pyo.Var(model.K, model.M, model.T, within=pyo.NonNegativeReals, doc="Total fixed operating cost [million USD]")
    model.ICint = pyo.Var(model.K, model.Mnew, model.C, model.T, within=pyo.NonNegativeReals, doc="Investment cost with integration discounts [million USD]")
    model.FCint = pyo.Var(model.K, model.Mnew, model.C, model.T, within=pyo.NonNegativeReals, doc="Fixed cost with integration discounts [million USD]")
    model.F1 = pyo.Var(model.R, model.S, model.J, model.K, model.M, model.T, within=pyo.NonNegativeReals, doc="Raw material flow from forest j to node k")
    model.F2 = pyo.Var(model.B, model.J, model.K, model.M, model.T, within=pyo.NonNegativeReals, doc="Sawmill residue flow from j to node k")
    model.F3 = pyo.Var(model.I, model.K, model.M, model.K, model.M, model.T, within=pyo.NonNegativeReals, doc="Intermediate flow from plant (k,m) to plant (kk,mm)")
    model.F4 = pyo.Var(model.O, model.K, model.M, model.L, model.T, within=pyo.NonNegativeReals, doc="Final product flow from node k to market l")
    model.TC = pyo.Var(model.T, within=pyo.NonNegativeReals, doc="Transport cost in t [USD]")
    model.FC = pyo.Var(model.T, within=pyo.NonNegativeReals, doc="Fixed cost in t [USD]")
    model.BC = pyo.Var(model.T, within=pyo.NonNegativeReals, doc="Biomass cost in t [USD]")
    model.VC = pyo.Var(model.T, within=pyo.NonNegativeReals, doc="Variable processing cost in t [USD]")
    model.CC = pyo.Var(model.T, within=pyo.NonNegativeReals, doc="Capital cost in t [USD]")
    model.ICexp_si = pyo.Var(model.K, model.Mnew, model.C, model.Mfix, model.T, within=pyo.NonNegativeReals, doc="Integrated part of the investment cost [million USD]")
    model.ICexp_no = pyo.Var(model.K, model.Mnew, model.C, model.T, within=pyo.NonNegativeReals, doc="Non-integrated part of the investment cost [million USD]")
    model.ICint_si = pyo.Var(model.K, model.Mnew, model.C, model.Mfix, model.T, within=pyo.NonNegativeReals, doc="Integrated part of the discounted investment cost [million USD]")
    model.ICint_no = pyo.Var(model.K, model.Mnew, model.C, model.T, within=pyo.NonNegativeReals, doc="Non-integrated part of the discounted investment cost [million USD]")
    model.FCexp_si = pyo.Var(model.K, model.Mnew, model.C, model.Mfix, model.T, within=pyo.NonNegativeReals, doc="Integrated part of the fixed cost [million USD]")
    model.FCexp_no = pyo.Var(model.K, model.Mnew, model.C, model.T, within=pyo.NonNegativeReals, doc="Non-integrated part of the fixed cost [million USD]")
    model.FCint_si = pyo.Var(model.K, model.Mnew, model.C, model.Mfix, model.T, within=pyo.NonNegativeReals, doc="Integrated part of the discounted fixed cost [million USD]")
    model.FCint_no = pyo.Var(model.K, model.Mnew, model.C, model.T, within=pyo.NonNegativeReals, doc="Non-integrated part of the discounted fixed cost [million USD]")

    model.TPC = pyo.Var(within=pyo.Reals, doc="Total present cost [USD]")
    model.Slack_F3 = pyo.Var(model.I, model.K, model.M, model.T, within=pyo.NonNegativeReals, doc="Disposal of intermediate i at plant (k,m), only for i = BL")

    print("\nDefining constraints...")

    model.Disjunction2_Integration_Hull = pyo.ConstraintList()
    for k, m, c, t in model.K * model.Mnew * model.C * model.T:
        if (m, k) in model.Set_MK:
            sum_exp_si = sum(model.ICexp_si[k, m, c, mm, t] for m_new, mm in model.Mint if m_new == m and (mm, k) in model.Mfix_K)
            sum_int_si = sum(model.ICint_si[k, m, c, mm, t] for m_new, mm in model.Mint if m_new == m and (mm, k) in model.Mfix_K)
            sum_fck_exp_si = sum(model.FCexp_si[k, m, c, mm, t] for m_new, mm in model.Mint if m_new == m and (mm, k) in model.Mfix_K)
            sum_fck_int_si = sum(model.FCint_si[k, m, c, mm, t] for m_new, mm in model.Mint if m_new == m and (mm, k) in model.Mfix_K)

            model.Disjunction2_Integration_Hull.add(model.ICexp[k, m, c, t] == sum_exp_si + model.ICexp_no[k, m, c, t])
            model.Disjunction2_Integration_Hull.add(model.ICint[k, m, c, t] == sum_int_si + model.ICint_no[k, m, c, t])
            model.Disjunction2_Integration_Hull.add(model.FCexp[k, m, c, t] == sum_fck_exp_si + model.FCexp_no[k, m, c, t])
            model.Disjunction2_Integration_Hull.add(model.FCint[k, m, c, t] == sum_fck_int_si + model.FCint_no[k, m, c, t])

            model.Disjunction2_Integration_Hull.add(model.ICint_no[k, m, c, t] == model.ICexp_no[k, m, c, t])
            model.Disjunction2_Integration_Hull.add(model.FCint_no[k, m, c, t] == model.FCexp_no[k, m, c, t])

            model.Disjunction2_Integration_Hull.add(model.ICexp_no[k, m, c, t] <= model.M5[m, c, k] * (1 - model.z_int[k, m, c]))
            model.Disjunction2_Integration_Hull.add(model.FCexp_no[k, m, c, t] <= model.M6[m, c, k] * (1 - model.z_int[k, m, c]))

            for m_new, mm in model.Mint:
                if m_new == m and (mm, k) in model.Mfix_K:
                    model.Disjunction2_Integration_Hull.add(model.ICint_si[k, m, c, mm, t] == model.Pi1[m, mm] * model.ICexp_si[k, m, c, mm, t])
                    model.Disjunction2_Integration_Hull.add(model.FCint_si[k, m, c, mm, t] == model.Pi2[m, mm] * model.FCexp_si[k, m, c, mm, t])
                    model.Disjunction2_Integration_Hull.add(model.ICexp_si[k, m, c, mm, t] <= (model.M5[m, c, k]) * model.t_int[k, m, c, mm])
                    model.Disjunction2_Integration_Hull.add(model.FCexp_si[k, m, c, mm, t] <= model.M6[m, c, k] * model.t_int[k, m, c, mm])

    model.Disjunction1_Capacity_BigM = pyo.ConstraintList()
    for k, m in model.K * model.M:
        if m in model.Mnew and (m, k) in model.Set_MK:
            for c, t in model.C * model.T:
                model.Disjunction1_Capacity_BigM.add(model.ACexp[k, m, c, t] == model.Delta[m, k] * model.Epsilo[c, m] * model.y[k, m, c, t])
                model.Disjunction1_Capacity_BigM.add(model.ACexp[k, m, c, t] <= model.M4[m, c, k] * model.y[k, m, c, t])
                model.Disjunction1_Capacity_BigM.add(model.ICexp[k, m, c, t] == model.ACexp[k, m, c, t] * model.beta2[m, c, k])
                model.Disjunction1_Capacity_BigM.add(model.FCexp[k, m, c, t] == model.ACexp[k, m, c, t] * model.gamma2[m, c, k])

    t0_emp = model.T.first()
    Nmax_emp = int(round(pyo.value(model.N_max)))

    def _r_points_emp(m):
        eps_vals = sorted({round(pyo.value(model.Epsilo[c, m]), 10) for c in model.C if pyo.value(model.Epsilo[c, m]) > 0})
        sums = {0.0}
        for _ in range(Nmax_emp):
            sums |= {round(s_ + e_, 10) for s_ in sums for e_ in eps_vals}
        return sorted(sums)

    model.Emp_set = pyo.Set(dimen=2, initialize=[(k, m) for k in model.K for m in model.Mnew
                                                 if (m, k) in model.Set_MK and (k, m) in param_V_emp],
                            doc='Plants (k,m) with an employment curve')
    model.r_emp = pyo.Var(model.Emp_set, model.T, within=pyo.NonNegativeReals, doc='Cumulative plant capacity / capacity c3')
    model.Open_emp = pyo.Var(model.Emp_set, model.T, bounds=(0, 1), doc='1 if the plant exists in t')
    model.G_emp = pyo.Var(model.Emp_set, model.T, within=pyo.NonNegativeReals, doc='r**b of the JEDI employment curve')

    model.Employment_Constraints = pyo.ConstraintList()
    pw_idx_emp, pw_pts_emp = [], {}
    for (k, m) in model.Emp_set:
        b_m = pyo.value(model.b_emp[m])
        points = _r_points_emp(m)
        social_factor = model.OUR[k] / model.EAP[k]
        for t in model.T:
            model.r_emp[k, m, t].setub(points[-1])
            model.Employment_Constraints.add(
                model.r_emp[k, m, t] == sum(model.Epsilo[c, m] * model.y[k, m, c, tp] for c in model.C for tp in model.T if tp <= t))
            y_t = sum(model.y[k, m, c, t] for c in model.C)
            if t == t0_emp:
                model.Employment_Constraints.add(model.Open_emp[k, m, t] == y_t)
            else:
                model.Employment_Constraints.add(model.Open_emp[k, m, t] >= model.Open_emp[k, m, t - 1])
                model.Employment_Constraints.add(model.Open_emp[k, m, t] >= y_t)
                model.Employment_Constraints.add(model.Open_emp[k, m, t] <= model.Open_emp[k, m, t - 1] + y_t)
            if abs(b_m - 1.0) < 1e-12:
                model.Employment_Constraints.add(model.G_emp[k, m, t] == model.r_emp[k, m, t])
            else:
                pw_idx_emp.append((k, m, t))
                pw_pts_emp[(k, m, t)] = points
            model.Employment_Constraints.add(
                model.JC[k, m, t] == social_factor * (model.F_emp[m] * model.Open_emp[k, m, t] + model.V_emp[k, m] * model.G_emp[k, m, t]))

    def _g_emp_rule(model, k, m, t, x):
        return float(x) ** pyo.value(model.b_emp[m]) if x > 0 else 0.0

    if pw_idx_emp:
        model.PW_employment_idx = pyo.Set(dimen=3, initialize=pw_idx_emp)
        model.PW_employment = pyo.Piecewise(model.PW_employment_idx, model.G_emp, model.r_emp,
                                            pw_pts=pw_pts_emp, pw_constr_type='EQ',
                                            pw_repn='SOS2', f_rule=_g_emp_rule)

    for k in model.K:
        for m in model.Mnew:
            if (k, m) not in model.Emp_set:
                for t in model.T:
                    model.JC[k, m, t].fix(0)

    model.Logic_Constraints = pyo.ConstraintList()
    model.Expansion_Discount = pyo.ConstraintList()
    for k in model.K:
        for m in model.Mnew:
            if (m, k) in model.Set_MK:
                for t in model.T:
                    t_prev = [tp for tp in model.T if tp < t]
                    model.Logic_Constraints.add(
                        model.E[k, m, t] <= sum(model.y[k, m, c, tp] for c in model.C for tp in t_prev))
                    for c in model.C:
                        for tp in t_prev:
                            model.Logic_Constraints.add(model.E[k, m, t] >= model.y[k, m, c, tp])
                    I_kmt = sum(model.ICint[k, m, c, t] for c in model.C)
                    model.Expansion_Discount.add(model.Xdisc[k, m, t] <= I_kmt)
                    model.Expansion_Discount.add(model.Xdisc[k, m, t] <= model.Mdisc[m, k] * model.E[k, m, t])
                    model.Expansion_Discount.add(model.Xdisc[k, m, t] >= I_kmt - model.Mdisc[m, k] * (1 - model.E[k, m, t]))

    model.Fixed_Cost_And_Capacity = pyo.ConstraintList()
    for k, m, t in model.K * model.M * model.T:
        if m in model.Mnew and (m, k) in model.Set_MK:
            model.Fixed_Cost_And_Capacity.add(model.FCK[k, m, t] >= sum(model.FCint[k, m, c, tt] for c in model.C for tt in model.T if tt <= t))

            inflow_sum = sum(model.F1[r, s, j, k, m, t] for (m_tech, r, s) in model.M_RS if m_tech == m for j in model.J) + \
                sum(model.F2[b, j, k, m, t] for (m_tech, b) in model.M_SR if m_tech == m for j in model.J) + \
                sum(model.F3[i, kk, mm, k, m, t] for i in model.I for kk in model.K for mm in model.M if (mm, i) in model.I_M and (mm, kk) in model.Set_MK and (m, i) in model.M_I)

            model.Fixed_Cost_And_Capacity.add(inflow_sum <= model.AC[k, m, t])
            model.Fixed_Cost_And_Capacity.add(inflow_sum >= model.Lambda_util * model.AC[k, m, t])

    for k, m in model.K * model.Mnew:
        if (m, k) in model.Set_MK:
            model.Logic_Constraints.add(sum(model.y[k, m, c, t] for c in model.C for t in model.T) <= model.N_max)
            for c in model.C:
                model.Logic_Constraints.add(model.z_int[k, m, c] <= sum(model.y[k, m, c, t] for t in model.T))
                model.Logic_Constraints.add(sum(model.t_int[k, m, c, mm] for m_new, mm in model.Mint if m_new == m) == model.z_int[k, m, c])
        elif m in model.Mnew and (m, k) not in model.Set_MK:
            for c in model.C:
                for t in model.T:
                    model.Logic_Constraints.add(model.y[k, m, c, t] == 0)

    for k, m, t in model.K * model.Mnew * model.T:
        if (m, k) in model.Set_MK:
            model.Logic_Constraints.add(sum(model.y[k, m, c, t] for c in model.C) <= 1)

    Mnew_list = list(model.Mnew)
    for k in model.K:
        valid_techs_k = [m for m in Mnew_list if (m, k) in model.Set_MK]
        for idx_m, m in enumerate(valid_techs_k):
            for m_prime in valid_techs_k[idx_m + 1:]:
                for c in model.C:
                    for c_prime in model.C:
                        for t in model.T:
                            for t_prime in model.T:
                                model.Logic_Constraints.add(
                                    model.y[k, m, c, t] + model.y[k, m_prime, c_prime, t_prime] <= 1)

    for k, m in model.K * model.M:
        if m in model.Mfix and (m, k) not in model.Mfix_K:
            model.x_fix[k, m].fix(0)
    for k, m, c, mm in model.K * model.Mnew * model.C * model.Mfix:
        if (m, k) in model.Set_MK:
            model.Logic_Constraints.add(model.t_int[k, m, c, mm] <= model.x_fix[k, mm])

    model.Forest_Management = pyo.ConstraintList()
    t_initial = model.T.first()
    for j, s in model.J * model.S:
        model.Forest_Management.add(model.Landav[j, s, t_initial] == model.Gamma1[s, j] + model.Landexp[j, s, t_initial] - model.Landharv[j, s, t_initial])
        for t in model.T:
            if t > t_initial:
                model.Forest_Management.add(model.Landav[j, s, t] == model.Landav[j, s, t - 1] + model.Landexp[j, s, t] - model.Landharv[j, s, t])
            model.Forest_Management.add(model.Q[j, 'Logs', s, t] <= model.Landharv[j, s, t] * model.Alpha['Logs', s, j, t])
            model.Forest_Management.add(model.Q[j, 'Forest_residues', s, t] <= model.Landharv[j, s, t] * model.Alpha['Forest_residues', s, j, t])
            model.Forest_Management.add(model.Q[j, 'Forest_residues', s, t] <= model.Q[j, 'Logs', s, t])
        model.Forest_Management.add(sum(model.Landexp[j, s, t] for t in model.T) <= 1.5 * model.Gamma1[s, j])

    model.Capacity_Constraints = pyo.ConstraintList()
    for k, m in model.K * model.M:
        if m in model.Mnew and (m, k) in model.Set_MK:
            model.Capacity_Constraints.add(model.AC[k, m, t_initial] == sum(model.ACexp[k, m, c, t_initial] for c in model.C))
            for t in model.T:
                if t > t_initial:
                    model.Capacity_Constraints.add(model.AC[k, m, t] == model.AC[k, m, t - 1] + sum(model.ACexp[k, m, c, t] for c in model.C))
    for k, m, t in model.K * model.Mfix * model.T:
        if (m, k) in model.Mfix_K:
            inflow_sum_fixed = sum(model.F1[r, s, j, k, m, t] for (m_tech, r, s) in model.M_RS if m_tech == m for j in model.J)
            model.Capacity_Constraints.add(inflow_sum_fixed <= model.Delta_fixed[m, k] * model.x_fix[k, m])

    model.Material_Balance = pyo.ConstraintList()
    for j, r, s, t in model.J * model.R * model.S * model.T:
        outflow_F1 = sum(model.F1[r, s, j, k, m, t] for k, m in model.K * model.M if (m, r, s) in model.M_RS and (m, k) in model.Set_MK)
        model.Material_Balance.add(outflow_F1 <= model.Q[j, r, s, t])
    for j, b, t in model.J * model.B * model.T:
        outflow_F2 = sum(model.F2[b, j, k, m, t] for k, m in model.K * model.M if (m, b) in model.M_SR and (m, k) in model.Set_MK)
        model.Material_Balance.add(outflow_F2 <= model.Fi[j, b, t])
    for k, m, i, t in model.K * model.M * model.I * model.T:
        if (m, i) in model.I_M and (m, k) in model.Set_MK:
            outflow_F3 = sum(model.F3[i, k, m, kk, mm, t] for kk, mm in model.K * model.M if (mm, i) in model.M_I and (mm, kk) in model.Set_MK)
            prod_from_F1 = sum(model.Sigma4[r, s, i, m] * model.F1[r, s, j, k, m, t] for j in model.J for (m_tech, r, s) in model.M_RS if m_tech == m)
            prod_from_F2 = sum(model.Sigma5[b, i, m] * model.F2[b, j, k, m, t] for j in model.J for (m_tech, b) in model.M_SR if m_tech == m)
            if i == 'BL':
                model.Material_Balance.add(outflow_F3 + model.Slack_F3['BL', k, m, t] == prod_from_F1 + prod_from_F2)
            else:
                expr = (outflow_F3 == prod_from_F1 + prod_from_F2)
                if expr is True:
                    print(f"[Material_Balance] trivial F3 balance (0 = 0) skipped: i={i}, k={k}, m={m}, t={t}")
                    continue
                elif expr is False:
                    raise ValueError(f"[Material_Balance] infeasible F3 balance: i={i}, k={k}, m={m}, t={t}")
                model.Material_Balance.add(expr)

    for k, m, o, t in model.K * model.M * model.O * model.T:
        if (m, o) in model.MO_O and (m, k) in model.Set_MK:
            outflow_F4 = sum(model.F4[o, k, m, l, t] for l in model.L if (k, l) in model.Set_KL and (o, l) in model.O_L)
            prod_from_F1 = sum(model.Sigma1[r, s, o, m] * model.F1[r, s, j, k, m, t] for j in model.J for (m_tech, r, s) in model.M_RS if m_tech == m)
            prod_from_F2 = sum(model.Sigma2[b, o, m] * model.F2[b, j, k, m, t] for j in model.J for (m_tech, b) in model.M_SR if m_tech == m)
            prod_from_F3 = sum(model.Sigma3[i, o, m] * sum(model.F3[i, kk, mm, k, m, t] for kk, mm in model.K * model.M if (mm, i) in model.I_M and (mm, kk) in model.Set_MK) for i in model.I if (m, i) in model.M_I)
            expr = (outflow_F4 == prod_from_F1 + prod_from_F2 + prod_from_F3)
            if expr is True:
                print(f"[Material_Balance] trivial F4 balance (0 = 0) skipped: k={k}, m={m}, o={o}, t={t}")
                continue
            elif expr is False:
                raise ValueError(f"[Material_Balance] infeasible F4 balance: k={k}, m={m}, o={o}, t={t}")
            model.Material_Balance.add(expr)

    o_first = model.O.first()
    for o, l, t in model.O * model.L * model.T:
        if (o, l) in model.O_L:
            total_inflow = sum(model.F4[o, k, m, l, t] for k, m in model.K * model.M if (k, l) in model.Set_KL and (m, o) in model.MO_O and (m, k) in model.Set_MK)
            if o == o_first:
                model.Material_Balance.add(total_inflow == model.Chi[o, t, l])
            else:
                model.Material_Balance.add(total_inflow >= 0.8 * model.Chi[o, t, l])
                model.Material_Balance.add(total_inflow <= 1.2 * model.Chi[o, t, l])

    model.Recipe_Constraints = pyo.ConstraintList()
    for k, m, o, t in model.K * model.M * model.O * model.T:
        if (m, o) in model.MO_O and (m, k) in model.Set_MK:
            output_of_o = sum(model.F4[o, k, m, l, t] for l in model.L if (k, l) in model.Set_KL and (o, l) in model.O_L)
            for r, s in model.R * model.S:
                if (m, r, s) in model.M_RS:
                    input_of_rs = sum(model.F1[r, s, j, k, m, t] for j in model.J)
                    expr = model.Sigma1[r, s, o, m] * input_of_rs >= model.Eta1_low[r, s, o, m] * output_of_o
                    if not isinstance(expr, bool):
                        model.Recipe_Constraints.add(expr)
            for b in model.B:
                if (m, b) in model.M_SR:
                    input_of_b = sum(model.F2[b, j, k, m, t] for j in model.J)
                    expr = model.Sigma2[b, o, m] * input_of_b >= model.Eta2_low[b, o, m] * output_of_o
                    if not isinstance(expr, bool):
                        model.Recipe_Constraints.add(expr)
            for i in model.I:
                if (m, i) in model.M_I:
                    input_of_i = sum(model.F3[i, kk, mm, k, m, t] for kk, mm in model.K * model.M if (mm, i) in model.I_M and (mm, kk) in model.Set_MK and (m, i) in model.M_I)
                    expr = model.Sigma3[i, o, m] * input_of_i >= model.Eta3_low[i, o, m] * output_of_o
                    if not isinstance(expr, bool):
                        model.Recipe_Constraints.add(expr)

    model.Objective_Components = pyo.ConstraintList()
    model.Objective_Components.add(model.TPC == sum((model.CC[t] + model.FC[t] + model.BC[t] + model.VC[t] + model.TC[t]) / (1 + model.Ro)**(t - 1) for t in model.T))
    for t in model.T:
        model.Objective_Components.add(
            model.CC[t] == sum(model.ICint[k, m, c, t] for k, m, c in model.K * model.Mnew * model.C if (m, k) in model.Set_MK) * COST_SCALE
            + sum(model.Landexp[j, s, t] * model.Beta1[s] for j, s in model.J * model.S)
            - model.Pi_exp * sum(model.Xdisc[k, m, t] for k, m in model.K * model.Mnew if (m, k) in model.Set_MK) * COST_SCALE
        )
        model.Objective_Components.add(model.FC[t] == sum(model.FCK[k, m, t] for k, m in model.K * model.Mnew if (m, k) in model.Set_MK) * COST_SCALE)
        model.Objective_Components.add(model.BC[t] == sum(model.Theta0[b] * model.F2[b, j, k, m, t] for b, j, k, m in model.B * model.J * model.K * model.M if (m, k) in model.Set_MK and (m, b) in model.M_SR) + sum(model.Thetacost[s] * model.Landharv[j, s, t] for j, s in model.J * model.S))
        model.Objective_Components.add(model.VC[t] == sum(model.Theta2[r, s, m] * model.F1[r, s, j, k, m, t] for r, s, j, k, m in model.R * model.S * model.J * model.K * model.M if (m, r, s) in model.M_RS and (m, k) in model.Set_MK) + sum(model.Theta3[b, m] * model.F2[b, j, k, m, t] for b, j, k, m in model.B * model.J * model.K * model.M if (m, b) in model.M_SR and (m, k) in model.Set_MK) + sum(model.Theta4[i, mm] * model.F3[i, k, m, kk, mm, t] for i, k, m, kk, mm in model.I * model.K * model.M * model.K * model.M if (m, i) in model.I_M and (mm, i) in model.M_I and (mm, kk) in model.Set_MK and (m, k) in model.Set_MK))
        model.Objective_Components.add(model.TC[t] == sum(model.Tau1[r, s] * model.Ro1[j, k] * model.F1[r, s, j, k, m, t] for r, s, j, k, m in model.R * model.S * model.J * model.K * model.M if (m, r, s) in model.M_RS and (m, k) in model.Set_MK) + sum(model.Tau2[b] * model.Ro1[j, k] * model.F2[b, j, k, m, t] for b, j, k, m in model.B * model.J * model.K * model.M if (m, b) in model.M_SR and (m, k) in model.Set_MK) + sum(model.Tau3[i] * model.Ro2[k, kk] * model.F3[i, k, m, kk, mm, t] for i, k, m, kk, mm in model.I * model.K * model.M * model.K * model.M if (m, i) in model.I_M and (mm, i) in model.M_I and (m, k) in model.Set_MK and (mm, kk) in model.Set_MK) + sum(model.Tau4[o] * model.Ro3[k, l] * model.F4[o, k, m, l, t] for o, k, m, l in model.O * model.K * model.M * model.L if (k, l) in model.Set_KL and (m, o) in model.MO_O and (m, k) in model.Set_MK and (o, l) in model.O_L))

    if 'BL' in model.I:
        for k_from, m_from, k_to, m_to, t in model.K * model.M * model.K * model.M * model.T:
            if k_from != k_to:
                model.F3['BL', k_from, m_from, k_to, m_to, t].fix(0)

    methanol_techs = ['BLG', 'BL-PO-G']
    baseline_capacity = 'c3'
    for k, m, c in model.K * model.Mnew * model.C:
        if m in methanol_techs and c != baseline_capacity:
            for t in model.T:
                model.y[k, m, c, t].fix(0)

    def economic_value_rule(model, k):
        return model.EV[k] == sum(model.F4[o, k, m, l, t] * model.price[o]
                                  for o in model.O for m in model.M for l in model.L for t in model.T
                                  if (k, l) in model.Set_KL and (m, o) in model.MO_O
                                  and (m, k) in model.Set_MK and (o, l) in model.O_L) * model.MPI[k]
    model.economic_value = pyo.Constraint(model.K, rule=economic_value_rule, doc='EV[k] in million USD')

    def economic_value_total_rule(model):
        return model.EV_obj == sum(model.EV[k] for k in model.K)
    model.economic_value_total = pyo.Constraint(rule=economic_value_total_rule)

    def jobs_created_rule(model):
        return model.JC_obj == sum(model.JC[k, m, t] for (k, m) in model.Emp_set for t in model.T)
    model.jobs_created = pyo.Constraint(rule=jobs_created_rule)

    def social_index_rule(model):
        return model.SI == w1 * (model.JC_obj / max_JC) + w2 * (model.EV_obj / max_EV)
    model.social_index_def = pyo.Constraint(rule=social_index_rule)

    objective_expr = getattr(model, objective)
    sense_obj = getattr(pyo, sense)
    if objective in ('JC_obj', 'SI'):
        objective_expr = objective_expr * 10000

    model.Objective = pyo.Objective(expr=objective_expr, sense=sense_obj)
    print("Model built.")

    def save_results_to_excel(model, filename=None, include_zero=False, tol=1e-6):
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename = os.path.join(output_dir, f"results_{objective}_piexp-{pi_exp}_w1-{w1}_w2-{w2}_{timestamp}.xlsx")
        elif not os.path.isabs(filename):
            filename = os.path.join(output_dir, filename)
        print(f"Saving results to: {filename}")
        used_sheet_names = set()

        def _unique_sheet_name(base_name):
            base = str(base_name).replace(" ", "_")[:31]
            name = base
            suffix = 1
            while name in used_sheet_names:
                tail = f"_{suffix}"
                name = base[: max(1, 31 - len(tail))] + tail
                suffix += 1
            used_sheet_names.add(name)
            return name

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for var_obj in model.component_objects(pyo.Var, descend_into=True):
                var_name = var_obj.name if hasattr(var_obj, "name") else str(var_obj)
                if var_name.startswith('PW_employment'):
                    continue

                if var_name in MONETARY_VARS:
                    conversion = COST_SCALE
                elif var_name in ('JC', 'JC_obj'):
                    conversion = 1.0 / EAP_SCALE
                else:
                    conversion = 1.0

                rows = []
                col_names = []

                if var_obj.is_indexed():
                    try:
                        for cs in var_obj.index_set().subsets():
                            nm = str(cs.name)
                            if "." in nm:
                                nm = nm.split(".")[-1]
                            col_names.append(nm)
                    except Exception:
                        col_names = None

                    for idx, var in var_obj.items():
                        try:
                            val = pyo.value(var, exception=False)
                        except Exception:
                            val = None
                        if val is None:
                            continue
                        val = val * conversion
                        if (not include_zero) and (abs(val) <= tol):
                            continue
                        idx_list = [str(x) for x in idx] if isinstance(idx, tuple) else [str(idx)]
                        rows.append(idx_list + [val])

                    if not col_names:
                        nidx = len(rows[0]) - 1 if rows else 1
                        col_names = [f"idx_{i + 1}" for i in range(nidx)]
                    col_names_out = col_names + ["Value"]
                else:
                    try:
                        val = pyo.value(var_obj, exception=False)
                    except Exception:
                        val = None
                    if val is not None:
                        val = val * conversion
                    if val is None or ((not include_zero) and abs(val) <= tol):
                        rows = []
                    else:
                        rows = [[var_name, val]]
                    col_names_out = ["Variable", "Value"]

                if rows:
                    try:
                        df = pd.DataFrame(rows, columns=col_names_out)
                    except Exception:
                        df = pd.DataFrame(rows)
                    df.to_excel(writer, sheet_name=_unique_sheet_name(var_name), index=False)

            try:
                plant_rows = []
                for k, m, c in model.K * model.Mnew * model.C:
                    if (m, k) not in model.Set_MK:
                        continue
                    level_cap = pyo.value(model.Delta[m, k] * model.Epsilo[c, m])
                    increments = {t: (pyo.value(model.ACexp[k, m, c, t], exception=False) or 0.0) for t in model.T}
                    periods = [t for t, v in increments.items() if v > tol]
                    total_cap = sum(increments.values())
                    if total_cap <= tol:
                        continue
                    plant_rows.append({
                        "Node": k,
                        "Technology": m,
                        "Level": c,
                        "Ref_capacity": round(level_cap, 1),
                        "Installed_capacity": round(total_cap, 1),
                        "Single_level_check": "OK" if abs(total_cap - level_cap) <= max(1.0, tol * max(level_cap, 1)) else "CHECK",
                        "Build_periods": str(periods),
                        "N_builds": len(periods),
                        "Staged": "yes" if len(periods) > 1 else "no",
                    })
                if plant_rows:
                    pd.DataFrame(plant_rows).sort_values("Node").to_excel(
                        writer, sheet_name=_unique_sheet_name("plant_summary"), index=False)
            except Exception as e:
                print(f"    ! Error building plant_summary: {e}")

            try:
                jobs_rows, plant_jobs_rows = [], []
                for t in model.T:
                    total_pt = 0.0
                    for (k, mm) in model.Emp_set:
                        r_val = sum(pyo.value(model.Epsilo[c, mm]) * (1.0 if (pyo.value(model.y[k, mm, c, tp], exception=False) or 0.0) > 0.5 else 0.0)
                                    for c in model.C for tp in model.T if tp <= t)
                        if r_val <= 1e-9:
                            continue
                        pt_val = pyo.value(model.F_emp[mm]) + pyo.value(model.V_emp[k, mm]) * r_val ** pyo.value(model.b_emp[mm])
                        total_pt += pt_val
                        plant_jobs_rows.append({"Node": k, "Technology": mm, "Period": t,
                                                "r_cumulative_capacity": round(r_val, 4),
                                                "Jobs": round(pt_val, 2)})
                    jobs_rows.append({"Period": t, "Total_jobs": total_pt})

                pd.DataFrame(jobs_rows).to_excel(writer, sheet_name=_unique_sheet_name("jobs"), index=False)
                if plant_jobs_rows:
                    pd.DataFrame(plant_jobs_rows).to_excel(writer, sheet_name=_unique_sheet_name("jobs_by_plant"), index=False)
            except Exception as e:
                print(f"    ! Error building the jobs sheets: {e}")
        print(f"Results saved: {filename}")
        return filename

    results = None
    if solve:
        log_folder = os.path.join(output_dir, "solver_logs")
        os.makedirs(log_folder, exist_ok=True)

        solver_name_lower = str(solver_name).lower()
        if solver_name_lower not in ('cplex', 'gurobi'):
            raise ValueError(f"solver_name must be 'cplex' or 'gurobi'; got: {solver_name}")

        opt = pyo.SolverFactory(solver_name_lower)
        if solver_name_lower == 'gurobi':
            opt.options['TimeLimit'] = 20000
            opt.options['MIPGap'] = 0.0
            opt.options['Threads'] = 8
            if numeric_focus is not None:
                opt.options['NumericFocus'] = numeric_focus
                opt.options['DualReductions'] = 0
            if presolve is not None:
                opt.options['Presolve'] = presolve
            if mip_focus is not None:
                opt.options['MIPFocus'] = mip_focus
            if cuts is not None:
                opt.options['Cuts'] = cuts
        else:
            opt.options['timelimit'] = 3600
            opt.options['mipgap'] = 0.0
            opt.options['threads'] = 1
            if numeric_focus is not None or presolve is not None:
                print("-> numeric_focus and presolve are only applied with Gurobi.")

        run_tag = f"{objective}_piexp-{pi_exp}_w1-{w1}_w2-{w2}_{timestamp}"
        log_path = os.path.join(log_folder, f"log_{run_tag}.log")
        print(f"-> Solving with {solver_name_lower}; solver log: {log_path}")
        results = opt.solve(model, tee=True, logfile=log_path)

        usable = (pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit)
        termination = results.solver.termination_condition

        if termination in usable:
            report_constraint_violations(model, os.path.join(log_folder, f"violations_{run_tag}.txt"))

        print("\n" + "=" * 50)
        print("Optimization results")
        print("=" * 50)
        if termination in usable:
            try:
                if termination == pyo.TerminationCondition.optimal:
                    print("Optimal solution found.")
                else:
                    print("Time limit reached without proving optimality; the best solution found is saved (see the gap in the solver log).")
                obj_value = pyo.value(objective_expr)
                print(f"{sense.upper()} {objective} = {obj_value:,.6f}"
                      + (f"  (before x10000: {obj_value / 10000:,.6f})" if objective in ('JC_obj', 'SI') else "")
                      + ("  [million USD]" if objective in MONETARY_VARS else ""))
                print(f"TPC = {pyo.value(model.TPC):,.2f} USD")
                print(f"SI  = {pyo.value(model.SI):.6f}")
                print("\nPlants built:")
                for k, m, c, t in model.y:
                    if pyo.value(model.y[k, m, c, t]) > 0.5:
                        print(f"  - {m}, level {c}, node {k}, period {t}")
                save_results_to_excel(model)
            except Exception as e:
                print(f"  ! Could not read or save a usable solution: {e}")
        elif termination == pyo.TerminationCondition.infeasible:
            print("The model is infeasible. See the solver log.")
        else:
            print(f"No usable solution. Solver status: {results.solver.status}; termination: {termination}")

    return model, save_results_to_excel, results


def report_constraint_violations(model, output_file=None, top_n=20):
    violations = []
    for con in model.component_data_objects(pyo.Constraint, active=True):
        try:
            body_val = pyo.value(con.body, exception=False)
        except Exception:
            continue
        if body_val is None:
            continue
        viol = 0.0
        if con.has_lb():
            lb = pyo.value(con.lower)
            if lb is not None and body_val < lb:
                viol = max(viol, lb - body_val)
        if con.has_ub():
            ub = pyo.value(con.upper)
            if ub is not None and body_val > ub:
                viol = max(viol, body_val - ub)
        if viol > 1e-9:
            violations.append((viol, con.name))
    violations.sort(key=lambda x: x[0], reverse=True)

    lines = []
    if violations:
        lines.append(f"Top {min(top_n, len(violations))} constraints by violation ({len(violations)} with violation > 1e-9):")
        for viol, name in violations[:top_n]:
            lines.append(f"  {name}: violation = {viol:.6e}")
        blocks = Counter(name.split('[')[0] for _, name in violations)
        lines.append("\nConstraints with violation, by block:")
        for block, cnt in blocks.most_common():
            lines.append(f"  {block}: {cnt}")
    else:
        lines.append("No constraint violations above 1e-9.")

    text = "\n".join(lines)
    print(text)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return violations


# Normalization constants of SI: optimal values of EV_obj (million USD) and
# JC_obj (scaled by EAP_SCALE) obtained by maximizing each term separately
# with pi_exp = 0.2 and constant demand. To recompute them, run this script
# with --objective EV_obj and with --objective JC_obj and use the printed
# values (for JC_obj, the value before the x10000 factor).
MAX_EV = 6104.679405
MAX_JC = 29.761281


def main():
    parser = argparse.ArgumentParser(description="Solve the forest supply chain model for one objective.")
    parser.add_argument("--objective", default="TPC", choices=["TPC", "EV_obj", "JC_obj", "SI"])
    parser.add_argument("--sense", default=None, choices=["minimize", "maximize"],
                        help="default: minimize for TPC, maximize otherwise")
    parser.add_argument("--scenario", default="base_case", choices=["base_case", "growing_demand"])
    parser.add_argument("--pi-exp", type=float, default=0.2)
    parser.add_argument("--w1", type=float, default=1)
    parser.add_argument("--w2", type=float, default=1)
    parser.add_argument("--solver", default="gurobi", choices=["gurobi", "cplex"])
    parser.add_argument("--mip-focus", type=int, default=3)
    parser.add_argument("--cuts", type=int, default=2)
    parser.add_argument("--numeric-focus", type=int, default=None)
    parser.add_argument("--presolve", type=int, default=None)
    args = parser.parse_args()

    sense = args.sense or ("minimize" if args.objective == "TPC" else "maximize")
    build_model(
        args.w1, args.w2, MAX_EV, MAX_JC, args.objective, sense,
        pi_exp=args.pi_exp, solver_name=args.solver,
        numeric_focus=args.numeric_focus, presolve=args.presolve,
        mip_focus=args.mip_focus, cuts=args.cuts,
        data_dir=os.path.join(DATA_DIR, args.scenario),
    )


if __name__ == "__main__":
    main()

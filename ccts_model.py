"""
Module CCTS (Carbon Capture, Transport and Storage).
Regroupe la lecture des instances, la construction et la résolution du modèle MILP Pyomo.
"""

import math
import time
import pyomo.environ as pyo


def read_instance(path):
    """
    Lit un fichier d'instance CCTS et renvoie un dictionnaire complet des données.
    Corrige automatiquement la valeur de n_capture si incohérente avec N.
    """
    scalars = []
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                name, val = line.split('=')
                scalars.append((name.strip(), float(val)))
            else:
                rows.append([float(x) for x in line.split(',')])

    sc = dict(scalars[:5])
    T, N = int(sc['T']), int(sc['N'])
    n_s, n_h = int(sc['n_storage']), int(sc['n_hubs'])
    n_c = N - n_s - n_h

    data = {'T': T, 'N': N}
    data['storage'] = {int(r[0]): (r[1], r[2]) for r in rows[:n_s]}
    data['avail'] = {int(r[0]): int(r[3]) if len(r) > 3 else 0 for r in rows[:n_s]}
    data['capture'] = {int(r[0]): (r[1], r[2]) for r in rows[n_s:n_s+n_c]}
    data['E'] = {int(r[0]): r[3] for r in rows[n_s:n_s+n_c]}
    data['hubs'] = {int(r[0]): (r[1], r[2]) for r in rows[n_s+n_c:n_s+n_c+n_h]}

    w = rows[n_s+n_c+n_h:n_s+n_c+n_h+N]
    data['w_rail'] = {(i, j): int(w[i][j]) for i in range(N) for j in range(N)}

    techno = ['truck', 'rail', 'pipe']
    params = scalars[5:]
    data['beta_1'] = {l: params[3*k][1] for k, l in enumerate(techno)}
    data['beta_2'] = {l: params[3*k+1][1] for k, l in enumerate(techno)}
    data['gamma'] = {l: params[3*k+2][1] for k, l in enumerate(techno)}
    data['alpha_1'] = dict(params)['alpha_1']
    data['alpha_2'] = dict(params)['alpha_2']

    coords = {**data['storage'], **data['capture'], **data['hubs']}
    data['coords'] = coords
    data['L'] = {(i, j): math.dist(coords[i], coords[j]) for i in coords for j in coords}
    return data


def build_ccts_model(data, alpha_2=None, beta_1_truck=None, Q_T_ratio=0.2):
    """
    Construit le modèle d'optimisation Pyomo (ConcreteModel).
    Permet de surcharger optionnellement alpha_2, beta_1_truck et Q_T_ratio.
    """
    model = pyo.ConcreteModel(name="CCTS_Deployment")

    # 1. Ensembles
    model.NS = pyo.Set(initialize=sorted(data['storage']), doc='Sites de stockage')
    model.NC = pyo.Set(initialize=sorted(data['capture']), doc='Sites de captage')
    model.NT = pyo.Set(initialize=sorted(data['hubs']), doc='Hubs de transport')
    model.N = pyo.Set(initialize=sorted(data['coords']), doc='Tous les noeuds')
    model.A = pyo.Set(initialize=[(i, j) for i in model.N for j in model.N if i != j], doc='Arcs orientés')
    model.L = pyo.Set(initialize=['truck', 'rail', 'pipe'], doc='Technologies de transport')
    model.T = pyo.Set(initialize=range(data['T']), doc='Horizon temporel (annees)')

    # 2. Paramètres
    beta_1_dict = dict(data['beta_1'])
    if beta_1_truck is not None:
        beta_1_dict['truck'] = float(beta_1_truck)

    alpha_2_val = float(alpha_2) if alpha_2 is not None else float(data['alpha_2'])

    model.E = pyo.Param(model.NC, initialize=data['E'], doc='Emissions annuelles brutes')
    model.avail = pyo.Param(model.NS, initialize=data['avail'], doc='Annee disponibilite stockage')
    model.dist = pyo.Param(model.A, initialize={a: data['L'][a] for a in model.A}, doc='Distances')
    model.w_rail = pyo.Param(model.A, initialize={a: data['w_rail'][a] for a in model.A}, doc='Disponibilite rail')

    model.beta_1 = pyo.Param(model.L, initialize=beta_1_dict, doc='Cout operatoire par tonne-km')
    model.beta_2 = pyo.Param(model.L, initialize=data['beta_2'], doc='Cout operatoire fixe par tonne')
    model.gamma = pyo.Param(model.L, initialize=data['gamma'], doc='Facteur emission transport')

    model.alpha_1 = pyo.Param(initialize=data['alpha_1'], doc='Investissement pipeline par tonne capacite')
    model.alpha_2 = pyo.Param(initialize=alpha_2_val, doc='Investissement pipeline par km longueur')

    Q_init = sum(data['E'].values())
    Q_T = Q_T_ratio * Q_init
    slope = (Q_T - Q_init) / data['T']
    BigM = 0.9 * sum(data['E'].values())

    model.Q_init = pyo.Param(initialize=Q_init)
    model.Q_T = pyo.Param(initialize=Q_T)
    model.a = pyo.Param(initialize=slope)
    model.M = pyo.Param(initialize=BigM)

    # 3. Variables de décision
    model.CC = pyo.Var(model.NC, model.T, domain=pyo.Binary, doc='Installation captage a t')
    model.F = pyo.Var(model.L, model.A, model.T, domain=pyo.NonNegativeReals, doc='Flux CO2 sur arc a l annee t')
    model.P = pyo.Var(model.A, model.T, domain=pyo.NonNegativeReals, doc='Capacite pipeline installee a t')
    model.p = pyo.Var(model.A, model.T, domain=pyo.Binary, doc='Existence pipeline a t')

    # Fonctions utilitaires flux
    def outflow(m, node, t):
        return sum(m.F[l, node, j, t] for l in m.L for j in m.N if j != node)

    def inflow(m, node, t):
        return sum(m.F[l, j, node, t] for l in m.L for j in m.N if j != node)

    # 4. Contraintes
    def emissions_rule(m, t):
        E_sites = sum(m.E[i] * (1 - m.CC[i, t]) + 0.1 * m.E[i] * m.CC[i, t] for i in m.NC)
        E_trans = sum(m.gamma[l] * m.F[l, i, j, t] * m.dist[i, j] for l in m.L for (i, j) in m.A)
        return E_sites + E_trans <= m.Q_init + m.a * t
    model.emissions = pyo.Constraint(model.T, rule=emissions_rule)

    def hub_balance_rule(m, j, t):
        return inflow(m, j, t) == outflow(m, j, t)
    model.hub_balance = pyo.Constraint(model.NT, model.T, rule=hub_balance_rule)

    def capture_rule(m, i, t):
        return outflow(m, i, t) - inflow(m, i, t) == 0.9 * m.E[i] * m.CC[i, t]
    model.capture = pyo.Constraint(model.NC, model.T, rule=capture_rule)

    def storage_sink_rule(m, s, t):
        return outflow(m, s, t) == 0
    model.storage_sink = pyo.Constraint(model.NS, model.T, rule=storage_sink_rule)

    def storage_avail_rule(m, s, t):
        if t >= m.avail[s]:
            return pyo.Constraint.Skip
        return inflow(m, s, t) == 0
    model.storage_avail = pyo.Constraint(model.NS, model.T, rule=storage_avail_rule)

    def cc_irreversible_rule(m, i, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.CC[i, t] >= m.CC[i, t - 1]
    model.cc_irreversible = pyo.Constraint(model.NC, model.T, rule=cc_irreversible_rule)

    def rail_rule(m, i, j, t):
        return m.F['rail', i, j, t] <= m.M * m.w_rail[i, j]
    model.rail = pyo.Constraint(model.A, model.T, rule=rail_rule)

    def pipe_capacity_rule(m, i, j, t):
        return m.F['pipe', i, j, t] <= m.P[i, j, t]
    model.pipe_capacity = pyo.Constraint(model.A, model.T, rule=pipe_capacity_rule)

    def pipe_exists_rule(m, i, j, t):
        return m.P[i, j, t] <= m.M * m.p[i, j, t]
    model.pipe_exists = pyo.Constraint(model.A, model.T, rule=pipe_exists_rule)

    def pipe_keep_rule(m, i, j, t):
        if t == data['T'] - 1:
            return pyo.Constraint.Skip
        return m.p[i, j, t + 1] >= m.p[i, j, t]
    model.pipe_keep = pyo.Constraint(model.A, model.T, rule=pipe_keep_rule)

    def pipe_frozen_up_rule(m, i, j, t):
        if t == data['T'] - 1:
            return pyo.Constraint.Skip
        return m.P[i, j, t + 1] - m.P[i, j, t] <= m.M * (1 - m.p[i, j, t])
    model.pipe_frozen_up = pyo.Constraint(model.A, model.T, rule=pipe_frozen_up_rule)

    def pipe_frozen_down_rule(m, i, j, t):
        if t == data['T'] - 1:
            return pyo.Constraint.Skip
        return m.P[i, j, t] - m.P[i, j, t + 1] <= m.M * (1 - m.p[i, j, t])
    model.pipe_frozen_down = pyo.Constraint(model.A, model.T, rule=pipe_frozen_down_rule)

    # 5. Fonction objectif
    def objective_rule(m):
        operating = sum(m.beta_1[l] * m.F[l, i, j, t] * m.dist[i, j] + m.beta_2[l] * m.F[l, i, j, t]
                        for l in m.L for (i, j) in m.A for t in m.T)
        investment = sum(m.alpha_1 * m.P[i, j, t] + m.alpha_2 * m.dist[i, j] * m.p[i, j, t]
                         for (i, j) in m.A for t in m.T)
        return operating + investment
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model


def resoudre_ccts(data, alpha_2=None, beta_1_truck=None, Q_T_ratio=0.2,
                  mipgap=0.005, timelimit=300, tee=False):
    """
    Construit et résout le modèle avec Gurobi, puis extrait et renvoie
    toutes les métriques de résolution et les résultats décisionnels.
    """
    model = build_ccts_model(data, alpha_2=alpha_2, beta_1_truck=beta_1_truck, Q_T_ratio=Q_T_ratio)

    opt = pyo.SolverFactory('gurobi')
    if mipgap is not None:
        opt.options['MIPGap'] = float(mipgap)
    if timelimit is not None:
        opt.options['TimeLimit'] = float(timelimit)

    t_start = time.time()
    results = opt.solve(model, tee=tee)
    solve_duration = time.time() - t_start

    # Extraction des métriques de performance
    status = str(results.solver.termination_condition)
    nodes = None
    # Aucune solution chargée (ex. scénario infaisable) : on ne lit pas les variables
    has_solution = all(v.value is not None for v in model.CC.values())
    try:
        obj_val = pyo.value(model.objective) if has_solution else None
    except Exception:
        obj_val = None

    # Extraction sécurisée des informations solveur
    solver_s = results.solver[0] if len(results.solver) > 0 else {}
    msg = solver_s.get('Message', solver_s.get('Termination message', ''))


    # Extraction des métriques décisionnelles
    total_transport = {'truck': 0.0, 'rail': 0.0, 'pipe': 0.0}
    try:
        for l in (model.L if has_solution else []):
            total_transport[l] = sum(pyo.value(model.F[l, i, j, t]) for (i, j) in model.A for t in model.T)
    except Exception:
        pass

    # Nombre de pipelines distincts construits sur l'horizon
    pipelines_built = []
    try:
        for (i, j) in (model.A if has_solution else []):
            if any(pyo.value(model.p[i, j, t]) > 0.5 for t in model.T):
                first_year = min(t for t in model.T if pyo.value(model.p[i, j, t]) > 0.5)
                cap = pyo.value(model.P[i, j, first_year])
                pipelines_built.append({'arc': (i, j), 'year': first_year, 'capacity': cap})
    except Exception:
        pass

    # Émissions effectivement réalisées (sites + transport), année par année
    emissions_par_an = {}
    try:
        for t in (model.T if has_solution else []):
            E_sites = sum(pyo.value(model.E[i] * (1 - 0.9 * model.CC[i, t])) for i in model.NC)
            E_trans = sum(pyo.value(model.gamma[l] * model.F[l, i, j, t] * model.dist[i, j])
                          for l in model.L for (i, j) in model.A)
            emissions_par_an[t] = E_sites + E_trans
    except Exception:
        emissions_par_an = {}

    try:
        lower_bound = float(results.problem.lower_bound)
    except Exception:
        lower_bound = None

    return {
        'status': status,
        'cost': obj_val,
        'lower_bound': lower_bound,
        'emissions_par_an': emissions_par_an,
        'emissions_total': sum(emissions_par_an.values()) if emissions_par_an else None,
        'E_ref': pyo.value(model.Q_init) * len(model.T),
        'time_sec': solve_duration,
        'nodes': nodes,
        'transport_totals': total_transport,
        'num_pipelines': len(pipelines_built),
        'pipelines': pipelines_built,
        'model': model,
        'data': data
    }


"""
Module de visualisation et de validation pour le projet CCTS
(Carbon Capture, Transport and Storage).

Ce module propose :
1. verify_constraints : un audit indépendant de toutes les contraintes (Hint 2 du sujet).
2. plot_network : tracé cartographique 2D du réseau à une année t donnée.
3. plot_milestones : planche 2x2 des années pivots (t=0, 5, 14, 24) pour le rapport LaTeX.
4. plot_summary_dashboard : trajectoire d'émissions et mix modal de transport sur 25 ans.
5. plot_interactive_slider : exploration interactive année par année.
"""

import math
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np


# =============================================================================
# 1. EXTRACTION DE LA SOLUTION
# =============================================================================

def extract_solution(model_or_sol, data):
    """
    Extrait les valeurs de décision sous forme de dictionnaires Python standard,
    que l'entrée soit un ConcreteModel Pyomo ou déjà un dictionnaire de solution.
    """
    if isinstance(model_or_sol, dict) and 'CC' in model_or_sol and 'F' in model_or_sol:
        return model_or_sol

    # Cas d'un modèle Pyomo
    from pyomo.environ import value

    T = data['T']
    coords = data['coords']
    technos = ['truck', 'rail', 'pipe']

    CC = {}
    for i in data['capture']:
        CC[i] = [int(round(value(model_or_sol.CC[i, t]))) for t in range(T)]

    F = {}
    for l in technos:
        for i in coords:
            for j in coords:
                if i != j:
                    F[(l, i, j)] = [float(value(model_or_sol.F[l, i, j, t])) for t in range(T)]

    p = {}
    P = {}
    for i in coords:
        for j in coords:
            if i != j:
                p[(i, j)] = [int(round(value(model_or_sol.p[i, j, t]))) for t in range(T)]
                P[(i, j)] = [float(value(model_or_sol.P[i, j, t])) for t in range(T)]

    obj_val = float(value(model_or_sol.objective)) if hasattr(model_or_sol, 'objective') else None

    return {
        'CC': CC,
        'F': F,
        'p': p,
        'P': P,
        'cost': obj_val
    }


# =============================================================================
# 2. VALIDATION FORMELLE DES CONTRAINTES (Hint 2)
# =============================================================================

def verify_constraints(data, model_or_sol, tolerance=1e-4, verbose=True, Q_T_ratio=0.2):
    """
    Vérifie indépendamment de Pyomo l'intégralité des contraintes du sujet.
    Affiche un tableau récapitulatif clair (PASS / FAIL) avec violations éventuelles.
    """
    sol = extract_solution(model_or_sol, data)
    T = data['T']
    nodes = list(data['coords'].keys())
    technos = ['truck', 'rail', 'pipe']

    CC = sol['CC']
    F = sol['F']
    p = sol['p']
    P = sol['P']

    Q_init = sum(data['E'].values())
    Q_T = Q_T_ratio * Q_init
    a = (Q_T - Q_init) / T

    checks = {}

    # 1. Évacuation CO2 capté
    max_viol_capture = 0.0
    for i in data['capture']:
        for t in range(T):
            outflow = sum(F.get((l, i, j), [0]*T)[t] for l in technos for j in nodes if j != i)
            inflow = sum(F.get((l, j, i), [0]*T)[t] for l in technos for j in nodes if j != i)
            expected = 0.9 * data['E'][i] * CC[i][t]
            diff = abs((outflow - inflow) - expected)
            if diff > max_viol_capture:
                max_viol_capture = diff
    checks['1. Évacuation CO2 captage (outflow - inflow = 0.9*E*CC)'] = max_viol_capture

    # 2. Bilan de flux aux hubs
    max_viol_hub = 0.0
    for h in data['hubs']:
        for t in range(T):
            outflow = sum(F.get((l, h, j), [0]*T)[t] for l in technos for j in nodes if j != h)
            inflow = sum(F.get((l, j, h), [0]*T)[t] for l in technos for j in nodes if j != h)
            diff = abs(outflow - inflow)
            if diff > max_viol_hub:
                max_viol_hub = diff
    checks['2. Bilan de flux aux hubs (inflow = outflow)'] = max_viol_hub

    # 3. Puits de stockage (aucune sortie)
    max_viol_sink = 0.0
    for s in data['storage']:
        for t in range(T):
            outflow = sum(F.get((l, s, j), [0]*T)[t] for l in technos for j in nodes if j != s)
            if outflow > max_viol_sink:
                max_viol_sink = outflow
    checks['3. Sites de stockage : aucune sortie (outflow = 0)'] = max_viol_sink

    # 4. Disponibilité du stockage
    max_viol_avail = 0.0
    for s in data['storage']:
        avail_year = data['avail'].get(s, 0)
        for t in range(avail_year):
            inflow = sum(F.get((l, j, s), [0]*T)[t] for l in technos for j in nodes if j != s)
            if inflow > max_viol_avail:
                max_viol_avail = inflow
    checks['4. Stockage indisponible : aucune entrée avant t_avail'] = max_viol_avail

    # 5. Disponibilité réseau ferroviaire
    max_viol_rail = 0.0
    for i in nodes:
        for j in nodes:
            if i != j and data['w_rail'].get((i, j), 0) == 0:
                for t in range(T):
                    flow_rail = F.get(('rail', i, j), [0]*T)[t]
                    if flow_rail > max_viol_rail:
                        max_viol_rail = flow_rail
    checks['5. Lignes de train : flux nul si ligne absente (w_rail = 0)'] = max_viol_rail

    # 6. Capacité des pipelines
    max_viol_pipecap = 0.0
    for i in nodes:
        for j in nodes:
            if i != j:
                for t in range(T):
                    flow_pipe = F.get(('pipe', i, j), [0]*T)[t]
                    cap_pipe = P.get((i, j), [0]*T)[t]
                    if flow_pipe - cap_pipe > max_viol_pipecap:
                        max_viol_pipecap = flow_pipe - cap_pipe
    checks['6. Capacité pipeline : flux pipe <= P_ij(t)'] = max_viol_pipecap

    # 7. Activation pipeline (capacité nulle si non construit)
    max_viol_pipebuilt = 0.0
    for i in nodes:
        for j in nodes:
            if i != j:
                for t in range(T):
                    if p.get((i, j), [0]*T)[t] == 0:
                        cap = P.get((i, j), [0]*T)[t]
                        if cap > max_viol_pipebuilt:
                            max_viol_pipebuilt = cap
    checks['7. Cohérence pipeline : capacité = 0 si non construit (p_ij=0)'] = max_viol_pipebuilt

    # 8. Maintien en service des pipelines
    max_viol_pipekeep = 0.0
    for i in nodes:
        for j in nodes:
            if i != j:
                for t in range(T - 1):
                    p_curr = p.get((i, j), [0]*T)[t]
                    p_next = p.get((i, j), [0]*T)[t + 1]
                    if p_curr > p_next:
                        max_viol_pipekeep = max(max_viol_pipekeep, p_curr - p_next)
    checks['8. Maintien des pipelines : p_ij(t+1) >= p_ij(t)'] = max_viol_pipekeep

    # 9. Capacité pipeline figée après construction
    max_viol_pipefrozen = 0.0
    for i in nodes:
        for j in nodes:
            if i != j:
                for t in range(T - 1):
                    if p.get((i, j), [0]*T)[t] == 1:
                        P_curr = P.get((i, j), [0]*T)[t]
                        P_next = P.get((i, j), [0]*T)[t + 1]
                        diff = abs(P_next - P_curr)
                        if diff > max_viol_pipefrozen:
                            max_viol_pipefrozen = diff
    checks['9. Capacité pipeline figée : P_ij(t+1) == P_ij(t) une fois construit'] = max_viol_pipefrozen

    # 10. Irréversibilité du captage
    max_viol_cc_irrev = 0.0
    for i in data['capture']:
        for t in range(T - 1):
            if CC[i][t] > CC[i][t + 1]:
                max_viol_cc_irrev = max(max_viol_cc_irrev, CC[i][t] - CC[i][t + 1])
    checks['10. Irréversibilité du captage : CC_i(t+1) >= CC_i(t)'] = max_viol_cc_irrev

    # 11. Plafond annuel d'émissions
    max_viol_emiss = 0.0
    margins = []
    for t in range(T):
        E_sites = sum(data['E'][i] * (1 - CC[i][t]) + 0.1 * data['E'][i] * CC[i][t] for i in data['capture'])
        E_trans = sum(data['gamma'][l] * F.get((l, i, j), [0]*T)[t] * data['L'][(i, j)]
                      for l in technos for i in nodes for j in nodes if i != j)
        E_tot = E_sites + E_trans
        limit = Q_init + a * t
        diff = E_tot - limit
        margins.append(limit - E_tot)
        if diff > max_viol_emiss:
            max_viol_emiss = diff
    checks['11. Trajectoire d\'émissions : E_tot(t) <= Q_init + a*t'] = max_viol_emiss

    all_passed = all(v <= tolerance for v in checks.values())

    if verbose:
        print("\n" + "=" * 78)
        print(" AUDIT INDÉPENDANT DES CONTRAINTES DE LA SOLUTION (Hint 2)")
        print("=" * 78)
        print(f" {'Contrainte vérifiée':<58} | {'Écart Max':<10} | {'Statut'}")
        print("-" * 78)
        for name, viol in checks.items():
            status = "[OK] PASS" if viol <= tolerance else "[!!] FAIL"
            print(f" {name:<58} | {viol:<10.2e} | {status}")
        print("-" * 78)
        print(f" Marge minimale au plafond d'émissions : {min(margins):.2f} t/an")
        if all_passed:
            print(" >> RÉSULTAT GLOBAL : TOUTES LES CONTRAINTES SONT RESPECTÉES [VALIDE]")
        else:
            print(" >> RÉSULTAT GLOBAL : ATTENTION, CERTAINES CONTRAINTES SONT ENFREINTES")
        print("=" * 78 + "\n")

    return {
        'all_passed': all_passed,
        'checks': checks,
        'min_emission_margin': min(margins)
    }


# =============================================================================
# 3. TRACÉ CARTOGRAPHIQUE 2D DU RÉSEAU (plot_network)
# =============================================================================

def plot_network(data, model_or_sol, t, ax=None, title=None, show_legend=True):
    """
    Trace la carte géographique 2D du réseau à l'année t avec :
    - Nœuds positionnés aux coordonnées exactes (x, y).
    - Sites de captage (verts si actifs, gris sinon, taille proportionnelle à E_i).
    - Sites de stockage (bleu si disponible à t, rouge si indisponible).
    - Hubs de transport (losanges gris).
    - Réseau ferré existant (pointillés gris en arrière-plan).
    - Pipelines construits (traits pleins verts épais avec capacité S).
    - Flux actifs par technologie (Rail en bleu, Pipe en vert, Camion en orange).
    """
    sol = extract_solution(model_or_sol, data)
    coords = data['coords']
    nodes = list(coords.keys())
    T = data['T']
    technos = ['truck', 'rail', 'pipe']

    CC = sol['CC']
    F = sol['F']
    p = sol['p']
    P = sol['P']

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        created_fig = True

    # Couleurs harmonieuses
    c_rail = '#1976D2'   # Bleu roi
    c_pipe = '#2E7D32'   # Vert forêt
    c_truck = '#E65100'  # Orange vif

    # 1. Tracé du réseau ferré existant en arrière-plan
    for i in nodes:
        for j in nodes:
            if i < j and (data['w_rail'].get((i, j), 0) == 1 or data['w_rail'].get((j, i), 0) == 1):
                xi, yi = coords[i]
                xj, yj = coords[j]
                ax.plot([xi, xj], [yi, yj], color='#CFD8DC', linestyle='--', linewidth=1.8, zorder=1)

    # 2. Tracé des pipelines installés en année t
    for i in nodes:
        for j in nodes:
            if i != j and p.get((i, j), [0]*T)[t] == 1:
                xi, yi = coords[i]
                xj, yj = coords[j]
                cap = P.get((i, j), [0]*T)[t]
                # Conduit physique sous forme de trait épais
                ax.plot([xi, xj], [yi, yj], color='#A5D6A7', linewidth=6.0, alpha=0.6, zorder=2)

    # 3. Tracé des flux actifs de CO2 à l'année t
    # On calcule le flux max pour normaliser les épaisseurs de flèche
    all_flows_t = [F.get((l, i, j), [0]*T)[t] for l in ['truck', 'rail', 'pipe'] for i in nodes for j in nodes if i != j]
    max_flow = max(all_flows_t) if all_flows_t and max(all_flows_t) > 0 else 1.0

    for (l, i, j) in [(l, i, j) for l in ['truck', 'rail', 'pipe'] for i in nodes for j in nodes if i != j]:
        flow = F.get((l, i, j), [0]*T)[t]
        if flow > 1e-2:
            xi, yi = coords[i]
            xj, yj = coords[j]
            dx, dy = xj - xi, yj - yi
            dist = math.hypot(dx, dy)
            if dist < 1e-4:
                continue

            # Décalage léger selon la techno pour éviter la superposition exacte
            offset_factor = {'rail': -0.15, 'pipe': 0.0, 'truck': 0.15}[l]
            perp_x = -dy / dist * offset_factor
            perp_y = dx / dist * offset_factor

            start_x, start_y = xi + perp_x, yi + perp_y
            end_x, end_y = xj + perp_x, yj + perp_y

            color = {'rail': c_rail, 'pipe': c_pipe, 'truck': c_truck}[l]
            lw = 1.8 + 3.2 * (flow / max_flow)

            arrow = patches.FancyArrowPatch(
                (start_x, start_y), (end_x, end_y),
                arrowstyle='-|>',
                mutation_scale=14,
                color=color,
                linewidth=lw,
                zorder=4,
                shrinkA=14, shrinkB=14
            )
            ax.add_patch(arrow)

            # Annotation du flux au milieu de l'arc
            mx = (start_x + end_x) / 2.0
            my = (start_y + end_y) / 2.0
            ax.text(
                mx, my, f"{flow:.0f}",
                fontsize=8, fontweight='bold', color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor='white', edgecolor=color, alpha=0.85, lw=0.8),
                ha='center', va='center', zorder=6
            )

    # 4. Tracé des nœuds
    # A. Hubs de transport
    for h in data['hubs']:
        x, y = coords[h]
        ax.scatter(x, y, s=260, marker='D', facecolor='#ECEFF1', edgecolor='#455A64', linewidth=2.0, zorder=5)
        ax.text(x, y - 0.45, f"Hub {h}", fontsize=9, fontweight='bold', ha='center', va='top', zorder=6)

    # B. Sites de captage
    for c in data['capture']:
        x, y = coords[c]
        is_active = (CC[c][t] == 1)
        ei = data['E'][c]
        size = 180 + 0.9 * ei
        if is_active:
            fc = '#C8E6C9'  # Vert clair
            ec = '#2E7D32'  # Bordure vert foncé
            label_text = f"Site {c}\nCap: {0.9*ei:.0f}t"
        else:
            fc = '#F5F5F5'  # Gris très clair
            ec = '#9E9E9E'  # Gris
            label_text = f"Site {c}\n(0t)"

        ax.scatter(x, y, s=size, marker='o', facecolor=fc, edgecolor=ec, linewidth=2.2, zorder=5)
        ax.text(x, y, str(c), fontsize=9, fontweight='bold', color='#1B5E20' if is_active else '#424242',
                ha='center', va='center', zorder=6)

        # Décalage intelligent du texte pour éviter la collision entre sites proches (ex: site 4 et site 6)
        y_offset = -0.55
        if c == 4 and 6 in data['capture']:
            y_offset = -0.45  # un peu plus haut
        elif c == 6 and 4 in data['capture']:
            y_offset = -0.65  # un peu plus bas

        ax.text(x, y + y_offset, label_text, fontsize=7.5, ha='center', va='top', zorder=6)

    # C. Sites de stockage
    for s in data['storage']:
        x, y = coords[s]
        is_avail = (t >= data['avail'].get(s, 0))
        # Calcul du CO2 reçu cette année
        co2_in = sum(F.get((l, j, s), [0]*T)[t] for l in technos for j in nodes if j != s)
        if is_avail:
            fc = '#BBDEFB'  # Bleu clair
            ec = '#0D47A1'  # Bleu nuit
            stat_lbl = f"Reçu: {co2_in:.0f}t"
        else:
            fc = '#FFCDD2'  # Rouge pâle
            ec = '#B71C1C'  # Rouge foncé
            stat_lbl = f"Dispo t={data['avail'].get(s, 0)}"

        ax.scatter(x, y, s=340, marker='s', facecolor=fc, edgecolor=ec, linewidth=2.5, zorder=5)
        ax.text(x, y, f"S{s}", fontsize=9, fontweight='bold', color='#0D47A1' if is_avail else '#B71C1C',
                ha='center', va='center', zorder=6)
        ax.text(x, y + 0.65, f"Stockage {s}\n{stat_lbl}", fontsize=8, fontweight='bold',
                color=ec, ha='center', va='bottom', zorder=6)

    # Paramètres d'affichage du graphique
    all_x = [c[0] for c in coords.values()]
    all_y = [c[1] for c in coords.values()]
    margin_x = max(1.5, (max(all_x) - min(all_x)) * 0.12)
    margin_y = max(1.5, (max(all_y) - min(all_y)) * 0.15)
    ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
    ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.4, color='#90A4AE')

    # Titre avec bilan de l'année
    E_sites = sum(data['E'][i] * (1 - CC[i][t]) + 0.1 * data['E'][i] * CC[i][t] for i in data['capture'])
    E_trans = sum(data['gamma'][l] * F.get((l, i, j), [0]*T)[t] * data['L'][(i, j)]
                  for l in technos for i in nodes for j in nodes if i != j)
    E_tot = E_sites + E_trans
    Q_init = sum(data['E'].values())
    Q_T = 0.2 * Q_init
    a = (Q_T - Q_init) / T
    lim_t = Q_init + a * t

    if title is None:
        title = f"Année t = {t} | Émissions : {E_tot:.1f} t/an (Plafond : {lim_t:.1f} t/an)"
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    # Légende explicative
    if show_legend:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#C8E6C9', markeredgecolor='#2E7D32',
                   markersize=10, label='Captage actif (90%)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#F5F5F5', markeredgecolor='#9E9E9E',
                   markersize=10, label='Captage inactif'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#BBDEFB', markeredgecolor='#0D47A1',
                   markersize=10, label='Stockage opérationnel'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFCDD2', markeredgecolor='#B71C1C',
                   markersize=10, label='Stockage non ouvert'),
            Line2D([0], [0], color='#CFD8DC', linestyle='--', linewidth=2, label='Ligne ferroviaire'),
            Line2D([0], [0], color='#A5D6A7', linewidth=4, label='Pipeline construit'),
            Line2D([0], [0], color=c_rail, linewidth=2.5, label='Flux Train'),
            Line2D([0], [0], color=c_pipe, linewidth=2.5, label='Flux Pipeline'),
            Line2D([0], [0], color=c_truck, linewidth=2.5, label='Flux Camion'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=8.5, frameon=True, facecolor='#FAFAFA', edgecolor='#CFD8DC')

    if created_fig:
        plt.tight_layout()
        plt.show()

    return ax


# =============================================================================
# 4. PLANCHE 2x2 DES JALONS CLÉS POUR LE RAPPORT (plot_milestones)
# =============================================================================

def plot_milestones(data, model_or_sol, years=None, save_path=None):
    """
    Génère une figure 2x2 des années charnières de déploiement pour le rapport LaTeX.
    Par défaut : t=0 (initial), t=5 (début rail), t=14 (mise en service stockage 7), t=24 (horizon final).
    """
    if years is None:
        # On choisit intelligemment les années selon les dates de stockage
        s2_avail = max(data['avail'].values()) if data['avail'] else 14
        years = [0, 5, s2_avail, data['T'] - 1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (t, ax) in enumerate(zip(years, axes)):
        is_last = (idx == 1) # Légende sur le panneau supérieur droit
        plot_network(data, model_or_sol, t=t, ax=ax, show_legend=is_last)

    plt.suptitle("Évolution du déploiement de l'infrastructure CCTS aux étapes clés",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Figure des jalons enregistrée sous : {save_path}")

    plt.show()
    return fig


# =============================================================================
# 5. DASHBOARD TEMPOREL DES ÉMISSIONS ET DU MIX MODAL (plot_summary)
# =============================================================================

def plot_summary_dashboard(data, model_or_sol, save_path=None):
    """
    Trace un tableau de bord à 2 panneaux sur l'horizon de 25 ans :
    - Panneau 1 : Trajectoire des émissions réelles vs plafond imposé Q(t).
    - Panneau 2 : Évolution annuelle des volumes de CO2 transportés par mode (Camion, Train, Pipe).
    """
    sol = extract_solution(model_or_sol, data)
    T = data['T']
    nodes = list(data['coords'].keys())
    technos = ['truck', 'rail', 'pipe']

    CC = sol['CC']
    F = sol['F']

    Q_init = sum(data['E'].values())
    Q_T = 0.2 * Q_init
    a = (Q_T - Q_init) / T

    years = list(range(T))
    targets = [Q_init + a * t for t in years]

    emiss_residual = []
    emiss_uncaptured = []
    emiss_trans = []
    emiss_tot = []

    flow_truck = []
    flow_rail = []
    flow_pipe = []

    for t in years:
        res = sum(0.1 * data['E'][i] * CC[i][t] for i in data['capture'])
        unc = sum(data['E'][i] * (1 - CC[i][t]) for i in data['capture'])
        tra = sum(data['gamma'][l] * F.get((l, i, j), [0]*T)[t] * data['L'][(i, j)]
                  for l in technos for i in nodes for j in nodes if i != j)
        emiss_residual.append(res)
        emiss_uncaptured.append(unc)
        emiss_trans.append(tra)
        emiss_tot.append(res + unc + tra)

        flow_truck.append(sum(F.get(('truck', i, j), [0]*T)[t] for i in nodes for j in nodes if i != j))
        flow_rail.append(sum(F.get(('rail', i, j), [0]*T)[t] for i in nodes for j in nodes if i != j))
        flow_pipe.append(sum(F.get(('pipe', i, j), [0]*T)[t] for i in nodes for j in nodes if i != j))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- PANNEAU 1 : TRAJECTOIRE CARBONE ---
    ax1.plot(years, targets, color='#D32F2F', linestyle='--', linewidth=2.2, label='Plafond d\'émissions (Qinit + a*t)')
    ax1.plot(years, emiss_tot, color='#1B5E20', linewidth=2.5, marker='o', markersize=5, label='Émissions réelles totales')
    ax1.fill_between(years, emiss_tot, targets, color='#C8E6C9', alpha=0.35, label='Marge de réduction atteinte')

    ax1.set_title("Trajectoire de réduction des émissions de CO2 (25 ans)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Année t", fontsize=10)
    ax1.set_ylabel("Émissions annuelles (t CO2 / an)", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.set_xticks(range(0, T, 2))
    ax1.legend(loc='upper right', fontsize=9)

    # --- PANNEAU 2 : MIX MODAL DE TRANSPORT ---
    c_rail = '#1976D2'
    c_pipe = '#2E7D32'
    c_truck = '#E65100'

    p1 = ax2.bar(years, flow_rail, color=c_rail, label='Train', width=0.7)
    p2 = ax2.bar(years, flow_pipe, bottom=flow_rail, color=c_pipe, label='Pipeline', width=0.7)
    bottom_truck = [flow_rail[t] + flow_pipe[t] for t in years]
    p3 = ax2.bar(years, flow_truck, bottom=bottom_truck, color=c_truck, label='Camion', width=0.7)

    # Annotation de la mise en service du stockage 2 si présent
    s_avail = max(data['avail'].values()) if data['avail'] else None
    if s_avail and s_avail > 0:
        ax2.axvline(x=s_avail - 0.5, color='#7B1FA2', linestyle=':', linewidth=1.8)
        ax2.text(s_avail - 0.4, max(bottom_truck)*0.9, f"Ouverture Stockage 7\n(t = {s_avail})",
                 fontsize=8.5, color='#7B1FA2', fontweight='bold')

    ax2.set_title("Évolution du mix de transport de CO2 par technologie", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Année t", fontsize=10)
    ax2.set_ylabel("CO2 transporté par an (t/an)", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.set_xticks(range(0, T, 2))
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Dashboard de synthèse enregistré sous : {save_path}")

    plt.show()
    return fig


# =============================================================================
# 6. VISUALISATION INTERACTIVE AVEC SLIDER TEMPOREL (plot_interactive_slider)
# =============================================================================

def plot_interactive_slider(data, model_or_sol):
    """
    Crée une fenêtre interactive avec un slider temporel permettant de faire défiler
    les années t = 0 à T-1 de manière dynamique.
    Très utile dans un environnement Jupyter interactif (%matplotlib widget) ou en script standalone.
    """
    from matplotlib.widgets import Slider

    T = data['T']
    fig, ax = plt.subplots(figsize=(11, 8.5))
    plt.subplots_adjust(bottom=0.18, right=0.82)

    initial_t = 0
    plot_network(data, model_or_sol, t=initial_t, ax=ax, show_legend=True)

    # Position du slider en bas de la figure
    ax_slider = plt.axes([0.20, 0.06, 0.55, 0.03], facecolor='#ECEFF1')
    slider = Slider(
        ax=ax_slider,
        label='Année t ',
        valmin=0,
        valmax=T - 1,
        valinit=initial_t,
        valstep=1,
        color='#1976D2'
    )

    def update(val):
        t_val = int(slider.val)
        ax.clear()
        plot_network(data, model_or_sol, t=t_val, ax=ax, show_legend=False)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    return fig, slider


# =============================================================================
# 7. VISUALISATION INTERACTIVE PLOTLY (plot_interactive_plotly)
# =============================================================================

def plot_interactive_plotly(data, model_or_sol, save_html='figures/reseau_interactif.html'):
    """
    Crée une figure interactive Plotly complète avec un slider d'années (t=0..T-1),
    un bouton de lecture automatique (Play/Pause 25 ans), et des info-bulles détaillées.
    Fonctionne directement dans Jupyter Notebook sans backend GUI et s'exporte en fichier HTML autonome.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[!] Le module 'plotly' n'est pas installé. Lancez : pip install plotly")
        return None

    sol = extract_solution(model_or_sol, data)
    T = data['T']
    coords = data['coords']
    nodes = list(coords.keys())
    technos = ['truck', 'rail', 'pipe']

    CC = sol['CC']
    F = sol['F']
    p = sol['p']
    P = sol['P']

    Q_init = sum(data['E'].values())
    Q_T = 0.2 * Q_init
    a = (Q_T - Q_init) / T

    def build_traces_for_year(t):
        traces = []

        # 1. Fond ferroviaire
        rx, ry = [], []
        for i in nodes:
            for j in nodes:
                if i < j and (data['w_rail'].get((i, j), 0) == 1 or data['w_rail'].get((j, i), 0) == 1):
                    rx.extend([coords[i][0], coords[j][0], None])
                    ry.extend([coords[i][1], coords[j][1], None])
        traces.append(go.Scatter(
            x=rx, y=ry, mode='lines',
            line=dict(color='#CFD8DC', width=2, dash='dot'),
            name='Ligne ferroviaire', hoverinfo='none'
        ))

        # 2. Pipelines construits à t
        px, py, p_hover = [], [], []
        for i in nodes:
            for j in nodes:
                if i != j and p.get((i, j), [0]*T)[t] == 1:
                    px.extend([coords[i][0], coords[j][0], None])
                    py.extend([coords[i][1], coords[j][1], None])
                    cap = P.get((i, j), [0]*T)[t]
                    p_hover.extend([f"Pipeline {i} -> {j} | Capacité : {cap:.1f} t/an"] * 3)
        traces.append(go.Scatter(
            x=px, y=py, mode='lines',
            line=dict(color='#A5D6A7', width=6),
            name='Pipeline construit', hoverinfo='text', hovertext=p_hover
        ))

        # 3. Flux actifs par technologie
        colors = {'rail': '#1976D2', 'pipe': '#2E7D32', 'truck': '#E65100'}
        names = {'rail': 'Flux Train', 'pipe': 'Flux Pipeline', 'truck': 'Flux Camion'}

        for l in ['rail', 'pipe', 'truck']:
            fx, fy, ftext = [], [], []
            for i in nodes:
                for j in nodes:
                    if i != j:
                        flow = F.get((l, i, j), [0]*T)[t]
                        if flow > 1e-2:
                            fx.extend([coords[i][0], coords[j][0], None])
                            fy.extend([coords[i][1], coords[j][1], None])
                            ftext.extend([f"{names[l]} : {i} -> {j} ({flow:.1f} t/an)"] * 3)
            traces.append(go.Scatter(
                x=fx, y=fy, mode='lines',
                line=dict(color=colors[l], width=4),
                name=names[l], hoverinfo='text', hovertext=ftext
            ))

        # 4. Sites de stockage
        sx, sy, stext, scolor = [], [], [], []
        for s in data['storage']:
            sx.append(coords[s][0])
            sy.append(coords[s][1])
            avail = (t >= data['avail'].get(s, 0))
            inflow = sum(F.get((l, j, s), [0]*T)[t] for l in technos for j in nodes if j != s)
            scolor.append('#1976D2' if avail else '#D32F2F')
            status_str = "Opérationnel" if avail else f"Fermé (ouvre en t={data['avail'][s]})"
            stext.append(f"<b>Stockage {s}</b><br>Statut : {status_str}<br>CO2 reçu cette année : {inflow:.1f} t/an")
        traces.append(go.Scatter(
            x=sx, y=sy, mode='markers+text',
            marker=dict(symbol='square', size=22, color=scolor, line=dict(color='white', width=2)),
            text=[f"S{s}" for s in data['storage']], textposition='middle center',
            textfont=dict(color='white', size=11, family='Arial Black'),
            name='Stockage', hoverinfo='text', hovertext=stext
        ))

        # 5. Sites de captage
        cx, cy, ctext, ccolor, csize = [], [], [], [], []
        for c in data['capture']:
            cx.append(coords[c][0])
            cy.append(coords[c][1])
            is_act = (CC[c][t] == 1)
            ei = data['E'][c]
            ccolor.append('#2E7D32' if is_act else '#9E9E9E')
            csize.append(16 + ei * 0.04)
            status_cc = "Captage ACTIF (90%)" if is_act else "Inactif (aucun captage)"
            co2_cap = 0.9 * ei if is_act else 0.0
            ctext.append(f"<b>Site de captage {c}</b><br>Émissions brutes : {ei:.0f} t/an<br>Statut : {status_cc}<br>CO2 capté : {co2_cap:.1f} t/an")
        traces.append(go.Scatter(
            x=cx, y=cy, mode='markers+text',
            marker=dict(symbol='circle', size=csize, color=ccolor, line=dict(color='white', width=1.5)),
            text=[f"{c}" for c in data['capture']], textposition='middle center',
            textfont=dict(color='white', size=10, family='Arial Black'),
            name='Captage', hoverinfo='text', hovertext=ctext
        ))

        # 6. Hubs
        if data['hubs']:
            hx = [coords[h][0] for h in data['hubs']]
            hy = [coords[h][1] for h in data['hubs']]
            htext = [f"<b>Hub de transport {h}</b>" for h in data['hubs']]
            traces.append(go.Scatter(
                x=hx, y=hy, mode='markers+text',
                marker=dict(symbol='diamond', size=18, color='#455A64', line=dict(color='white', width=1.5)),
                text=[f"H{h}" for h in data['hubs']], textposition='middle center',
                textfont=dict(color='white', size=9),
                name='Hub transit', hoverinfo='text', hovertext=htext
            ))

        return traces

    # Construction des frames d'animation pour chaque année
    frames = []
    steps = []
    for t in range(T):
        E_sites = sum(data['E'][i] * (1 - CC[i][t]) + 0.1 * data['E'][i] * CC[i][t] for i in data['capture'])
        E_trans = sum(data['gamma'][l] * F.get((l, i, j), [0]*T)[t] * data['L'][(i, j)]
                      for l in technos for i in nodes for j in nodes if i != j)
        E_tot = E_sites + E_trans
        lim_t = Q_init + a * t
        t_title = f"Réseau CCTS — Année t = {t} | Émissions : {E_tot:.1f} t/an (Plafond : {lim_t:.1f} t/an)"

        fr_traces = build_traces_for_year(t)
        frames.append(go.Frame(data=fr_traces, name=str(t), layout=go.Layout(title_text=t_title)))
        steps.append({
            'args': [[str(t)], {'frame': {'duration': 400, 'redraw': True}, 'mode': 'immediate'}],
            'label': f"{t}",
            'method': 'animate'
        })

    initial_traces = build_traces_for_year(0)
    fig = go.Figure(data=initial_traces, frames=frames)

    fig.update_layout(
        title=dict(text=f"Réseau CCTS — Année t = 0 | Émissions : {Q_init:.1f} t/an (Plafond : {Q_init:.1f} t/an)",
                   font=dict(size=14, color='#263238')),
        xaxis=dict(showgrid=True, gridcolor='#ECEFF1', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#ECEFF1', zeroline=False, scaleanchor='x', scaleratio=1),
        plot_bgcolor='#FAFAFA',
        width=950, height=750,
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {'prefix': 'Année : t = ', 'font': {'size': 14, 'color': '#1976D2'}},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.05,
            'y': 0,
            'steps': steps
        }],
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 600, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}],
                    'label': '▶ Lecture (25 ans)',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                    'label': '❚❚ Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 60},
            'showactive': False,
            'type': 'buttons',
            'x': 0.05,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }]
    )

    if save_html:
        os.makedirs(os.path.dirname(save_html) or '.', exist_ok=True)
        fig.write_html(save_html)
        print(f"[OK] Fichier HTML interactif sauvegardé sous : {save_html}")

    return fig



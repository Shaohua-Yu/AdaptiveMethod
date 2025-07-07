
import math, random, copy, time, sys
from collections import defaultdict
import numpy as np
import multiprocessing as mp
from functools import partial

V1_MAX = 3
V2_MAX = 4
V_S_MAX = defaultdict(lambda: 4)

BIG_PENALTY   = 1_000_000
OVER_PENALTY  = 10_000
WEIGHT_REACTION = .1
EPS = 1e-6


class Location:
    def __init__(self, idx, loc_id, tp, x, y,
                 demand=0, capacity=float("inf")):
        self.idx = idx
        self.id = loc_id
        self.type = tp
        self.x, self.y = x, y
        self.demand  = demand
        self.capacity = capacity

class Route:
    def __init__(self, name, echelon, cap):
        self.name  = name
        self.echelon = echelon
        self.capacity = cap
        self.nodes = []
        self.cost  = 0.0
        self.load  = 0.0
        self.payload = defaultdict(float)

    def __repr__(self):
        seq = " -> ".join(map(str, self.nodes))
        if self.echelon == 1:
            pl = ", ".join(f"{sid}:{q:.0f}" for sid, q in self.payload.items())
            return f"{self.name}: {seq} | cost={self.cost:.1f} load={self.load:.1f} [{pl}]"
        else:
            return f"{self.name}: {seq} | cost={self.cost:.1f} load={self.load:.1f}"

    def update_load_cost(self, L, M):
        if len(self.nodes) < 2:
            self.cost = self.load = 0.0
            return
        self.cost = sum(
            M[L[a].idx, L[b].idx] for a, b in zip(self.nodes[:-1], self.nodes[1:])
        )
        if self.echelon == 2:
            self.load = sum(L[c].demand for c in self.nodes[1:-1])
        else:
            self.load = sum(self.payload.values())

class Solution:
    def __init__(self):
        self.R1 = []
        self.R2 = defaultdict(list)
        self.unassigned_customers = []
        self.total_cost  = float("inf")
        self.infeas_cost = float("inf")
    def deepcopy(self):      return copy.deepcopy(self)
    def all_served_customers(self):
        s = set(); _ = [s.update(r.nodes[1:-1])
             for lst in self.R2.values() for r in lst]; return s
    def l1_truck_used(self):         return len(self.R1)
    def l2_truck_used(self):         return sum(len(v) for v in self.R2.values())
    def l2_truck_used_at(self, sat): return len(self.R2[sat])


def build_distance_matrix(locs):
    n = len(locs)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(locs[i].x - locs[j].x,
                           locs[i].y - locs[j].y)
            M[i, j] = M[j, i] = d
    return M
def dist(id1, id2, L, M):
    return M[L[id1].idx, L[id2].idx]


def calc_R2_cost_load(r, L, M):
    cost = sum(dist(r.nodes[i], r.nodes[i + 1], L, M)
               for i in range(len(r.nodes) - 1))
    load = sum(L[c].demand for c in r.nodes[1:-1])
    return cost, load

def evaluate(sol: Solution, L, M):
    cost = infeas = 0.0
    sat_demand = defaultdict(float)
    sat_inflow = defaultdict(float)

    for sid, routes in sol.R2.items():
        for r in routes:
            r.cost, r.load = calc_R2_cost_load(r, L, M)
            cost += r.cost
            excess = max(0, r.load - r.capacity)
            infeas += OVER_PENALTY * excess
            sat_demand[sid] += r.load

    for r in sol.R1:
        r.update_load_cost(L, M)
        cost += r.cost
        infeas += OVER_PENALTY * max(0, r.load - r.capacity)
        for sid, qty in r.payload.items():
            sat_inflow[sid] += qty

    for sid in set(list(sat_demand.keys()) + list(sat_inflow.keys())):
        diff = sat_inflow[sid] - sat_demand[sid]
        if abs(diff) > EPS:
            infeas += OVER_PENALTY * abs(diff)

    if sol.l1_truck_used() > V1_MAX:
        infeas += BIG_PENALTY * (sol.l1_truck_used() - V1_MAX)
    if sol.l2_truck_used() > V2_MAX:
        infeas += BIG_PENALTY * (sol.l2_truck_used() - V2_MAX)

    cost += len(sol.unassigned_customers) * BIG_PENALTY

    sol.total_cost  = cost + infeas
    sol.infeas_cost = infeas
    return sol.total_cost


def is_feasible(sol): return sol.infeas_cost == 0

def l1_two_opt_route(route: Route, L, M):
    nodes = route.nodes
    best_gain = 0.0
    best_i = best_j = None
    for i in range(1, len(nodes) - 2):
        for j in range(i + 1, len(nodes) - 1):
            a, b = nodes[i - 1], nodes[i]
            c, d = nodes[j], nodes[j + 1]
            gain = (dist(a, b, L, M) + dist(c, d, L, M)
                    - dist(a, c, L, M) - dist(b, d, L, M))
            if gain > best_gain + EPS:
                best_gain, best_i, best_j = gain, i, j
    if best_i is not None:
        route.nodes = (nodes[:best_i] +
                       list(reversed(nodes[best_i:best_j + 1])) +
                       nodes[best_j + 1:])
        return True
    return False

def improve_L1_routes(sol: Solution, L, M, max_loop=20):
    loop, improved = 0, True
    while improved and loop < max_loop:
        improved = False
        for r in sol.R1:
            if l1_two_opt_route(r, L, M):
                improved = True
        loop += 1
    evaluate(sol, L, M)
    return sol

def l2_two_opt_route(route: Route, L, M):
    nodes = route.nodes
    best_gain = 0.0
    best_i = best_j = None
    for i in range(1, len(nodes) - 2):
        for j in range(i + 1, len(nodes) - 1):
            a, b = nodes[i - 1], nodes[i]
            c, d = nodes[j], nodes[j + 1]
            gain = (dist(a, b, L, M) + dist(c, d, L, M)
                    - dist(a, c, L, M) - dist(b, d, L, M))
            if gain > best_gain + EPS:
                best_gain, best_i, best_j = gain, i, j
    if best_i is not None:
        route.nodes = (nodes[:best_i] +
                       list(reversed(nodes[best_i:best_j + 1])) +
                       nodes[best_j + 1:])
        route.update_load_cost(L, M)
        return True
    return False

def improve_L2_routes(sol: Solution, L, M, max_loop=20):
    loop, improved = 0, True
    while improved and loop < max_loop:
        improved = False
        for rs in sol.R2.values():
            for r in rs:
                if l2_two_opt_route(r, L, M):
                    improved = True
        loop += 1

    evaluate(sol, L, M)
    return sol


def clarke_wright_savings(depot_id,
                          customer_locs,
                          capacity,
                          L, M,
                          echelon,
                          route_prefix,
                          max_vehicles=float('inf')):

    if not customer_locs:
        return [], []

    cust_ids = [c.id for c in customer_locs]
    demand   = {c.id: c.demand for c in customer_locs}
    savings = []
    for i in range(len(cust_ids)):
        for j in range(i + 1, len(cust_ids)):
            ci, cj = cust_ids[i], cust_ids[j]
            s_ij = (dist(depot_id, ci, L, M)
                    + dist(depot_id, cj, L, M)
                    - dist(ci, cj, L, M))
            if s_ij > EPS:
                savings.append((s_ij, ci, cj))
    savings.sort(reverse=True)


    routes     = {}
    route_load = {}
    ends       = {}
    for cid in cust_ids:
        if demand[cid] > capacity + EPS:
            continue
        nds = [depot_id, cid, depot_id]
        routes[cid] = nds
        route_load[tuple(nds)] = demand[cid]
        ends[cid] = (True, True)

    merged = set()

    for _, ci, cj in savings:
        ri = routes.get(ci)
        rj = routes.get(cj)
        if ri is None or rj is None or ri is rj:
            continue

        if not ends[ci][1] or not ends[cj][0]:
            continue

        load_new = route_load[tuple(ri)] + route_load[tuple(rj)]
        if load_new > capacity + EPS:
            continue

        new_nodes = ri[:-1] + rj[1:]
        new_tuple = tuple(new_nodes)

        merged.add(tuple(ri))
        merged.add(tuple(rj))

        for n in ri[1:-1]:
            routes.pop(n, None)
            ends.pop(n, None)
        for n in rj[1:-1]:
            routes.pop(n, None)
            ends.pop(n, None)
        route_load.pop(tuple(ri), None)
        route_load.pop(tuple(rj), None)

        for n in new_nodes[1:-1]:
            routes[n] = new_nodes
            ends[n]   = (n == new_nodes[1], n == new_nodes[-2])
        route_load[new_tuple] = load_new

    unique_routes = {}
    for cid, nds in routes.items():
        key = tuple(nds)
        if key not in merged:
            unique_routes[key] = nds

    res_routes = []
    for idx, nds in enumerate(unique_routes.values(), 1):
        r = Route(f"{route_prefix}_{idx}", echelon, capacity)
        r.nodes = list(nds)
        if echelon == 1:
            for nid in r.nodes[1:-1]:
                r.payload[nid] = L[nid].demand
        r.update_load_cost(L, M)
        res_routes.append(r)
    served = {n for r in res_routes for n in r.nodes[1:-1]}
    unassigned = [cid for cid in cust_ids if cid not in served]

    assert set(cust_ids) == served | set(unassigned), \
        "Some customers were lost during C-W merging!"

    res_routes.sort(key=lambda r: r.name)
    return res_routes, unassigned


def _build_sat_chunks(sat_demand: dict, cap):

    chunks = []
    for sid, dem in sat_demand.items():
        remain = dem
        while remain > EPS:
            qty = min(remain, cap)
            chunks.append((sid, qty))
            remain -= qty
    return chunks


def _giant_tour_cheapest_insertion(chunks, L, M, depot_id=0):

    if not chunks:
        return []

    seed = max(chunks, key=lambda ck: dist(depot_id, ck[0], L, M))
    tour = [seed]
    rem  = [ck for ck in chunks if ck is not seed]

    while rem:
        best = None
        for ck in rem:
            sid = ck[0]
            for pos in range(len(tour) + 1):
                prev = depot_id if pos == 0 else tour[pos - 1][0]
                nxt  = depot_id if pos == len(tour) else tour[pos][0]
                inc  = (dist(prev, sid, L, M) +
                        dist(sid,  nxt, L, M) -
                        dist(prev, nxt, L, M))
                if best is None or inc < best[0]:
                    best = (inc, ck, pos)
        _, ck_best, insert_pos = best
        tour.insert(insert_pos, ck_best)
        rem.remove(ck_best)

    return tour


def _split_greedy(tour, cap):

    groups, cur, load = [], [], 0.0
    for sid, qty in tour:
        if load + qty > cap + EPS:
            groups.append(cur)
            cur, load = [], 0.0
        cur.append((sid, qty))
        load += qty
    if cur:
        groups.append(cur)
    return groups


def _routes_from_groups(groups, depot_id, cap, L, M, prefix="L1"):

    routes = []
    for idx, grp in enumerate(groups, 1):
        r = Route(f"{prefix}_{idx}", 1, cap)
        r.nodes = [depot_id] + [sid for sid, _ in grp] + [depot_id]
        r.payload = defaultdict(float)
        for sid, qty in grp:
            r.payload[sid] += qty
        r.update_load_cost(L, M)
        routes.append(r)
    return routes


def _improve_split_routes(routes, L, M, cap, max_loop=50):

    loop = 0
    improved = True
    while improved and loop < max_loop:
        improved = False
        for r_from in routes:
            for i in range(1, len(r_from.nodes) - 1):
                sid = r_from.nodes[i]
                qty = r_from.payload[sid]
                for r_to in routes:
                    if r_from is r_to:
                        continue
                    if r_to.load + qty > cap + EPS:
                        continue
                    for pos in range(1, len(r_to.nodes)):
                        gain = (dist(r_from.nodes[i - 1], sid, L, M) +
                                dist(sid, r_from.nodes[i + 1], L, M) -
                                dist(r_from.nodes[i - 1], r_from.nodes[i + 1], L, M) +
                                dist(r_to.nodes[pos - 1], sid, L, M) +
                                dist(sid, r_to.nodes[pos], L, M) -
                                dist(r_to.nodes[pos - 1], r_to.nodes[pos], L, M))
                        if gain < -EPS:
                            r_from.nodes.pop(i)
                            r_from.payload[sid] -= qty
                            r_from.update_load_cost(L, M)

                            r_to.nodes.insert(pos, sid)
                            r_to.payload[sid] += qty
                            r_to.update_load_cost(L, M)

                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            loop += 1
            continue
        for r1 in routes:
            for i in range(1, len(r1.nodes) - 1):
                sid1 = r1.nodes[i]
                q1   = r1.payload[sid1]
                for r2 in routes:
                    if r1 is r2:
                        continue
                    for j in range(1, len(r2.nodes) - 1):
                        sid2 = r2.nodes[j]
                        q2   = r2.payload[sid2]
                        if (r1.load - q1 + q2 > cap + EPS or
                            r2.load - q2 + q1 > cap + EPS):
                            continue
                        delta = 0
                        delta += (dist(r1.nodes[i - 1], sid2, L, M) +
                                  dist(sid2, r1.nodes[i + 1], L, M) -
                                  dist(r1.nodes[i - 1], sid1, L, M) -
                                  dist(sid1, r1.nodes[i + 1], L, M))
                        delta += (dist(r2.nodes[j - 1], sid1, L, M) +
                                  dist(sid1, r2.nodes[j + 1], L, M) -
                                  dist(r2.nodes[j - 1], sid2, L, M) -
                                  dist(sid2, r2.nodes[j + 1], L, M))
                        if delta < -EPS:
                            r1.nodes[i], r2.nodes[j] = sid2, sid1
                            r1.payload[sid1] -= q1; r1.payload[sid2] += q2
                            r2.payload[sid2] -= q2; r2.payload[sid1] += q1
                            r1.update_load_cost(L, M)
                            r2.update_load_cost(L, M)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        loop += 1
    return routes


def greedy_initial(depot, sats, custs, L, M):
    sol = Solution()
    sol.R2 = defaultdict(list)
    sol.R1 = []
    sol.unassigned_customers = []

    assign = defaultdict(list)
    for c in custs:
        sid = min(sats, key=lambda s: dist(c.id, s.id, L, M)).id
        assign[sid].append(c)

    all_un = []
    for sat in sats:
        sid = sat.id
        routes, unass = clarke_wright_savings(
            depot_id      = sid,
            customer_locs = assign.get(sid, []),
            capacity      = L2_CAP,
            L=L, M=M,
            echelon       = 2,
            route_prefix  = f"L2_{sid}",
            max_vehicles  = V_S_MAX[sid]
        )
        sol.R2[sid] = routes
        all_un.extend(unass)

    sol.unassigned_customers = all_un

    for cid in sol.unassigned_customers[:]:
        best = None
        for sid, routes in sol.R2.items():
            for r in routes:
                if r.load + L[cid].demand > L2_CAP + EPS:
                    continue

                for pos in range(1, len(r.nodes)):
                    inc = (dist(r.nodes[pos-1], cid, L, M)
                           + dist(cid, r.nodes[pos], L, M)
                           - dist(r.nodes[pos-1], r.nodes[pos], L, M))
                    if best is None or inc < best[0]:
                        best = (inc, r, pos)
        if best:
            _, r, pos = best
            safe_insert_L2(r, pos, cid, L, M)
            sol.unassigned_customers.remove(cid)
        else:
            all_r = [r for rs in sol.R2.values() for r in rs]
            if not all_r:
                raise RuntimeError("无法插入 R2 ——没有任何路线！")
            r = min(all_r, key=lambda x: x.load)
            safe_insert_L2(r, len(r.nodes)-1, cid, L, M)
            sol.unassigned_customers.remove(cid)

    flag = float("inf")
    while flag > V2_MAX:
        sol = route_removal(sol, 5, L, M)
        sol = random_customer_insertion(sol, sats, depot, L, M)
        flag = sol.l2_truck_used()
    sat_demand = defaultdict(float)
    for sid, rs in sol.R2.items():
        sat_demand[sid] = sum(r.load for r in rs)

    if not any(v > EPS for v in sat_demand.values()):
        evaluate(sol, L, M)
        return sol

    chunks = _build_sat_chunks(sat_demand, L1_CAP)
    giant = _giant_tour_cheapest_insertion(chunks, L, M, depot.id)
    groups_split = _split_greedy(giant, L1_CAP)
    routes_split = _routes_from_groups(groups_split, depot.id,
                                       L1_CAP, L, M, prefix="L1S")
    routes_split = _improve_split_routes(routes_split, L, M, L1_CAP)

    groups_oab = [[ck] for ck in giant]
    routes_oab = _routes_from_groups(groups_oab, depot.id,
                                     L1_CAP, L, M, prefix="L1O")

    def _cost_with_penalty(routes):
        cost = sum(r.cost for r in routes)
        if len(routes) > V1_MAX:
            cost += BIG_PENALTY * (len(routes) - V1_MAX)
        return cost

    if _cost_with_penalty(routes_split) <= _cost_with_penalty(routes_oab):
        sol.R1 = routes_split[:]
    else:
        sol.R1 = routes_oab[:]

    evaluate(sol, L, M)
    return sol

def safe_insert_L2(r: Route, pos: int, cid: int, L, M):
    r.nodes.insert(pos, cid); r.update_load_cost(L, M)

def safe_insert_L1_sat(r: Route, pos: int, sid: int, qty: float, L, M):
    r.nodes.insert(pos, sid)
    r.payload[sid] += qty
    r.update_load_cost(L, M)

def worst_customer_removal(sol, q, L, M):
    s = sol.deepcopy(); cand = []
    for sid, routes in s.R2.items():
        for ridx, r in enumerate(routes):
            for idx in range(1, len(r.nodes) - 1):
                cid = r.nodes[idx]
                prev, nxt = r.nodes[idx - 1], r.nodes[idx + 1]
                save = (dist(prev, cid, L, M) +
                        dist(cid, nxt, L, M) -
                        dist(prev, nxt, L, M))
                cand.append((-save, sid, ridx, idx, cid))
    cand.sort(); q = min(q, len(cand))
    rem = set(c[4] for c in cand[:q])
    for sid, routes in s.R2.items():
        for r in routes[:]:
            if any(c in rem for c in r.nodes[1:-1]):
                r.nodes = [n for n in r.nodes if n not in rem]
                if len(r.nodes) <= 2: routes.remove(r)
                else:                 r.update_load_cost(L, M)
    s.unassigned_customers.extend(rem); evaluate(s, L, M); return s

def random_customer_removal(sol, q, L, M):
    s = sol.deepcopy(); served = list(s.all_served_customers())
    q = min(q, len(served)); rem = set(random.sample(served, q))
    for sid, rts in s.R2.items():
        for r in rts[:]:
            if any(c in rem for c in r.nodes[1:-1]):
                r.nodes = [n for n in r.nodes if n not in rem]
                if len(r.nodes) <= 2: rts.remove(r)
                else:                 r.update_load_cost(L, M)
    s.unassigned_customers.extend(rem); evaluate(s, L, M); return s

def route_removal(sol, q, L, M):
    s = sol.deepcopy()
    all_routes = [(sid, r) for sid, rs in s.R2.items() for r in rs]
    if not all_routes: return s
    random.shuffle(all_routes); removed = 0
    for sid, r in all_routes:
        s.R2[sid].remove(r); s.unassigned_customers.extend(r.nodes[1:-1])
        removed += len(r.nodes) - 2
        if removed >= q: break
    evaluate(s, L, M); return s

def satellite_removal(sol, q, L, M):
    s = sol.deepcopy(); sats = [sid for sid, rs in s.R2.items() if rs]
    if not sats: return s
    sid = random.choice(sats)
    for r in s.R2[sid]:
        s.unassigned_customers.extend(r.nodes[1:-1])
    s.R2[sid] = []
    for r in s.R1[:]:
        if sid in r.nodes:
            r.nodes = [n for n in r.nodes if n != sid]
            if len(r.nodes) <= 2: s.R1.remove(r)
            else:                 r.update_load_cost(L, M)
    evaluate(s, L, M); return s

def related_customer_removal(sol, q, L, M, theta=0.3):
    s = sol.deepcopy(); served = list(s.all_served_customers())
    if not served: return s
    q = min(q, len(served)); seed = random.choice(served); remove = [seed]
    while len(remove) < q:
        cand = [c for c in served if c not in remove]
        if not cand: break
        dist_arr = np.array([dist(seed, c, L, M) for c in cand])
        prob = np.exp(-dist_arr / (theta + EPS)); prob /= prob.sum()
        pick = random.choices(cand, weights=prob, k=1)[0]; remove.append(pick)
    for sid, rts in s.R2.items():
        for r in rts[:]:
            if any(c in remove for c in r.nodes[1:-1]):
                r.nodes = [n for n in r.nodes if n not in remove]
                if len(r.nodes) <= 2: rts.remove(r)
                else:                 r.update_load_cost(L, M)
    s.unassigned_customers.extend(remove); evaluate(s, L, M); return s


def insert_sat_into_best_L1(sat_id, s: Solution, add_qty, depot, L, M):
    for r in s.R1:
        if sat_id in r.nodes:
            if r.load + add_qty <= L1_CAP + EPS:
                r.payload[sat_id] += add_qty
                r.update_load_cost(L, M)
                return True
            else:
                return False
    best = None
    for r in s.R1:
        if r.load + add_qty > L1_CAP + EPS:
            continue
        for pos in range(1, len(r.nodes)):
            inc = (dist(r.nodes[pos - 1], sat_id, L, M) +
                   dist(sat_id, r.nodes[pos], L, M) -
                   dist(r.nodes[pos - 1], r.nodes[pos], L, M))
            if best is None or inc < best[0]:
                best = (inc, r, pos)

    if best:
        _, r, pos = best
        safe_insert_L1_sat(r, pos, sat_id, add_qty, L, M)
        return True
    return False

def try_best_insertion(cid, s: Solution, sats, depot, L, M):
    cust = L[cid]; best = None
    for sid, routes in s.R2.items():
        for ridx, r in enumerate(routes):
            if r.load + cust.demand > L2_CAP: continue
            for pos in range(1, len(r.nodes)):
                d = (dist(r.nodes[pos - 1], cid, L, M) +
                     dist(cid, r.nodes[pos], L, M) -
                     dist(r.nodes[pos - 1], r.nodes[pos], L, M))
                if best is None or d < best[0]:
                    best = (d, sid, ridx, pos)
    if best is None:
        for sat in sorted(sats, key=lambda st: dist(cid, st.id, L, M)):
            if s.l2_truck_used() >= V2_MAX: break
            if s.l2_truck_used_at(sat.id) >= V_S_MAX[sat.id]: continue
            if not any(sat.id in r.nodes for r in s.R1):
                ok = insert_sat_into_best_L1(sat.id, s, cust.demand,
                                             depot, L, M)
                if not ok:
                    if s.l1_truck_used() >= V1_MAX: continue
                    r1 = Route(f"NEWL1_{sat.id}", 1, L1_CAP)
                    r1.nodes = [depot.id, sat.id, depot.id]
                    r1.load = cust.demand; r1.update_load_cost(L, M)
                    s.R1.append(r1)
            newR = Route(f"NEWL2_{cid}", 2, L2_CAP)
            newR.nodes = [sat.id, cid, sat.id]; newR.update_load_cost(L, M)
            s.R2[sat.id].append(newR); return True
    else:
        _, sid, ridx, pos = best
        safe_insert_L2(s.R2[sid][ridx], pos, cid, L, M)
        return True
    return False


def basic_greedy_customer_insertion(sol, sats, depot, L, M):
    s = sol.deepcopy(); random.shuffle(s.unassigned_customers); fail = []
    for cid in s.unassigned_customers:
        if not try_best_insertion(cid, s, sats, depot, L, M):
            fail.append(cid)
    s.unassigned_customers = fail; evaluate(s, L, M); return s

def regret_k_customer_insertion(sol, sats, depot, L, M, k=3):
    s = sol.deepcopy()
    while s.unassigned_customers:
        best = None
        for cid in s.unassigned_customers:
            ins = []
            for sid, routes in s.R2.items():
                for ridx, r in enumerate(routes):
                    if r.load + L[cid].demand > L2_CAP: continue
                    for pos in range(1, len(r.nodes)):
                        d = (dist(r.nodes[pos - 1], cid, L, M) +
                             dist(cid, r.nodes[pos], L, M) -
                             dist(r.nodes[pos - 1], r.nodes[pos], L, M))
                        ins.append((d, sid, ridx, pos))
            if not ins: continue
            ins.sort(key=lambda x: x[0])
            regret = ins[min(k - 1, len(ins) - 1)][0] - ins[0][0]
            if best is None or regret > best[0]:
                best = (regret, cid, ins[0])
        if best is None: break
        _, cid, (d, sid, ridx, pos) = best
        safe_insert_L2(s.R2[sid][ridx], pos, cid, L, M)
        s.unassigned_customers.remove(cid)
    evaluate(s, L, M); return s

def random_customer_insertion(sol, sats, depot, L, M):
    s = sol.deepcopy(); random.shuffle(s.unassigned_customers); fail = []
    for cid in s.unassigned_customers:
        opts = []
        for sid, routes in s.R2.items():
            for ridx, r in enumerate(routes):
                if r.load + L[cid].demand > L2_CAP: continue
                for pos in range(1, len(r.nodes)):
                    opts.append((sid, ridx, pos))
        if opts:
            sid, ridx, pos = random.choice(opts)
            safe_insert_L2(s.R2[sid][ridx], pos, cid, L, M)
        else:
            if not try_best_insertion(cid, s, sats, depot, L, M):
                fail.append(cid)
    s.unassigned_customers = fail; evaluate(s, L, M); return s


def build_new_L2_routes(sol, sats, depot, L, M,
                        max_extra_per_route=4,
                        cost_increase_tol=1e4):
    s = sol.deepcopy()
    def choose_sat(cid):
        order = sorted(sats, key=lambda st: dist(cid, st.id, L, M))
        for sat in order:
            if s.l2_truck_used() >= V2_MAX: break
            if s.l2_truck_used_at(sat.id) >= V_S_MAX[sat.id]: continue
            return sat
        return None
    c_left = s.unassigned_customers[:]; random.shuffle(c_left)
    for seed in c_left:
        if seed not in s.unassigned_customers: continue
        sat = choose_sat(seed);
        if sat is None: continue
        if not any(sat.id in r.nodes for r in s.R1):
            ok = insert_sat_into_best_L1(sat.id, s, L[seed].demand,
                                         depot, L, M)
            if not ok:
                if s.l1_truck_used() >= V1_MAX: continue
                r1 = Route(f"NL1_{sat.id}", 1, L1_CAP)
                r1.nodes = [depot.id, sat.id, depot.id]
                r1.load  = L[seed].demand; r1.update_load_cost(L, M)
                s.R1.append(r1)
        newR = Route(f"NL2_{seed}", 2, L2_CAP)
        newR.nodes = [sat.id, seed, sat.id]; newR.update_load_cost(L, M)
        s.R2[sat.id].append(newR); s.unassigned_customers.remove(seed)
        # 再塞点
        sorted_left = sorted(s.unassigned_customers,
                             key=lambda c: dist(c, sat.id, L, M))
        ins_cnt = 0
        for cid in sorted_left:
            if ins_cnt >= max_extra_per_route: break
            if newR.load + L[cid].demand > L2_CAP: continue
            best_pos = best_inc = None
            for pos in range(1, len(newR.nodes)):
                inc = (dist(newR.nodes[pos - 1], cid, L, M) +
                       dist(cid, newR.nodes[pos], L, M) -
                       dist(newR.nodes[pos - 1], newR.nodes[pos], L, M))
                if best_inc is None or inc < best_inc:
                    best_inc, best_pos = inc, pos
            if best_pos and best_inc < cost_increase_tol:
                safe_insert_L2(newR, best_pos, cid, L, M)
                s.unassigned_customers.remove(cid); ins_cnt += 1
    evaluate(s, L, M); return s


def _init_weights(n_d, n_r, init=10.0):
    return np.ones(n_d)*init, np.zeros(n_d), np.ones((n_d,n_r))*init, np.zeros((n_d,n_r))
def _reset_scores(d_s, rt_s): d_s[:] = 0.0; rt_s[:] = 0.0
def _roulette(arr):
    cdf = (arr/arr.sum()).cumsum(); return int((cdf>random.random()).argmax())
def _select_ops(d_w, rt_w): d=_roulette(d_w); r=_roulette(rt_w[d]); return d,r
def _update_w(d_w, rt_w, d_s, rt_s, rho):
    d_w[:]  = rho*d_w  + (1-rho)*d_s
    rt_w[:] = rho*rt_w + (1-rho)*rt_s

def table_alns(initial, destroy_ops, repair_ops,
               depot, sats, L, M,
               *, r1=6, r2=4, r3=2, r4=0,
               rho=.95, phi=.95,
               T0=1_000, epochs=10_000,
               max_non_imp=200, seed=11):
    random.seed(seed); np.random.seed(seed)
    cur = initial.deepcopy(); best = cur.deepcopy()
    n_d, n_r = len(destroy_ops), len(repair_ops)
    d_w, d_s, rt_w, rt_s = _init_weights(n_d,n_r)
    T = T0; non_imp=0; hist=[]
    state = True
    for ep in range(1, epochs+1):
        _reset_scores(d_s, rt_s)
        d_id, r_id = _select_ops(d_w, rt_w)
        d_op, r_op = destroy_ops[d_id], repair_ops[r_id]
        served = len(cur.all_served_customers())
        q = random.randint(max(1,int(.30*served)),
                           max(2,int(.50*served)))
        cand = r_op(d_op(cur, q, L, M), sats, depot, L, M)
        improve_L2_routes(cand, L, M)
        improve_L1_routes(cand, L, M)

        delta = cand.total_cost - cur.total_cost
        accepted=False
        if delta < -EPS:
            cur = cand; accepted=True; score=r2
            if is_feasible(cur) and cur.total_cost < best.total_cost:
                best = cur.deepcopy(); score=r1; non_imp=0
            else: non_imp+=1
        elif math.exp(-delta/max(T,1e-3))>=random.random():
            cur=cand; accepted=True; score=r3; non_imp+=1
        else:
            score=r4; non_imp+=1
        d_s[d_id]+=score; rt_s[d_id,r_id]+=score
        T*=phi; _update_w(d_w, rt_w, d_s, rt_s, rho)
        if non_imp>=max_non_imp:
            if state:
                max_non_imp = max_non_imp + ep
                state = False
            cur=greedy_initial(depot,sats,
                    [L[c] for c in L if L[c].type=="customer"],L,M)
            d_w, d_s, rt_w, rt_s = _init_weights(n_d, n_r)
            non_imp=0; T=T0
        hist.append(best.total_cost)
    return best, hist


def normal_alns(initial, destroy_ops, repair_ops,
                depot, sats, L, M,
                *, r1=6, r2=4, r3=2, r4=0,
                rho=.95, phi=.95,
                T0=1_000, epochs=10_000,
                max_non_imp=200, seed=11):

    random.seed(seed);
    np.random.seed(seed)
    cur = initial.deepcopy();
    best = cur.deepcopy()
    n_d, n_r = len(destroy_ops), len(repair_ops)

    d_w = np.ones(n_d) * 10.0
    d_s = np.zeros(n_d)
    r_w = np.ones(n_r) * 10.0
    r_s = np.zeros(n_r)

    T = T0;
    non_imp = 0;
    hist = []
    state = True

    for ep in range(1, epochs + 1):

        d_s[:] = 0.0
        r_s[:] = 0.0
        d_id = _roulette(d_w)
        r_id = _roulette(r_w)

        d_op, r_op = destroy_ops[d_id], repair_ops[r_id]
        served = len(cur.all_served_customers())
        q = random.randint(max(1, int(.30 * served)),
                           max(2, int(.50 * served)))
        cand = r_op(d_op(cur, q, L, M), sats, depot, L, M)
        improve_L2_routes(cand, L, M)
        improve_L1_routes(cand, L, M)

        delta = cand.total_cost - cur.total_cost
        accepted = False
        if delta < -EPS:
            cur = cand;
            accepted = True;
            score = r2
            if is_feasible(cur) and cur.total_cost < best.total_cost:
                best = cur.deepcopy();
                score = r1;
                non_imp = 0
            else:
                non_imp += 1
        elif math.exp(-delta / max(T, 1e-3)) >= random.random():
            cur = cand;
            accepted = True;
            score = r3;
            non_imp += 1
        else:
            score = r4;
            non_imp += 1

        d_s[d_id] += score
        r_s[r_id] += score

        T *= phi

        d_w[:] = rho * d_w + (1 - rho) * d_s
        r_w[:] = rho * r_w + (1 - rho) * r_s

        if non_imp >= max_non_imp:
            if state:
                max_non_imp = max_non_imp + ep
                state = False
            cur = greedy_initial(depot, sats,
                                 [L[c] for c in L if L[c].type == "customer"], L, M)
            d_w = np.ones(n_d) * 10.0
            d_s = np.zeros(n_d)
            r_w = np.ones(n_r) * 10.0
            r_s = np.zeros(n_r)
            non_imp = 0;
            T = T0
        hist.append(best.total_cost)
    return best, hist


def reward_alns(initial, destroy_ops, repair_ops,
                depot, sats, L, M,
                *, r1=6, r2=4, r3=2, r4=0,
                rho=.95, phi=.95,
                T0=1_000, epochs=10_000,
                max_non_imp=200, seed=11):

    random.seed(seed);
    np.random.seed(seed)
    cur = initial.deepcopy();
    best = cur.deepcopy()
    n_d, n_r = len(destroy_ops), len(repair_ops)

    d_w = np.ones(n_d) * 10.0
    d_s = np.zeros(n_d)
    r_w = np.ones(n_r) * 10.0
    r_s = np.zeros(n_r)

    T = T0;
    non_imp = 0;
    hist = []

    setting_dif = initial.total_cost / 100
    large_delta = 100
    w_fix = 3
    state = True

    for ep in range(1, epochs + 1):
        d_s[:] = 0.0
        r_s[:] = 0.0
        d_id = _roulette(d_w)
        r_id = _roulette(r_w)

        d_op, r_op = destroy_ops[d_id], repair_ops[r_id]
        served = len(cur.all_served_customers())
        q = random.randint(max(1, int(.30 * served)),
                           max(2, int(.50 * served)))
        cand = r_op(d_op(cur, q, L, M), sats, depot, L, M)
        improve_L2_routes(cand, L, M)
        improve_L1_routes(cand, L, M)

        delta = cand.total_cost - cur.total_cost
        accepted = False
        if delta < -EPS:
            cur = cand;
            accepted = True;
            score = r2
            if is_feasible(cur) and cur.total_cost < best.total_cost:
                target_dif = best.total_cost - cur.total_cost
                dynamic_r1 = max(w_fix, min(0.1 * large_delta,
                                            w_fix + round(large_delta * non_imp / max_non_imp) +
                                            round(target_dif / setting_dif)))

                best = cur.deepcopy();
                score = dynamic_r1;
                non_imp = 0
            else:
                non_imp += 1
        elif math.exp(-delta / max(T, 1e-3)) >= random.random():
            cur = cand;
            accepted = True;
            score = r3;
            non_imp += 1
        else:
            score = r4;
            non_imp += 1

        d_s[d_id] += score
        r_s[r_id] += score
        T *= phi
        d_w[:] = rho * d_w + (1 - rho) * d_s
        r_w[:] = rho * r_w + (1 - rho) * r_s

        if non_imp >= max_non_imp:
            if state:
                max_non_imp = max_non_imp + ep
                state = False
            cur = greedy_initial(depot, sats,
                                 [L[c] for c in L if L[c].type == "customer"], L, M)
            d_w = np.ones(n_d) * 10.0
            d_s = np.zeros(n_d)
            r_w = np.ones(n_r) * 10.0
            r_s = np.zeros(n_r)
            non_imp = 0;
            T = T0
        hist.append(best.total_cost)
    return best, hist


def pair_alns(initial, destroy_ops, repair_ops,
              depot, sats, L, M,
              *, r1=6, r2=4, r3=2, r4=0,
              rho=.95, phi=.95,
              T0=1_000, epochs=10_000,
              max_non_imp=200, seed=11):

    random.seed(seed);
    np.random.seed(seed)
    cur = initial.deepcopy();
    best = cur.deepcopy()
    n_d, n_r = len(destroy_ops), len(repair_ops)

    dr_w = np.ones(n_d * n_r) * 10.0
    dr_s = np.zeros(n_d * n_r)

    T = T0;
    non_imp = 0;
    hist = []
    state = True

    for ep in range(1, epochs + 1):
        dr_s[:] = 0.0
        dr_id = _roulette(dr_w)
        d_id = dr_id // n_r
        r_id = dr_id % n_r

        d_op, r_op = destroy_ops[d_id], repair_ops[r_id]
        served = len(cur.all_served_customers())
        q = random.randint(max(1, int(.30 * served)),
                           max(2, int(.50 * served)))
        cand = r_op(d_op(cur, q, L, M), sats, depot, L, M)
        improve_L2_routes(cand, L, M)
        improve_L1_routes(cand, L, M)

        delta = cand.total_cost - cur.total_cost
        accepted = False
        if delta < -EPS:
            cur = cand;
            accepted = True;
            score = r2
            if is_feasible(cur) and cur.total_cost < best.total_cost:
                best = cur.deepcopy();
                score = r1;
                non_imp = 0
            else:
                non_imp += 1
        elif math.exp(-delta / max(T, 1e-3)) >= random.random():
            cur = cand;
            accepted = True;
            score = r3;
            non_imp += 1
        else:
            score = r3;
            non_imp += 1

        dr_s[dr_id] += score
        T *= phi
        dr_w[:] = rho * dr_w + (1 - rho) * dr_s

        if non_imp >= max_non_imp:
            if state:
                max_non_imp = max_non_imp + ep
                state = False
            cur = greedy_initial(depot, sats,
                                 [L[c] for c in L if L[c].type == "customer"], L, M)
            dr_w = np.ones(n_d * n_r) * 10.0
            dr_s = np.zeros(n_d * n_r)
            non_imp = 0;
            T = T0
        hist.append(best.total_cost)
    return best, hist


def solve_instance(name,
                   depot_xy, hubs_xy,
                   cust_xy, cust_demand,
                   L1_CAP, L2_CAP,
                   opt=None,
                   seed=18,
                   alns_params=None,
                   alns_mode="table"):
    random.seed(seed); np.random.seed(seed)

    idx = 0
    depot = Location(idx, 0, "depot", *depot_xy); idx += 1
    satellites = []
    for i, (x,y) in enumerate(hubs_xy, 1):
        satellites.append(Location(idx, 100+i, "satellite", x, y,
                                   capacity=float('inf')))
        idx += 1
    customers = []
    for i, ((x,y), d) in enumerate(zip(cust_xy, cust_demand), 1):
        customers.append(Location(idx, i, "customer", x, y, demand=d))
        idx += 1

    all_locs = [depot] + satellites + customers
    for i, loc in enumerate(all_locs): loc.idx = i
    L = {loc.id: loc for loc in all_locs}
    M = build_distance_matrix(all_locs)

    init = greedy_initial(depot, satellites, customers, L, M)
    improve_L2_routes(init, L, M)
    print(f"\n>>> Instance {name}")
    print(f"    initial cost = {init.total_cost:.4f}")
    if opt is not None:
        print(f"    known Opt    = {opt:.4f}")

    params = {
        'epochs': 400000, 'max_non_imp': 1000,
        'r1':6, 'r2':4, 'r3':2, 'r4':0,
        'rho': .99, 'phi': .95, 'T0': 1000,
    }
    if alns_params:
        params.update(alns_params)

    destroy_ops = [
        worst_customer_removal,
        random_customer_removal,
        route_removal,
        satellite_removal,
        related_customer_removal,
    ]
    repair_ops = [
        basic_greedy_customer_insertion,
        lambda s,sts,dep,L,M: regret_k_customer_insertion(s,sts,dep,L,M,k=3),
        random_customer_insertion,
        build_new_L2_routes,
    ]

    if alns_mode == "normal":
        best, _ = normal_alns(
            init, destroy_ops, repair_ops,
            depot, satellites, L, M,
            epochs=params['epochs'],
            max_non_imp=params['max_non_imp'],
            r1=params['r1'], r2=params['r2'],
            r3=params['r3'], r4=params['r4'],
            rho=params['rho'], phi=params['phi'],
            T0=params['T0'],
            seed=seed
        )
    elif alns_mode == "reward":
        best, _ = reward_alns(
            init, destroy_ops, repair_ops,
            depot, satellites, L, M,
            epochs=params['epochs'],
            max_non_imp=params['max_non_imp'],
            r1=params['r1'], r2=params['r2'],
            r3=params['r3'], r4=params['r4'],
            rho=params['rho'], phi=params['phi'],
            T0=params['T0'],
            seed=seed
        )
    elif alns_mode == "pair":
        best, _ = pair_alns(
            init, destroy_ops, repair_ops,
            depot, satellites, L, M,
            epochs=params['epochs'],
            max_non_imp=params['max_non_imp'],
            r1=params['r1'], r2=params['r2'],
            r3=params['r3'], r4=params['r4'],
            rho=params['rho'], phi=params['phi'],
            T0=params['T0'],
            seed=seed
        )
    else:  # 默认使用table模式
        best, _ = table_alns(
            init, destroy_ops, repair_ops,
            depot, satellites, L, M,
            epochs=params['epochs'],
            max_non_imp=params['max_non_imp'],
            r1=params['r1'], r2=params['r2'],
            r3=params['r3'], r4=params['r4'],
            rho=params['rho'], phi=params['phi'],
            T0=params['T0'],
            seed=seed
        )


L1_CAP = None
L2_CAP = None
def run_with_seed(instance_data, seed, alns_mode="table"):
    global L1_CAP, L2_CAP
    name, depot, hubs, custs, demands, l1cap, l2cap, opt = instance_data
    L1_CAP, L2_CAP = l1cap, l2cap
    start_time = time.time()
    idx = 0
    depot_obj = Location(idx, 0, "depot", *depot);
    idx += 1
    satellites = []
    for i, (x, y) in enumerate(hubs, 1):
        satellites.append(Location(idx, 100 + i, "satellite", x, y,
                                   capacity=float('inf')))
        idx += 1
    customers = []
    for i, ((x, y), d) in enumerate(zip(custs, demands), 1):
        customers.append(Location(idx, i, "customer", x, y, demand=d))
        idx += 1

    all_locs = [depot_obj] + satellites + customers
    for i, loc in enumerate(all_locs): loc.idx = i
    L = {loc.id: loc for loc in all_locs}
    M = build_distance_matrix(all_locs)
    init = greedy_initial(depot_obj, satellites, customers, L, M)
    improve_L2_routes(init, L, M)

    params = {
        'epochs': 200, 'max_non_imp': 100,
        'r1': 6, 'r2': 4, 'r3': 2, 'r4': 0,
        'rho': .99, 'phi': .95, 'T0': 1000,
    }

    destroy_ops = [
        worst_customer_removal,
        random_customer_removal,
        route_removal,
        satellite_removal,
        related_customer_removal,
    ]
    repair_ops = [
        basic_greedy_customer_insertion,
        lambda s, sts, dep, L, M: regret_k_customer_insertion(s, sts, dep, L, M, k=3),
        random_customer_insertion,
        build_new_L2_routes,
    ]

    if alns_mode == "normal":
        best, _ = normal_alns(
            init, destroy_ops, repair_ops,
            depot_obj, satellites, L, M,
            epochs=params['epochs'],
            max_non_imp=params['max_non_imp'],
            r1=params['r1'], r2=params['r2'],
            r3=params['r3'], r4=params['r4'],
            rho=params['rho'], phi=params['phi'],
            T0=params['T0'],
            seed=seed
        )
    elif alns_mode == "reward":
        best, _ = reward_alns(
            init, destroy_ops, repair_ops,
            depot_obj, satellites, L, M,
            epochs=params['epochs'],
            max_non_imp=params['max_non_imp'],
            r1=params['r1'], r2=params['r2'],
            r3=params['r3'], r4=params['r4'],
            rho=params['rho'], phi=params['phi'],
            T0=params['T0'],
            seed=seed
        )
    elif alns_mode == "pair":
        best, _ = pair_alns(
            init, destroy_ops, repair_ops,
            depot_obj, satellites, L, M,
            epochs=params['epochs'],
            max_non_imp=params['max_non_imp'],
            r1=params['r1'], r2=params['r2'],
            r3=params['r3'], r4=params['r4'],
            rho=params['rho'], phi=params['phi'],
            T0=params['T0'],
            seed=seed
        )
    else:  # 默认使用table模式
        best, _ = table_alns(
            init, destroy_ops, repair_ops,
            depot_obj, satellites, L, M,
            epochs=params['epochs'],
            max_non_imp=params['max_non_imp'],
            r1=params['r1'], r2=params['r2'],
            r3=params['r3'], r4=params['r4'],
            rho=params['rho'], phi=params['phi'],
            T0=params['T0'],
            seed=seed
        )

    runtime = time.time() - start_time

    gap = (best.total_cost - opt) / opt * 100 if opt is not None else None

    result = {
        "name": name,
        "seed": seed,
        "cost": best.total_cost,
        "gap": gap,
        "l1_trucks": best.l1_truck_used(),
        "l2_trucks": best.l2_truck_used(),
        "runtime": runtime,
        "opt": opt,
    }

    return result

if __name__ == "__main__":
    instances = [
        # name, depot, hubs, customers, demands, L1_CAP,  L2_CAP, Opt
        ("Set2_E-n22-k4-s8-14",
         [145,215],
         [[142,239],[146,208]],
         [[151,264],[159,261],[130,254],[128,252],[163,247],
          [146,246],[161,242],[142,239],[163,236],[148,232],
          [128,231],[156,217],[129,214],[146,208],[164,208],
          [141,206],[147,193],[164,193],[129,189],[155,185],
          [139,182]],
         [1100,700,800,1400,2100,400,800,100,500,600,1200,
          1300,1300,300,900,2100,1000,900,2500,1800,700],
         15000, 6000,
         384.9558780193329
        ),
        ("Set2_E-n22-k4-s9-19",
         [145,215],
         [[163,236],[129,189]],
         # customers_set 和 demands_set 同上
         [[151,264],[159,261],[130,254],[128,252],[163,247],
          [146,246],[161,242],[142,239],[163,236],[148,232],
          [128,231],[156,217],[129,214],[146,208],[164,208],
          [141,206],[147,193],[164,193],[129,189],[155,185],
          [139,182]],
         [1100,700,800,1400,2100,400,800,100,500,600,1200,
          1300,1300,300,900,2100,1000,900,2500,1800,700],
         15000, 6000,
         470.60154604911804
        ),
        ("Set2_E-n22-k4-s10-14",
         [145,215],
         [[148,232],[146,208]],
         [[151,264],[159,261],[130,254],[128,252],[163,247],
          [146,246],[161,242],[142,239],[163,236],[148,232],
          [128,231],[156,217],[129,214],[146,208],[164,208],
          [141,206],[147,193],[164,193],[129,189],[155,185],
          [139,182]],
         [1100,700,800,1400,2100,400,800,100,500,600,1200,
          1300,1300,300,900,2100,1000,900,2500,1800,700],
         15000, 6000,
         371.4985330104828
        ),
        ("Set2_E-n22-k4-s11-12",
         [145,215],
         [[128,231],[156,217]],
         [[151,264],[159,261],[130,254],[128,252],[163,247],
          [146,246],[161,242],[142,239],[163,236],[148,232],
          [128,231],[156,217],[129,214],[146,208],[164,208],
          [141,206],[147,193],[164,193],[129,189],[155,185],
          [139,182]],
         [1100,700,800,1400,2100,400,800,100,500,600,1200,
          1300,1300,300,900,2100,1000,900,2500,1800,700],
         15000, 6000,
         427.2204830646515
        ),
        ("Set2_E-n22-k4-s12-16",
         [145,215],
         [[156,217],[141,206]],
         [[151,264],[159,261],[130,254],[128,252],[163,247],
          [146,246],[161,242],[142,239],[163,236],[148,232],
          [128,231],[156,217],[129,214],[146,208],[164,208],
          [141,206],[147,193],[164,193],[129,189],[155,185],
          [139,182]],
         [1100,700,800,1400,2100,400,800,100,500,600,1200,
          1300,1300,300,900,2100,1000,900,2500,1800,700],
         15000, 6000,
         392.78265738487244
        ),
        ("Set2_E-n33-k4-s1-9",
         [292,495],
         [[298,427],[324,433]],
         [[298,427],[309,445],[307,464],[336,475],[320,439],[321,437],
          [322,437],[323,433],[324,433],[323,429],[314,435],[311,442],
          [304,427],[293,421],[296,418],[261,384],[297,410],[315,407],
          [314,406],[321,391],[321,398],[314,394],[313,378],[304,382],
          [295,402],[283,406],[279,399],[271,401],[264,414],[277,439],
          [290,434],[319,433]],
         [700,400,400,1200,40,80,2000,900,600,750,1500,150,250,1600,
          450,700,550,650,200,400,300,1300,700,750,1400,4000,600,1000,
          500,2500,1700,1100],
         20000, 8000,
         730.162536740303
        ),
        ("Set2_E-n33-k4-s2-13",
         [292,495],
         [[309,445],[304,427]],
         [[298,427],[309,445],[307,464],[336,475],[320,439],[321,437],
          [322,437],[323,433],[324,433],[323,429],[314,435],[311,442],
          [304,427],[293,421],[296,418],[261,384],[297,410],[315,407],
          [314,406],[321,391],[321,398],[314,394],[313,378],[304,382],
          [295,402],[283,406],[279,399],[271,401],[264,414],[277,439],
          [290,434],[319,433]],
         [700,400,400,1200,40,80,2000,900,600,750,1500,150,250,1600,
          450,700,550,650,200,400,300,1300,700,750,1400,4000,600,1000,
          500,2500,1700,1100],
         20000, 8000,
         714.6327868700027
        ),
        ("Set2_E-n33-k4-s3-17",
         [292,495],
         [[307,464],[297,410]],
         [[298,427],[309,445],[307,464],[336,475],[320,439],[321,437],
          [322,437],[323,433],[324,433],[323,429],[314,435],[311,442],
          [304,427],[293,421],[296,418],[261,384],[297,410],[315,407],
          [314,406],[321,391],[321,398],[314,394],[313,378],[304,382],
          [295,402],[283,406],[279,399],[271,401],[264,414],[277,439],
          [290,434],[319,433]],
         [700,400,400,1200,40,80,2000,900,600,750,1500,150,250,1600,
          450,700,550,650,200,400,300,1300,700,750,1400,4000,600,1000,
          500,2500,1700,1100],
         20000, 8000,
         707.4831744432449
        ),
        ("Set2_E-n33-k4-s4-5",
         [292,495],
         [[336,475],[320,439]],
         [[298,427],[309,445],[307,464],[336,475],[320,439],[321,437],
          [322,437],[323,433],[324,433],[323,429],[314,435],[311,442],
          [304,427],[293,421],[296,418],[261,384],[297,410],[315,407],
          [314,406],[321,391],[321,398],[314,394],[313,378],[304,382],
          [295,402],[283,406],[279,399],[271,401],[264,414],[277,439],
          [290,434],[319,433]],
         [700,400,400,1200,40,80,2000,900,600,750,1500,150,250,1600,
          450,700,550,650,200,400,300,1300,700,750,1400,4000,600,1000,
          500,2500,1700,1100],
         20000, 8000,
         778.737855553627
        ),
        ("Set2_E-n33-k4-s7-25",
         [292,495],
         [[322,437],[295,402]],
         [[298,427],[309,445],[307,464],[336,475],[320,439],[321,437],
          [322,437],[323,433],[324,433],[323,429],[314,435],[311,442],
          [304,427],[293,421],[296,418],[261,384],[297,410],[315,407],
          [314,406],[321,391],[321,398],[314,394],[313,378],[304,382],
          [295,402],[283,406],[279,399],[271,401],[264,414],[277,439],
          [290,434],[319,433]],
         [700,400,400,1200,40,80,2000,900,600,750,1500,150,250,1600,
          450,700,550,650,200,400,300,1300,700,750,1400,4000,600,1000,
          500,2500,1700,1100],
         20000, 8000,
         756.8460360765457
        ),
        ("Set2_E-n33-k4-s14-22",
         [292,495],
         [[293,421],[314,394]],
         [[298,427],[309,445],[307,464],[336,475],[320,439],[321,437],
          [322,437],[323,433],[324,433],[323,429],[314,435],[311,442],
          [304,427],[293,421],[296,418],[261,384],[297,410],[315,407],
          [314,406],[321,391],[321,398],[314,394],[313,378],[304,382],
          [295,402],[283,406],[279,399],[271,401],[264,414],[277,439],
          [290,434],[319,433]],
         [700,400,400,1200,40,80,2000,900,600,750,1500,150,250,1600,
          450,700,550,650,200,400,300,1300,700,750,1400,4000,600,1000,
          500,2500,1700,1100],
         20000, 8000,
         779.0508366823196
        ),
    ]

    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    selected_instances = instances

    test_mode = "table"  # "table", "normal", "reward", "pair"
    all_results = []

    for instance_data in selected_instances:
        instance_name = instance_data[0]
        with mp.Pool(processes=min(len(seeds), mp.cpu_count())) as pool:
            tasks = [(instance_data, seed, test_mode) for seed in seeds]
            results = pool.starmap(run_with_seed, tasks)
            all_results.extend(results)

        instance_results = [r for r in all_results if r["name"] == instance_name]
        avg_cost = sum(r["cost"] for r in instance_results) / len(instance_results)
        best_cost = min(r["cost"] for r in instance_results)
        worst_cost = max(r["cost"] for r in instance_results)
        avg_runtime = sum(r["runtime"] for r in instance_results) / len(instance_results)


        opt = instance_data[7]
        best_gap = (best_cost - opt) / opt * 100 if opt is not None else None

        print(f"\n======= Summary for {instance_name} =======")
        print(f"Optimal value: {opt:.4f}")
        print(f"Best cost: {best_cost:.4f}")
        print(f"Worst cost: {worst_cost:.4f}")
        print(f"Average cost: {avg_cost:.4f}")
        if opt is not None:
            print(f"Best gap: {best_gap:.2f}%")
            print(f"Average gap: {sum(r['gap'] for r in instance_results) / len(instance_results):.2f}%")
        print(f"Average runtime: {avg_runtime:.2f} seconds")

    print("\n======= Overall Summary =======")
    print(
        f"{'Instance':<20} {'Opt':<10} {'Best Cost':<10} {'Avg Cost':<10} {'Best Gap':<10} {'Avg Gap':<10} {'Avg Time':<10}")
    print("-" * 90)

    for instance_data in selected_instances:
        instance_name = instance_data[0]
        opt = instance_data[7]
        instance_results = [r for r in all_results if r["name"] == instance_name]

        avg_cost = sum(r["cost"] for r in instance_results) / len(instance_results)
        best_cost = min(r["cost"] for r in instance_results)
        avg_runtime = sum(r["runtime"] for r in instance_results) / len(instance_results)

        avg_gap = "-"
        best_gap = "-"
        if opt is not None:
            avg_gap = f"{(avg_cost - opt) / opt * 100:.2f}%"
            best_gap = f"{(best_cost - opt) / opt * 100:.2f}%"

        print(
            f"{instance_name:<20} {opt if opt else '-':<10.4f} {best_cost:<10.4f} {avg_cost:<10.4f} {best_gap:<10} {avg_gap:<10} {avg_runtime:<10.2f}s")
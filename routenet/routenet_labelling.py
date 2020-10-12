import gurobipy as gb
import time
from iteration_utilities import deepflatten

from utils import *


def print_modelling_info(model,num_quests,failure_rate,modelling_time):
    print('=======================')
    print('Number of quests:', num_quests)
    print("Failure rate:", failure_rate)
    print("Optimizing time", model.runtime)
    print("Modelling time", modelling_time)

def solve_slice(num_edges, num_quests, num_paths, sp, capacity, print_info=False):
    total_start = time.time()
    with gb.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gb.Model(env=env) as model:
            slices = model.addVars(range(num_edges), range(num_quests), lb=0, ub=max(capacity), vtype=gb.GRB.INTEGER,
                                   name='slice')
            select_paths = model.addVars(range(num_quests), range(num_paths), lb=0, vtype=gb.GRB.BINARY, name='select_paths')
            edge_flow = model.addVars(range(num_edges), range(num_quests), lb=0, ub=1000, vtype=gb.GRB.INTEGER, name='flow')

            # diff = model.addVars(range(num_edges), range(num_quests), name='diff')
            success = model.addVars(range(num_edges), range(num_quests), vtype=gb.GRB.BINARY, name='success')
            scc = model.addVars(range(num_quests), vtype=gb.GRB.BINARY, name='scc')

            model.update()

            # objective
            model.setObjective(gb.quicksum(scc), gb.GRB.MAXIMIZE)

            # constraints
            model.addConstrs(select_paths.sum(q, '*') == 1 for q in range(num_quests))
            model.addConstrs(slices.sum(e, '*') == capacity[e] for e in range(num_edges))
            model.addConstrs(scc[q] == gb.min_(success.select('*', q)) for q in range(num_quests))
            model.addConstrs(edge_flow[e, q] == gb.quicksum(select_paths[q, p] * sp[q, p, e] for p in range(num_paths))
                             for q in range(num_quests) for e in range(num_edges))

            # conditional constraints
            model.addConstrs((success[e, q]) * (slices[e, q].ub - edge_flow[e, q].lb) >= slices[e, q] - edge_flow[e, q]
                             for e in range(num_edges) for q in range(num_quests))
            model.addConstrs(
                (success[e, q]) * 1 + (1 - success[e, q]) * (slices[e, q].lb - edge_flow[e, q].ub) <= slices[e, q] - edge_flow[
                    e, q]
                for e in range(num_edges) for q in range(num_quests))

            model.optimize()

            failure_rate = (num_quests - model.objVal) / num_quests
            fails = num_quests - model.objVal

            res_selct = np.zeros([num_quests, num_paths], dtype=np.int)

            for q in range(num_quests):
                for p in range(num_paths):
                    var = model.getVarByName(f"select_paths[{q},{p}]")
                    if var.x == 1:
                        res_selct[q, p] = 1

            modelling_time = time.time() - total_start

            if print_info:
                print_modelling_info(model,num_quests,failure_rate,modelling_time)

            # update capacity
            res_scc = np.ones([num_quests], dtype=np.int)
            res_flow = np.zeros([num_edges, num_quests], dtype=np.int)
            for q in range(num_quests):
                v = model.getVarByName(f"scc[{q}]")
                if v.x == 0:
                    res_scc[q] = 0

            for e in range(num_edges):
                for q in range(num_quests):
                    vf = model.getVarByName(f"flow[{e},{q}]")
                    if vf.x != 0:
                        res_flow[e, q] = vf.x
            occupy = np.multiply(res_flow, res_scc)

    return res_selct, occupy


def gen_label(graph, flow, capacity, num_paths, num_quests=1,print_info=False):
    num_edges = nx.number_of_edges(graph)
    edges = pd.DataFrame(graph.edges, columns=['src', 'dst'])

    shortest_path = []
    for q in range(num_quests):
        k_sp = k_shortest_paths(graph, source=flow[q][0], target=flow[q][1], k=num_paths)
        for i in range(num_paths - len(k_sp)):
            k_sp.append(k_sp[0])
        shortest_path.append(k_sp)

    # print('shortest paths (nodes): ',shortest_path)
    sp = np.zeros([num_quests, num_paths, num_edges], dtype=np.int)

    paths = []
    idx_list = []
    seqs = []
    for q in range(num_quests):
        for p in range(num_paths):
            path = shortest_path[q][p]
            pp = []
            for i in range(len(path) - 1):
                idx = edges[((edges.src == path[i]) & (edges.dst == path[i + 1])) |
                            ((edges.dst == path[i]) & (edges.src == path[i + 1]))].index[0]
                pp.append(idx)
                sp[q, p, idx] = flow[q][2]
            paths.append(pp)
    for i in range(len(paths)):
        seqs.append(list(range(len(paths[i]))))
        for j in range(len(paths[i])):
            idx_list.append(i)
    label, occupy = solve_slice(num_edges, num_quests, num_paths, sp, capacity,print_info)
    paths = list(deepflatten(paths))
    seqs = list(deepflatten(seqs))
    flow_size=np.zeros([num_quests,num_paths])
    for i in range(num_quests):
        flow_size[i,:]=flow[i][2]
    # transform label
    # for i in range(num_paths):
    #     l.append([label[0][i]])
    flow_size=flow_size.flatten()

    # reform labels
    l=[]
    for i in range(len(label)):
        l.append(np.argwhere(label[i]==1)[0][0])
    return paths, idx_list, seqs, l, occupy, flow_size

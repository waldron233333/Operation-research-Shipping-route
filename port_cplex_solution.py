import sys
from docplex.mp.model import Model

# ==============================================================================
# 1. 场景与数据配置
# ==============================================================================

T_horizon = 15
Time_Periods = list(range(T_horizon + 1))

Ports_CN = ['SHA', 'NGB']
Ports_US = ['LA', 'LB']
Ports_EU = ['ROT', 'HAM']
Ports_Trans = ['MX', 'SIN']
All_Ports = Ports_CN + Ports_US + Ports_EU + Ports_Trans

Fleet_Data = {
    'Ship_CN_1':    {'type': 'CN', 'cap': 20000, 'start': 'SHA'},
    'Ship_CN_2':    {'type': 'CN', 'cap': 20000, 'start': 'SHA'},
    'Ship_Other_1': {'type': 'Other', 'cap': 15000, 'start': 'SHA'}, 
    'Ship_Other_2': {'type': 'Other', 'cap': 15000, 'start': 'SHA'}   
}
Ships = list(Fleet_Data.keys())

# --- 需求定义 (包含你的新需求) ---
Demand_Segments = {
    ('SHA', 'LA'): [(5, 20000), (10, 50000), (15, 80000)],
    ('SHA', 'ROT'): [(8, 15000), (15, 30000)],
    ('ROT', 'LA'): [(12, 10000)], # 欧洲 -> 美国
    ('LA', 'ROT'): [(12, 10000)]  # 美国 -> 欧洲 (你关注的这条)
}
Commodities = list(Demand_Segments.keys())

def get_travel_time(src, dst):
    if src == dst: return 0
    # 区域内
    if (src in Ports_CN and dst in Ports_CN) or \
       (src in Ports_US and dst in Ports_US) or \
       (src in Ports_EU and dst in Ports_EU): return 1
    # 跨大洋
    if (src in Ports_CN and dst in Ports_US) or (src in Ports_US and dst in Ports_CN): return 3
    if (src in Ports_CN and dst in Ports_EU) or (src in Ports_EU and dst in Ports_CN): return 4
    # 欧美跨大西洋 (关键路径)
    if (src in Ports_EU and dst in Ports_US) or (src in Ports_US and dst in Ports_EU): return 2 
    # 中转
    if src in Ports_CN and dst == 'MX': return 2
    if src == 'MX' and dst in Ports_US: return 1
    if src in Ports_US and dst == 'MX': return 1
    if src == 'MX' and dst in Ports_CN: return 2
    if src in Ports_CN and dst == 'SIN': return 1
    if src == 'SIN' and dst in Ports_US: return 3
    if src == 'SIN' and dst in Ports_EU: return 3
    return 99

# 费用
Weekly_Op_Cost = 50
Wait_Cost = 10
Time_Penalty = 0.5
Excess_Penalty = 0.05
Cargo_Holding_Cost = 0.001 

def get_policy_fee(ship_name, src, dst):
    ship_type = Fleet_Data[ship_name]['type']
    if ship_type == 'CN' and src in Ports_CN and dst in Ports_US:
        return 1000 
    return 20

Valid_Routes = [(i, j) for i in All_Ports for j in All_Ports if i != j and get_travel_time(i, j) < 20]

# ==============================================================================
# 2. 模型构建 (修复版)
# ==============================================================================

mdl = Model(name="Strict_Transshipment")

# --- 变量生成 ---
keys_y = []
keys_z = []
for k in Ships:
    for (i, j) in Valid_Routes:
        tau = get_travel_time(i, j)
        for t in Time_Periods:
            if t + tau <= T_horizon:
                keys_y.append((k, i, j, t))
                for c in Commodities:
                    keys_z.append((k, i, j, t, c))

keys_x = [(k, p, t) for k in Ships for p in All_Ports for t in Time_Periods]
keys_z_stay = [(k, p, t, c) for k in Ships for p in All_Ports for t in Time_Periods for c in Commodities]

x = mdl.binary_var_dict(keys_x, name="X")
y = mdl.binary_var_dict(keys_y, name="Y")
z = mdl.continuous_var_dict(keys_z, lb=0, name="Z")
z_stay = mdl.continuous_var_dict(keys_z_stay, lb=0, name="Z_Stay")

# --- 目标函数 ---
obj_terms = []
for (k, i, j, t) in keys_y:
    tau = get_travel_time(i, j)
    fee = get_policy_fee(k, i, j)
    cost = (Weekly_Op_Cost * tau) + fee + (t * Time_Penalty)
    obj_terms.append(y[k, i, j, t] * cost)

for k in Ships:
    for p in All_Ports:
        for t in Time_Periods:
            obj_terms.append(x[k, p, t] * Wait_Cost)

for key in keys_z: obj_terms.append(z[key] * Cargo_Holding_Cost)
for key in keys_z_stay: obj_terms.append(z_stay[key] * Cargo_Holding_Cost)

# --- 约束条件 ---

# 1. 船只流平衡
for k in Ships:
    start_node = Fleet_Data[k]['start']
    for p in All_Ports:
        out_voyage = mdl.sum(y[k, p, j, 0] for j in All_Ports if (k, p, j, 0) in y)
        mdl.add_constraint(x[k, p, 0] + out_voyage == (1 if p == start_node else 0))

    for p in All_Ports:
        for t in range(1, T_horizon + 1):
            in_stay = x[k, p, t - 1]
            in_arrive_list = []
            for origin in All_Ports:
                if (origin, p) in Valid_Routes:
                    tau = get_travel_time(origin, p)
                    t_dept = t - tau
                    if (k, origin, p, t_dept) in y:
                        in_arrive_list.append(y[k, origin, p, t_dept])
            
            out_stay = x[k, p, t]
            out_depart_list = []
            if t < T_horizon:
                for dest in All_Ports:
                    if (k, p, dest, t) in y:
                        out_depart_list.append(y[k, p, dest, t])
            mdl.add_constraint(in_stay + mdl.sum(in_arrive_list) == out_stay + mdl.sum(out_depart_list))

# 2. 容量约束
for (k, i, j, t) in keys_y:
    cargo_sum = mdl.sum(z[k, i, j, t, c] for c in Commodities)
    mdl.add_constraint(cargo_sum <= Fleet_Data[k]['cap'] * y[k, i, j, t])
for (k, p, t) in keys_x:
    cargo_sum = mdl.sum(z_stay[k, p, t, c] for c in Commodities)
    mdl.add_constraint(cargo_sum <= Fleet_Data[k]['cap'] * x[k, p, t])

# 3. 货物守恒 (修复瞬移BUG的核心)
print("构建货物守恒约束...")
for k in Ships:
    for comm in Commodities:
        (c_src, c_dst) = comm
        
        for p in All_Ports:
            for t in range(1, T_horizon + 1):
                # Inflow
                flow_in = z_stay[k, p, t-1, comm]
                arr_list = []
                for origin in All_Ports:
                    if (origin, p) in Valid_Routes:
                        tau = get_travel_time(origin, p)
                        t_dept = t - tau
                        if (k, origin, p, t_dept, comm) in z:
                            arr_list.append(z[k, origin, p, t_dept, comm])
                lhs = flow_in + mdl.sum(arr_list)

                # Outflow
                flow_out = z_stay[k, p, t, comm]
                dept_list = []
                if t < T_horizon:
                    for dest in All_Ports:
                        if (k, p, dest, t, comm) in z:
                            dept_list.append(z[k, p, dest, t, comm])
                rhs = flow_out + mdl.sum(dept_list)

                # 约束逻辑
                if p == c_src:
                    # 【起点】: 流出 >= 流入 (差值就是从港口装上船的)
                    mdl.add_constraint(rhs >= lhs)
                elif p == c_dst:
                    # 【终点】: 必须强制卸货 (流出 == 0)
                    mdl.add_constraint(rhs == 0)
                else:
                    # 【非起点非终点】: 严禁装货！(流出 == 流入)
                    # 之前是 lhs == rhs，这里必须严格，防止在上海凭空装洛杉矶的货
                    mdl.add_constraint(lhs == rhs)

# 4. 需求满足
for (src, dst), segments in Demand_Segments.items():
    this_comm = (src, dst)
    for (deadline, needed_amt) in segments:
        total_delivered = 0
        for k in Ships:
            for origin in All_Ports:
                if (origin, dst) in Valid_Routes:
                    tau = get_travel_time(origin, dst)
                    for t_dept in Time_Periods:
                        if (k, origin, dst, t_dept, this_comm) in z:
                            if t_dept + tau <= deadline:
                                total_delivered += z[k, origin, dst, t_dept, this_comm]
        
        excess = mdl.continuous_var(lb=0)
        mdl.add_constraint(total_delivered >= needed_amt)
        mdl.add_constraint(excess >= total_delivered - needed_amt)
        obj_terms.append(excess * Excess_Penalty)

mdl.minimize(mdl.sum(obj_terms))

# ==============================================================================
# 3. 求解与输出
# ==============================================================================
sol = mdl.solve(log_output=False)

if sol:
    print(f"\nStatus: Optimal | Obj: {sol.objective_value:.2f}")
    
    events = []
    for (k, i, j, t) in keys_y:
        val_y = y[k, i, j, t].solution_value
        if val_y > 0.5:
            tau = get_travel_time(i, j)
            fee = get_policy_fee(k, i, j)
            
            c_str = []
            for c in Commodities:
                val_z = z[k, i, j, t, c].solution_value
                if val_z > 0.1:
                    c_str.append(f"{c[0]}->{c[1]}:{int(round(val_z))}")
            
            cargo_disp = ", ".join(c_str) if c_str else "Empty"
            
            # 智能判断：这是什么类型的航线？
            route_type = "Re-position" # 默认为调机/空载
            if c_str:
                if i == 'SHA' and j == 'LA': route_type = "Direct(HighFee)"
                elif i == 'MX' and j == 'LA': route_type = "Transshipment"
                elif i == 'SHA' and j == 'ROT' and "LA->ROT" in cargo_disp: route_type = "Consolidation" # 拼单
                elif i == 'LA' and j == 'SHA' and "LA->ROT" in cargo_disp: route_type = "Relay (Backhaul)" # 回程捎带
                else: route_type = "Direct"

            events.append({
                't': t, 's': k, 'a': f"{i} -> {j}", 'c': cargo_disp, 
                'info': f"{route_type} | Fee={fee}", 'raw_t': t
            })
            
    events.sort(key=lambda x: x['t'])
    
    print("\n" + "="*120)
    print(f"{'Time':<4} | {'Ship':<15} | {'Action':<15} | {'Cargo Onboard':<40} | {'Route Analysis'}")
    print("="*120)
    for e in events:
        print(f"{e['t']:<4} | {e['s']:<15} | {e['a']:<15} | {e['c']:<40} | {e['info']}")
    print("-" * 120)

    # 打印 LA->ROT 货物的特殊追踪
    print("\n>>> 追踪: LA -> ROT 货物去哪了?")
    found = False
    for e in events:
        if "LA->ROT" in e['c']:
            found = True
            print(f"  Week {e['t']}: {e['s']} 在 {e['a']} 携带了 LA->ROT 货物 ({e['info']})")
    if not found:
        print("  未发现 LA->ROT 货物移动 (可能需求未满足或通过其他方式)")

else:
    print("No Solution.")
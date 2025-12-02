import sys
from docplex.mp.model import Model

# ==============================================================================
# 1. 场景配置 (保持业务逻辑不变)
# ==============================================================================

T_horizon = 15
Time_Periods = list(range(T_horizon + 1))

Ports_CN = ['SHA', 'NGB']
Ports_US = ['LA', 'LB']
Ports_EU = ['ROT', 'HAM']
Ports_Trans = ['MX', 'SIN']
All_Ports = Ports_CN + Ports_US + Ports_EU + Ports_Trans

Fleet_Data = {
    'Ship_CN_1':    {'type': 'CN', 'cap': 20000, 'start': 'SHA', 'cost': 50},
    'Ship_CN_2':    {'type': 'CN', 'cap': 20000, 'start': 'SHA', 'cost': 50},
    'Ship_Other_1': {'type': 'Other', 'cap': 15000, 'start': 'SHA', 'cost': 40},
    'Ship_Other_2': {'type': 'Other', 'cap': 15000, 'start': 'SHA', 'cost': 40}
}
Ships = list(Fleet_Data.keys())

Demand_Segments = {
    ('SHA', 'LA'): [(5, 20000), (10, 50000), (15, 80000)],
    ('SHA', 'ROT'): [(8, 15000), (15, 30000)],
    ('ROT', 'LA'): [(12, 10000)]
}
Commodities = list(Demand_Segments.keys())

def get_travel_time(src, dst):
    if src == dst: return 0
    # 区域内 (1周)
    if (src in Ports_CN and dst in Ports_CN) or \
       (src in Ports_US and dst in Ports_US) or \
       (src in Ports_EU and dst in Ports_EU): return 1
    # 跨大洋
    if (src in Ports_CN and dst in Ports_US) or (src in Ports_US and dst in Ports_CN): return 3
    if (src in Ports_CN and dst in Ports_EU) or (src in Ports_EU and dst in Ports_CN): return 4
    if (src in Ports_EU and dst in Ports_US) or (src in Ports_US and dst in Ports_EU): return 2
    # 中转
    if src in Ports_CN and dst == 'MX': return 2
    if src == 'MX' and dst in Ports_US: return 1
    if src in Ports_US and dst == 'MX': return 1 # 回程
    if src == 'MX' and dst in Ports_CN: return 2 # 回程
    
    # 移除无效路径 (比如 ROT->MX, SIN->MX)，直接返回大数
    return 99

# 费用参数
Wait_Cost = 10
Time_Penalty = 0.5
Excess_Penalty = 0.05
Cargo_Cost = 0.001

def get_policy_fee(ship_name, src, dst):
    ship_type = Fleet_Data[ship_name]['type']
    if ship_type == 'CN' and src in Ports_CN and dst in Ports_US:
        return 1000
    return 20

# 优化1：严格剪枝，只保留小于10周的航线
Valid_Routes = [(i, j) for i in All_Ports for j in All_Ports if i != j and get_travel_time(i, j) < 10]

# ==============================================================================
# 2. 优化后的模型构建
# ==============================================================================

mdl = Model(name="Optimized_Logistics")

# 优化2：求解器参数调整 (解决解不出来的问题)
mdl.parameters.timelimit = 60          # 限制最长计算 60秒
mdl.parameters.mip.tolerances.mipgap = 0.05  # 允许 5% 的误差 (只要解足够好就行，不用完美)
mdl.parameters.emphasis.mip = 1        # 强调寻找可行解 (Feasibility)

print("生成变量索引...")

# --- 稀疏变量生成 (Sparse Key Generation) ---
# 只生成物理上可行的时间段，不生成多余的 key
keys_y = []
keys_z = [] 

for k in Ships:
    for (i, j) in Valid_Routes:
        tau = get_travel_time(i, j)
        for t in Time_Periods:
            # 严格时间过滤：超时的航程不生成变量
            if t + tau <= T_horizon:
                keys_y.append((k, i, j, t))
                for c in Commodities:
                    keys_z.append((k, i, j, t, c))

keys_x = [(k, p, t) for k in Ships for p in All_Ports for t in Time_Periods]
keys_z_stay = [(k, p, t, c) for k in Ships for p in All_Ports for t in Time_Periods for c in Commodities]

print(f"变量规模预估: 航行变量 {len(keys_y)}, 货物变量 {len(keys_z) + len(keys_z_stay)}")

# --- 定义变量 (不命名以节省内存) ---
# 优化3：移除 name="..." 参数，极大降低内存占用
x = mdl.binary_var_dict(keys_x)          
y = mdl.binary_var_dict(keys_y)          
z = mdl.continuous_var_dict(keys_z, lb=0) 
z_stay = mdl.continuous_var_dict(keys_z_stay, lb=0)

# --- 目标函数 ---
obj_terms = []

# 1. 航行成本
for (k, i, j, t) in keys_y:
    tau = get_travel_time(i, j)
    fee = get_policy_fee(k, i, j)
    cost = (Fleet_Data[k]['cost'] * tau) + fee + (t * Time_Penalty)
    obj_terms.append(y[k, i, j, t] * cost)

# 2. 停泊成本
for k in Ships:
    for p in All_Ports:
        for t in Time_Periods:
            obj_terms.append(x[k, p, t] * Wait_Cost)

# 3. 货物持有成本 (防止幽灵货)
for key in keys_z:
    obj_terms.append(z[key] * Cargo_Cost)
for key in keys_z_stay:
    obj_terms.append(z_stay[key] * Cargo_Cost)

# --- 约束条件 ---
print("构建约束...")

# 1. 船只流平衡
for k in Ships:
    start_node = Fleet_Data[k]['start']
    
    # t=0
    for p in All_Ports:
        out_voyage = mdl.sum(y[k, p, j, 0] for j in All_Ports if (k, p, j, 0) in y)
        mdl.add_constraint(x[k, p, 0] + out_voyage == (1 if p == start_node else 0))

    # t=1..T
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

# 2. 容量耦合
for (k, i, j, t) in keys_y:
    cargo_sum = mdl.sum(z[k, i, j, t, c] for c in Commodities)
    mdl.add_constraint(cargo_sum <= Fleet_Data[k]['cap'] * y[k, i, j, t])

for (k, p, t) in keys_x:
    cargo_sum = mdl.sum(z_stay[k, p, t, c] for c in Commodities)
    mdl.add_constraint(cargo_sum <= Fleet_Data[k]['cap'] * x[k, p, t])

# 3. 货物守恒 (含强制卸货逻辑)
for k in Ships:
    for comm in Commodities:
        (c_src, c_dst) = comm
        for p in All_Ports:
            for t in range(1, T_horizon + 1):
                # In
                flow_in = z_stay[k, p, t-1, comm]
                arr_list = []
                for origin in All_Ports:
                    if (origin, p) in Valid_Routes:
                        tau = get_travel_time(origin, p)
                        t_dept = t - tau
                        if (k, origin, p, t_dept, comm) in z:
                            arr_list.append(z[k, origin, p, t_dept, comm])
                lhs = flow_in + mdl.sum(arr_list)

                # Out
                flow_out = z_stay[k, p, t, comm]
                dept_list = []
                if t < T_horizon:
                    for dest in All_Ports:
                        if (k, p, dest, t, comm) in z:
                            dept_list.append(z[k, p, dest, t, comm])
                rhs = flow_out + mdl.sum(dept_list)

                # Logic
                if p == c_src:
                    mdl.add_constraint(rhs >= lhs) # Load
                elif p == c_dst:
                    mdl.add_constraint(rhs == 0)   # Force Unload (Fix loop bug)
                else:
                    mdl.add_constraint(lhs == rhs) # Conservation

# 4. 需求满足
for (src, dst), segments in Demand_Segments.items():
    this_comm = (src, dst)
    for (deadline, needed_amt) in segments:
        
        delivered = []
        for k in Ships:
            for origin in All_Ports:
                if (origin, dst) in Valid_Routes:
                    tau = get_travel_time(origin, dst)
                    for t_dept in Time_Periods:
                        if t_dept + tau <= deadline:
                            if (k, origin, dst, t_dept, this_comm) in z:
                                delivered.append(z[k, origin, dst, t_dept, this_comm])
        
        total_delivered = mdl.sum(delivered)
        excess = mdl.continuous_var(lb=0) # 匿名变量省内存
        
        mdl.add_constraint(total_delivered >= needed_amt)
        mdl.add_constraint(excess >= total_delivered - needed_amt)
        obj_terms.append(excess * Excess_Penalty)

mdl.minimize(mdl.sum(obj_terms))

# ==============================================================================
# 3. 求解
# ==============================================================================
print("开始求解 (限时60秒)...")
sol = mdl.solve(log_output=True) # 打开日志看进度

if sol:
    print(f"\nStatus: Solved | Obj: {sol.objective_value:.2f}")
    
    def get_val(var): return var.solution_value

    events = []
    for (k, i, j, t) in keys_y:
        val_y = y[k, i, j, t].solution_value
        if val_y > 0.5:
            tau = get_travel_time(i, j)
            fee = get_policy_fee(k, i, j)
            
            # Cargo
            c_str = []
            load = 0
            for c in Commodities:
                val_z = z[k, i, j, t, c].solution_value
                if val_z > 0.1:
                    c_str.append(f"{c[0]}->{c[1]}:{int(round(val_z))}")
                    load += val_z
            
            load_info = f"Load:{int(round(load))} | " + ",".join(c_str) if load > 0.1 else "Empty"
            fee_tag = "(!HIGH!)" if fee >= 500 else "(Low)"
            
            events.append({
                't': t,
                's': f"{k} ({Fleet_Data[k]['type']})",
                'a': f"{i} -> {j}",
                'c': load_info,
                'i': f"{fee_tag} Fee={fee}"
            })
            
    events.sort(key=lambda x: x['t'])
    
    print("\n" + "="*120)
    print(f"{'Time':<4} | {'Ship':<20} | {'Action':<20} | {'Cargo':<50} | {'Info'}")
    print("="*120)
    for e in events:
        print(f"{e['t']:<4} | {e['s']:<20} | {e['a']:<20} | {e['c']:<50} | {e['i']}")
    print("-" * 120)
    
else:
    print("未找到解 (可能是内存不足或约束冲突，尝试减少 T_horizon 或减少港口数量)")
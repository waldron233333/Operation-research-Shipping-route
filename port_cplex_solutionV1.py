import sys
from docplex.mp.model import Model
import config  # <--- 导入你的配置文件

# ==============================================================================
# 1. 场景配置 (从 Config 读取)
# ==============================================================================

print(f"Loading configuration... Horizon: {config.T_HORIZON} weeks")

T_horizon = config.T_HORIZON
Time_Periods = list(range(T_horizon + 1))

# 港口引用
Ports_CN = config.PORTS_CN
Ports_US = config.PORTS_US
Ports_EU = config.PORTS_EU
Ports_Trans = config.PORTS_TRANS
All_Ports = config.ALL_PORTS

# 船队引用
Fleet_Data = config.FLEET_DATA_DICT
Ships = config.SHIPS_LIST

# 需求引用
Demand_Segments = config.DEMAND_SEGMENTS
Commodities = list(Demand_Segments.keys())

# 费用参数引用
Wait_Cost = config.COST_PARAMS['WAIT_COST']
Time_Penalty = config.COST_PARAMS['TIME_PENALTY']
Excess_Penalty = config.COST_PARAMS['EXCESS_PENALTY']
Cargo_Holding_Cost = config.COST_PARAMS['CARGO_COST']


# --- 核心逻辑函数适配 ---

def get_travel_time(src, dst):
    if src == dst: return 0

    # 1. 优先查表 (Config中的真实航运数据)
    if (src, dst) in config.TRANSIT_TIMES:
        return config.TRANSIT_TIMES[(src, dst)]

    # 2. 区域内移动 (默认 1 周)
    # 用于如 SHA->NGB 移船，或 LAX->LGB
    if (src in Ports_CN and dst in Ports_CN) or \
            (src in Ports_US and dst in Ports_US) or \
            (src in Ports_EU and dst in Ports_EU) or \
            (src in Ports_Trans and dst in Ports_Trans):
        return 1

    # 3. 未定义的跨区航线 (视为不可达)
    return 99


def get_policy_fee(ship_name, src, dst):
    ship_type = Fleet_Data[ship_name]['type']

    # 政策逻辑：中国船 (CN) 去往 美国港口 (US)
    # 这里假设只要目的地是美国港口，且船只类型受限，就罚款
    if ship_type == 'CN' and dst in Ports_US:
        return config.COST_PARAMS['POLICY_FEE_HIGH']

    return config.COST_PARAMS['POLICY_FEE_LOW']


# --- 路径剪枝 ---
# 只保留时间合理的航线，减少变量数量 (例如限制单程小于10周)
# 真实数据中亚欧绕行约需 7周 (45天)，所以阈值设为 10 是安全的
Valid_Routes = [(i, j) for i in All_Ports for j in All_Ports
                if i != j and get_travel_time(i, j) < 10]

print(f"有效航线数量: {len(Valid_Routes)}")
print(f"船队规模: {len(Ships)} 艘")

# ==============================================================================
# 2. 模型构建
# ==============================================================================

mdl = Model(name="Global_Fleet_Opt")

# 求解参数优化 (针对20艘船的大规模问题)
mdl.parameters.timelimit = 120  # 限时 120秒
mdl.parameters.mip.tolerances.mipgap = 0.02  # 允许 2% 的最优性差距
mdl.parameters.emphasis.mip = 1  # 强调寻找可行解

# --- 变量生成 ---
print("生成变量...")
keys_y = []
keys_z = []
for k in Ships:
    for (i, j) in Valid_Routes:
        tau = get_travel_time(i, j)
        for t in Time_Periods:
            # 严格的时间截止检查
            if t + tau <= T_horizon:
                keys_y.append((k, i, j, t))
                # 只有当这条航线可能运输某种货物时才生成 cargo 变量 (可进一步优化，这里暂全生成)
                for c in Commodities:
                    keys_z.append((k, i, j, t, c))

keys_x = [(k, p, t) for k in Ships for p in All_Ports for t in Time_Periods]
keys_z_stay = [(k, p, t, c) for k in Ships for p in All_Ports for t in Time_Periods for c in Commodities]

# 使用 dict 变量 (节省内存)
x = mdl.binary_var_dict(keys_x)
y = mdl.binary_var_dict(keys_y)
z = mdl.continuous_var_dict(keys_z, lb=0)
z_stay = mdl.continuous_var_dict(keys_z_stay, lb=0)

print(f"变量总数: ~{len(keys_y) + len(keys_z) + len(keys_x) + len(keys_z_stay)}")

# --- 目标函数 ---
obj_terms = []

# 1. 航行成本 (运营费 + 政策费 + 时间惩罚)
for (k, i, j, t) in keys_y:
    tau = get_travel_time(i, j)
    fee = get_policy_fee(k, i, j)
    # 注意：这里使用了 Config 中每艘船特定的 cost
    ship_op_cost = Fleet_Data[k]['cost']

    cost = (ship_op_cost * tau) + fee + (t * Time_Penalty)
    obj_terms.append(y[k, i, j, t] * cost)

# 2. 停泊成本
for k in Ships:
    for p in All_Ports:
        for t in Time_Periods:
            obj_terms.append(x[k, p, t] * Wait_Cost)

# 3. 货物持有成本
for key in keys_z: obj_terms.append(z[key] * Cargo_Holding_Cost)
for key in keys_z_stay: obj_terms.append(z_stay[key] * Cargo_Holding_Cost)

# --- 约束条件 ---
print("添加约束...")

# 1. 船只流平衡
for k in Ships:
    start_node = Fleet_Data[k]['start']

    # t=0 初始状态
    for p in All_Ports:
        out_voyage = mdl.sum(y[k, p, j, 0] for j in All_Ports if (k, p, j, 0) in y)
        mdl.add_constraint(x[k, p, 0] + out_voyage == (1 if p == start_node else 0))

    # t=1..T 动态平衡
    for p in All_Ports:
        for t in range(1, T_horizon + 1):
            in_stay = x[k, p, t - 1]
            in_arrive_list = []

            # 寻找所有可能的来源港
            # 优化：只遍历 Valid_Routes 中以此为终点的路线
            possible_origins = [src for (src, dst) in Valid_Routes if dst == p]

            for origin in possible_origins:
                tau = get_travel_time(origin, p)
                t_dept = t - tau
                if (k, origin, p, t_dept) in y:
                    in_arrive_list.append(y[k, origin, p, t_dept])

            out_stay = x[k, p, t]
            out_depart_list = []
            if t < T_horizon:
                possible_dests = [dst for (src, dst) in Valid_Routes if src == p]
                for dest in possible_dests:
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

# 3. 货物守恒
for k in Ships:
    for comm in Commodities:
        (c_src, c_dst) = comm
        for p in All_Ports:
            for t in range(1, T_horizon + 1):
                # Inflow
                flow_in = z_stay[k, p, t - 1, comm]
                arr_list = []
                # 优化来源遍历
                possible_origins = [src for (src, dst) in Valid_Routes if dst == p]
                for origin in possible_origins:
                    tau = get_travel_time(origin, p)
                    t_dept = t - tau
                    if (k, origin, p, t_dept, comm) in z:
                        arr_list.append(z[k, origin, p, t_dept, comm])
                lhs = flow_in + mdl.sum(arr_list)

                # Outflow
                flow_out = z_stay[k, p, t, comm]
                dept_list = []
                if t < T_horizon:
                    possible_dests = [dst for (src, dst) in Valid_Routes if src == p]
                    for dest in possible_dests:
                        if (k, p, dest, t, comm) in z:
                            dept_list.append(z[k, p, dest, t, comm])
                rhs = flow_out + mdl.sum(dept_list)

                # 逻辑
                if p == c_src:
                    mdl.add_constraint(rhs >= lhs)  # Load
                elif p == c_dst:
                    mdl.add_constraint(rhs == 0)  # Force Unload
                else:
                    mdl.add_constraint(lhs == rhs)  # Conservation

# 4. 需求满足
for (src, dst), segments in Demand_Segments.items():
    this_comm = (src, dst)
    for (deadline, needed_amt) in segments:
        total_delivered = 0
        for k in Ships:
            # 查找所有能到达 dst 的路径
            possible_origins = [s for (s, d) in Valid_Routes if d == dst]
            for origin in possible_origins:
                tau = get_travel_time(origin, dst)
                for t_dept in Time_Periods:
                    # 只要出发+航程 <= 截止时间，且变量存在
                    if t_dept + tau <= deadline:
                        if (k, origin, dst, t_dept, this_comm) in z:
                            total_delivered += z[k, origin, dst, t_dept, this_comm]

        excess = mdl.continuous_var(lb=0)
        mdl.add_constraint(total_delivered >= needed_amt)
        mdl.add_constraint(excess >= total_delivered - needed_amt)
        obj_terms.append(excess * Excess_Penalty)

mdl.minimize(mdl.sum(obj_terms))

# ==============================================================================
# 3. 求解与输出
# ==============================================================================
print("开始求解...")
sol = mdl.solve(log_output=True)

if sol:
    print(f"\nStatus: Solved | Obj: {sol.objective_value:.2f}")

    events = []
    for (k, i, j, t) in keys_y:
        val_y = y[k, i, j, t].solution_value
        if val_y > 0.5:
            tau = get_travel_time(i, j)
            fee = get_policy_fee(k, i, j)

            c_str = []
            total_load = 0
            for c in Commodities:
                val_z = z[k, i, j, t, c].solution_value
                if val_z > 1.0:  # 过滤掉极小值
                    c_str.append(f"{c[0]}->{c[1]}:{int(round(val_z))}")
                    total_load += val_z

            cargo_disp = ", ".join(c_str) if c_str else "Empty"

            # 简单的路由分析
            route_info = "Standard"
            if fee >= 500:
                route_info = "!!HIGH PENALTY!!"
            elif i in Ports_Trans:
                route_info = "Transshipment"
            elif total_load < 100:
                route_info = "Relay/Reposition"

            events.append({
                't': t, 's': f"{k}({Fleet_Data[k]['type']})",
                'a': f"{i} -> {j}", 'c': cargo_disp,
                'info': f"{route_info} | Arr: t+{tau}", 'raw_t': t
            })

    events.sort(key=lambda x: x['t'])

    print("\n" + "=" * 130)
    print(f"{'Time':<4} | {'Ship':<20} | {'Action':<20} | {'Cargo Onboard':<50} | {'Info'}")
    print("=" * 130)
    for e in events:
        print(f"{e['t']:<4} | {e['s']:<20} | {e['a']:<20} | {e['c']:<50} | {e['info']}")
    print("-" * 130)

    # 简单的需求达成统计
    print("\n>>> 需求满足情况概览:")
    for comm, segments in Demand_Segments.items():
        print(f"Route {comm}:")
        # 简单取最后一个segment看最终完成情况
        last_seg = segments[-1]
        print(f"  Target by Week {last_seg[0]}: {last_seg[1]} TEU")

else:
    print("No Solution Found. 建议检查 Config 中的需求量是否超过了 20艘船的总运力，或者时间窗口 T_Horizon 是否太短。")
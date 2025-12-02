import math

# ==============================================================================
# 1. 全局仿真设置
# ==============================================================================
# 仿真周期 (周)
# 建议：20-25周，足够覆盖亚欧航线(单程6周)的往返
T_HORIZON = 24

# ==============================================================================
# 2. 港口定义
# ==============================================================================
PORTS_CN = ['SHA', 'NGB', 'YTN']  # 上海, 宁波, 盐田
PORTS_US = ['LAX', 'LGB', 'NYC']  # 洛杉矶, 长滩, 纽约
PORTS_EU = ['RTM', 'HAM', 'ANT']  # 鹿特丹, 汉堡, 安特卫普
PORTS_TRANS = ['SIN', 'PUS', 'MX']  # 新加坡, 釜山, 墨西哥(曼萨尼约)

ALL_PORTS = PORTS_CN + PORTS_US + PORTS_EU + PORTS_TRANS

# ==============================================================================
# 3. 航线与需求数据 (基于2024年真实数据估算)
# ==============================================================================
# 格式: (起点, 终点): {'annual_teu': 年度总量, 'days': 航行天数}
# 注意：这里的数据是双向总量，后续代码会自动除以2来计算单向需求
RAW_ROUTE_DATA = {
    # --- 跨太平洋 (Transpacific) ---
    ('SHA', 'LAX'): {'annual_teu': 2150000, 'days': 16},  # 直航快船
    ('NGB', 'LAX'): {'annual_teu': 1850000, 'days': 17},
    ('YTN', 'LAX'): {'annual_teu': 1600000, 'days': 15},
    ('SHA', 'NYC'): {'annual_teu': 900000, 'days': 35},  # 经巴拿马

    # --- 亚欧线 (绕行好望角模式) ---
    ('SHA', 'RTM'): {'annual_teu': 1300000, 'days': 42},
    ('NGB', 'HAM'): {'annual_teu': 880000, 'days': 45},

    # --- 亚洲内部/中转 (Intra-Asia / Feeder) ---
    ('SHA', 'SIN'): {'annual_teu': 1550000, 'days': 7},  # 极大的中转流
    ('SHA', 'PUS'): {'annual_teu': 780000, 'days': 2},  # 釜山快线
    ('SHA', 'MX'): {'annual_teu': 400000, 'days': 20},  # 假设数据：去墨西哥的直航

    # --- 中转连接段 (用于模型寻找跳板) ---
    # 新加坡 -> 欧美
    ('SIN', 'LAX'): {'annual_teu': 0, 'days': 26},  # 新加坡直发美西
    ('SIN', 'RTM'): {'annual_teu': 0, 'days': 36},  # 新加坡直发欧洲

    # 釜山 -> 美国
    ('PUS', 'LAX'): {'annual_teu': 0, 'days': 13},  # 釜山去美西很快

    # 墨西哥 -> 美国
    ('MX', 'LAX'): {'annual_teu': 0, 'days': 4},  # 短途接驳

    # 欧洲 -> 美国 (跨大西洋)
    ('RTM', 'NYC'): {'annual_teu': 600000, 'days': 10},
}

# ==============================================================================
# 4. 船队配置 (20艘船模拟)
# ==============================================================================
# 假设均为 15,000 TEU 级别的 Neo-Panamax 船只
# Cost 单位：万美元/周 (假设燃油+运营约 150万/周 -> 缩放为 150)
FLEET_CONFIG = []

# 10艘中国造/中国籍船 (受限)
for i in range(1, 11):
    FLEET_CONFIG.append({
        'name': f'Ship_CN_{i}',
        'type': 'CN',
        'cap': 15000,
        'start': 'SHA',  # 假设初始都在上海等待指令
        'cost': 150  # 运营成本
    })

# 10艘非中国船 (非受限)
for i in range(1, 11):
    FLEET_CONFIG.append({
        'name': f'Ship_Other_{i}',
        'type': 'Other',
        'cap': 15000,
        'start': 'SHA',  # 初始也在上海，或可设为 SIN/RTM
        'cost': 140  # 假设由于技术或管理优势，成本略低一点点，或者持平
    })

# ==============================================================================
# 5. 费用与惩罚参数 (单位：万美元)
# ==============================================================================
COST_PARAMS = {
    'WAIT_COST': 20,  # 停泊/闲置成本 (每周)
    'TIME_PENALTY': 1,  # 时间折旧/利息成本
    'EXCESS_PENALTY': 0.1,  # 超运/库存持有成本 (每TEU)
    'CARGO_COST': 0.005,  # 货物在途资金占用成本

    # 政策关税 (关键参数)
    'POLICY_FEE_HIGH': 2000,  # 针对CN船去美国的重罚 (单航次或折算后)
    # 如果按箱罚 $500, 1.5万箱就是 750万美元 -> 750
    # 这里设为 2000 (2000万) 以确保产生绝对阻吓
    'POLICY_FEE_LOW': 50  # 正常港口挂靠费
}


# ==============================================================================
# 6. 数据处理辅助函数
# ==============================================================================

def get_weekly_demand_segments(t_horizon):
    """
    根据年度TEU自动计算每周需求，并生成分段累计目标
    """
    segments = {}

    for (src, dst), data in RAW_ROUTE_DATA.items():
        annual = data['annual_teu']
        if annual == 0: continue  # 纯中转路段没有原生需求

        # 核心逻辑：年度总量的一半 (单向) / 52周
        weekly_demand = (annual / 2) / 52
        weekly_demand = int(round(weekly_demand))

        # 生成累计检查点 (每4周检查一次)
        # 例如: 第4周需完成4周的量，第8周需完成8周的量
        checkpoints = []
        for t in range(4, t_horizon + 1, 4):
            cum_amt = weekly_demand * t
            checkpoints.append((t, cum_amt))

        # 补上最后一周 (如果没被4整除)
        if t_horizon % 4 != 0:
            checkpoints.append((t_horizon, weekly_demand * t_horizon))

        segments[(src, dst)] = checkpoints

    return segments


def get_transit_time_matrix():
    """
    将航行天数转换为周数 (Matrix Dictionary)
    """
    matrix = {}
    for (src, dst), data in RAW_ROUTE_DATA.items():
        days = data['days']
        weeks = math.ceil(days / 7.0)  # 向上取整，不足一周按一周算
        matrix[(src, dst)] = weeks

        # 假设回程时间相同 (简化)
        matrix[(dst, src)] = weeks

    return matrix


# ==============================================================================
# 7. 导出给求解器的数据对象
# ==============================================================================
# 实际计算出的字典
DEMAND_SEGMENTS = get_weekly_demand_segments(T_HORIZON)
TRANSIT_TIMES = get_transit_time_matrix()

# 将 Fleet 列表转为字典格式，适配原代码
FLEET_DATA_DICT = {ship['name']: ship for ship in FLEET_CONFIG}
SHIPS_LIST = list(FLEET_DATA_DICT.keys())

# 打印预览 (调试用)
if __name__ == "__main__":
    print(f"Simulation Horizon: {T_HORIZON} weeks")
    print(f"Total Ships: {len(SHIPS_LIST)}")
    print("\n--- Sample Transit Times (Weeks) ---")
    print(f"SHA -> LAX: {TRANSIT_TIMES.get(('SHA', 'LAX'), 'N/A')}")
    print(f"SHA -> RTM: {TRANSIT_TIMES.get(('SHA', 'RTM'), 'N/A')}")
    print(f"SHA -> SIN: {TRANSIT_TIMES.get(('SHA', 'SIN'), 'N/A')}")

    print("\n--- Sample Demand Segments (Cumulative TEU) ---")
    if ('SHA', 'LAX') in DEMAND_SEGMENTS:
        print(f"SHA -> LAX (Weekly avg ~{(RAW_ROUTE_DATA[('SHA', 'LAX')]['annual_teu'] / 2 / 52):.0f}):")
        for pt in DEMAND_SEGMENTS[('SHA', 'LAX')]:
            print(f"  Week {pt[0]}: Need {pt[1]}")
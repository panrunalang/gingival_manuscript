"""
计算问卷Cronbach's α系数 - 补充表
对问卷条目（Q10-Q20）进行编码并计算内部一致性信度
纯Python，不依赖第三方库
"""
import csv
import math

# ─── 1. 读取数据 ───
with open('results/survey.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

n = len(rows)
print(f"有效样本量: {n}")

# ─── 2. 准确列名 ───
COL_Q10 = '10、术前是否通过一些途径了解过？ '
COL_Q11 = '11、术前是否焦虑甚至恐惧？ '
COL_Q12 = '12、术前医护是否与您沟通过手术方案，例如手术必要性、流程、费用等？'
COL_Q13 = '13、术前是否与您签署了手术知情同意书？ '
COL_Q14 = '14、是否了解牙周膜龈手术治疗前需进行牙周基础治疗(洁牙、刮治等)？ '
COL_Q15 = '15、术后反应如何？ '
COL_Q16 = '16、对牙周膜龈手术治疗的认知程度会直接影响您是否接受手术治疗： '
COL_Q17 = '17、手术费用会直接影响您的选择： '
COL_Q18 = '18、心理负担会直接影响您的选择： '
COL_Q19 = '19、全身状况(血压、血糖、心脏等)是否能承受会影响您的选择 '
COL_Q20 = '20、本次手术治疗是否会影响到您今后的方案选择 '

ALL_COLS = [COL_Q10, COL_Q11, COL_Q12, COL_Q13, COL_Q14, COL_Q15,
            COL_Q16, COL_Q17, COL_Q18, COL_Q19, COL_Q20]

# ─── 3. 编码映射 ───
def encode(rows, col, mapping):
    """对一列应用编码映射，返回数值列表"""
    result = []
    for r in rows:
        raw = r[col]
        if raw in mapping:
            result.append(mapping[raw])
        else:
            raise ValueError(f"未映射的值 [{col}]: {repr(raw)}")
    return result

# Q10: 术前了解途径 (0-3, 越高越主动通过专业渠道了解)
q10_map = {
    'D.没了解过': 0,
    'B.身边人': 1,
    'A.网络': 2,
    'C.主治医生或护士': 3,
}

# Q11: 术前焦虑程度 (0-3, 越高焦虑越重)
q11_map = {
    'A.完全没有': 0,
    'B.轻度焦虑：情绪紧张不安，莫名担心、慌乱': 1,
    'C. 中度焦虑：持续的紧张难安，莫名担心，心慌心悸': 2,
}

# Q12: 术前沟通情况 (0-1, 越高沟通越清楚)
q12_map = {
    'B.有告知但不十分清楚': 0,
    'C.有告知且完全明白': 1,
}

# Q13: 知情同意签署 (0-2, 越高越规范签署)
q13_map = {
    'A. 没有签署': 0,
    'C. 记不清了': 1,
    'B. 有签署': 2,
}

# Q14: 了解基础治疗必要性 (0-1)
q14_map = {
    'B.否': 0,
    'A是': 1,
}

# Q15: 术后反应 (0-2, 越高中越轻)
q15_map = {
    'C.明显不适：明显疼痛、肿胀、需口服止疼药才能缓解': 0,
    'B.轻微症状：少量出血、疼痛、肿胀': 1,
    'A.无特殊不适': 2,
}

# Q16-Q19: 是否影响 (0=否, 1=是)
q16_map = {'B:否': 0, 'A:是': 1}
q17_map = {'B:否': 0, 'A:是': 1}
q18_map = {'B:否': 0, 'A:是': 1}
q19_map = {'B:否': 0, 'A:是': 1}

# Q20: 对今后选择的影响 (0-2, 越高影响越小)
q20_map = {
    'C.比较影响，即使其他部位有必要，还是需要考虑很久才能接受': 0,
    'B.轻微影响，若其他部位有必要，还是会积极接受手术方案': 1,
    'A.完全不影响': 2,
}

# 执行编码
data = {}
data['Q10'] = encode(rows, COL_Q10, q10_map)
data['Q11'] = encode(rows, COL_Q11, q11_map)
data['Q12'] = encode(rows, COL_Q12, q12_map)
data['Q13'] = encode(rows, COL_Q13, q13_map)
data['Q14'] = encode(rows, COL_Q14, q14_map)
data['Q15'] = encode(rows, COL_Q15, q15_map)
data['Q16'] = encode(rows, COL_Q16, q16_map)
data['Q17'] = encode(rows, COL_Q17, q17_map)
data['Q18'] = encode(rows, COL_Q18, q18_map)
data['Q19'] = encode(rows, COL_Q19, q19_map)
data['Q20'] = encode(rows, COL_Q20, q20_map)

print("编码完成，无未映射值")
print()

# ─── 4. 统计工具函数 ───
def mean(vals):
    return sum(vals) / len(vals)

def variance(vals, ddof=1):
    m = mean(vals)
    return sum((x - m) ** 2 for x in vals) / (len(vals) - ddof)

def covariance(x, y):
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (len(x) - 1)

def pearson_r(x, y):
    sx = math.sqrt(variance(x))
    sy = math.sqrt(variance(y))
    if sx == 0 or sy == 0:
        return 0.0
    return covariance(x, y) / (sx * sy)


def cronbach_alpha(data_dict, item_keys, label=""):
    """计算Cronbach's α系数及item-total统计"""
    k = len(item_keys)
    if k < 2:
        return None

    # 数据矩阵
    matrix = [data_dict[k_] for k_ in item_keys]
    # 每个变量的方差
    item_vars = [variance(col) for col in matrix]
    # 总分
    total = [sum(matrix[j][i] for j in range(k)) for i in range(n)]
    total_var = variance(total)

    # α
    alpha = (k / (k - 1)) * (1 - sum(item_vars) / total_var)

    # 条目间平均相关系数
    rs = []
    for i in range(k):
        for j in range(i + 1, k):
            rs.append(pearson_r(matrix[i], matrix[j]))
    mean_r = sum(rs) / len(rs)
    alpha_std = (k * mean_r) / (1 + (k - 1) * mean_r)

    # 输出标题
    print(f"{'─' * 20} {label} {'─' * (50 - len(label))}")
    print(f"  N = {n}, k = {k}")
    print(f"  Cronbach's α  = {alpha:.4f}")
    if alpha_std == alpha:
        pass
    else:
        print(f"  标准化 α     = {alpha_std:.4f}")
    print(f"  条目间平均 r = {mean_r:.4f}")
    print(f"  总分 Mean±SD = {mean(total):.2f}±{math.sqrt(total_var):.2f}")
    print()

    # 逐项删除表
    print(f"  {'Item':<6} {'编码范围':<10} {'Mean':>6} {'SD':>6} {'Item-Total r':>12} {'α if del':>9}")
    print(f"  {'─' * 55}")
    for idx, key in enumerate(item_keys):
        vals = matrix[idx]
        m = mean(vals)
        sd = math.sqrt(variance(vals))
        vmin, vmax = min(vals), max(vals)
        # item-total相关
        rest = [total[i] - vals[i] for i in range(n)]
        r_it = pearson_r(vals, rest)
        # 删除后α
        remaining = [matrix[j] for j in range(k) if j != idx]
        k2 = len(remaining)
        rv = [variance(c) for c in remaining]
        rt = [sum(remaining[j][i] for j in range(k2)) for i in range(n)]
        rtv = variance(rt)
        alpha_if = (k2 / (k2 - 1)) * (1 - sum(rv) / rtv)
        print(f"  {key:<6} {vmin}-{vmax:<8} {m:>6.2f} {sd:>6.2f} {r_it:>12.3f} {alpha_if:>9.4f}")
    print()
    return alpha


# ─── 5. 按维度计算 ───

print("=" * 70)
print("问卷条目编码信息")
print("=" * 70)
print()
print(f"  {'Q':<6} {'条目标题':<35} {'编码':<12} {'Mean±SD'}")
print(f"  {'─' * 70}")
for q_key, col_name, mapping in [
    ('Q10', '了解途径', q10_map),
    ('Q11', '焦虑程度', q11_map),
    ('Q12', '术前沟通', q12_map),
    ('Q13', '知情同意', q13_map),
    ('Q14', '基础治疗认知', q14_map),
    ('Q15', '术后反应', q15_map),
    ('Q16', '认知→决策', q16_map),
    ('Q17', '费用→决策', q17_map),
    ('Q18', '心理→决策', q18_map),
    ('Q19', '全身→决策', q19_map),
    ('Q20', '未来方案意愿', q20_map),
]:
    vals = data[q_key]
    m = mean(vals)
    sd = math.sqrt(variance(vals))
    range_str = f"{min(vals)}-{max(vals)} ({len(mapping)}级)"
    print(f"  {q_key:<6} {col_name:<35} {range_str:<12} {m:.2f}±{sd:.2f}")
print()

# ── 维度A: 认知状况（Q10, Q11, Q12, Q14, Q15）──
print("=" * 70)
cronbach_alpha(data, ['Q10', 'Q11', 'Q12', 'Q14', 'Q15'],
               "维度A: 认知状况 (Awareness)")

# ── 维度A2: 认知状况（不含Q15术后反应）──
cronbach_alpha(data, ['Q10', 'Q11', 'Q12', 'Q14'],
               "维度A2: 认知状况 (去除Q15)")

# ── 维度B: 决策影响因素（Q16, Q17, Q18, Q19, Q20）──
cronbach_alpha(data, ['Q16', 'Q17', 'Q18', 'Q19', 'Q20'],
               "维度B: 决策影响因素 (Decision Factors)")

# ── 维度C: 整体Q10-Q20 ──
cronbach_alpha(data, ['Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15',
                       'Q16', 'Q17', 'Q18', 'Q19', 'Q20'],
               "维度C: 整体问卷 Q10-Q20")

# ── 维度D: 无Q13的整体 ──
cronbach_alpha(data, ['Q10', 'Q11', 'Q12', 'Q14', 'Q15',
                       'Q16', 'Q17', 'Q18', 'Q19', 'Q20'],
               "维度D: 整体 (去除Q13)")

# ── 维度E: 术前认知（Q10, Q11, Q12）──
cronbach_alpha(data, ['Q10', 'Q11', 'Q12'],
               "维度E: 术前认知")

print("=" * 70)

import streamlit as st
import pandas as pd
import io

# 设置页面配置
st.set_page_config(page_title="市场数据自动化处理 (修复版)", layout="wide")

st.title("📊 市场数据自动化清洗与报表生成 (修复版)")
st.markdown("""
**修复说明**：已修复 2025年天猫品牌数据未应用系数的问题。

本工具生成 **12组数据** (3个平台 × 4类指标)：
1. **平台**：京东+天猫 / 京东 / 天猫
2. **指标**：
   - **销额** (Top N 销额品牌)
   - **销量** (Top N 销量品牌)
   - **Top 销额品牌销量** (Top N 销额品牌的销量趋势)
   - **ASP** (Top N 销额品牌的客单价对比)
""")

# --- 侧边栏配置 ---
st.sidebar.header("⚙️ 参数设置")

# 1. 系数设置
st.sidebar.subheader("1. 系数设置 (Coefficients)")
with st.sidebar.expander("点击展开系数设置", expanded=True):
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**2024年**")
        coeff_tm_24 = st.number_input("24年 天猫系数", value=0.82, format="%.2f")
        coeff_jd_24 = st.number_input("24年 京东系数", value=0.87, format="%.2f")
    with col_c2:
        st.markdown("**2025年**")
        coeff_tm_25 = st.number_input("25年 天猫系数", value=0.72, format="%.2f")
        coeff_jd_25 = st.number_input("25年 京东系数", value=0.87, format="%.2f")

# 2. 截取设置
st.sidebar.subheader("2. 报表输出设置")
top_n = st.sidebar.number_input("输出 Top N 品牌", value=15, min_value=5, max_value=100)

# 3. 品牌合并规则
st.sidebar.subheader("3. 品牌合并规则")
default_rules = """华为,鸿蒙:华为
paulmann p,paulmann:paulmann
明基,麦朵尔:明基"""
rules_input = st.sidebar.text_area("输入规则 (格式: 关键词1,关键词2:目标名)", value=default_rules, height=150)

# 解析合并规则
merge_rules = {}
if rules_input:
    for line in rules_input.split('\n'):
        if ':' in line:
            keys_str, target = line.split(':', 1)
            keys = [k.strip() for k in keys_str.split(',') if k.strip()]
            target = target.strip()
            for k in keys:
                merge_rules[k.lower()] = target


def clean_brand_name_dynamic(name, rules):
    if pd.isna(name): return "Unknown"
    name_str = str(name).strip()
    name_lower = name_str.lower()

    # 优先匹配规则
    for key, target in rules.items():
        if key in name_lower:
            return target
    return name_str


# --- 主页面：文件上传 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📂 2024年 数据上传")
    file_tm_ov_24 = st.file_uploader("24年 天猫大盘", type=['csv', 'xlsx'], key="tm_ov_24")
    file_tm_br_24 = st.file_uploader("24年 天猫品牌", type=['csv', 'xlsx'], key="tm_br_24")
    file_jd_ov_24 = st.file_uploader("24年 京东大盘", type=['csv', 'xlsx'], key="jd_ov_24")
    file_jd_br_24 = st.file_uploader("24年 京东品牌", type=['csv', 'xlsx'], key="jd_br_24")

with col2:
    st.subheader("📂 2025年 数据上传")
    file_tm_ov_25 = st.file_uploader("25年 天猫大盘", type=['csv', 'xlsx'], key="tm_ov_25")
    file_tm_br_25 = st.file_uploader("25年 天猫品牌", type=['csv', 'xlsx'], key="tm_br_25")
    file_jd_ov_25 = st.file_uploader("25年 京东大盘", type=['csv', 'xlsx'], key="jd_ov_25")
    file_jd_br_25 = st.file_uploader("25年 京东品牌", type=['csv', 'xlsx'], key="jd_br_25")


# --- 数据处理逻辑 ---
def load_file(uploaded_file):
    if uploaded_file is None: return None
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"读取文件失败: {uploaded_file.name}, 错误: {e}")
        return None


def process_data():
    # 1. 读取数据
    dfs = {
        'tm_ov_24': load_file(file_tm_ov_24), 'tm_br_24': load_file(file_tm_br_24),
        'jd_ov_24': load_file(file_jd_ov_24), 'jd_br_24': load_file(file_jd_br_24),
        'tm_ov_25': load_file(file_tm_ov_25), 'tm_br_25': load_file(file_tm_br_25),
        'jd_ov_25': load_file(file_jd_ov_25), 'jd_br_25': load_file(file_jd_br_25)
    }

    # 检查文件是否齐全
    if any(v is None for v in dfs.values()):
        st.warning("请上传所有8个文件以开始处理。")
        return None

    # 2. 清洗日期和品牌
    def clean_common(df):
        # 日期
        date_col = None
        for col in df.columns:
            if '月份' in col or 'date' in col.lower():
                date_col = col
                break
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df['date_norm'] = df[date_col].dt.to_period('M').dt.to_timestamp()
        else:
            st.error("无法在文件中找到日期/月份列")
            return df

        # 品牌
        if '品牌名称' in df.columns:
            df['品牌名称'] = df['品牌名称'].apply(lambda x: clean_brand_name_dynamic(x, merge_rules))
        return df

    for k, df in dfs.items():
        dfs[k] = clean_common(df)

    # 3. 应用系数 (严格检查每一行)
    # 24天猫
    dfs['tm_ov_24']['销售额'] *= coeff_tm_24;
    dfs['tm_ov_24']['销量'] *= coeff_tm_24
    dfs['tm_br_24']['销售额'] *= coeff_tm_24;
    dfs['tm_br_24']['销量'] *= coeff_tm_24
    # 25天猫 (这里修复了之前漏掉的 tm_br_25)
    dfs['tm_ov_25']['销售额'] *= coeff_tm_25;
    dfs['tm_ov_25']['销量'] *= coeff_tm_25
    dfs['tm_br_25']['销售额'] *= coeff_tm_25;
    dfs['tm_br_25']['销量'] *= coeff_tm_25
    # 24京东
    dfs['jd_ov_24']['销售额'] *= coeff_jd_24;
    dfs['jd_ov_24']['销量'] *= coeff_jd_24
    dfs['jd_br_24']['销售额'] *= coeff_jd_24;
    dfs['jd_br_24']['销量'] *= coeff_jd_24
    # 25京东
    dfs['jd_ov_25']['销售额'] *= coeff_jd_25;
    dfs['jd_ov_25']['销量'] *= coeff_jd_25
    dfs['jd_br_25']['销售额'] *= coeff_jd_25;
    dfs['jd_br_25']['销量'] *= coeff_jd_25

    # 4. 合并大盘与品牌数据
    def concat_years(df24, df25, is_brand=False):
        cols = ['date_norm', '销售额', '销量']
        if is_brand: cols.append('品牌名称')

        d1 = df24[cols].copy()
        d2 = df25[cols].copy()

        combined = pd.concat([d1, d2])
        if is_brand:
            combined = combined.groupby(['品牌名称', 'date_norm'])[['销售额', '销量']].sum().reset_index()
        return combined

    jd_ov = concat_years(dfs['jd_ov_24'], dfs['jd_ov_25'])
    tm_ov = concat_years(dfs['tm_ov_24'], dfs['tm_ov_25'])
    jd_br = concat_years(dfs['jd_br_24'], dfs['jd_br_25'], is_brand=True)
    tm_br = concat_years(dfs['tm_br_24'], dfs['tm_br_25'], is_brand=True)

    # 5. 合并平台
    total_ov = pd.merge(jd_ov, tm_ov, on='date_norm', how='outer', suffixes=('_jd', '_tm')).fillna(0)
    total_ov['销售额'] = total_ov['销售额_jd'] + total_ov['销售额_tm']
    total_ov['销量'] = total_ov['销量_jd'] + total_ov['销量_tm']

    total_br = pd.merge(jd_br, tm_br, on=['品牌名称', 'date_norm'], how='outer', suffixes=('_jd', '_tm')).fillna(0)
    total_br['销售额'] = total_br['销售额_jd'] + total_br['销售额_tm']
    total_br['销量'] = total_br['销量_jd'] + total_br['销量_tm']

    # --- 核心处理逻辑 ---
    def process_platform_metrics(df_ov, df_br, ov_name, top_n):
        results = {}

        # 1. 全量透视表
        def get_pivot(val_col):
            df_ov_temp = df_ov.copy()
            df_ov_temp['品牌名称'] = ov_name
            combined = pd.concat(
                [df_ov_temp[['品牌名称', 'date_norm', val_col]], df_br[['品牌名称', 'date_norm', val_col]]])
            pivot = combined.pivot_table(index='品牌名称', columns='date_norm', values=val_col, aggfunc='sum').fillna(0)

            # 计算合计
            cols_24 = [c for c in pivot.columns if c.year == 2024]
            cols_25 = [c for c in pivot.columns if c.year == 2025]
            pivot['24年合计'] = pivot[cols_24].sum(axis=1)
            pivot['25年合计'] = pivot[cols_25].sum(axis=1)
            pivot['累计同比'] = pivot.apply(
                lambda r: (r['25年合计'] - r['24年合计']) / r['24年合计'] if r['24年合计'] != 0 else 0, axis=1)
            return pivot

        pivot_sales = get_pivot('销售额')
        pivot_vol = get_pivot('销量')

        # 2. 获取 Top N 名单 (排除大盘)
        top_sales_brands = pivot_sales.drop(index=ov_name).sort_values('25年合计', ascending=False).head(
            top_n).index.tolist()
        top_vol_brands = pivot_vol.drop(index=ov_name).sort_values('25年合计', ascending=False).head(
            top_n).index.tolist()

        # 3. 筛选与排序函数
        def filter_sort_add_ov(pivot, brand_list):
            row_ov = pivot.loc[[ov_name]] if ov_name in pivot.index else pd.DataFrame()
            valid_brands = [b for b in brand_list if b in pivot.index]
            row_brands = pivot.loc[valid_brands]
            return pd.concat([row_ov, row_brands])

        # 生成表1: 销额 (按销额排名)
        results['Sales_TopSales'] = filter_sort_add_ov(pivot_sales, top_sales_brands)

        # 生成表2: 销量 (按销量排名)
        results['Vol_TopVol'] = filter_sort_add_ov(pivot_vol, top_vol_brands)

        # 生成表3: Top 销额品牌销量 (按销额排名，看销量)
        results['Vol_TopSales'] = filter_sort_add_ov(pivot_vol, top_sales_brands)

        # 生成表4: ASP (按销额排名)
        target_brands = [ov_name] + top_sales_brands
        target_brands = [b for b in target_brands if b in pivot_sales.index]

        s_data = pivot_sales.loc[target_brands, ['24年合计', '25年合计']]
        v_data = pivot_vol.loc[target_brands, ['24年合计', '25年合计']]

        asp_df = pd.DataFrame(index=target_brands)
        asp_df['24年销额'] = s_data['24年合计']
        asp_df['24年销量'] = v_data['24年合计']
        asp_df['24年均价'] = asp_df['24年销额'] / asp_df['24年销量'].replace(0, 1)

        asp_df['25年销额'] = s_data['25年合计']
        asp_df['25年销量'] = v_data['25年合计']
        asp_df['25年均价'] = asp_df['25年销额'] / asp_df['25年销量'].replace(0, 1)

        asp_df['均价同比'] = (asp_df['25年均价'] - asp_df['24年均价']) / asp_df['24年均价'].replace(0, 1)
        results['ASP'] = asp_df

        return results

    # 分别处理 3 个平台
    res_total = process_platform_metrics(total_ov, total_br, '京东+天猫', top_n)
    res_jd = process_platform_metrics(jd_ov, jd_br, '京东', top_n)
    res_tm = process_platform_metrics(tm_ov, tm_br, '天猫', top_n)

    # 汇总输出
    final_output = {
        '京东+天猫_销额': res_total['Sales_TopSales'],
        '京东+天猫_销量': res_total['Vol_TopVol'],
        '京东+天猫_Top销额品牌销量': res_total['Vol_TopSales'],
        '京东+天猫_ASP': res_total['ASP'],

        '京东_销额': res_jd['Sales_TopSales'],
        '京东_销量': res_jd['Vol_TopVol'],
        '京东_Top销额品牌销量': res_jd['Vol_TopSales'],
        '京东_ASP': res_jd['ASP'],

        '天猫_销额': res_tm['Sales_TopSales'],
        '天猫_销量': res_tm['Vol_TopVol'],
        '天猫_Top销额品牌销量': res_tm['Vol_TopSales'],
        '天猫_ASP': res_tm['ASP']
    }

    return final_output


# --- 按钮与输出 ---
if st.button("🚀 开始处理并生成报表"):
    with st.spinner("正在处理数据..."):
        report_data = process_data()

    if report_data:
        st.success("处理完成！数据已校验。")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            date_cols = pd.date_range('2024-01-01', '2025-12-01', freq='MS')
            str_cols = [d.strftime('%Y-%m-%d') for d in date_cols]

            for sheet_name, df in report_data.items():
                if 'ASP' in sheet_name:
                    df.to_excel(writer, sheet_name=sheet_name)
                else:
                    df.columns = [c.strftime('%Y-%m-%d') if isinstance(c, pd.Timestamp) else c for c in df.columns]
                    for c in str_cols:
                        if c not in df.columns: df[c] = 0

                    final_cols = str_cols + ['24年合计', '25年合计', '累计同比']
                    final_cols = [c for c in final_cols if c in df.columns]

                    df_final = df[final_cols].reset_index()
                    df_final.to_excel(writer, sheet_name=sheet_name, index=False)

        output.seek(0)
        st.download_button("📥 下载 Excel 表格 (12组数据)", data=output, file_name="Final_Market_12Groups_Fixed.xlsx")

        st.subheader("数据预览 (京东+天猫 销额)")
        st.dataframe(report_data['京东+天猫_销额'].style.format("{:,.0f}"))
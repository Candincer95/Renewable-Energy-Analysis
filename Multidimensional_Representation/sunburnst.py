import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# 1. PAGE SETUP & CSS
# ---------------------------------------------------------
st.set_page_config(page_title="Renewable Energy Sunburst", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B, #FCA048);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .insight-section {
        background: #111118;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        color: #EEE;
    }
    .highlight { color: #FCA048; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATA PREPARATION (STATISTICALLY CORRECT & ASYMMETRIC)
# ---------------------------------------------------------
@st.cache_data
def load_hierarchical_data(file_path='renewable_capacity_timeseries.csv'):
    df = pd.read_csv(file_path)
    latest_date = df['day'].max()
    df_latest = df[df['day'] == latest_date].drop('day', axis=1)
    melted = df_latest.melt(var_name='col', value_name='capacity')
    
    exclude = ['_wind_capacity', 'GB-UKM_']
    df_leaf = melted[~melted['col'].str.contains('|'.join(exclude))].copy()
    df_leaf = df_leaf[df_leaf['capacity'] > 0]

    data = []
    for _, row in df_leaf.iterrows():
        col, cap = row['col'], row['capacity']
        prefix = col.split('_')[0]
        energy_raw = col.replace(prefix + '_', '').replace('_capacity', '')
        
        country_map = {'CH': 'Switzerland', 'DE': 'Germany', 'DK': 'Denmark', 'FR': 'France', 'SE': 'Sweden'}
        
        is_uk = prefix.startswith('GB-')
        if is_uk:
            country = 'United Kingdom'
            region = 'Great Britain' if prefix == 'GB-GBN' else 'Northern Ireland'
        else:
            country = country_map.get(prefix, prefix)
            region = None 
            
        if 'wind' in energy_raw:
            category = 'Wind'
            detail = 'Onshore' if 'onshore' in energy_raw else 'Offshore'
        else:
            category = energy_raw.capitalize()
            detail = None 
            
        l1 = country
        if is_uk:
            l2 = region
            l3 = category
            l4 = detail
        else:
            l2 = category
            l3 = detail
            l4 = None
            
        data.append({
            'Country': country, 'Region': region, 'Energy': category, 'Detail': detail,
            'Level1': l1, 'Level2': l2, 'Level3': l3, 'Level4': l4, 'Capacity': cap
        })
        
    return pd.DataFrame(data), latest_date

df_sb, l_date = load_hierarchical_data()

# ---------------------------------------------------------
# 3. HEADER & INTERACTIVE FILTERS
# ---------------------------------------------------------
st.markdown('<div class="main-title">🌀 Hierarchical Breakdown: Regional & Technical Depth</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">Drill-down analysis of the UK regions and wind energy specifications · Data: {l_date}</div>', unsafe_allow_html=True)

selected_countries = st.multiselect("🌐 Select Countries to Analyze:", df_sb['Country'].dropna().unique(), default=df_sb['Country'].dropna().unique())

filtered_df = df_sb[df_sb['Country'].isin(selected_countries)]

if filtered_df.empty:
    st.warning("⚠️ Please select at least one country.")
    st.stop()

# ---------------------------------------------------------
# 4. SUNBURST CHART GENERATION WITH CUSTOM LEGEND
# ---------------------------------------------------------
ENERGY_COLORS = {
    'Wind': '#E63946', 'Solar': '#FFD166', 'Hydro': '#4FC3F7', 
    'Bioenergy': '#52B788', 'Geothermal': '#F4845F', 'Marine': '#7B5EA7'
}

fig = px.sunburst(
    filtered_df,
    path=['Level1', 'Level2', 'Level3', 'Level4'],
    values='Capacity',
    color='Energy',
    color_discrete_map={**ENERGY_COLORS, '(?)': '#222233'}
)

fig.update_traces(
    textinfo="label+percent entry", 
    insidetextorientation='radial',
    marker=dict(line=dict(color='#111118', width=1.5)), 
    hovertemplate="<b>%{label}</b><br>Capacity: %{value:,.0f} MW<br>Share of Parent: %{percentParent:.1%}<extra></extra>"
)

# LEGEND HACK: Plotly'i kandırmak için görünmez işaretçiler ekliyoruz
for energy_name, hex_color in ENERGY_COLORS.items():
    if energy_name in filtered_df['Energy'].unique(): # Sadece veri setinde olanları göster
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=hex_color, symbol='square'),
            name=energy_name,
            showlegend=True
        ))

fig.update_layout(
    height=750,
    margin=dict(t=20, l=0, r=0, b=20),
    font=dict(family="Inter, Arial Black", size=14),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=True, # Legend'i zorla açtık
    legend=dict(
        title="<b>Energy Sources</b>",
        font=dict(size=14, color="white"),
        bgcolor="rgba(0,0,0,0)",
        yanchor="middle", y=0.5,
        xanchor="left", x=1.05
    )
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 5. STRATEGIC ANALYSIS & INSIGHTS
# ---------------------------------------------------------
wind_data = filtered_df[filtered_df['Energy'] == 'Wind']
total_wind = wind_data['Capacity'].sum()

if total_wind > 0:
    offshore_wind = wind_data[wind_data['Detail'] == 'Offshore']['Capacity'].sum()
    offshore_share = (offshore_wind / total_wind) * 100
    offshore_by_country = wind_data[wind_data['Detail'] == 'Offshore'].groupby('Country')['Capacity'].sum().sort_values(ascending=False)
    top_offshore_countries = offshore_by_country[offshore_by_country > 0].head(2).index.tolist()
    leaders_str = " and ".join(top_offshore_countries) if top_offshore_countries else "selected countries"
else:
    offshore_share = 0
    leaders_str = "N/A"

uk_insight = ""
if 'United Kingdom' in selected_countries:
    uk_data = filtered_df[filtered_df['Country'] == 'United Kingdom']
    uk_total = uk_data['Capacity'].sum()
    nir_share = (uk_data[uk_data['Region'] == 'Northern Ireland']['Capacity'].sum() / uk_total) * 100 if uk_total > 0 else 0
    uk_insight = f"* **United Kingdom's Internal Dynamics:** The UK is not a monolithic block. The regional split demonstrates that **Northern Ireland** contributes <span class='highlight'>{nir_share:.1f}%</span> to the UK's overall renewable portfolio."
else:
    country_wind_focus = (filtered_df[filtered_df['Energy']=='Wind'].groupby('Country')['Capacity'].sum() / filtered_df.groupby('Country')['Capacity'].sum()) * 100
    top_wind_country = country_wind_focus.idxmax() if not country_wind_focus.empty else "N/A"
    top_wind_val = country_wind_focus.max() if not country_wind_focus.empty else 0
    uk_insight = f"* **National Wind Focus:** With the UK excluded, **{top_wind_country}** stands out with the highest proportional reliance on wind energy, allocating <span class='highlight'>{top_wind_val:.1f}%</span> of its portfolio to this source."

st.markdown('<div class="insight-section">', unsafe_allow_html=True)
st.markdown("### 📊 Structural Insights & Deductions")
st.markdown(f"""
This Sunburst chart provides a deep dive into the anatomical structure of the renewable energy capacity. By breaking down the top-level country data, we reveal the sub-regional distributions and technical specifications that are invisible in a standard area or bar chart.

* **The Wind Technology Split:** While Wind energy is a massive aggregate figure, drilling down reveals that **Offshore Wind** accounts for <span class="highlight">{offshore_share:.1f}%</span> of the total wind capacity in the selected scope. This highlights a strategic shift towards marine environments, currently driven by **{leaders_str}**.
{uk_insight}
* **Asymmetrical Specialization:** Notice how certain branches end early (e.g., Solar), while others branch out deeply (e.g., Wind -> Offshore/Onshore). This proves that the complexity of the grid is highly dependent on the energy source type.
* **How to interact:** Click on any country to zoom in and isolate its national portfolio. Click in the center circle to zoom back out.
""")
st.markdown('</div>', unsafe_allow_html=True)
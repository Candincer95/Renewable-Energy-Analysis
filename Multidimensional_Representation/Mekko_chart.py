import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Renewable Energy Mekko Chart",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00C9A7, #4FC3F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: #1E1E2E; border: 1px solid #333; border-radius: 12px;
        padding: 1rem 1.2rem; text-align: center;
    }
    .metric-val { font-size: 1.4rem; font-weight: 700; color: #4FC3F7; }
    .metric-label { font-size: 0.75rem; color: #888; margin-top: 0.2rem; }
    .info-box {
        background: #1A1A2E; border-left: 3px solid #00C9A7;
        border-radius: 6px; padding: 0.7rem 1rem; font-size: 0.82rem;
        color: #bbb; margin-bottom: 1rem;
    }
    .insight-section {
        background: #111118;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    latest_date = df['day'].max()
    df_latest = df[df['day'] == latest_date].drop('day', axis=1)
    df_melted = df_latest.melt(var_name='Col_Name', value_name='Capacity_MW')

    def parse_column(col_name):
        if col_name.startswith('GB-'):
            return col_name[:6], col_name[7:].replace('_capacity', '')
        return col_name[:2], col_name[3:].replace('_capacity', '')

    df_melted[['Country_Code', 'Energy_Type']] = df_melted['Col_Name'].apply(lambda x: pd.Series(parse_column(x)))
    
    df_clean = df_melted[~df_melted['Country_Code'].isin(['GB-GBN', 'GB-NIR'])].copy()

    MAIN_NON_WIND = ['bioenergy', 'geothermal', 'hydro', 'marine', 'solar']
    records = []

    for country in df_clean['Country_Code'].unique():
        df_c = df_clean[df_clean['Country_Code'] == country]
        for energy in MAIN_NON_WIND:
            val = df_c[df_c['Energy_Type'] == energy]['Capacity_MW'].sum()
            if val > 0:
                records.append({'Country_Code': country, 'Energy_Type': energy.capitalize(), 'Capacity_MW': val})

        wind_total = df_c[df_c['Energy_Type'] == 'wind']['Capacity_MW'].sum()
        if wind_total > 0:
            records.append({'Country_Code': country, 'Energy_Type': 'Wind', 'Capacity_MW': wind_total})
        else:
            sub_wind = df_c[df_c['Energy_Type'].isin(['wind_onshore', 'wind_offshore'])]['Capacity_MW'].sum()
            if sub_wind > 0:
                records.append({'Country_Code': country, 'Energy_Type': 'Wind', 'Capacity_MW': sub_wind})

    df_final = pd.DataFrame(records)
    country_map = {'CH': 'Switzerland', 'DE': 'Germany', 'DK': 'Denmark', 'FR': 'France', 'SE': 'Sweden', 'GB-UKM': 'UK'}
    short_map = {'CH': 'CH', 'DE': 'DE', 'DK': 'DK', 'FR': 'FR', 'SE': 'SE', 'GB-UKM': 'GB'}
    df_final['Country'] = df_final['Country_Code'].map(country_map)
    df_final['Country_Short'] = df_final['Country_Code'].map(short_map)
    return df_final.dropna(subset=['Country']), latest_date

df, latest_date = load_and_clean_data('renewable_capacity_timeseries.csv')


st.markdown('<div class="main-title">🌍 Renewable Energy — Mekko Chart</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">Multidimensional capacity analysis · Data: {latest_date}</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([3, 1.5, 1.5])
with col_a:
    secilen_ulkeler = st.multiselect("🌐 Countries", sorted(df['Country'].unique()), default=df['Country'].unique())
with col_b:
    secilen_enerjiler = st.multiselect("⚡ Energy Sources", sorted(df['Energy_Type'].unique()), default=df['Energy_Type'].unique())
with col_c:
    goster_mod = st.radio("📊 Label Mode", ["Percentage (%)", "Capacity (MW)"])

df_filtered = df[df['Country'].isin(secilen_ulkeler) & df['Energy_Type'].isin(secilen_enerjiler)]

if df_filtered.empty:
    st.warning("⚠️ Please select at least one country and energy type."); st.stop()


total_mw = df_filtered["Capacity_MW"].sum()
top_country = df_filtered.groupby('Country')['Capacity_MW'].sum().idxmax()
top_energy = df_filtered.groupby('Energy_Type')['Capacity_MW'].sum().idxmax()

m1, m2, m3, m4, m5 = st.columns(5)
metrics = [
    (f"{total_mw/1000:,.1f} GW", "Total Capacity"),
    (str(len(secilen_ulkeler)), "Countries"),
    (str(len(secilen_enerjiler)), "Energy Sources"),
    (top_country, "Market Leader"),
    (top_energy, "Dominant Source"),
]
for col, (val, label) in zip([m1, m2, m3, m4, m5], metrics):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


country_totals = df_filtered.groupby('Country')['Capacity_MW'].sum().sort_values(ascending=False)
raw_widths = (country_totals.values / country_totals.sum()) * 100
widths_pct = np.maximum(raw_widths, 5.0)
widths_pct = (widths_pct / widths_pct.sum()) * 100
left_edges = np.append([0], np.cumsum(widths_pct)[:-1])
centers = left_edges + widths_pct / 2

ENERGY_COLORS = {'Bioenergy': '#52B788', 'Geothermal': '#F4845F', 'Hydro': '#4FC3F7', 'Marine': '#7B5EA7', 'Solar': '#FFD166', 'Wind': '#E63946'}

fig = go.Figure()
for enerji in sorted(df_filtered['Energy_Type'].unique()):
    y_vals, labels, hovers = [], [], []
    for ulke in country_totals.index:
        val = df_filtered[(df_filtered['Country'] == ulke) & (df_filtered['Energy_Type'] == enerji)]['Capacity_MW'].sum()
        pct = (val / country_totals[ulke]) * 100 if country_totals[ulke] > 0 else 0
        y_vals.append(pct)
        if pct > 5:
            labels.append(f"{val/1000:.1f}GW" if goster_mod == "Capacity (MW)" and val >= 1000 else f"{val:.0f}MW" if goster_mod == "Capacity (MW)" else f"{pct:.1f}%")
        else: labels.append("")
        hovers.append(f"<b>{ulke}</b><br>{enerji}: {val:,.0f} MW ({pct:.1f}%)")

    fig.add_trace(go.Bar(
        name=enerji, x=left_edges, y=y_vals, width=widths_pct, offset=0,
        marker=dict(color=ENERGY_COLORS.get(enerji, '#AAA'), line=dict(color='#0D0D0D', width=1.5)),
        text=labels, textposition='inside', insidetextanchor='middle',
        hovertext=hovers, hoverinfo='text'
    ))

fig.update_layout(
    barmode='stack', bargap=0, height=650, margin=dict(l=60, r=40, t=50, b=100),
    plot_bgcolor='#111118', paper_bgcolor='#111118', font=dict(color='#EEE'),
    xaxis=dict(
        type='linear', range=[0, 100], tickmode='array', tickvals=list(centers),
        ticktext=[f"<b>{df[df['Country']==c]['Country_Short'].iloc[0]}</b><br>{(w/widths_pct.sum()*100):.1f}%" for c, w in zip(country_totals.index, raw_widths)],
        showgrid=False, zeroline=False
    ),
    yaxis=dict(range=[0, 100], gridcolor='#333')
)

st.plotly_chart(fig, use_container_width=True)


st.markdown('<div class="insight-section">', unsafe_allow_html=True)
st.markdown("### 📊 Strategic Analysis & Key Insights")

# Dinamik İçgörü Hesaplama
# En uzmanlaşmış ülke (kendi portföyünde tek bir enerji tipi en yüksek olan)
portfolio_max = df_filtered.copy()
portfolio_max['Share'] = portfolio_max.apply(lambda x: (x['Capacity_MW'] / country_totals[x['Country']]) * 100, axis=1)
specialist = portfolio_max.loc[portfolio_max['Share'].idxmax()]

st.markdown(f"""
This Mekko Chart provides a multi-dimensional perspective on the European renewable energy landscape as of **{latest_date}**.

- **Market Dominance (Column Width):** The width of each column represents a country's total renewable capacity relative to the selected group. Currently, **{top_country}** holds the largest market share.
- **Energy Mix (Segment Height):** The vertical segments illustrate national strategic priorities. For instance, **{specialist['Country']}** shows the highest level of specialization with **{specialist['Energy_Type']}** making up **{specialist['Share']:.1f}%** of its portfolio.
- **Portfolio Diversity:** Wider columns typically represent established energy hubs like **Germany** or the **UK**, while narrower columns highlight smaller but often more specialized markets like **Denmark** or **Switzerland**.
- **Dominant Resource:** Across the selected region, **{top_energy}** remains the primary driver of the green transition, accounting for the largest total installed capacity.
""")
st.markdown('</div>', unsafe_allow_html=True)


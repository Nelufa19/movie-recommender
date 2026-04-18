import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse
import sklearn.metrics as skm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG — Amazon Prime theme
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# AMAZON PRIME CSS THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Amazon Ember', Arial, sans-serif;
    background-color: #0F171E;
    color: #FFFFFF;
}
.stApp { background-color: #0F171E; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #1A242F;
    border-right: 1px solid #00A8E1;
}
section[data-testid="stSidebar"] * { color: #FFFFFF !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1A242F, #0F171E);
    border: 1px solid #00A8E1;
    border-radius: 10px;
    padding: 16px;
}
[data-testid="metric-container"] label { color: #00A8E1 !important; font-size: 13px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important; font-size: 28px; font-weight: 700;
}

/* ── Headers ── */
h1 { color: #00A8E1 !important; font-size: 2rem !important; }
h2 { color: #00A8E1 !important; }
h3 { color: #FFFFFF !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background-color: #1A242F; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #AAAAAA !important; }
.stTabs [aria-selected="true"] { color: #00A8E1 !important; border-bottom: 2px solid #00A8E1; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00A8E1, #0073A8);
    color: #FFFFFF; border: none; border-radius: 6px;
    font-weight: 600; padding: 10px 24px;
    transition: all 0.2s;
}
.stButton > button:hover { background: linear-gradient(135deg, #0073A8, #005580); }

/* ── Selectbox / Slider ── */
.stSelectbox > div > div, .stNumberInput > div > div {
    background-color: #1A242F !important; color: #FFFFFF !important;
    border: 1px solid #00A8E1 !important; border-radius: 6px;
}
.stSlider > div { color: #00A8E1; }

/* ── DataFrames ── */
.stDataFrame { border: 1px solid #00A8E1; border-radius: 8px; }

/* ── Divider ── */
hr { border-color: #00A8E1; }

/* ── Success / Info boxes ── */
.stSuccess { background-color: #1A3A2A; border-left: 4px solid #00C853; }
.stInfo    { background-color: #1A2A3A; border-left: 4px solid #00A8E1; }

/* ── Card style ── */
.prime-card {
    background: #1A242F;
    border: 1px solid #00A8E1;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.prime-title {
    font-size: 1.5rem; font-weight: 700;
    color: #00A8E1; margin-bottom: 4px;
}
.prime-sub { color: #AAAAAA; font-size: 0.85rem; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0F171E 0%, #1A3A5C 50%, #00A8E1 100%);
    border-radius: 12px; padding: 32px 40px; margin-bottom: 24px;
    border: 1px solid #00A8E1;
}
.hero-banner h1 { font-size: 2.4rem !important; margin: 0; }
.hero-banner p  { color: #CCCCCC; margin: 8px 0 0; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME DEFAULTS
# ─────────────────────────────────────────────
PRIME_BLUE   = "#00A8E1"
PRIME_DARK   = "#0F171E"
PRIME_CARD   = "#1A242F"
PRIME_ACCENT = "#FF9900"   # Amazon orange accent

CHART_LAYOUT = dict(
    paper_bgcolor=PRIME_CARD,
    plot_bgcolor=PRIME_CARD,
    font=dict(color="#FFFFFF", family="Arial"),
    title_font=dict(color=PRIME_BLUE, size=16),
    xaxis=dict(gridcolor="#2A3A4A", color="#AAAAAA"),
    yaxis=dict(gridcolor="#2A3A4A", color="#AAAAAA"),
    legend=dict(bgcolor=PRIME_DARK, bordercolor=PRIME_BLUE, borderwidth=1),
    margin=dict(t=50, b=40, l=40, r=20),
)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(movies_path, ratings_path, users_path):
    moviesData  = pd.read_csv(movies_path,  converters={'Year-Of-Release': lambda x: pd.to_numeric(x, errors='coerce')}).dropna()
    moviesData['Year-Of-Release'] = moviesData['Year-Of-Release'].astype(int)
    ratingsData = pd.read_csv(ratings_path).dropna()
    ratingsData['Movie-Rating'] = ratingsData['Movie-Rating'].astype(int)
    userData    = pd.read_csv(users_path).dropna()
    return moviesData, ratingsData, userData

@st.cache_data(show_spinner=False)
def build_combined(_moviesData, _ratingsData, _userData):
    combined = pd.merge(_ratingsData, _moviesData, on='Movie-ID')
    combined = pd.merge(combined, _userData, on='User-ID')
    combined = combined.drop('Year-Of-Release', axis=1)
    combined = combined[['User-ID', 'Movie-Rating', 'Movie-ID', 'Movie-Title', 'Num-Ratings', 'Avg-Rating']]
    MovieID_dict = {mid: i for i, mid in enumerate(combined['Movie-ID'].unique())}
    combined['Movie-ID-Key'] = combined['Movie-ID'].map(MovieID_dict)
    return combined, MovieID_dict

@st.cache_resource(show_spinner=False)
def build_matrix(_combined):
    return scipy.sparse.coo_matrix((
        _combined['Movie-Rating'],
        (_combined['User-ID'], _combined['Movie-ID-Key'])
    ))

# ─────────────────────────────────────────────
# RECOMMENDER FUNCTION
# ─────────────────────────────────────────────
def knowledge_based_recommender(combined_df, moviesData, user_id, matrix, n=10):
    try:
        sims = skm.pairwise.cosine_similarity(
            matrix.getrow(user_id), matrix
        ).ravel()
        sims[user_id] = 0
        top_users   = sims.argsort()[::-1][:5]
        movie_ids   = combined_df.loc[combined_df['User-ID'].isin(top_users)]['Movie-ID']
        recs        = moviesData[moviesData['Movie-ID'].isin(movie_ids)].drop_duplicates('Movie-ID').head(n)
        return recs
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0'>
        <span style='font-size:2.5rem'>🎬</span>
        <div style='font-size:1.2rem; font-weight:700; color:#00A8E1'>Movie Recommender</div>
        <div style='font-size:0.75rem; color:#888'>Netflix Prize Dataset</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📂 Data Paths")
    movies_path  = st.text_input("Movies.csv",  value="Movies.csv")
    ratings_path = st.text_input("Ratings.csv", value="Ratings.csv")
    users_path   = st.text_input("Users.csv",   value="Users.csv")

    load_btn = st.button("🚀 Load Data", use_container_width=True)

    st.divider()
    st.markdown("### ⚙️ Recommender Settings")
    n_recs    = st.slider("Number of Recommendations", 5, 20, 10)
    min_year  = st.slider("Min Release Year", 1950, 2005, 1990)
    min_rating = st.slider("Min Avg Rating", 1.0, 5.0, 3.0, 0.5)

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#666; text-align:center'>
    Built with ❤️ using Streamlit<br>
    Amazon Prime Theme
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <h1>🎬 Movie Recommender Dashboard</h1>
    <p>Knowledge-Based · Reinforcement Learning · Data Analytics · Netflix Prize Dataset</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if load_btn or st.session_state.data_loaded:
    try:
        with st.spinner("Loading data..."):
            moviesData, ratingsData, userData = load_data(movies_path, ratings_path, users_path)
            combined_df, MovieID_dict         = build_combined(moviesData, ratingsData, userData)
            matrix                            = build_matrix(combined_df)
        st.session_state.data_loaded = True
        st.success(f"✅  Data loaded — {len(moviesData):,} movies · {len(ratingsData):,} ratings · {len(userData):,} users")
    except FileNotFoundError as e:
        st.error(f"❌  File not found: {e}\n\nUpdate the paths in the sidebar.")
        st.stop()

    # ── TABS ─────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔍 Recommender",
        "📈 Analytics",
        "🔗 Correlations",
        "🗂️ Data Explorer"
    ])

    # ═══════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ═══════════════════════════════════════════
    with tab1:
        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🎬 Total Movies",   f"{len(moviesData):,}")
        c2.metric("⭐ Total Ratings",  f"{len(ratingsData):,}")
        c3.metric("👥 Total Users",    f"{len(userData):,}")
        c4.metric("📅 Year Range",     f"{int(moviesData['Year-Of-Release'].min())}–{int(moviesData['Year-Of-Release'].max())}")
        c5.metric("⭐ Avg Rating",      f"{ratingsData['Movie-Rating'].mean():.2f} / 5")

        st.divider()

        col_a, col_b = st.columns(2)

        # Rating distribution bar chart
        with col_a:
            rating_counts = ratingsData['Movie-Rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Star Rating', 'y': 'Number of Ratings'},
                title="⭐ Rating Distribution",
                color=rating_counts.values,
                color_continuous_scale=[[0, "#1A242F"], [0.5, PRIME_BLUE], [1, PRIME_ACCENT]],
            )
            fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
            fig.update_traces(marker_line_color=PRIME_BLUE, marker_line_width=1)
            st.plotly_chart(fig, use_container_width=True)

        # Movies per decade
        with col_b:
            moviesData['Decade'] = (moviesData['Year-Of-Release'] // 10 * 10).astype(str) + 's'
            decade_counts = moviesData['Decade'].value_counts().sort_index()
            fig2 = px.bar(
                x=decade_counts.index,
                y=decade_counts.values,
                labels={'x': 'Decade', 'y': 'Number of Movies'},
                title="📅 Movies by Decade",
                color=decade_counts.values,
                color_continuous_scale=[[0, "#1A242F"], [0.5, PRIME_BLUE], [1, PRIME_ACCENT]],
            )
            fig2.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)

        # Rating pie chart
        with col_c:
            fig3 = px.pie(
                values=rating_counts.values,
                names=[f"{r} Star{'s' if r>1 else ''}" for r in rating_counts.index],
                title="⭐ Rating Share",
                color_discrete_sequence=[PRIME_BLUE, PRIME_ACCENT, "#00C853", "#FF5252", "#AA00FF"],
                hole=0.4,
            )
            fig3.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

        # User activity distribution
        with col_d:
            fig4 = px.histogram(
                userData,
                x='Num-Ratings',
                nbins=50,
                title="👥 User Activity Distribution (# Ratings per User)",
                labels={'Num-Ratings': 'Number of Ratings', 'count': 'Users'},
                color_discrete_sequence=[PRIME_BLUE],
            )
            fig4.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig4, use_container_width=True)

        # Top 15 most rated movies
        top_movies = (
            ratingsData.groupby('Movie-ID')
            .agg(total_ratings=('Movie-Rating', 'count'), avg_rating=('Movie-Rating', 'mean'))
            .reset_index()
            .merge(moviesData[['Movie-ID', 'Movie-Title']], on='Movie-ID')
            .sort_values('total_ratings', ascending=False)
            .head(15)
        )
        fig5 = px.bar(
            top_movies,
            x='total_ratings',
            y='Movie-Title',
            orientation='h',
            title="🏆 Top 15 Most Rated Movies",
            labels={'total_ratings': 'Total Ratings', 'Movie-Title': ''},
            color='avg_rating',
            color_continuous_scale=[[0, "#1A242F"], [0.5, PRIME_BLUE], [1, PRIME_ACCENT]],
            color_continuous_midpoint=3,
        )
        fig5.update_layout(**CHART_LAYOUT, height=450,
                           coloraxis_colorbar=dict(title="Avg ⭐", tickfont=dict(color='white')))
        fig5.update_yaxes(autorange='reversed')
        st.plotly_chart(fig5, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 2 — RECOMMENDER
    # ═══════════════════════════════════════════
    with tab2:
        st.markdown("### 🔍 Knowledge-Based Movie Recommender")
        st.markdown("Uses **cosine similarity** between users to find the 5 most similar users, then recommends their top-rated movies.")

        col_r1, col_r2 = st.columns([2, 1])
        with col_r1:
            valid_users = combined_df['User-ID'].unique().tolist()
            default_user = int(combined_df['User-ID'].value_counts().index[0])
            user_id_input = st.number_input(
                "Enter User ID", min_value=int(min(valid_users)),
                max_value=int(max(valid_users)), value=default_user, step=1
            )
        with col_r2:
            st.markdown("<br>", unsafe_allow_html=True)
            recommend_btn = st.button("🎬 Get Recommendations", use_container_width=True)

        if recommend_btn:
            with st.spinner("Finding similar users and recommending movies..."):
                recs = knowledge_based_recommender(
                    combined_df, moviesData, int(user_id_input), matrix, n=n_recs
                )

            if recs.empty:
                st.warning("No recommendations found for this user. Try a different User ID.")
            else:
                # Apply filters from sidebar
                recs = recs[recs['Year-Of-Release'] >= min_year]

                st.markdown(f"#### 🎯 Top {len(recs)} Recommendations for User `{user_id_input}`")

                for _, row in recs.iterrows():
                    st.markdown(f"""
                    <div class='prime-card'>
                        <div class='prime-title'>🎬 {row['Movie-Title']}</div>
                        <div class='prime-sub'>📅 Year: {int(row['Year-Of-Release'])} &nbsp;|&nbsp; 🆔 Movie ID: {row['Movie-ID']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Visualise recommendations
                fig_rec = px.bar(
                    recs,
                    x='Movie-Title',
                    y='Year-Of-Release',
                    title=f"📅 Release Year of Recommended Movies",
                    labels={'Movie-Title': 'Movie', 'Year-Of-Release': 'Year'},
                    color='Year-Of-Release',
                    color_continuous_scale=[[0, PRIME_BLUE], [1, PRIME_ACCENT]],
                )
                fig_rec.update_layout(**CHART_LAYOUT, xaxis_tickangle=-35)
                fig_rec.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_rec, use_container_width=True)

        st.divider()
        st.markdown("### 👤 User Profile")
        uid_profile = st.number_input("User ID for Profile", value=default_user, step=1, key='profile_uid')
        if st.button("🔎 Show Profile"):
            user_movies = ratingsData[ratingsData['User-ID'] == uid_profile].merge(
                moviesData, on='Movie-ID'
            )
            user_info = userData[userData['User-ID'] == uid_profile]

            if user_movies.empty:
                st.warning("User not found or has no ratings.")
            else:
                ua, ub, uc = st.columns(3)
                ua.metric("Movies Rated", len(user_movies))
                ub.metric("Avg Given Rating", f"{user_movies['Movie-Rating'].mean():.2f}")
                uc.metric("Platform Avg Rating", f"{user_info['Avg-Rating'].values[0]:.2f}")

                fig_p = px.bar(
                    user_movies,
                    x='Movie-Title',
                    y='Movie-Rating',
                    title="User's Rating History",
                    color='Movie-Rating',
                    color_continuous_scale=[[0,'#FF5252'],[0.5, PRIME_BLUE],[1, '#00C853']],
                )
                fig_p.update_layout(**CHART_LAYOUT, xaxis_tickangle=-35, coloraxis_showscale=False)
                st.plotly_chart(fig_p, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 3 — ANALYTICS
    # ═══════════════════════════════════════════
    with tab3:
        st.markdown("### 📈 Deep Analytics")

        # Movies over time line chart
        yearly = moviesData.groupby('Year-Of-Release').size().reset_index(name='count')
        fig_y = px.line(
            yearly, x='Year-Of-Release', y='count',
            title="📅 Movies Released Per Year",
            labels={'Year-Of-Release': 'Year', 'count': 'Movies'},
            color_discrete_sequence=[PRIME_BLUE],
            markers=True,
        )
        fig_y.update_layout(**CHART_LAYOUT)
        fig_y.update_traces(line_width=2, marker_size=4)
        st.plotly_chart(fig_y, use_container_width=True)

        col_e, col_f = st.columns(2)

        # Avg rating per year
        with col_e:
            avg_by_year = (
                ratingsData.merge(moviesData, on='Movie-ID')
                .groupby('Year-Of-Release')['Movie-Rating']
                .mean()
                .reset_index()
            )
            fig_ay = px.area(
                avg_by_year, x='Year-Of-Release', y='Movie-Rating',
                title="⭐ Average Rating by Release Year",
                labels={'Year-Of-Release': 'Year', 'Movie-Rating': 'Avg Rating'},
                color_discrete_sequence=[PRIME_BLUE],
            )
            fig_ay.update_layout(**CHART_LAYOUT)
            fig_ay.add_hline(y=ratingsData['Movie-Rating'].mean(),
                             line_dash="dash", line_color=PRIME_ACCENT,
                             annotation_text="Overall Avg", annotation_font_color=PRIME_ACCENT)
            st.plotly_chart(fig_ay, use_container_width=True)

        # Ratings count per movie histogram
        with col_f:
            ratings_per_movie = ratingsData.groupby('Movie-ID').size().reset_index(name='count')
            fig_rpm = px.histogram(
                ratings_per_movie, x='count', nbins=50,
                title="📊 Ratings Per Movie Distribution",
                labels={'count': 'Number of Ratings', 'y': 'Movies'},
                color_discrete_sequence=[PRIME_ACCENT],
            )
            fig_rpm.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig_rpm, use_container_width=True)

        # Top rated movies (min 50 ratings)
        st.markdown("#### 🏅 Top Rated Movies (minimum 50 ratings)")
        top_rated = (
            ratingsData.groupby('Movie-ID')
            .agg(avg=('Movie-Rating','mean'), count=('Movie-Rating','count'))
            .reset_index()
            .query('count >= 50')
            .merge(moviesData, on='Movie-ID')
            .sort_values('avg', ascending=False)
            .head(20)
        )
        fig_tr = px.bar(
            top_rated, x='avg', y='Movie-Title', orientation='h',
            title="🏅 Top 20 Highest Rated Movies (≥50 ratings)",
            labels={'avg': 'Average Rating', 'Movie-Title': ''},
            color='count',
            color_continuous_scale=[[0, PRIME_BLUE],[1, PRIME_ACCENT]],
        )
        fig_tr.update_layout(**CHART_LAYOUT, height=500,
                             coloraxis_colorbar=dict(title="# Ratings", tickfont=dict(color='white')))
        fig_tr.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_tr, use_container_width=True)

        # User avg rating distribution
        col_g, col_h = st.columns(2)
        with col_g:
            fig_ua = px.histogram(
                userData, x='Avg-Rating', nbins=40,
                title="📊 User Average Rating Distribution",
                labels={'Avg-Rating': 'Avg Rating Given', 'count': 'Users'},
                color_discrete_sequence=[PRIME_BLUE],
            )
            fig_ua.update_layout(**CHART_LAYOUT)
            fig_ua.add_vline(x=userData['Avg-Rating'].mean(), line_dash='dash',
                             line_color=PRIME_ACCENT,
                             annotation_text=f"Mean {userData['Avg-Rating'].mean():.2f}",
                             annotation_font_color=PRIME_ACCENT)
            st.plotly_chart(fig_ua, use_container_width=True)

        with col_h:
            # Rating given vs user's avg rating scatter (sample)
            sample = ratingsData.sample(min(3000, len(ratingsData)), random_state=42).merge(userData, on='User-ID')
            fig_sc = px.scatter(
                sample, x='Avg-Rating', y='Movie-Rating',
                title="🔵 Individual Rating vs User Avg Rating",
                labels={'Avg-Rating': 'User Avg Rating', 'Movie-Rating': 'This Rating'},
                color='Movie-Rating',
                color_continuous_scale=[[0,'#FF5252'],[0.5, PRIME_BLUE],[1,'#00C853']],
                opacity=0.5,
            )
            fig_sc.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig_sc, use_container_width=True)

        # Decade box plot
        merged_decade = ratingsData.merge(moviesData, on='Movie-ID')
        merged_decade['Decade'] = (merged_decade['Year-Of-Release'] // 10 * 10).astype(str) + 's'
        fig_box = px.box(
            merged_decade.sort_values('Year-Of-Release'),
            x='Decade', y='Movie-Rating',
            title="📦 Rating Distribution by Decade",
            labels={'Decade': 'Decade', 'Movie-Rating': 'Rating'},
            color='Decade',
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_box.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_box, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 4 — CORRELATIONS
    # ═══════════════════════════════════════════
    with tab4:
        st.markdown("### 🔗 Correlations & Relationships")

        # Build enriched df
        enriched = (
            ratingsData.groupby('Movie-ID')
            .agg(total_ratings=('Movie-Rating','count'), avg_rating=('Movie-Rating','mean'),
                 rating_std=('Movie-Rating','std'))
            .reset_index()
            .merge(moviesData, on='Movie-ID')
        )
        enriched['rating_std'] = enriched['rating_std'].fillna(0)

        # Scatter: total ratings vs avg rating
        col_i, col_j = st.columns(2)
        with col_i:
            fig_corr1 = px.scatter(
                enriched, x='total_ratings', y='avg_rating',
                title="📊 Total Ratings vs Avg Rating per Movie",
                labels={'total_ratings':'Total Ratings','avg_rating':'Avg Rating'},
                color='avg_rating',
                color_continuous_scale=[[0,'#FF5252'],[0.5,PRIME_BLUE],[1,'#00C853']],
                trendline='ols',
                hover_data=['Movie-Title'],
                opacity=0.7,
            )
            fig_corr1.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig_corr1, use_container_width=True)
            corr1 = enriched['total_ratings'].corr(enriched['avg_rating'])
            st.info(f"📐 Pearson Correlation: **{corr1:.4f}**")

        # Scatter: year vs avg rating
        with col_j:
            fig_corr2 = px.scatter(
                enriched, x='Year-Of-Release', y='avg_rating',
                title="📅 Release Year vs Avg Rating",
                labels={'Year-Of-Release':'Year','avg_rating':'Avg Rating'},
                color='total_ratings',
                color_continuous_scale=[[0,PRIME_BLUE],[1,PRIME_ACCENT]],
                trendline='ols',
                hover_data=['Movie-Title'],
                opacity=0.7,
            )
            fig_corr2.update_layout(**CHART_LAYOUT,
                                    coloraxis_colorbar=dict(title="# Ratings", tickfont=dict(color='white')))
            st.plotly_chart(fig_corr2, use_container_width=True)
            corr2 = enriched['Year-Of-Release'].corr(enriched['avg_rating'])
            st.info(f"📐 Pearson Correlation: **{corr2:.4f}**")

        # Correlation heatmap
        corr_df = enriched[['total_ratings','avg_rating','rating_std','Year-Of-Release']].corr()
        fig_heat = px.imshow(
            corr_df,
            text_auto='.3f',
            title="🌡️ Feature Correlation Heatmap",
            color_continuous_scale=[[0,'#FF5252'],[0.5,'#1A242F'],[1,PRIME_BLUE]],
            zmin=-1, zmax=1,
            labels=dict(color="Correlation"),
        )
        fig_heat.update_layout(**CHART_LAYOUT, height=400)
        fig_heat.update_traces(textfont_size=14)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Scatter: rating std vs avg rating (rating consistency)
        col_k, col_l = st.columns(2)
        with col_k:
            fig_corr3 = px.scatter(
                enriched, x='avg_rating', y='rating_std',
                title="🎯 Rating Consistency vs Avg Rating",
                labels={'avg_rating':'Avg Rating','rating_std':'Rating Std Dev'},
                color='total_ratings',
                color_continuous_scale=[[0,PRIME_BLUE],[1,PRIME_ACCENT]],
                trendline='ols',
                hover_data=['Movie-Title'],
                opacity=0.7,
            )
            fig_corr3.update_layout(**CHART_LAYOUT,
                                    coloraxis_colorbar=dict(title="# Ratings", tickfont=dict(color='white')))
            st.plotly_chart(fig_corr3, use_container_width=True)
            corr3 = enriched['avg_rating'].corr(enriched['rating_std'])
            st.info(f"📐 Pearson Correlation: **{corr3:.4f}**")

        # User: num ratings vs avg rating
        with col_l:
            user_sample = userData.sample(min(5000, len(userData)), random_state=42)
            fig_corr4 = px.scatter(
                user_sample, x='Num-Ratings', y='Avg-Rating',
                title="👥 User Activity vs Avg Rating Given",
                labels={'Num-Ratings':'# Ratings by User','Avg-Rating':'User Avg Rating'},
                color='Avg-Rating',
                color_continuous_scale=[[0,'#FF5252'],[0.5,PRIME_BLUE],[1,'#00C853']],
                trendline='ols',
                opacity=0.5,
            )
            fig_corr4.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig_corr4, use_container_width=True)
            corr4 = userData['Num-Ratings'].corr(userData['Avg-Rating'])
            st.info(f"📐 Pearson Correlation: **{corr4:.4f}**")

        # Violin plot: rating distribution per star
        fig_vio = px.violin(
            ratingsData, y='Movie-Rating', x='Movie-Rating',
            title="🎻 Rating Violin Plot",
            labels={'Movie-Rating':'Star Rating'},
            color='Movie-Rating',
            color_discrete_sequence=[PRIME_BLUE, PRIME_ACCENT, '#00C853', '#FF5252', '#AA00FF'],
            box=True,
        )
        fig_vio.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_vio, use_container_width=True)

        # 3D scatter: year, avg_rating, total_ratings
        st.markdown("#### 🧊 3D: Year × Avg Rating × Total Ratings")
        fig_3d = px.scatter_3d(
            enriched.sample(min(1000, len(enriched)), random_state=42),
            x='Year-Of-Release', y='avg_rating', z='total_ratings',
            color='avg_rating',
            color_continuous_scale=[[0,'#FF5252'],[0.5,PRIME_BLUE],[1,'#00C853']],
            hover_data=['Movie-Title'],
            title="🧊 3D Scatter: Year × Avg Rating × Total Ratings",
            labels={'Year-Of-Release':'Year','avg_rating':'Avg Rating','total_ratings':'Total Ratings'},
            opacity=0.8,
        )
        fig_3d.update_layout(
            paper_bgcolor=PRIME_CARD,
            font=dict(color='white'),
            scene=dict(
                xaxis=dict(backgroundcolor=PRIME_DARK, gridcolor='#2A3A4A', color='white'),
                yaxis=dict(backgroundcolor=PRIME_DARK, gridcolor='#2A3A4A', color='white'),
                zaxis=dict(backgroundcolor=PRIME_DARK, gridcolor='#2A3A4A', color='white'),
            ),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 5 — DATA EXPLORER
    # ═══════════════════════════════════════════
    with tab5:
        st.markdown("### 🗂️ Data Explorer")

        subtab1, subtab2, subtab3, subtab4 = st.tabs(["🎬 Movies", "⭐ Ratings", "👥 Users", "🔀 Combined"])

        with subtab1:
            st.markdown(f"**{len(moviesData):,} movies** — filter and explore")
            year_filter = st.slider("Filter by Year", int(moviesData['Year-Of-Release'].min()),
                                    int(moviesData['Year-Of-Release'].max()),
                                    (1990, 2005))
            df_show = moviesData[(moviesData['Year-Of-Release'] >= year_filter[0]) &
                                 (moviesData['Year-Of-Release'] <= year_filter[1])]
            st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=400)
            st.download_button("⬇️ Download Movies.csv", df_show.to_csv(index=False),
                               "movies_filtered.csv", "text/csv")

        with subtab2:
            st.markdown(f"**{len(ratingsData):,} ratings** — sample view")
            rating_filter = st.multiselect("Filter by Rating", [1,2,3,4,5], default=[1,2,3,4,5])
            df_r = ratingsData[ratingsData['Movie-Rating'].isin(rating_filter)]
            st.dataframe(df_r.sample(min(500, len(df_r)), random_state=42).reset_index(drop=True),
                         use_container_width=True, height=400)
            st.download_button("⬇️ Download Ratings.csv", ratingsData.to_csv(index=False),
                               "ratings.csv", "text/csv")

        with subtab3:
            st.markdown(f"**{len(userData):,} users**")
            st.dataframe(userData.reset_index(drop=True), use_container_width=True, height=400)
            st.download_button("⬇️ Download Users.csv", userData.to_csv(index=False),
                               "users.csv", "text/csv")

        with subtab4:
            st.markdown(f"**{len(combined_df):,} combined records**")
            st.dataframe(combined_df.sample(min(500, len(combined_df)), random_state=42).reset_index(drop=True),
                         use_container_width=True, height=400)

else:
    # ── LANDING STATE ──
    st.markdown("""
    <div style='text-align:center; padding:60px 20px'>
        <div style='font-size:5rem'>🎬</div>
        <div style='font-size:1.6rem; color:#00A8E1; font-weight:700; margin:16px 0'>
            Welcome to the Movie Recommender Dashboard
        </div>
        <div style='color:#AAAAAA; font-size:1rem; max-width:500px; margin:0 auto'>
            Set the paths to your <code>Movies.csv</code>, <code>Ratings.csv</code>, and
            <code>Users.csv</code> in the sidebar, then click <strong>🚀 Load Data</strong> to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, icon, title, desc in [
        (col1, "📊", "Overview", "KPIs, rating distributions, top movies"),
        (col2, "🔍", "Recommender", "Knowledge-based collaborative filtering"),
        (col3, "📈", "Analytics", "Deep dive into trends and patterns"),
        (col4, "🔗", "Correlations", "Feature relationships and heatmaps"),
    ]:
        with col:
            st.markdown(f"""
            <div class='prime-card' style='text-align:center'>
                <div style='font-size:2rem'>{icon}</div>
                <div style='color:#00A8E1; font-weight:600; margin:8px 0'>{title}</div>
                <div style='color:#888; font-size:0.8rem'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

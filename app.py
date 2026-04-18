import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse
import sklearn.metrics as skm
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CineAI · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif!important;background-color:#08141E!important;color:#F0F4F8!important}
.stApp{background:#08141E!important}
.block-container{padding:0!important;max-width:100%!important}
.main>div{padding:0!important}
#MainMenu,footer,header{visibility:hidden}
[data-testid="stToolbar"]{display:none}
[data-testid="collapsedControl"]{display:none}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#08141E}::-webkit-scrollbar-thumb{background:#1C6EA4;border-radius:3px}
.topnav{background:linear-gradient(180deg,#0D1F2D,transparent);padding:18px 40px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:999;border-bottom:1px solid rgba(28,110,164,0.2);backdrop-filter:blur(12px)}
.logo{font-size:1.5rem;font-weight:800;background:linear-gradient(135deg,#1C6EA4,#00C9FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.5px}
.logo span{-webkit-text-fill-color:#FF6B35}
.nav-links{display:flex;gap:32px}
.nav-link{color:#8899AA;font-size:0.875rem;font-weight:500;cursor:pointer;transition:color 0.2s}
.nav-link:hover,.nav-link.active{color:#00C9FF}
.hero{background:linear-gradient(135deg,#08141E 0%,#0D2137 40%,#102840 70%,#08141E 100%);padding:56px 40px 48px;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;top:-50%;right:-10%;width:500px;height:500px;background:radial-gradient(circle,rgba(28,110,164,0.15) 0%,transparent 70%);pointer-events:none}
.hero-eyebrow{font-size:0.75rem;font-weight:600;color:#1C6EA4;letter-spacing:2px;text-transform:uppercase;margin-bottom:14px}
.hero-title{font-size:clamp(1.8rem,4vw,3rem);font-weight:800;line-height:1.15;letter-spacing:-1px;color:#F0F4F8;margin-bottom:14px}
.hero-title .accent{background:linear-gradient(135deg,#1C6EA4,#00C9FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-sub{font-size:0.95rem;color:#6B8294;max-width:500px;line-height:1.7;margin-bottom:32px}
.hero-stats{display:flex;gap:36px;flex-wrap:wrap}
.hero-stat-val{font-size:1.5rem;font-weight:700;color:#F0F4F8;letter-spacing:-0.5px;line-height:1}
.hero-stat-lbl{font-size:0.72rem;color:#6B8294;font-weight:500;margin-top:4px}
.hero-stat-sep{width:1px;background:rgba(255,255,255,0.08);align-self:stretch}
.content{padding:0 40px 60px}
.section-header{display:flex;align-items:baseline;gap:12px;margin:36px 0 18px}
.section-title{font-size:1.1rem;font-weight:700;color:#F0F4F8;letter-spacing:-0.3px;white-space:nowrap}
.section-sub{font-size:0.78rem;color:#4A6275}
.section-line{flex:1;height:1px;background:linear-gradient(90deg,rgba(28,110,164,0.3),transparent);margin-left:8px}
.kpi-card{background:#0D1F2D;border:1px solid rgba(28,110,164,0.2);border-radius:14px;padding:20px 22px;position:relative;overflow:hidden;transition:border-color 0.2s,transform 0.2s}
.kpi-card:hover{border-color:rgba(28,110,164,0.6);transform:translateY(-2px)}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#1C6EA4,#00C9FF);opacity:0.6}
.kpi-icon{font-size:1.3rem;margin-bottom:10px}
.kpi-val{font-size:1.65rem;font-weight:700;color:#F0F4F8;letter-spacing:-0.5px;line-height:1}
.kpi-lbl{font-size:0.7rem;color:#4A6275;font-weight:500;margin-top:6px;text-transform:uppercase;letter-spacing:0.8px}
.kpi-delta{font-size:0.72rem;color:#2ECC71;font-weight:600;margin-top:4px}
.chart-card{background:#0D1F2D;border:1px solid rgba(28,110,164,0.15);border-radius:16px;padding:22px;transition:border-color 0.2s;margin-bottom:4px}
.chart-card:hover{border-color:rgba(28,110,164,0.35)}
.chart-title{font-size:0.88rem;font-weight:600;color:#C5D5E0;margin-bottom:3px}
.chart-sub{font-size:0.73rem;color:#4A6275;margin-bottom:14px}
.movie-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:14px;margin-top:8px}
.movie-card{background:#0A1929;border:1px solid rgba(28,110,164,0.15);border-radius:12px;overflow:hidden;transition:transform 0.25s,border-color 0.25s,box-shadow 0.25s;cursor:pointer}
.movie-card:hover{transform:translateY(-6px) scale(1.02);border-color:#1C6EA4;box-shadow:0 20px 40px rgba(0,0,0,0.5)}
.movie-poster{width:100%;aspect-ratio:2/3;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:2.8rem;position:relative;overflow:hidden}
.movie-poster::after{content:'';position:absolute;inset:0;background:linear-gradient(to bottom,transparent 50%,rgba(8,20,30,0.9) 100%)}
.movie-poster-id{position:absolute;bottom:8px;left:10px;font-size:0.62rem;color:rgba(255,255,255,0.35);font-weight:500;z-index:2}
.movie-body{padding:12px}
.movie-title-card{font-size:0.8rem;font-weight:600;color:#D0E0EE;line-height:1.4;margin-bottom:5px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.movie-year{font-size:0.68rem;color:#4A6275;font-weight:500}
.movie-badge{display:inline-block;background:rgba(28,110,164,0.2);color:#00C9FF;font-size:0.63rem;padding:2px 8px;border-radius:99px;font-weight:600;margin-top:6px;border:1px solid rgba(0,201,255,0.2)}
.rec-panel{background:#0D1F2D;border:1px solid rgba(28,110,164,0.2);border-radius:16px;padding:28px}
.corr-badge{display:inline-flex;align-items:center;gap:6px;background:#0A1929;border:1px solid rgba(28,110,164,0.25);border-radius:8px;padding:7px 13px;font-size:0.78rem;color:#C5D5E0;font-weight:500;margin-top:8px}
.corr-val{font-weight:700;color:#00C9FF;font-size:0.88rem}
.stTextInput>div>div>input{background:#0A1929!important;color:#F0F4F8!important;border:1px solid rgba(28,110,164,0.3)!important;border-radius:10px!important;padding:10px 14px!important;font-family:'Inter',sans-serif!important}
.stTextInput>div>div>input:focus{border-color:#1C6EA4!important;box-shadow:0 0 0 3px rgba(28,110,164,0.15)!important}
.stTextInput>div>div>input::placeholder{color:#3A5265!important}
.stSelectbox>div>div{background:#0A1929!important;border:1px solid rgba(28,110,164,0.3)!important;border-radius:10px!important;color:#F0F4F8!important}
.stNumberInput>div>div>input{background:#0A1929!important;color:#F0F4F8!important;border:1px solid rgba(28,110,164,0.3)!important;border-radius:10px!important;font-family:'Inter',sans-serif!important}
.stButton>button{background:linear-gradient(135deg,#1C6EA4,#0E4C78)!important;color:#FFFFFF!important;border:none!important;border-radius:10px!important;padding:11px 24px!important;font-weight:600!important;font-size:0.875rem!important;font-family:'Inter',sans-serif!important;transition:all 0.2s!important;box-shadow:0 4px 15px rgba(28,110,164,0.3)!important;width:100%}
.stButton>button:hover{background:linear-gradient(135deg,#2280C0,#1C6EA4)!important;box-shadow:0 6px 20px rgba(28,110,164,0.4)!important;transform:translateY(-1px)!important}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid rgba(28,110,164,0.15)!important;gap:0!important;padding:0!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#4A6275!important;font-weight:500!important;font-size:0.875rem!important;padding:13px 22px!important;border-radius:0!important;border-bottom:2px solid transparent!important;font-family:'Inter',sans-serif!important}
.stTabs [aria-selected="true"]{color:#00C9FF!important;border-bottom:2px solid #1C6EA4!important;background:transparent!important}
.stTabs [data-baseweb="tab-panel"]{padding:28px 0 0!important}
.stSlider>div>div>div{background:#1C6EA4!important}
.stSlider label{color:#8899AA!important;font-size:0.8rem!important}
[data-testid="stDataFrame"]{border:1px solid rgba(28,110,164,0.2)!important;border-radius:12px!important;overflow:hidden}
hr{border-color:rgba(28,110,164,0.15)!important}
.stSuccess{background:rgba(46,204,113,0.08)!important;border-left:3px solid #2ECC71!important;border-radius:8px!important}
.stInfo{background:rgba(28,110,164,0.08)!important;border-left:3px solid #1C6EA4!important;border-radius:8px!important}
.stWarning{background:rgba(255,107,53,0.08)!important;border-left:3px solid #FF6B35!important;border-radius:8px!important}
.stDownloadButton>button{background:transparent!important;color:#1C6EA4!important;border:1px solid rgba(28,110,164,0.4)!important;border-radius:8px!important;font-size:0.8rem!important;padding:7px 14px!important;width:auto!important;box-shadow:none!important}
</style>
""", unsafe_allow_html=True)

BG="#0D1F2D"; BG2="#0A1929"; BLUE="#1C6EA4"; CYAN="#00C9FF"
ORANGE="#FF6B35"; GREEN="#2ECC71"; TEXT="#C5D5E0"; MUTED="#4A6275"; GRID="#0F2336"
SCALE=[[0,BG2],[0.5,BLUE],[1,CYAN]]
SCALE2=[[0,"#8B0000"],[0.5,BLUE],[1,"#00FF88"]]

def CL(h=360,**kw):
    return dict(paper_bgcolor=BG,plot_bgcolor=BG,
                font=dict(family="Inter",color=TEXT,size=12),
                xaxis=dict(gridcolor=GRID,linecolor=GRID,color=MUTED,tickfont=dict(size=11)),
                yaxis=dict(gridcolor=GRID,linecolor=GRID,color=MUTED,tickfont=dict(size=11)),
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT,size=11)),
                margin=dict(t=40,b=36,l=46,r=16),height=h,**kw)

ICONS=["🎬","🎥","🍿","🎭","🎞️","🌟","🎦","📽️","🎪","🎠","🌌","🔮"]
PALETTES=[("0D2137","1C4E7A"),("1A0A2E","4A1880"),("0A2218","1A6644"),
          ("2A0A0A","7A2020"),("1A1A0A","6A6A20"),("0A1A2A","1A5A7A")]
def icon(mid): return ICONS[int(mid)%len(ICONS)]
def grad(mid): a,b=PALETTES[int(mid)%len(PALETTES)]; return f"#{a}",f"#{b}"

@st.cache_data(show_spinner=False)
def load(mp,rp,up):
    M=pd.read_csv(mp,converters={'Year-Of-Release':lambda x:pd.to_numeric(x,errors='coerce')}).dropna()
    M['Year-Of-Release']=M['Year-Of-Release'].astype(int)
    R=pd.read_csv(rp).dropna(); R['Movie-Rating']=R['Movie-Rating'].astype(int)
    U=pd.read_csv(up).dropna()
    return M,R,U

@st.cache_data(show_spinner=False)
def build_combined(_M,_R,_U):
    df=pd.merge(_R,_M,on='Movie-ID'); df=pd.merge(df,_U,on='User-ID')
    df=df.drop('Year-Of-Release',axis=1)
    df=df[['User-ID','Movie-Rating','Movie-ID','Movie-Title','Num-Ratings','Avg-Rating']]
    d={mid:i for i,mid in enumerate(df['Movie-ID'].unique())}
    df['Movie-ID-Key']=df['Movie-ID'].map(d)
    return df,d

@st.cache_resource(show_spinner=False)
def build_mat(_df):
    return scipy.sparse.coo_matrix((_df['Movie-Rating'],(_df['User-ID'],_df['Movie-ID-Key'])))

def rec(combined,movies,uid,mat,n=12):
    try:
        s=skm.pairwise.cosine_similarity(mat.getrow(uid),mat).ravel()
        s[uid]=0
        top=s.argsort()[::-1][:5]
        ids=combined.loc[combined['User-ID'].isin(top),'Movie-ID']
        return movies[movies['Movie-ID'].isin(ids)].drop_duplicates('Movie-ID').head(n)
    except: return pd.DataFrame()

# ── TOP NAV ──
st.markdown("""
<div class="topnav">
  <div class="logo">Cine<span>AI</span></div>
  <div class="nav-links">
    <span class="nav-link active">Dashboard</span>
    <span class="nav-link">Discover</span>
    <span class="nav-link">Analytics</span>
  </div>
  <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#1C6EA4,#00C9FF);
      display:flex;align-items:center;justify-content:center;font-size:0.85rem;font-weight:700">U</div>
</div>
""", unsafe_allow_html=True)

# ── FILE CONFIG — hardcoded, no user input needed ──
mp      = "Movies.csv"
rp      = "Ratings.csv"
up      = "Users.csv"
go_load = True

if go_load or st.session_state.get('loaded'):
    try:
        with st.spinner("Initialising recommender engine..."):
            M,R,U=load(mp,rp,up)
            combined,MovieID_dict=build_combined(M,R,U)
            mat=build_mat(combined)
        st.session_state['loaded']=True

        stats=(R.groupby('Movie-ID').agg(total=('Movie-Rating','count'),avg=('Movie-Rating','mean'),
               std=('Movie-Rating','std')).reset_index().merge(M,on='Movie-ID'))
        stats['std']=stats['std'].fillna(0)

        # ── HERO ──
        st.markdown(f"""
        <div class="hero">
          <div class="hero-eyebrow">🎬 AI-Powered Movie Intelligence</div>
          <div class="hero-title">Your personal<br><span class="accent">cinema universe</span></div>
          <div class="hero-sub">Discover movies you will love using knowledge-based collaborative filtering trained on the Netflix Prize dataset.</div>
          <div class="hero-stats">
            <div><div class="hero-stat-val">{len(M):,}</div><div class="hero-stat-lbl">Movies</div></div>
            <div class="hero-stat-sep"></div>
            <div><div class="hero-stat-val">{len(R):,}</div><div class="hero-stat-lbl">Ratings</div></div>
            <div class="hero-stat-sep"></div>
            <div><div class="hero-stat-val">{len(U):,}</div><div class="hero-stat-lbl">Users</div></div>
            <div class="hero-stat-sep"></div>
            <div><div class="hero-stat-val">{R['Movie-Rating'].mean():.2f}⭐</div><div class="hero-stat-lbl">Avg Rating</div></div>
            <div class="hero-stat-sep"></div>
            <div><div class="hero-stat-val">{int(M['Year-Of-Release'].min())}–{int(M['Year-Of-Release'].max())}</div><div class="hero-stat-lbl">Year Span</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="content">', unsafe_allow_html=True)
        t1,t2,t3,t4,t5=st.tabs(["  🏠  Overview  ","  🎯  Recommender  ","  📊  Analytics  ","  🔗  Correlations  ","  🗂️  Explorer  "])

        # ════ TAB 1 — OVERVIEW ════
        with t1:
            kpis=[("🎬",f"{len(M):,}","Total Movies","Netflix Prize catalogue"),
                  ("⭐",f"{len(R):,}","Total Ratings",f"Avg {R['Movie-Rating'].mean():.2f} stars"),
                  ("👥",f"{len(U):,}","Unique Users","Active reviewers"),
                  ("🏆",f"{stats['avg'].max():.2f}★","Peak Avg Rating","Best-rated movie"),
                  ("📅",f"{int(M['Year-Of-Release'].max())}","Latest Release",f"From {int(M['Year-Of-Release'].min())}")]
            cols=st.columns(5)
            for col,(ic,val,lbl,sub) in zip(cols,kpis):
                col.markdown(f'<div class="kpi-card"><div class="kpi-icon">{ic}</div><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div><div class="kpi-delta">{sub}</div></div>',unsafe_allow_html=True)

            st.markdown('<div class="section-header"><div class="section-title">Rating & Catalogue Overview</div><div class="section-line"></div></div>',unsafe_allow_html=True)
            c1,c2=st.columns(2)
            rc=R['Movie-Rating'].value_counts().sort_index()

            with c1:
                fig=go.Figure(go.Bar(x=rc.index,y=rc.values,
                    marker=dict(color=rc.values,colorscale=SCALE,line=dict(color=BLUE,width=1)),
                    hovertemplate="<b>%{x} Stars</b><br>%{y:,} ratings<extra></extra>"))
                fig.update_layout(**CL(320),bargap=0.25)
                st.markdown('<div class="chart-card"><div class="chart-title">⭐ Rating Distribution</div><div class="chart-sub">How users rate movies across the dataset</div>',unsafe_allow_html=True)
                st.plotly_chart(fig,use_container_width=True,config=dict(displayModeBar=False))
                st.markdown('</div>',unsafe_allow_html=True)

            with c2:
                M2=M.copy(); M2['Decade']=(M2['Year-Of-Release']//10*10).astype(str)+"s"
                dc=M2['Decade'].value_counts().sort_index()
                fig2=go.Figure(go.Bar(x=dc.index,y=dc.values,
                    marker=dict(color=dc.values,colorscale=[[0,BG2],[0.5,BLUE],[1,ORANGE]],
                                line=dict(color=BG,width=0.5)),
                    hovertemplate="<b>%{x}</b><br>%{y} movies<extra></extra>"))
                fig2.update_layout(**CL(320),bargap=0.25)
                st.markdown('<div class="chart-card"><div class="chart-title">📅 Movies by Decade</div><div class="chart-sub">Catalogue distribution across release decades</div>',unsafe_allow_html=True)
                st.plotly_chart(fig2,use_container_width=True,config=dict(displayModeBar=False))
                st.markdown('</div>',unsafe_allow_html=True)

            c3,c4=st.columns(2)
            with c3:
                fig3=go.Figure(go.Pie(values=rc.values,labels=[f"{r}★" for r in rc.index],hole=0.55,
                    marker=dict(colors=[BG2,"#0E3D6E",BLUE,"#0EA5E9",CYAN],line=dict(color="#08141E",width=2)),
                    hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>"))
                fig3.add_annotation(text=f"{R['Movie-Rating'].mean():.1f}★",x=0.5,y=0.5,
                    font=dict(size=20,color="white",family="Inter"),showarrow=False)
                fig3.update_layout(**CL(320),showlegend=True)
                st.markdown('<div class="chart-card"><div class="chart-title">🍩 Rating Share</div><div class="chart-sub">Proportion of each star rating</div>',unsafe_allow_html=True)
                st.plotly_chart(fig3,use_container_width=True,config=dict(displayModeBar=False))
                st.markdown('</div>',unsafe_allow_html=True)

            with c4:
                fig4=go.Figure(go.Histogram(x=U['Num-Ratings'],nbinsx=40,
                    marker=dict(color=BLUE,opacity=0.8,line=dict(color=BG,width=0.3)),
                    hovertemplate="<b>%{x} ratings</b><br>%{y} users<extra></extra>"))
                fig4.update_layout(**CL(320))
                st.markdown('<div class="chart-card"><div class="chart-title">👥 User Activity Distribution</div><div class="chart-sub">How many ratings each user has given</div>',unsafe_allow_html=True)
                st.plotly_chart(fig4,use_container_width=True,config=dict(displayModeBar=False))
                st.markdown('</div>',unsafe_allow_html=True)

            top15=stats.nlargest(15,'total')
            fig5=go.Figure(go.Bar(x=top15['total'],y=top15['Movie-Title'],orientation='h',
                marker=dict(color=top15['avg'],colorscale=SCALE2,cmin=1,cmax=5,
                            colorbar=dict(title=dict(text="Avg★",font=dict(color=TEXT)),tickfont=dict(color=TEXT))),
                text=[f"  ★{v:.1f}" for v in top15['avg']],textposition='inside',textfont=dict(color='white',size=11),
                hovertemplate="<b>%{y}</b><br>%{x:,} ratings<extra></extra>"))
            fig5.update_layout(**CL(420),yaxis=dict(autorange='reversed',gridcolor=GRID,color=MUTED))
            st.markdown('<br><div class="chart-card"><div class="chart-title">🏆 Top 15 Most Rated Movies</div><div class="chart-sub">Colour intensity = average star rating</div>',unsafe_allow_html=True)
            st.plotly_chart(fig5,use_container_width=True,config=dict(displayModeBar=False))
            st.markdown('</div>',unsafe_allow_html=True)

        # ════ TAB 2 — RECOMMENDER ════
        with t2:
            st.markdown('<div class="section-header"><div class="section-title">Knowledge-Based Recommender</div><div class="section-line"></div><div class="section-sub">Cosine similarity · Top-5 user neighbours</div></div>',unsafe_allow_html=True)
            st.markdown('<div class="rec-panel">',unsafe_allow_html=True)
            ri1,ri2,ri3=st.columns([2,2,1])
            default_uid=int(combined['User-ID'].value_counts().index[0])
            uid_in=ri1.number_input("User ID",min_value=1,max_value=int(combined['User-ID'].max()),value=default_uid,step=1)
            n_recs=ri2.slider("Recommendations",4,20,12)
            ri3.markdown("<br>",unsafe_allow_html=True)
            go_btn=ri3.button("🎯  Recommend",use_container_width=True)

            if go_btn:
                with st.spinner("Finding your perfect movies..."):
                    recs=rec(combined,M,int(uid_in),mat,n_recs)
                if recs.empty:
                    st.warning("No recommendations found. Try a different User ID.")
                else:
                    st.markdown(f"<br><div style='font-size:0.8rem;color:{MUTED};margin-bottom:14px'>Showing <strong style='color:{TEXT}'>{len(recs)}</strong> recommendations for User <strong style='color:{CYAN}'>{uid_in}</strong></div>",unsafe_allow_html=True)
                    st.markdown('<div class="movie-grid">',unsafe_allow_html=True)
                    for _,row in recs.iterrows():
                        c1f,c2f=grad(row['Movie-ID'])
                        st.markdown(f"""
                        <div class="movie-card">
                          <div class="movie-poster" style="background:linear-gradient(135deg,{c1f},{c2f})">
                            <span style="font-size:2.8rem;z-index:1">{icon(row['Movie-ID'])}</span>
                            <div class="movie-poster-id">ID #{row['Movie-ID']}</div>
                          </div>
                          <div class="movie-body">
                            <div class="movie-title-card">{row['Movie-Title']}</div>
                            <div class="movie-year">📅 {int(row['Year-Of-Release'])}</div>
                            <div class="movie-badge">Recommended</div>
                          </div>
                        </div>""",unsafe_allow_html=True)
                    st.markdown('</div>',unsafe_allow_html=True)

                    fig_rec=go.Figure(go.Bar(x=recs['Movie-Title'],y=recs['Year-Of-Release'],
                        marker=dict(color=recs['Year-Of-Release'],colorscale=[[0,BLUE],[1,CYAN]],
                                    line=dict(color=BG,width=1)),
                        hovertemplate="<b>%{x}</b><br>Released: %{y}<extra></extra>"))
                    fig_rec.update_layout(**CL(280),xaxis=dict(tickangle=-32,gridcolor=GRID,color=MUTED))
                    st.markdown('<br><div class="chart-card"><div class="chart-title">Release Years of Recommended Movies</div>',unsafe_allow_html=True)
                    st.plotly_chart(fig_rec,use_container_width=True,config=dict(displayModeBar=False))
                    st.markdown('</div>',unsafe_allow_html=True)

            st.markdown('</div>',unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:32px"><div class="section-title">User Profile Lookup</div><div class="section-line"></div></div>',unsafe_allow_html=True)
            pp1,pp2=st.columns([3,1])
            uid_p=pp1.number_input("User ID for profile",value=default_uid,step=1)
            pp2.markdown("<br>",unsafe_allow_html=True)
            prof_btn=pp2.button("🔎  View Profile",use_container_width=True)
            if prof_btn:
                um=R[R['User-ID']==uid_p].merge(M,on='Movie-ID')
                ui=U[U['User-ID']==uid_p]
                if um.empty: st.warning("User not found.")
                else:
                    pm1,pm2,pm3=st.columns(3)
                    pm1.metric("Movies Rated",len(um))
                    pm2.metric("Avg Rating Given",f"{um['Movie-Rating'].mean():.2f} ★")
                    pm3.metric("Platform Avg",f"{ui['Avg-Rating'].values[0]:.2f} ★" if len(ui) else "N/A")
                    fig_p=go.Figure(go.Bar(x=um['Movie-Title'],y=um['Movie-Rating'],
                        marker=dict(color=um['Movie-Rating'],colorscale=SCALE2,cmin=1,cmax=5,
                                    line=dict(color=BG,width=0.5)),
                        hovertemplate="<b>%{x}</b><br>%{y}★<extra></extra>"))
                    fig_p.update_layout(**CL(280),xaxis=dict(tickangle=-35,gridcolor=GRID))
                    st.markdown('<div class="chart-card"><div class="chart-title">Rating History</div>',unsafe_allow_html=True)
                    st.plotly_chart(fig_p,use_container_width=True,config=dict(displayModeBar=False))
                    st.markdown('</div>',unsafe_allow_html=True)

        # ════ TAB 3 — ANALYTICS ════
        with t3:
            st.markdown('<div class="section-header"><div class="section-title">Deep Analytics</div><div class="section-line"></div></div>',unsafe_allow_html=True)
            yearly=M.groupby('Year-Of-Release').size().reset_index(name='count')
            fig_y=go.Figure(go.Scatter(x=yearly['Year-Of-Release'],y=yearly['count'],fill='tozeroy',mode='lines',
                line=dict(color=BLUE,width=2),fillcolor="rgba(28,110,164,0.12)",
                hovertemplate="<b>%{x}</b><br>%{y} movies<extra></extra>"))
            fig_y.update_layout(**CL(260))
            st.markdown('<div class="chart-card"><div class="chart-title">Production Timeline</div><div class="chart-sub">Number of movies released each year</div>',unsafe_allow_html=True)
            st.plotly_chart(fig_y,use_container_width=True,config=dict(displayModeBar=False))
            st.markdown('</div>',unsafe_allow_html=True)

            ca,cb=st.columns(2)
            with ca:
                avy=R.merge(M,on='Movie-ID').groupby('Year-Of-Release')['Movie-Rating'].mean().reset_index()
                fig_avy=go.Figure(go.Scatter(x=avy['Year-Of-Release'],y=avy['Movie-Rating'],fill='tozeroy',
                    mode='lines+markers',line=dict(color=CYAN,width=2),fillcolor="rgba(0,201,255,0.07)",
                    marker=dict(size=4,color=CYAN),hovertemplate="<b>%{x}</b><br>%{y:.2f}★<extra></extra>"))
                fig_avy.add_hline(y=R['Movie-Rating'].mean(),line_dash="dash",line_color=ORANGE,
                    annotation_text=f"Avg {R['Movie-Rating'].mean():.2f}★",annotation_font_color=ORANGE,annotation_position="top right")
                fig_avy.update_layout(**CL(300))
                st.markdown('<div class="chart-card"><div class="chart-title">Avg Rating by Release Year</div><div class="chart-sub">Does era affect how well movies are rated?</div>',unsafe_allow_html=True)
                st.plotly_chart(fig_avy,use_container_width=True,config=dict(displayModeBar=False))
                st.markdown('</div>',unsafe_allow_html=True)

            with cb:
                rpm=R.groupby('Movie-ID').size().reset_index(name='count')
                fig_rpm=go.Figure(go.Histogram(x=rpm['count'],nbinsx=40,
                    marker=dict(color=ORANGE,opacity=0.85,line=dict(color=BG,width=0.5))))
                fig_rpm.update_layout(**CL(300))
                st.markdown('<div class="chart-card"><div class="chart-title">Ratings Per Movie</div><div class="chart-sub">How many ratings does each movie receive?</div>',unsafe_allow_html=True)
                st.plotly_chart(fig_rpm,use_container_width=True,config=dict(displayModeBar=False))
                st.markdown('</div>',unsafe_allow_html=True)

            top_r=stats.query('total>=50').nlargest(20,'avg')
            fig_tr=go.Figure(go.Bar(x=top_r['avg'],y=top_r['Movie-Title'],orientation='h',
                marker=dict(color=top_r['total'],colorscale=[[0,BLUE],[1,CYAN]],
                            colorbar=dict(title=dict(text="# Ratings",font=dict(color=TEXT)),tickfont=dict(color=TEXT))),
                text=[f"  {v:.2f}★" for v in top_r['avg']],textposition='inside',textfont=dict(color='white',size=10),
                hovertemplate="<b>%{y}</b><br>%{x:.2f}★<extra></extra>"))
            fig_tr.update_layout(**CL(460),yaxis=dict(autorange='reversed'))
            st.markdown('<br><div class="chart-card"><div class="chart-title">Top 20 Highest Rated Movies (≥50 ratings)</div><div class="chart-sub">Colour = number of ratings received</div>',unsafe_allow_html=True)
            st.plotly_chart(fig_tr,use_container_width=True,config=dict(displayModeBar=False))
            st.markdown('</div>',unsafe_allow_html=True)

            md=R.merge(M,on='Movie-ID'); md['Decade']=(md['Year-Of-Release']//10*10).astype(str)+"s"
            fig_box=px.box(md.sort_values('Year-Of-Release'),x='Decade',y='Movie-Rating',color='Decade',
                color_discrete_sequence=[BG2,"#0E3D6E",BLUE,"#0EA5E9",CYAN,ORANGE,GREEN])
            fig_box.update_layout(**CL(340))
            st.markdown('<br><div class="chart-card"><div class="chart-title">Rating Box Plot by Decade</div><div class="chart-sub">Median, spread, and outliers per era</div>',unsafe_allow_html=True)
            st.plotly_chart(fig_box,use_container_width=True,config=dict(displayModeBar=False))
            st.markdown('</div>',unsafe_allow_html=True)

            cc,cd=st.columns(2)
            with cc:
                fig_ua=go.Figure(go.Histogram(x=U['Avg-Rating'],nbinsx=40,
                    marker=dict(color=GREEN,opacity=0.8,line=dict(color=BG,width=0.3))))
                fig_ua.add_vline(x=U['Avg-Rating'].mean(),line_dash="dash",line_color=ORANGE,
                    annotation_text=f"Mean {U['Avg-Rating'].mean():.2f}★",annotation_font_color=ORANGE)
                fig_ua.update_layout(**CL(300))
                st.markdown('<div class="chart-card"><div class="chart-title">User Generosity Distribution</div><div class="chart-sub">Average rating each user gives</div>',unsafe_allow_html=True)
                st.plotly_chart(fig_ua,use_container_width=True,config=dict(displayModeBar=False))
                st.markdown('</div>',unsafe_allow_html=True)

            with cd:
                samp=R.sample(min(2000,len(R)),random_state=42).merge(U,on='User-ID')
                fig_sc=go.Figure(go.Scatter(x=samp['Avg-Rating'],y=samp['Movie-Rating'],mode='markers',
                    marker=dict(color=samp['Movie-Rating'],colorscale=SCALE2,cmin=1,cmax=5,size=4,opacity=0.5),
                    hovertemplate="Avg: %{x:.1f}★<br>Rating: %{y}★<extra></extra>"))
                fig_sc.update_layout(**CL(300))
                st.markdown('<div class="chart-card"><div class="chart-title">Individual Rating vs User Average</div><div class="chart-sub">Does a user\'s typical rating predict each rating?</div>',unsafe_allow_html=True)
                st.plotly_chart(fig_sc,use_container_width=True,config=dict(displayModeBar=False))
                st.markdown('</div>',unsafe_allow_html=True)

        # ════ TAB 4 — CORRELATIONS ════
        with t4:
            st.markdown('<div class="section-header"><div class="section-title">Correlations & Relationships</div><div class="section-line"></div></div>',unsafe_allow_html=True)
            corr_df=stats[['total','avg','std','Year-Of-Release']].rename(
                columns={'total':'Total Ratings','avg':'Avg Rating','std':'Rating Std','Year-Of-Release':'Release Year'}).corr()
            fig_heat=px.imshow(corr_df,text_auto='.3f',
                color_continuous_scale=[[0,'#7B0000'],[0.5,BG2],[1,BLUE]],zmin=-1,zmax=1,aspect='auto')
            fig_heat.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,font=dict(family="Inter",color=TEXT),
                height=320,margin=dict(t=16,b=16,l=16,r=16),
                coloraxis_colorbar=dict(tickfont=dict(color=TEXT),title=dict(text="r",font=dict(color=TEXT))))
            fig_heat.update_traces(textfont=dict(size=14,color="white"))
            st.markdown('<div class="chart-card"><div class="chart-title">🌡️ Feature Correlation Heatmap</div><div class="chart-sub">Pearson r between all numeric features</div>',unsafe_allow_html=True)
            st.plotly_chart(fig_heat,use_container_width=True,config=dict(displayModeBar=False))
            st.markdown('</div>',unsafe_allow_html=True)

            pairs=[(('total','avg','Total Ratings','Avg Rating','Popularity vs Quality',"Do popular movies rate better?"),
                    ('Year-Of-Release','avg','Release Year','Avg Rating','Era vs Quality',"Do newer films rate higher?")),
                   (('avg','std','Avg Rating','Rating Std','Consistency vs Quality',"Are polarising movies rated lower?"),
                    ('Num-Ratings','Avg-Rating','User # Ratings','User Avg Rating','Activity vs Generosity',"Do prolific users rate differently?"))]
            for row_pair in pairs:
                c1,c2=st.columns(2)
                for col,(xc,yc,xl,yl,title,sub) in zip([c1,c2],row_pair):
                    src=stats if xc in stats.columns else U.sample(min(3000,len(U)),random_state=42)
                    fig_c=px.scatter(src,x=xc,y=yc,trendline='ols',color=yc,color_continuous_scale=SCALE2,
                        opacity=0.55,labels={xc:xl,yc:yl},hover_data=['Movie-Title'] if 'Movie-Title' in src.columns else None)
                    fig_c.update_traces(marker=dict(size=5))
                    fig_c.update_layout(**CL(300),coloraxis_showscale=False)
                    r=src[xc].corr(src[yc])
                    with col:
                        st.markdown(f'<div class="chart-card"><div class="chart-title">{title}</div><div class="chart-sub">{sub}</div>',unsafe_allow_html=True)
                        st.plotly_chart(fig_c,use_container_width=True,config=dict(displayModeBar=False))
                        st.markdown(f'<div class="corr-badge">Pearson r <span class="corr-val">{r:+.4f}</span></div>',unsafe_allow_html=True)
                        st.markdown('</div>',unsafe_allow_html=True)

            fig_vio=go.Figure()
            for rat in sorted(R['Movie-Rating'].unique()):
                sub_r=R[R['Movie-Rating']==rat]
                fig_vio.add_trace(go.Violin(y=sub_r['Movie-Rating'],x=[f"{rat}★"]*len(sub_r),name=f"{rat}★",
                    box_visible=True,meanline_visible=True,fillcolor=BLUE,line_color=CYAN,opacity=0.7))
            fig_vio.update_layout(**CL(320))
            st.markdown('<br><div class="chart-card"><div class="chart-title">🎻 Rating Violin Plot</div><div class="chart-sub">Distribution shape for each star rating</div>',unsafe_allow_html=True)
            st.plotly_chart(fig_vio,use_container_width=True,config=dict(displayModeBar=False))
            st.markdown('</div>',unsafe_allow_html=True)

            s3d=stats.sample(min(600,len(stats)),random_state=42)
            fig_3d=go.Figure(go.Scatter3d(x=s3d['Year-Of-Release'],y=s3d['avg'],z=s3d['total'],mode='markers',
                marker=dict(size=4,color=s3d['avg'],colorscale=SCALE2,cmin=1,cmax=5,opacity=0.8,
                            colorbar=dict(title=dict(text="Avg★",font=dict(color=TEXT)),tickfont=dict(color=TEXT))),
                text=s3d['Movie-Title'],
                hovertemplate="<b>%{text}</b><br>Year: %{x}<br>Rating: %{y:.2f}★<br>Total: %{z}<extra></extra>"))
            fig_3d.update_layout(paper_bgcolor=BG,font=dict(family="Inter",color=TEXT),
                scene=dict(xaxis=dict(backgroundcolor=BG2,gridcolor=GRID,color=MUTED,title="Year"),
                           yaxis=dict(backgroundcolor=BG2,gridcolor=GRID,color=MUTED,title="Avg Rating"),
                           zaxis=dict(backgroundcolor=BG2,gridcolor=GRID,color=MUTED,title="Total Ratings")),
                height=480,margin=dict(t=16,b=0,l=0,r=0))
            st.markdown('<br><div class="chart-card"><div class="chart-title">🧊 3D: Year × Avg Rating × Total Ratings</div><div class="chart-sub">Hover to see movie names · Drag to rotate</div>',unsafe_allow_html=True)
            st.plotly_chart(fig_3d,use_container_width=True,config=dict(displayModeBar=False))
            st.markdown('</div>',unsafe_allow_html=True)

        # ════ TAB 5 — EXPLORER ════
        with t5:
            st.markdown('<div class="section-header"><div class="section-title">Data Explorer</div><div class="section-line"></div></div>',unsafe_allow_html=True)
            e1,e2,e3=st.tabs(["🎬  Movies","⭐  Ratings","👥  Users"])
            with e1:
                sc1,sc2=st.columns(2)
                yr=sc1.slider("Year range",int(M['Year-Of-Release'].min()),int(M['Year-Of-Release'].max()),(1990,2005))
                srch=sc2.text_input("Search title",placeholder="e.g. Titanic")
                mf=M[(M['Year-Of-Release']>=yr[0])&(M['Year-Of-Release']<=yr[1])]
                if srch: mf=mf[mf['Movie-Title'].str.contains(srch,case=False,na=False)]
                st.markdown(f"<div style='font-size:0.78rem;color:{MUTED};margin-bottom:8px'>{len(mf):,} results</div>",unsafe_allow_html=True)
                st.dataframe(mf.reset_index(drop=True),use_container_width=True,height=380)
                st.download_button("⬇️ Download filtered movies",mf.to_csv(index=False),"movies_filtered.csv","text/csv")
            with e2:
                rf=st.multiselect("Filter by rating",[1,2,3,4,5],default=[1,2,3,4,5])
                rd=R[R['Movie-Rating'].isin(rf)]
                st.dataframe(rd.sample(min(500,len(rd)),random_state=42).reset_index(drop=True),use_container_width=True,height=380)
                st.download_button("⬇️ Download ratings",R.to_csv(index=False),"ratings.csv","text/csv")
            with e3:
                st.dataframe(U.reset_index(drop=True),use_container_width=True,height=380)
                st.download_button("⬇️ Download users",U.to_csv(index=False),"users.csv","text/csv")

        st.markdown('</div>',unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.error(f"File not found: {e}\n\nCheck the paths in the Configure panel above.")

else:
    st.markdown(f"""
    <div style='text-align:center;padding:80px 20px'>
        <div style='font-size:5rem;margin-bottom:16px'>🎬</div>
        <div style='font-size:2rem;font-weight:800;background:linear-gradient(135deg,#1C6EA4,#00C9FF);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px'>CineAI · Movie Recommender</div>
        <div style='color:#4A6275;font-size:1rem;max-width:440px;margin:0 auto 40px;line-height:1.7'>
            Enter the paths to your three CSV files in the panel above and click <strong style='color:#C5D5E0'>Load</strong> to launch the dashboard.
        </div>
    </div>
    """, unsafe_allow_html=True)

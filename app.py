import os
import sys
import copy
from pathlib import Path
import json
import streamlit.components.v1 as components
import plotly.express as px
import time

ROOT = Path(__file__).parent
NOTEBOOKS_DIR = ROOT / "notebooks"

sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import joblib

from src.simulation import simulate_tournament, run_monte_carlo

st.set_page_config(
    page_title="WC 2026 Simulator",
    page_icon="⚽",
    layout="wide",
)

# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    models_dir = ROOT / "models"
    return {
        "model_h":        joblib.load(models_dir / "home_goals_model.joblib"),
        "model_a":        joblib.load(models_dir / "away_goals_model.joblib"),
        "country_elo":    joblib.load(models_dir / "final_elo.joblib"),
        "features":       joblib.load(models_dir / "model_features.joblib"),
        "team_to_confed": joblib.load(models_dir / "team_to_confed.joblib"),
        "df_groups":      joblib.load(models_dir / "df_groups.joblib"),
        "df_fixtures":    joblib.load(models_dir / "df_groups_fixtures.joblib"),
    }

m = load_models()

# ── Page header ───────────────────────────────────────────────────────────────

st.title("FIFA World Cup 2026 Simulator")
st.caption(
    "Monte Carlo simulation")

tab_mc, tab_sim = st.tabs(["Win Probabilities", "Simulate Tournament"])

# TAB 1 — MONTE CARLO
with tab_mc:
    ctrl_col, _ = st.columns([1, 3])
    with ctrl_col:
        n_sims = st.slider("Number of simulations", 100, 2000, 500, step=100)
        run_mc = st.button("Run Monte Carlo", use_container_width=True)

    if run_mc:
        bar = st.progress(0, text="Starting…")
        start_time = time.time()

        def progress_callback(i, n):
            elapsed = time.time() - start_time
            avg_per_sim = elapsed / i
            remaining = avg_per_sim * (n - i)
            mins, secs = divmod(int(remaining), 60)
            eta = f"{mins}m {secs}s" if mins else f"{secs}s"
            bar.progress(i / n, text=f"{i}/{n} simulations — est. {eta} remaining")
        
        df_mc = run_monte_carlo(
        n_sims,
        m["model_h"], m["model_a"],
        copy.deepcopy(m["country_elo"]),
        m["team_to_confed"],
        m["features"],
        m["df_groups"],
        m["df_fixtures"],
        progress_callback=progress_callback,
    )
        bar.empty()
        st.session_state["mc_results"] = df_mc

    if "mc_results" in st.session_state:
        df = st.session_state["mc_results"]

        st.subheader("Champion probability")
        chart_data = df[df["winner_%"] > 0].sort_values("winner_%", ascending=False)
        fig = px.bar(chart_data, x="team", y="winner_%", color_discrete_sequence=["#1a6530"])
        fig.update_layout(xaxis_title="", yaxis_title="Win %", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("All stages")
        display = df.rename(columns={
            "team":    "Team",
            "r32_%":   "R32 %",
            "r16_%":   "R16 %",
            "qf_%":    "QF %",
            "sf_%":    "SF %",
            "final_%": "Final %",
            "winner_%":"Win %",
        })
        win_max = float(display["Win %"].max())
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "R32 %":   st.column_config.ProgressColumn("R32 %",   min_value=0, max_value=100, format="%.1f%%"),
        "R16 %":   st.column_config.ProgressColumn("R16 %",   min_value=0, max_value=100, format="%.1f%%"),
        "QF %":    st.column_config.ProgressColumn("QF %",    min_value=0, max_value=100, format="%.1f%%"),
        "SF %":    st.column_config.ProgressColumn("SF %",    min_value=0, max_value=100, format="%.1f%%"),
        "Final %": st.column_config.ProgressColumn("Final %", min_value=0, max_value=100, format="%.1f%%"),
        "Win %":   st.column_config.ProgressColumn("Win %",   min_value=0, max_value=win_max, format="%.1f%%"),
            },
        )


# TAB 2 — SINGLE TOURNAMENT
with tab_sim:
    if st.button("Simulate Tournament"):
        with st.spinner("Simulating…"):
            result = simulate_tournament(
                m["model_h"], m["model_a"],
                copy.deepcopy(m["country_elo"]),
                m["team_to_confed"],
                m["features"],
                m["df_groups"],
                m["df_fixtures"],
            )
        st.session_state["sim_result"] = result

    if "sim_result" in st.session_state:
        result = st.session_state["sim_result"]
        summary = result["summary"]
        
        st.success(
            f"🏆 **{summary['winner']}** wins the World Cup!  "
            f"Runner-up: **{summary['runner_up']}**"
        )

        # ── Group Stage ───────────────────────────────────────────────────────
        with st.expander("Group Stage", expanded=True):
            group_tables: pd.DataFrame = result["group_tables"]
            groups = sorted(group_tables["group"].unique())

            all_thirds = set(group_tables[group_tables['group_rank'] == 3]['team'].tolist())
            qualified_thirds = [t for t in summary['r32_teams'] if t in all_thirds]

            cols = st.columns(3)
            for i, gname in enumerate(groups):
                with cols[i % 3]:
                    st.markdown(f"**Group {gname}**")
                    gdf = (
                        group_tables[group_tables["group"] == gname]
                        .sort_values("group_rank")
                        .reset_index(drop=True)
                    )
                    display_gdf = gdf[
                        ["team", "points", "wins", "draws", "losses",
                         "goals_for", "goals_against", "goal_difference"]
                    ].copy()
                    display_gdf.columns = ["Team", "Pts", "W", "D", "L", "GF", "GA", "GD"]
                    
                    def _row_style(row):
                        idx = row.name
                        team = gdf.iloc[idx]['team']
                        if idx < 2:
                            return ["background-color: #008000"] * len(row)
                        if idx == 2 and team in qualified_thirds:
                            return ["background-color: #FFA000"] * len(row)
                        return [""] * len(row)

                    st.dataframe(
                        display_gdf.style.apply(_row_style, axis=1),
                        use_container_width=True,
                        hide_index=True,
                        height=178,
                    )

            st.caption("Green = automatic qualifier (top 2) · Yellow = 3rd-place qualifier")

        # ── Knockout Bracket ──────────────────────────────────────────────────
        with st.expander("Knockout Stage", expanded=True):
            bracket = result["bracket"]
            bracket_data = {}
            for rnd in ['R32', 'R16', 'QF', 'SF', 'Final']:
                rnd_df = bracket[bracket['round'] == rnd]
                bracket_data[rnd] = [
                    {
                        'home': row['home_team'],
                        'away': row['away_team'],
                        'hg': int(row['home_goals']),
                        'ag': int(row['away_goals']),
                        'winner': row['winner'],
                        'pens': row['result_type'] == 'OT/Pens'
                    }
                    for _, row in rnd_df.iterrows()
                ]

            _BRACKET_HTML = """
            <style>
            *{box-sizing:border-box;margin:0;padding:0}
            body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:transparent;overflow:auto}
            #wrap{position:relative}
            .rl{position:absolute;font-size:10px;font-weight:700;text-transform:uppercase;
                letter-spacing:1px;color:#aaa;text-align:center;white-space:nowrap}
            .mc{position:absolute;background:#fff;border:1px solid #e0e0e0;
                border-radius:5px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08)}
            .mc.fin{border:2px solid #1a6530;box-shadow:0 2px 10px rgba(26,101,48,.2)}
            .tr{display:flex;align-items:center;padding:0 8px;gap:4px}
            .tr+.tr{border-top:1px solid #f0f0f0}
            .tn{flex:1;font-size:11.5px;color:#777;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
            .ts{font-size:11.5px;font-weight:700;color:#aaa;min-width:13px;text-align:right}
            .tr.w .tn{color:#1a6530;font-weight:700}
            .tr.w .ts{color:#1a6530}
            .tr.w { background: #f0faf3; }
            .pr{font-size:9.5px;color:#aaa;text-align:center;padding:2px;
                background:#f8f8f8;border-top:1px solid #f0f0f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
            #svg{position:absolute;top:0;left:0;pointer-events:none;overflow:visible}
            </style>
            <div id="wrap"></div>
            <script>
            // R32 data comes in match-number order (73,74,...,88).
            // We reorder to display positions where each adjacent pair feeds the same R16 match.
            // R32[i] refers to the i-th match in the raw data array (0-indexed):
            //   0=m73,1=m74,2=m75,3=m76,4=m77,5=m78,6=m79,7=m80,
            //   8=m81,9=m82,10=m83,11=m84,12=m85,13=m86,14=m87,15=m88
            // R16 raw order: 0=m89,1=m90,2=m91,3=m92,4=m93,5=m94,6=m95,7=m96
            const R32O=[1,4,0,2,10,11,8,9,3,5,6,7,13,15,12,14];
            const R16O=[0,1,4,5,2,3,6,7];

            const rounds=[
            R32O.map(i=>BRACKET_DATA.R32[i]),
            R16O.map(i=>BRACKET_DATA.R16[i]),
            BRACKET_DATA.QF,
            BRACKET_DATA.SF,
            BRACKET_DATA.Final
            ];
            const RNAMES=['Round of 32','Round of 16','Quarter-finals','Semi-finals','Final'];

            // Layout
            const BH=70, CH=50, PH=14, CW=180, CG=34, LH=24, NB=16;
            const TH=LH+NB*BH;
            const TW=5*CW+4*CG;
            const CX=Array.from({length:5},(_,i)=>i*(CW+CG));

            function slH(ri){return NB*BH/rounds[ri].length}
            function yC(ri,si){return LH+(si+.5)*slH(ri)}

            const wrap=document.getElementById('wrap');
            wrap.style.cssText='width:'+TW+'px;height:'+TH+'px;position:relative';

            // Round labels
            RNAMES.forEach((n,ri)=>{
            const el=document.createElement('div');
            el.className='rl';
            el.style.cssText='left:'+CX[ri]+'px;top:4px;width:'+CW+'px';
            el.textContent=n;
            wrap.appendChild(el);
            });

            // Match cards — all absolutely positioned
            rounds.forEach((ms,ri)=>{
            ms.forEach((m,si)=>{
                if(!m)return;
                const yc=yC(ri,si);
                const h=CH+(m.pens?PH:0);
                const hw=m.winner===m.home;
                const rh=CH/2;
                const card=document.createElement('div');
                card.className='mc'+(ri===4?' fin':'');
                card.style.cssText='left:'+CX[ri]+'px;top:'+(yc-h/2)+'px;width:'+CW+'px;height:'+h+'px';
                card.innerHTML=
                '<div class="tr'+(hw?' w':'')+'" style="height:'+rh+'px">'+
                    '<span class="tn">'+m.home+'</span><span class="ts">'+m.hg+'</span></div>'+
                '<div class="tr'+(hw?'':' w')+'" style="height:'+rh+'px">'+
                    '<span class="tn">'+m.away+'</span><span class="ts">'+m.ag+'</span></div>'+
                (m.pens?'<div class="pr">'+m.winner+' wins on penalties</div>':'');
                wrap.appendChild(card);
            });
            });

            // SVG connector lines — drawn last so they sit on top
            const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
            svg.id='svg';svg.setAttribute('width',TW);svg.setAttribute('height',TH);
            wrap.appendChild(svg);

            function conn(ri,si,ri2,si2){
            const x1=CX[ri]+CW, y1=yC(ri,si), x2=CX[ri2], y2=yC(ri2,si2), mx=(x1+x2)/2;
            const p=document.createElementNS('http://www.w3.org/2000/svg','path');
            p.setAttribute('d','M'+x1+','+y1+' L'+mx+','+y1+' L'+mx+','+y2+' L'+x2+','+y2);
            p.setAttribute('stroke','#ddd');p.setAttribute('stroke-width','1.5');p.setAttribute('fill','none');
            svg.appendChild(p);
            }

            // Each round feeds the next: adjacent pairs → single target
            for(let ri=0;ri<4;ri++){
            const nN=rounds[ri+1].length, ppt=rounds[ri].length/nN;
            for(let ni=0;ni<nN;ni++)
                for(let pi=0;pi<ppt;pi++)
                conn(ri,ni*ppt+pi,ri+1,ni);
            }
            </script>
            """
            components.html(
                        f'<script>const BRACKET_DATA={json.dumps(bracket_data)};</script>' + _BRACKET_HTML,
                        height=1180,
                        scrolling=True,
                    )

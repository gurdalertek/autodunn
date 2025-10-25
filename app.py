# app.py — KW + Dunn pipeline + Heatmap + Box/Violin + Directed Network (Plotly)
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import networkx as nx
import plotly.graph_objects as go
# from plotly.io import to_image

from plotly.io import to_image

# ---------------------------------------------------------------------
# Helper: safely display Plotly figures inside Streamlit
# ---------------------------------------------------------------------
def _show_plotly_fig(fig, note_label="plot"):
    import plotly.graph_objects as go
    try:
        if isinstance(fig, go.Figure):
            st.plotly_chart(fig, use_container_width=True)
        elif isinstance(fig, (dict, list)):
            st.plotly_chart(go.Figure(fig), use_container_width=True)
        else:
            st.error(f"{note_label} renderer returned a {type(fig).__name__}, expected a Plotly Figure.")
    except Exception as e:
        st.error(f"Failed to render {note_label}: {e}")

# ---------------------------------------------------------------------
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".matplotlib-cache"))
st.set_page_config(page_title="Kruskal–Wallis & Dunn", layout="wide")
st.title("Kruskal–Wallis & Dunn — End-to-End Analysis")

st.caption("💡 Developed by Dr. Gurdal Ertek, source code under https://github.com/gurdalertek/autodunn")
st.caption("💡 The app runs best when there are 2-20 distinct group label values for the categorical factor, with at least 10 observations for each value. If you have more distinct values, consider reducing the number of distinct values by combining values with few observations under the group label 'Other' ")
st.caption("💡 You can visualize the exported `.dot` or `.svg` graph/network files using Graphviz Viewer, Gephi, yEd, or online: https://dreampuf.github.io/GraphvizOnline/")

st.markdown("""
Upload a **single data file** (`.csv`, `.xls`, or `.xlsx`).  
**Assumptions:** first column = *categorical factor*, second column = *numeric response*, first row = *column titles*.
""")

# ---------- Single uploader (CSV / Excel) ----------
data_file = st.file_uploader("Data file (CSV / XLS / XLSX)", type=["csv", "xls", "xlsx"])

# A tiny wrapper so the rest of the app can keep calling `.getvalue()`
class _MemUpload:
    def __init__(self, b: bytes):
        self._b = b
    def getvalue(self) -> bytes:
        return self._b

def _read_any(data_file):
    """Read CSV or Excel to pandas DataFrame."""
    if data_file is None:
        return None
    name = (data_file.name or "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(data_file)
    elif name.endswith(".xls") or name.endswith(".xlsx"):
        # read first sheet by default
        df = pd.read_excel(data_file)
    else:
        # try CSV as a fallback
        df = pd.read_csv(data_file)
    return df

if not data_file:
    st.stop()

# ---------- Read and normalize to the expected two columns ----------
df_raw = _read_any(data_file)
if df_raw is None or df_raw.shape[1] < 2:
    st.error("The data file must contain at least two columns (factor, response).")
    st.stop()

# Use first two columns, preserve given headers
factor_col   = str(df_raw.columns[0])
response_col = str(df_raw.columns[1])

# Coerce response to numeric, keep original order
df_use = df_raw[[factor_col, response_col]].copy()
df_use[response_col] = pd.to_numeric(df_use[response_col], errors="coerce")

# Build a minimal attributeinfo table the existing pipeline expects
# rows: (numeric/response) and (nominal/factor)
ai = pd.DataFrame({
    "attrname":          [response_col, factor_col],
    "attrtype":          ["numeric",    "nominal"],
    "factororresponse":  ["response",   "factor"],
})

# Convert both to CSV bytes for the existing analyze_all(...) calls
_dataset_bytes   = df_use.to_csv(index=False).encode("utf-8")
_attribute_bytes = ai.to_csv(index=False).encode("utf-8")

# Expose as objects with .getvalue() so the rest of the code remains unchanged
dataset_file   = _MemUpload(_dataset_bytes)
attrinfo_file  = _MemUpload(_attribute_bytes)

# A small heads-up for the user
with st.expander("Detected columns (from your single file)"):
    st.write(f"**Factor (categorical)**: `{factor_col}`")
    st.write(f"**Response (numeric)**: `{response_col}`")
    st.write("Tip: If this isn't correct, reorder/rename the first two columns in your file and re-upload.")


# ---------------- Helpers ----------------
def unique_sorted(series): return sorted(pd.unique(series.astype(str)))

def build_symmetric_matrix(sub):
    if sub.empty: return None
    sub = sub.copy()
    sub["group_i"] = sub["group_i"].astype(str)
    sub["group_j"] = sub["group_j"].astype(str)
    mat = sub.pivot_table(index="group_i", columns="group_j", values="p_adj", aggfunc="min")
    idx = unique_sorted(pd.Index(mat.index).append(pd.Index(mat.columns)))
    mat = mat.reindex(index=idx, columns=idx)
    sym = np.fmin(mat.values, mat.T.values)
    np.fill_diagonal(sym, 1.0)
    return pd.DataFrame(sym, index=mat.index, columns=mat.columns)

def altair_heatmap(df, title):
    long_df = (
        df.reset_index()
          .melt(id_vars=df.index.name or "index", var_name="group_j", value_name="p_adj")
          .rename(columns={df.index.name or "index": "group_i"})
    )

    base = alt.Chart(long_df).mark_rect().encode(
        x=alt.X("group_j:N", sort=unique_sorted(long_df["group_j"]), title="group_j"),
        y=alt.Y("group_i:N", sort=unique_sorted(long_df["group_i"]), title="group_i"),
        color=alt.Color(
            "p_adj:Q",
            scale=alt.Scale(scheme="redblue", domain=[0, 1]),
            title="Adjusted p-value"
        ),
        tooltip=["group_i", "group_j", alt.Tooltip("p_adj:Q", format=".4f")]
    ).properties(title=title, width=600, height=600)

    # --- Adaptive text color ---
    text = base.mark_text(fontSize=9).encode(
        text=alt.Text("p_adj:Q", format=".4f"),
        color=alt.condition(
            "(datum.p_adj < 0.2) || (datum.p_adj > 0.8)",
            alt.value("white"),
            alt.value("black")
        )
    )

    return base + text



# ------------- Analysis (KW + Dunn) -------------
@st.cache_data(show_spinner=True)
def analyze_all(data_csv, attr_csv, run_dunn, alpha):
    from scipy import stats
    import scikit_posthocs as sp

    data = pd.read_csv(io.BytesIO(data_csv))
    ai = pd.read_csv(io.BytesIO(attr_csv))

    ai["attrname"] = ai["attrname"].astype(str)
    ai["attrtype"] = ai["attrtype"].astype(str).str.lower()
    ai["factororresponse"] = ai["factororresponse"].astype(str).str.lower()

    nums = ai.query("attrtype=='numeric' and factororresponse=='response'")["attrname"].tolist()
    noms = ai.query("attrtype=='nominal' and factororresponse=='factor'")["attrname"].tolist()

    kw_rows, dunn_rows = [], []
    for f in noms:
        if f not in data.columns: continue
        for r in nums:
            if r not in data.columns: continue
            sub = data[[r, f]].dropna()
            y = pd.to_numeric(sub[r], errors="coerce")
            g = sub[f].astype("category")
            sub = pd.DataFrame({r: y, f: g}).dropna()
            if sub.empty or g.nunique() < 2: continue
            groups = [sub.loc[sub[f]==lvl, r].values for lvl in g.cat.categories if len(sub.loc[sub[f]==lvl, r])>0]
            if len(groups) < 2: continue
            H, p = stats.kruskal(*groups, nan_policy="omit")
            kw_rows.append(dict(responsevar=r, factorvar=f, chi2=float(H), p=float(p)))
            if run_dunn or p < alpha:
                for adj in ["fdr_bh", "holm-sidak", "bonferroni"]:
                    ph = sp.posthoc_dunn(sub, val_col=r, group_col=f, p_adjust=adj)
                    ph = (ph.rename_axis("group_i").reset_index()
                              .melt(id_vars="group_i", var_name="group_j", value_name="p_adj"))
                    ph = ph.loc[ph["group_i"] != ph["group_j"]]
                    ph["responsevar"], ph["factorvar"], ph["adjustment"] = r, f, adj
                    dunn_rows.append(ph)
    kw_df = pd.DataFrame(kw_rows)
    dunn_df = pd.concat(dunn_rows, ignore_index=True) if dunn_rows else pd.DataFrame(
        columns=["responsevar","factorvar","adjustment","group_i","group_j","p_adj"]
    )
    return kw_df, dunn_df

# --- Settings and run ---
st.markdown("### Analysis Settings")
run_dunn = st.checkbox("Always run Dunn post-hoc", value=True)
alpha = st.number_input("α threshold", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, format="%.4f")

if not dataset_file or not attrinfo_file:
    st.stop()

with st.spinner("Running analysis..."):
    kw_df, dunn_df = analyze_all(dataset_file.getvalue(), attrinfo_file.getvalue(), run_dunn, alpha)
st.success("Done.")

# ---------------- Dashboard ----------------
st.header("Dashboard")
if dunn_df.empty:
    st.warning("No Dunn results generated.")
    st.stop()

csel1, csel2, csel3, csel4 = st.columns(4)
resp = csel1.selectbox("Response", unique_sorted(dunn_df["responsevar"]))
fact = csel2.selectbox("Factor", unique_sorted(dunn_df["factorvar"]))
adj  = csel3.selectbox("Adjustment", unique_sorted(dunn_df["adjustment"]))
a_sig = csel4.number_input("Display α", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, format="%.4f")

sub = dunn_df.query(
    "responsevar == @resp and factorvar == @fact and adjustment == @adj"
)[["group_i","group_j","p_adj"]].dropna()
if sub.empty:
    st.info("No rows for this selection.")
    st.stop()

# --- compute group means for the selected factor/response (for direction) ---
raw_for_means = pd.read_csv(io.BytesIO(dataset_file.getvalue()))
raw_for_means[resp] = pd.to_numeric(raw_for_means[resp], errors="coerce")
means_map = raw_for_means.groupby(fact, dropna=False)[resp].mean().astype(float)

# --- attach numeric means (force float) ---
sub = sub.assign(
    mean_i=pd.to_numeric(sub["group_i"].map(means_map), errors="coerce").astype(float),
    mean_j=pd.to_numeric(sub["group_j"].map(means_map), errors="coerce").astype(float)
)


# Heatmap
st.subheader("Heatmap (Adjusted p-values)")
mat = build_symmetric_matrix(sub)
if mat is not None:
    st.altair_chart(altair_heatmap(mat, f"{resp} ~ {fact} • {adj}"), use_container_width=True)

# Significant pairs
# --- Significant Comparisons filtered by mean direction ---
st.subheader("Significant Comparisons (mean_i > mean_j and p_adj < α)")

# Ensure means are numeric
sub["mean_i"] = pd.to_numeric(sub["mean_i"], errors="coerce").astype(float)
sub["mean_j"] = pd.to_numeric(sub["mean_j"], errors="coerce").astype(float)

# Keep only valid, directional, and significant pairs
sig = (
    sub.dropna(subset=["mean_i", "mean_j"])
       .loc[(sub["p_adj"] < a_sig) & (sub["mean_i"] > sub["mean_j"])]
       .sort_values("p_adj")
       .reset_index(drop=True)
)

# Display table with means for clarity
if sig.empty:
    st.info("No significant comparisons where mean_i > mean_j.")
else:
    st.dataframe(sig[["group_i", "group_j", "mean_i", "mean_j", "p_adj"]], width="stretch")



# ---------------- Box / Violin (before network) ----------------
st.markdown("---")
st.header("Box / Violin from Raw Dataset")
raw_df = pd.read_csv(io.BytesIO(dataset_file.getvalue()))
ai_df  = pd.read_csv(io.BytesIO(attrinfo_file.getvalue()))

resp_candidates = ai_df.query("attrtype.str.lower()=='numeric' and factororresponse.str.lower()=='response'", engine="python")["attrname"].astype(str).tolist()
fact_candidates = ai_df.query("attrtype.str.lower()=='nominal' and factororresponse.str.lower()=='factor'", engine="python")["attrname"].astype(str).tolist()

with st.form("box_violin_form", clear_on_submit=False):
    colA, colB, colC = st.columns(3)
    resp_col = colA.selectbox("Response (numeric)", options=resp_candidates or list(raw_df.columns),
                              index=(resp_candidates or list(raw_df.columns)).index(resp)
                              if resp in (resp_candidates or list(raw_df.columns)) else 0)
    fact_col = colB.selectbox("Factor (categorical)", options=fact_candidates or list(raw_df.columns),
                              index=(fact_candidates or list(raw_df.columns)).index(fact)
                              if fact in (fact_candidates or list(raw_df.columns)) else 0)
    order_by = colC.selectbox("Order groups by", options=["median","mean","none"], index=0)
    submitted_bv = st.form_submit_button("Draw Box/Violin Plot")

if submitted_bv:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = raw_df.copy()
        df[resp_col] = pd.to_numeric(df[resp_col], errors="coerce")
        df = df[[resp_col, fact_col]].dropna()

        if df.empty:
            st.warning("No data after coercion / NA removal.")
        else:
            order = None
            if order_by == "median":
                order = df.groupby(fact_col)[resp_col].median().sort_values().index.tolist()
            elif order_by == "mean":
                order = df.groupby(fact_col)[resp_col].mean().sort_values().index.tolist()

            # --- Violin plot ---
            fig1, ax1 = plt.subplots(figsize=(10, 5))
  
            sns.violinplot(
                data=df,
                x=fact_col,
                y=resp_col,
                order=order,
                inner="box",
                ax=ax1,
                color="#FDBA74"   # light orange for all violins
            )
            ax1.set_title(f"Violin: {resp_col} ~ {fact_col}")
            ax1.tick_params(axis="x", rotation=45)
            st.pyplot(fig1, clear_figure=True)

            # --- Box plot ---
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.boxplot(
                data=df,
                x=fact_col,
                y=resp_col,
                order=order,
                ax=ax2,
                color="#FDBA74"   # light orange for all boxes
            )
            ax2.set_title(f"Box: {resp_col} ~ {fact_col}")
            ax2.tick_params(axis="x", rotation=45)
            st.pyplot(fig2, clear_figure=True)

    except Exception as e:
        st.error(f"Plot error: {e}")

# ---------------- Directed Significance Network (optional; at the end) ----------------
st.markdown("---")

with st.form("net_form", clear_on_submit=False):
    st.header("Optional: Directed Significance Network")
    
    colN1, colN2, colN3 = st.columns(3)
    
    # Define orientation variable (previously 'layout_dir')
    orientation = colN1.selectbox(
        "Orientation",
        ["Left → Right", "Top → Bottom"],
        index=0
    )
    
    edge_alpha = colN2.number_input(
        "Edge α (p_adj < α)",
        min_value=0.0001, max_value=0.5,
        value=float(a_sig),
        step=0.01,
        format="%.4f"
    )
    
    show_lbl = colN3.checkbox("Show edge labels", value=True)
    
    draw_net = st.form_submit_button("Draw Directed Network")

# ---------- Layout + Directed Network Builder ----------
def hierarchical_layout_directed(edges, direction="LR"):
    """
    Create hierarchical layout (directed), spacing nodes evenly and repelling slightly
    to reduce overlap.
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)
    UG = G.to_undirected()

    if UG.number_of_nodes() == 0:
        return {}

    core = nx.core_number(UG)
    levels = {}
    for n, c in core.items():
        levels.setdefault(c, []).append(n)
    level_keys = sorted(levels.keys(), reverse=True)

    pos = {}
    x_gap, y_gap = 1.0, 1.0
    for li, k in enumerate(level_keys):
        row = sorted(levels[k], key=lambda n:(-UG.degree(n), str(n)))
        m = len(row)
        for j, n in enumerate(row):
            if direction == "LR":
                pos[n] = (li * x_gap, (j - (m-1)/2) * y_gap)
            else:
                pos[n] = ((j - (m-1)/2) * x_gap, li * y_gap)

    # light repulsion to reduce overlap within same level
    def repel_once(nodes, horiz=True, min_dist=0.3, step=0.05):
        coords = [pos[n][0] if horiz else pos[n][1] for n in nodes]
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                ni, nj = nodes[i], nodes[j]
                xi, yi = pos[ni]
                xj, yj = pos[nj]
                dx, dy = xj - xi, yj - yi
                dist = np.hypot(dx, dy)
                if dist < min_dist and dist > 1e-9:
                    push = step * (min_dist - dist) / dist
                    pos[ni] = (xi - push*dx, yi - push*dy)
                    pos[nj] = (xj + push*dx, yj + push*dy)

    for k in level_keys:
        nodes = levels[k]
        for _ in range(3):
            repel_once(nodes, horiz=(direction == "LR"))

    # normalize positions
    xs = np.array([p[0] for p in pos.values()], float)
    ys = np.array([p[1] for p in pos.values()], float)
    if xs.ptp() > 0: xs = (xs - xs.mean()) / (xs.ptp() / 2)
    if ys.ptp() > 0: ys = (ys - ys.mean()) / (ys.ptp() / 2)
    for (node, _), x, y in zip(pos.items(), xs, ys):
        pos[node] = (float(x), float(y))
    return pos



from collections import deque

def hierarchical_layout_barycentric(DG, direction="LR", layer_gap=1.2, node_gap=1.0, passes=4):
    """
    Hierarchical layout for a directed graph using BFS layers + barycentric
    reordering (Sugiyama-style). Returns a {node: (x, y)} dict.
    - direction: "LR" (left→right) or "TB" (top→bottom)
    """
    # ----- 1) assign layers (sources→sinks) via BFS on directed edges
    sources = [n for n in DG.nodes() if DG.in_degree(n) == 0]
    if not sources:
        # pick a good pseudo-source if the graph has cycles
        sources = sorted(DG.nodes(), key=lambda n: (DG.out_degree(n) - DG.in_degree(n)), reverse=True)[:1]

    dist = {n: np.inf for n in DG.nodes()}
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)

    while q:
        u = q.popleft()
        for v in DG.successors(u):
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)

    maxd = int(max([d for d in dist.values() if np.isfinite(d)] + [0]))
    layers = [[] for _ in range(maxd + 1)]
    for n, d in dist.items():
        k = int(d if np.isfinite(d) else maxd)
        layers[k].append(n)

    # initial ordering by out-degree (stable)
    for k, L in enumerate(layers):
        layers[k] = sorted(L, key=lambda n: (-DG.out_degree(n), str(n)))

    # ----- 2) barycentric reordering passes to reduce crossings
    def _bc_order(curr, neigh_index):
        def center(n):
            idxs = [neigh_index[m] for m in curr_neigh(n) if m in neigh_index]
            return np.mean(idxs) if idxs else 1e9  # push isolated to the end
        curr.sort(key=center)

    for _ in range(max(0, passes)):
        # down pass: order each layer by barycenter of predecessors (above)
        for k in range(1, len(layers)):
            prev_index = {n: i for i, n in enumerate(layers[k - 1])}
            curr_neigh = lambda n: list(DG.predecessors(n))
            _bc_order(layers[k], prev_index)

        # up pass: order each layer by barycenter of successors (below)
        for k in range(len(layers) - 2, -1, -1):
            next_index = {n: i for i, n in enumerate(layers[k + 1])}
            curr_neigh = lambda n: list(DG.successors(n))
            _bc_order(layers[k], next_index)

    # ----- 3) assign coordinates
    pos = {}
    for li, L in enumerate(layers):
        m = max(1, len(L))
        for j, n in enumerate(L):
            sec = (j - (m - 1) / 2) * node_gap
            pri = li * layer_gap
            pos[n] = (pri, sec) if direction == "LR" else (sec, pri)

    # normalize to a pleasant range (NumPy 2.x–compatible)
    xs = np.array([p[0] for p in pos.values()], float)
    ys = np.array([p[1] for p in pos.values()], float)

    xs_range = np.ptp(xs)  # instead of xs.ptp()
    ys_range = np.ptp(ys)  # instead of ys.ptp()

    if xs_range > 0:
        xs = (xs - xs.mean()) / (xs_range / 2)
    if ys_range > 0:
        ys = (ys - ys.mean()) / (ys_range / 2)

    for (node, _), x, y in zip(pos.items(), xs, ys):
        pos[node] = (float(x), float(y))

    return pos


def build_directed_network_figure(sub_df, alpha, direction="LR", show_labels=True):
    """
    Build a directed significance network figure from sub_df having columns:
    ['group_i','group_j','p_adj'] already filtered to the selected response/factor/adjustment.
    Only edges with p_adj < alpha are drawn.
    """
    # ---- guard & sanitize
    if sub_df is None or sub_df.empty:
        return go.Figure().update_layout(
            title="Directed Significance Network",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
        )

    sdf = sub_df.copy()
    sdf["p_adj"] = pd.to_numeric(sdf["p_adj"], errors="coerce")
    sdf = sdf.dropna(subset=["group_i", "group_j", "p_adj"])
    sdf = sdf.loc[sdf["p_adj"] < float(alpha)]

    if sdf.empty:
        return go.Figure().update_layout(
            title="Directed Significance Network (no edges at current α)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
        )

    # ---- build the directed graph DG
    edges = [(str(r["group_i"]), str(r["group_j"]), float(r["p_adj"])) for _, r in sdf.iterrows()]
    DG = nx.DiGraph()
    for u, v, p in edges:
        if DG.has_edge(u, v):
            DG[u][v]["p"] = min(DG[u][v]["p"], p)  # keep the most significant
        else:
            DG.add_edge(u, v, p=p)

    # ---- hierarchical positions (Sugiyama-style helper you added earlier)
    pos = hierarchical_layout_barycentric(
        DG, direction=direction, layer_gap=1.2, node_gap=1.0, passes=4
    )

    nodes = list(DG.nodes())
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]

    # ---- node trace (labels above nodes)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=nodes,
        textposition="top center",
        textfont=dict(size=12, color="#333"),
        marker=dict(size=24, color="#E6F2FF", line=dict(width=1, color="#6AAFE6")),
        hoverinfo="text",
        name="nodes",
    )

    # ---- edge lines (light gray)
    line_x, line_y = [], []
    pvals, end_pts = [], []
    for u, v, data in DG.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        line_x += [x0, x1, None]
        line_y += [y0, y1, None]
        pvals.append(float(data.get("p", 1.0)))
        end_pts.append((x0, y0, x1, y1))

    edge_trace = go.Scatter(
        x=line_x, y=line_y,
        mode="lines",
        line=dict(width=2, color="rgba(180,180,180,0.4)"),
        hoverinfo="skip",
        name="edges",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Directed Significance Network",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
        dragmode="pan",
    )

    # ---- arrowheads + optional p-value labels
    # width by significance: -log10(p) scaled to [1..5]
    if pvals:
        w_raw = [-np.log10(max(p, 1e-300)) for p in pvals]
        wmin, wmax = min(w_raw), max(w_raw)
        widths = [1.0 + 4.0 * ((w - wmin) / (wmax - wmin) if wmax > wmin else 0.5) for w in w_raw]
    else:
        widths = []

    for (x0, y0, x1, y1), p, width in zip(end_pts, pvals, widths):
        # smaller arrowhead per your preference
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2,   # <= smaller tips
            arrowwidth=2, arrowcolor="rgba(180,180,180,0.6)",
            standoff=12,
        ) # arrowwidth=float(width)
        if show_labels:
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            fig.add_trace(go.Scatter(
                x=[mx], y=[my],
                text=[f"{p:.4f}"],
                mode="text",
                textfont=dict(size=10, color="#333"),
                hoverinfo="skip",
                showlegend=False,
            ))

    return fig


# Optional alias for backward compatibility
build_curved_directed_network = build_directed_network_figure

def make_dot_directed(sub_df, alpha):
    """Export DOT format (directed)."""
    sdf = sub_df.dropna().copy()
    sdf = sdf[sdf["p_adj"].astype(float) < alpha]
    if sdf.empty:
        return "digraph G { label='No significant edges'; }"
    lines = ["digraph G {"]
    for _, r in sdf.iterrows():
        a, b, p = str(r["group_i"]), str(r["group_j"]), float(r["p_adj"])
        lines.append(f'  "{a}" -> "{b}" [label="{p:.4f}"];')
    lines.append("}")
    return "\n".join(lines)




# --- Draw network (replace your current block with this) ---
if draw_net:
    # Resolve orientation safely
    ori_str = locals().get("orientation", "Left → Right")
    orient = "LR" if isinstance(ori_str, str) and ori_str.startswith("Left") else "TB"

    # Ensure means are numeric before comparisons
    sub["mean_i"] = pd.to_numeric(sub.get("mean_i"), errors="coerce")
    sub["mean_j"] = pd.to_numeric(sub.get("mean_j"), errors="coerce")

    # Keep only pairs where mean_i > mean_j and p_adj < α
    sub_net = (
        sub.dropna(subset=["mean_i", "mean_j"])
           .loc[(sub["p_adj"] < edge_alpha) & (sub["mean_i"] > sub["mean_j"])]
           .copy()
    )

    if sub_net.empty:
        st.info("No edges to draw at current settings (mean_i > mean_j and p_adj < α).")
        # DOT export (empty graph notice provided by make_dot_directed)
        dot_src = make_dot_directed(sub_net, edge_alpha)
        st.download_button(
            "⬇️ Download network as DOT",
            data=dot_src.encode("utf-8"),
            file_name=f"network_{resp}_{fact}_{adj}.dot",
            mime="text/vnd.graphviz"
        )
    else:
        with st.spinner("Rendering directed network..."):
            fig = build_directed_network_figure(sub_net, edge_alpha, direction=orient, show_labels=show_lbl)
            _show_plotly_fig(fig, note_label="directed network")

        # --- Interactive HTML export (no kaleido needed) ---
        html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
        st.download_button(
            "⬇️ Download network as HTML (interactive)",
            data=html_bytes,
            file_name=f"network_{resp}_{fact}_{adj}.html",
            mime="text/html",
        )


    # DOT export (matches the same filtered edges)
    if "make_dot_directed" in globals():
        dot_src = make_dot_directed(sub_net, edge_alpha)
    else:
        dot_src = make_dot_directed(sub_net, edge_alpha)
    
    st.download_button(
        "⬇️ Download network as DOT",
        data=dot_src.encode("utf-8"),
        file_name=f"network_{resp}_{fact}_{adj}.dot",
        mime="text/vnd.graphviz",
    )


    st.caption("💡 Only edges where the source group’s mean > target group’s mean are shown.")
    st.caption("💡 You can visualize `.dot` or `.svg` files using Graphviz Viewer, Gephi, yEd, or online: https://dreampuf.github.io/GraphvizOnline/")



# ---------------- Arc Diagram (Community) — node-colored arcs, no fill ----------------
st.markdown("---")
st.header("Arc Diagram (Node-colored arcs) — overrides the directed network")

with st.form("arc_comm_form", clear_on_submit=False):
    colA1, colA2, colA3, colA4 = st.columns([1,1,1,1])
    arc_alpha = colA1.number_input(
        "Arc α (keep pairs with p_adj < α)",
        min_value=0.0001, max_value=0.5, value=float(a_sig) if 'a_sig' in globals() else 0.05,
        step=0.01, format="%.4f"
    )
    curve_scale = colA2.slider("Arc curvature", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
    show_node_labels = colA3.checkbox("Show node labels", value=True)
    use_sig_width = colA4.checkbox("Thicker arcs for stronger significance", value=False)
    draw_arc = st.form_submit_button("Draw Arc Diagram")

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def _build_sig_edges_for_arc(sub_df: pd.DataFrame, alpha_thr: float):
    """
    Expects Dunn-style rows with: group_i, group_j, p_adj, mean_i, mean_j.
    Keeps significant directional pairs mean_i > mean_j.
    Returns: nodes_sorted (by mean desc), edges [(u,v,p)], means {node:mean}.
    """
    if sub_df is None or sub_df.empty:
        return [], [], {}

    df = sub_df.copy()
    for c in ["p_adj", "mean_i", "mean_j"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    mask = (df["p_adj"] < alpha_thr) & (df["mean_i"] > df["mean_j"])
    s = df.loc[mask].dropna(subset=["group_i", "group_j", "p_adj", "mean_i", "mean_j"]).copy()

    # Mean per node (average if appears multiple times)
    means_acc, counts = {}, {}
    for _, r in s.iterrows():
        gi, gj = str(r["group_i"]), str(r["group_j"])
        mi, mj = _safe_float(r["mean_i"]), _safe_float(r["mean_j"])
        if not np.isnan(mi):
            means_acc[gi] = means_acc.get(gi, 0.0) + mi
            counts[gi] = counts.get(gi, 0) + 1
        if not np.isnan(mj):
            means_acc[gj] = means_acc.get(gj, 0.0) + mj
            counts[gj] = counts.get(gj, 0) + 1
    means = {n: means_acc[n] / counts[n] for n in means_acc.keys()}

    # Fallback to include nodes (even if no surviving edges)
    if s.empty and "group_i" in df.columns and "group_j" in df.columns:
        for _, r in df.dropna(subset=["group_i"]).iterrows():
            means.setdefault(str(r["group_i"]), np.nan)
        for _, r in df.dropna(subset=["group_j"]).iterrows():
            means.setdefault(str(r["group_j"]), np.nan)

    nodes_sorted = sorted(
        means.keys(),
        key=lambda n: (np.isnan(means[n]), -means[n] if not np.isnan(means[n]) else 0, n)
    )

    # Unique edges (u,v)
    seen, edges = set(), []
    for _, r in s.iterrows():
        u, v = str(r["group_i"]), str(r["group_j"])
        p = _safe_float(r["p_adj"])
        if (u, v) not in seen:
            seen.add((u, v))
            edges.append((u, v, p))
    return nodes_sorted, edges, means

def _arc_points(x0, x1, height):
    import numpy as _np
    if x1 < x0:
        x0, x1 = x1, x0
    t = _np.linspace(0.0, 1.0, 40)
    xs = x0 + (x1 - x0) * t
    ys = height * np.sin(np.pi * t)
    return xs, ys

def _make_arc_figure(nodes_sorted, edges, means, curve_scale=1.0, show_labels=True, use_sig_width=True):
    import plotly.graph_objects as go
    import plotly.express as px

    x_pos = {n: i for i, n in enumerate(nodes_sorted)}
    baseline_y = 0.0

    # Outgoing degree per node
    outdeg = {n: 0 for n in nodes_sorted}
    for (u, _v, _p) in edges:
        if u in outdeg:
            outdeg[u] += 1

    # Color policy:
    # - Nodes with outdeg>0: assign unique color from a qualitative palette.
    # - Nodes with outdeg==0: gray.
    palette = px.colors.qualitative.Plotly  # uses default Plotly qualitative set
    active_nodes = [n for n in nodes_sorted if outdeg.get(n, 0) > 0]
    color_map_active = {n: palette[i % len(palette)] for i, n in enumerate(active_nodes)}
    GRAY = "#A0A0A0"
    node_line_color = {n: (color_map_active[n] if outdeg.get(n, 0) > 0 else GRAY) for n in nodes_sorted}

    # Node trace — no fill, outline only
    node_x = [x_pos[n] for n in nodes_sorted]
    node_y = [baseline_y] * len(nodes_sorted)
    node_text = [f"{n} ({means.get(n, float('nan')):.3g})" for n in nodes_sorted]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",  # text handled via annotations for rotation
        marker=dict(
            size=12,
            color="rgba(0,0,0,0)",           # transparent fill
            line=dict(
                width=2,
                color=[node_line_color[n] for n in nodes_sorted]  # outline colored per policy
            )
        ),
        hovertext=node_text,
        hoverinfo="text",
        name="nodes"
    )

    # Edge widths
    if use_sig_width and len(edges) > 0:
        widths = []
        for (_u, _v, p) in edges:
            if p is None or p <= 0:
                widths.append(6.0)
            else:
                widths.append(float(np.clip(-np.log10(p), 1.0, 6.0)))
    else:
        widths = [1.5] * len(edges)

    # Edge arcs — colored by source node line color
    edge_traces = []
    for k, (u, v, p) in enumerate(edges):
        xi, xj = x_pos.get(u), x_pos.get(v)
        if xi is None or xj is None or xi == xj:
            continue
        span = abs(xj - xi)
        xs, ys = _arc_points(xi, xj, height=curve_scale * 0.3 * max(1, span))
        edge_traces.append(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(width=widths[k], color=node_line_color.get(u, GRAY)),
                hoverinfo="text",
                text=[f"{u} → {v}<br>p_adj={p:.4g}" if p is not None else f"{u} → {v}"] * len(xs),
                name=f"{u}→{v}"
            )
        )

    fig = go.Figure(edge_traces + [node_trace])
    
    # Rotated node labels as annotations (works on all Plotly versions)
    if show_labels:
        for n in nodes_sorted:
            fig.add_annotation(
                x=x_pos[n],
                y=baseline_y,
                text=n,
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                textangle=-45,   # rotation like violin plot x-labels
                yshift=10        # little lift above the node marker
            )    
    
    fig.update_layout(
        height=480,
        margin=dict(l=40, r=20, t=40, b=60),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title=f"Arc Diagram — α={arc_alpha:g}"
    )
    return fig

def _to_dot(nodes_sorted, edges):
    lines = ["digraph G {", '  rankdir=LR;']
    lines += [f'  "{n}" [shape=ellipse];' for n in nodes_sorted]
    for u, v, _p in edges:
        lines.append(f'  "{u}" -> "{v}";')
    lines.append("}")
    return "\n".join(lines)

if draw_arc:
    try:
        nodes_sorted, edges, means = _build_sig_edges_for_arc(sub, arc_alpha)
        if len(nodes_sorted) == 0:
            st.info("No nodes to draw for the selected α.")
        else:
            fig_arc = _make_arc_figure(
                nodes_sorted, edges, means,
                curve_scale=curve_scale,
                show_labels=show_node_labels,
                use_sig_width=use_sig_width
            )
            _show_plotly_fig(fig_arc, note_label="Arc Diagram")

            dot_text = _to_dot(nodes_sorted, edges)
            st.download_button(
                "⬇️ Download DOT for Arc Diagram",
                data=dot_text.encode("utf-8"),
                file_name="arc_diagram_comm.dot",
                mime="text/vnd.graphviz"
            )
            st.caption("Nodes: outline only (no fill). Nodes without outgoing arcs are gray; each source colors its own arcs.")
    except Exception as e:
        st.error(f"Arc diagram error: {e}")

st.caption("💡 Reference for interpreting the arc diagram: https://doi.org/10.1109/aiccsa47632.2019.9035354")


# ---------------- Diffogram (Dunn-based; uses existing results only) ----------------
st.markdown("---")
st.header("Diffogram — From Dunn Results (uses current Adjustment & α)")

with st.form("diffogram_dunn_form", clear_on_submit=False):
    c1, c2 = st.columns([1,1])
    show_ticks = c1.checkbox("Show mean reference ticks/lines", value=True)
    only_sig   = c2.checkbox("Draw only significant pairs", value=False)  # NEW
    draw_diff  = st.form_submit_button("Draw Diffogram")

def _pairs_from_dunn(sub_df: pd.DataFrame, means_map: pd.Series) -> list:
    """
    Build unordered pairs (i,j) with:
      - mi, mj from means_map
      - p = min p_adj for (i,j) and (j,i) from Dunn table
    Returns list of dicts: {'i','j','mi','mj','p'}
    """
    if sub_df is None or sub_df.empty:
        return []

    s = sub_df.copy()
    s["group_i"] = s["group_i"].astype(str)
    s["group_j"] = s["group_j"].astype(str)
    s["p_adj"]   = pd.to_numeric(s["p_adj"], errors="coerce")

    # make symmetric min p for each unordered pair
    s["a"] = np.where(s["group_i"] < s["group_j"], s["group_i"], s["group_j"])
    s["b"] = np.where(s["group_i"] < s["group_j"], s["group_j"], s["group_i"])
    sym = (s.groupby(["a","b"], as_index=False)["p_adj"].min()
             .rename(columns={"a":"i","b":"j","p_adj":"p"}))

    rows = []
    for _, r in sym.iterrows():
        i, j = str(r["i"]), str(r["j"])
        mi = float(pd.to_numeric(means_map.get(i), errors="coerce"))
        mj = float(pd.to_numeric(means_map.get(j), errors="coerce"))
        if np.isnan(mi) or np.isnan(mj):
            continue
        p  = float(pd.to_numeric(r["p"], errors="coerce"))
        rows.append(dict(i=i, j=j, mi=mi, mj=mj, p=p))
    return rows

def _segment_endpoints(mi, mj, ensure_no_cross: bool):
    """
    Build a short segment with slope -1 centered at (mi, mj).
    Crossing rule for y=x:
      crosses iff |mi - mj| <= M
    We set M just below/above |d| to guarantee cross/no-cross behavior.
    """
    d = abs(mi - mj)
    eps = max(1e-12, 1e-3 * (1.0 + d))  # small numeric cushion
    if ensure_no_cross:
        M = max(1e-6, d - eps)           # so |d| > M  ⇒ no cross (significant)
    else:
        M = d + eps                      # so |d| <= M ⇒ cross (not significant)
    dx = M / np.sqrt(2.0)
    x0, y0 = mi - dx, mj + dx
    x1, y1 = mi + dx, mj - dx
    return (x0, y0, x1, y1), M

def _build_diffogram_from_dunn(sub_df, means_map, alpha_display, show_ticks=True, only_sig=False):
    import plotly.graph_objects as go

    pairs = _pairs_from_dunn(sub_df, means_map)
    if not pairs:
        return go.Figure().update_layout(
            title="Diffogram (no pairs found for current selection)",
            xaxis=dict(visible=False), yaxis=dict(visible=False)
        )

    # collect all means to set axes and optional guides
    all_means = np.array(list({*(m for m in means_map.dropna().astype(float).values)}), float)
    mn, mx = float(np.min(all_means)), float(np.max(all_means))
    pad = 0.05 * (mx - mn if mx > mn else 1.0)
    lo, hi = mn - pad, mx + pad

    fig = go.Figure()

    # diagonal y=x
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(width=1.5, dash="dash", color="#555"),
        hoverinfo="skip", name="y = x (equal means)"
    ))

    # optional reference guides at each group mean
# optional reference guides at each group mean (labels via annotations to allow rotation safely)
if show_ticks:
    for g, m in sorted(means_map.dropna().items(), key=lambda kv: kv[1]):
        m = float(m)
        # guide lines
        fig.add_shape(type="line", x0=m, x1=m, y0=lo, y1=hi,
                      line=dict(color="rgba(120,120,120,0.25)", width=1))
        fig.add_shape(type="line", x0=lo, x1=hi, y0=m, y1=m,
                      line=dict(color="rgba(120,120,120,0.25)", width=1))

        # x-axis label (bottom) — horizontal
        fig.add_annotation(
            x=m, y=lo,
            text=str(g),
            showarrow=False,
            xanchor="center", yanchor="top",
            font=dict(size=11, color="#444"),
            yshift=-6  # nudge below the axis
        )

        # y-axis label (left) — vertical
        fig.add_annotation(
            x=lo, y=m,
            text=str(g),
            showarrow=False,
            xanchor="right", yanchor="middle",
            textangle=-90,                   # rotate safely via annotation
            font=dict(size=11, color="#444"),
            xshift=-6                        # nudge left of the axis
        )

    # draw segments: color by significance under current α (a_sig)
    sig_color  = "#2A9D8F"   # teal-green
    nsig_color = "#A0A0A0"   # gray
    sig_traces, nsig_traces = [], []

    for row in pairs:
        i, j, mi, mj, p = row["i"], row["j"], row["mi"], row["mj"], row["p"]
        significant = (p < float(alpha_display))
        if only_sig and not significant:
            continue

        (x0, y0, x1, y1), _M = _segment_endpoints(mi, mj, ensure_no_cross=significant)
        txt = f"{i} vs {j}: diff={mi-mj:.4g} • p_adj={p:.4g} • {'✔ significant' if significant else '✖ not significant'}"
        color = sig_color if significant else nsig_color
        trace = go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=3, color=color),
            hoverinfo="text", text=[txt, txt],
            name="Significant" if significant else "Not Significant",
            showlegend=True
        )
        (sig_traces if significant else nsig_traces).append(trace)

    # Add legend entries once (skip gray legend when only significant pairs are shown)
    if nsig_traces and not only_sig:
        fig.add_trace(nsig_traces[0])
        for tr in nsig_traces[1:]:
            tr.showlegend = False
            fig.add_trace(tr)
    if sig_traces:
        fig.add_trace(sig_traces[0])
        for tr in sig_traces[1:]:
            tr.showlegend = False
            fig.add_trace(tr)

    # mean–mean centers (optional faint points)
    centers_x, centers_y, centers_txt = [], [], []
    for row in pairs:
        i, j, mi, mj = row["i"], row["j"], row["mi"], row["mj"]
        centers_x.append(mi); centers_y.append(mj)
        centers_txt.append(f"({i},{j}) = ({mi:.4g},{mj:.4g})")
    fig.add_trace(go.Scatter(
        x=centers_x, y=centers_y, mode="markers",
        marker=dict(size=4, color="rgba(30,30,30,0.35)"),
        hovertext=centers_txt, hoverinfo="text", showlegend=False
    ))

    # Set square layout (height = width)
    fig.update_layout(
    autosize=False,
    width=700,
    height=1400,  # roughly twice as tall
    margin=dict(l=40, r=20, t=60, b=70),

        xaxis=dict(title=f"{resp} mean for {fact} (x-axis group)", range=[lo, hi], zeroline=False),
        yaxis=dict(title=f"{resp} mean for {fact} (y-axis group)", range=[lo, hi], zeroline=False),
        title=f"Diffogram — Dunn ({adj}) @ α={float(a_sig):g}  (rule: segment crosses y=x ⇒ NOT significant)",
        legend=dict(
            title="Line color meaning:",
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(180,180,180,0.5)",
            borderwidth=1
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

if draw_diff:
    try:
        # Use the SAME selection (resp, fact, adj) and a_sig already defined above.
        fig_diff = _build_diffogram_from_dunn(
            sub_df=sub,                # current Dunn results slice
            means_map=means_map,       # already computed group means (from your app)
            alpha_display=a_sig,
            show_ticks=show_ticks,
            only_sig=only_sig          # NEW
        )
        _show_plotly_fig(fig_diff, note_label="diffogram (Dunn)")

        # Download (interactive HTML)
        html_bytes = fig_diff.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
        st.download_button(
            "⬇️ Download Diffogram (Dunn) as HTML",
            data=html_bytes,
            file_name=f"diffogram_dunn_{resp}_{fact}_{adj}.html",
            mime="text/html",
        )
    except Exception as e:
        st.error(f"Diffogram error: {e}")

st.caption("💡 Reference for interpreting the diffogram: https://blogs.sas.com/content/iml/2017/10/18/diffogram-multiple-comparisons-sas.html")




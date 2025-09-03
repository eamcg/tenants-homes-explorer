# app.py — Tenants × Homes Explorer (CSV+GeoJSON)
# Single-scenario UX with a "Run analysis" button; custom filters only when Custom is chosen

import io, json
from typing import Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import streamlit as st
import matplotlib.pyplot as plt

# --------------- Page setup ---------------
st.set_page_config(page_title="Tenants × Homes Explorer", layout="wide")
st.markdown("""
<style>
.dataframe td, .dataframe th { padding: 6px 8px !important; }
.block-container { padding-top: 1.2rem; }
.small-muted { color:#6b7280; font-size:0.9rem; }
.card { padding: 0.9rem 1rem; border:1px solid #e5e7eb; border-radius:12px; background:#fff; }
.section { padding: 1rem; border:1px solid #e5e7eb; border-radius:12px; margin-bottom: 1rem; background:#fafafa; }
h1, h2, h3 { margin-bottom: 0.25rem; }
</style>
""", unsafe_allow_html=True)

st.title("Tenants × Homes Explorer")
st.markdown(
    "<div class='small-muted'>See how many tenants each home would get under a single matching rule. "
    "All scenarios require a <b>spatial match</b> (home inside tenant search area); the chosen filters apply on top.</div>",
    unsafe_allow_html=True,
)

# ============================================================
# Helpers (preprocessing & core logic)
# ============================================================
def geojson_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "geojson" in c.lower():
            return c
    raise ValueError("No GeoJSON column found. SQL export should alias geometry as e.g. `geom_geojson`.")

def df_geojson_to_gdf(df: pd.DataFrame, geojson_colname: str, crs="EPSG:4326") -> gpd.GeoDataFrame:
    geom = df[geojson_colname].apply(lambda s: shape(json.loads(s)) if pd.notnull(s) else None)
    return gpd.GeoDataFrame(df.drop(columns=[geojson_colname]), geometry=geom, crs=crs)

def prep_frames(homes_gdf: gpd.GeoDataFrame, areas_gdf: gpd.GeoDataFrame):
    for df, col in [(homes_gdf, "start_optimal"), (areas_gdf, "start_optimal")]:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    homes = homes_gdf.copy()
    tenants = areas_gdf.copy()
    if homes.crs and tenants.crs and homes.crs != tenants.crs:
        tenants = tenants.to_crs(homes.crs)

    # Always-required cols
    home_cols = ["home_id", "rent", "home_allows_pets", "shared", "start_optimal", "geometry"]
    tenant_cols = ["tenant_ad_id", "max_monthly_cost", "pets", "shared", "start_optimal", "geometry"]

    # Optional cols we’ll keep if present
    for opt in ["currency", "country", "market"]:
        if opt in homes.columns: home_cols.append(opt)
        if opt in tenants.columns: tenant_cols.append(opt)

    homes = homes[home_cols].copy()
    tenants = tenants[tenant_cols].copy()
    return homes, tenants

def spatial_candidates(homes: gpd.GeoDataFrame, tenants: gpd.GeoDataFrame, predicate: str) -> pd.DataFrame:
    candidates = gpd.sjoin(homes, tenants, predicate=predicate, how="left", lsuffix="home", rsuffix="tenant")
    cand = candidates.dropna(subset=["tenant_ad_id"]).copy()

    rename_map = {
        "tenant_ad_id": "t_tenant_ad_id",
        "max_monthly_cost": "t_max_monthly_cost",
        "pets": "t_pets",
        "shared_tenant": "t_shared",
        "start_optimal_tenant": "t_start_optimal",
        "shared_home": "home_shared",
        "start_optimal_home": "home_start_optimal",
        "currency_home": "home_currency",
        "currency_tenant": "t_currency",
        "country_home": "home_country",
        "country_tenant": "t_country",
        "market_home": "home_market",
        "market_tenant": "t_market",
    }
    cand.rename(columns=rename_map, inplace=True)

    # Fallbacks if suffixes didn’t appear
    if "home_currency" not in cand and "currency" in cand: cand["home_currency"] = cand["currency"]
    if "t_currency" not in cand and "currency" in cand:   cand["t_currency"]   = cand["currency"]
    if "home_country" not in cand and "country" in cand:  cand["home_country"] = cand["country"]
    if "t_country" not in cand and "country" in cand:     cand["t_country"]    = cand["country"]

    # --- Robust fallbacks so downstream code always finds the columns ---

    # Tenant ID
    if "t_tenant_ad_id" not in cand:
        if "tenant_ad_id" in cand:
            cand.rename(columns={"tenant_ad_id": "t_tenant_ad_id"}, inplace=True)
        elif "tenant_id" in cand:
            cand.rename(columns={"tenant_id": "t_tenant_ad_id"}, inplace=True)

    # Shared / start_optimal (if not already handled)
    if "home_shared" not in cand and "shared_home" in cand:
        cand.rename(columns={"shared_home": "home_shared"}, inplace=True)
    if "t_shared" not in cand and "shared_tenant" in cand:
        cand.rename(columns={"shared_tenant": "t_shared"}, inplace=True)
    if "home_start_optimal" not in cand and "start_optimal_home" in cand:
        cand.rename(columns={"start_optimal_home": "home_start_optimal"}, inplace=True)
    if "t_start_optimal" not in cand and "start_optimal_tenant" in cand:
        cand.rename(columns={"start_optimal_tenant": "t_start_optimal"}, inplace=True)

    # Currency / country (if you added those)
    if "home_currency" not in cand and "currency_home" in cand:
        cand.rename(columns={"currency_home": "home_currency"}, inplace=True)
    if "t_currency" not in cand and "currency_tenant" in cand:
        cand.rename(columns={"currency_tenant": "t_currency"}, inplace=True)
    if "home_country" not in cand and "country_home" in cand:
        cand.rename(columns={"country_home": "home_country"}, inplace=True)
    if "t_country" not in cand and "country_tenant" in cand:
        cand.rename(columns={"country_tenant": "t_country"}, inplace=True)

    return cand

def _tenant_id_col(df: pd.DataFrame) -> Optional[str]:
    """Find a tenant id column across likely names."""
    for c in ["t_tenant_ad_id", "tenant_ad_id", "t_tenant_id", "tenant_id"]:
        if c in df.columns:
            return c
    return None

def distribution_for_case(
    cand_df: pd.DataFrame,
    homes: pd.DataFrame,
    # NEW: explicit subset rules
    homes_shared_only: bool = False,        # keep only homes where home_shared == True
    homes_no_pets_only: bool = False,       # keep only homes where home_allows_pets == False
    tenants_require_shared: bool = False,   # keep only tenants where t_shared == True
    tenants_require_no_pets: bool = False,  # keep only tenants where t_pets == False
    # existing rules
    rent_mode: str = "exact",               # "exact" or "plus5"
    date_within_days: Optional[int] = None,
) -> pd.DataFrame:
    df = cand_df.copy()

    if "home_currency" in df and "t_currency" in df:
        df = df[(df["home_currency"].isna()) | (df["t_currency"].isna()) | (df["home_currency"] == df["t_currency"])]

    # --- Home subset filters (explicit) ---
    if homes_shared_only:
        if "home_shared" in df:
            df = df[df["home_shared"] == True]
        else:
            df = df.iloc[0:0].copy()

    if homes_no_pets_only:
        if "home_allows_pets" in df:
            df = df[df["home_allows_pets"] == False]
        else:
            df = df.iloc[0:0].copy()

    # --- Tenant subset filters (explicit) ---
    if tenants_require_shared:
        if "t_shared" in df:
            df = df[df["t_shared"] == True]
        else:
            df = df.iloc[0:0].copy()

    if tenants_require_no_pets:
        if "t_pets" in df:
            df = df[df["t_pets"] == False]
        else:
            df = df.iloc[0:0].copy()

    # --- Rent rule ---
    if rent_mode == "exact":
        df = df[df["t_max_monthly_cost"] >= df["rent"]]
    elif rent_mode == "plus5":
        df = df[(df["t_max_monthly_cost"] * 1.05) >= df["rent"]]
    else:
        raise ValueError("rent_mode must be 'exact' or 'plus5'")

    # --- Date window (fill NaNs with today) ---
    if date_within_days is not None and date_within_days > 0:
        today = pd.Timestamp.today().normalize()
        h = df["home_start_optimal"].fillna(today)
        t = df["t_start_optimal"].fillna(today)
        df = df[((t - h).abs() <= pd.Timedelta(days=date_within_days))].copy()

    # --- Unique tenants per home & distribution ---
    if len(df) > 0:
        
        tid = _tenant_id_col(df)
        if tid and len(df) > 0:
            per_home = (
                df.groupby("home_id")[tid]
                .nunique()
                .rename("tenant_count")
                .reset_index()
            )
        else:
            per_home = pd.DataFrame(columns=["home_id", "tenant_count"])

    else:
        per_home = pd.DataFrame(columns=["home_id","tenant_count"])

    all_homes = homes[["home_id"]].drop_duplicates()
    per_home = all_homes.merge(per_home, on="home_id", how="left").fillna({"tenant_count":0})
    per_home["tenant_count"] = per_home["tenant_count"].astype(int)

    dist = (per_home.value_counts(subset=["tenant_count"]).rename("n_homes")
                  .reset_index().sort_values("tenant_count").reset_index(drop=True))
    total_homes = len(all_homes)
    dist["pct"] = (dist["n_homes"] / total_homes * 100).round(1)
    dist["cum_pct"] = (dist["n_homes"].cumsum() / total_homes * 100).round(1)
    return dist

def distribution_control(cand_df: pd.DataFrame, homes: pd.DataFrame) -> pd.DataFrame:
    df = cand_df.copy()
    if len(df) > 0:
        
        tid = _tenant_id_col(df)
        if tid and len(df) > 0:
            per_home = (
                df.groupby("home_id")[tid]
                .nunique()
                .rename("tenant_count")
                .reset_index()
            )
        else:
            per_home = pd.DataFrame(columns=["home_id", "tenant_count"])
        
    else:
        per_home = pd.DataFrame(columns=["home_id","tenant_count"])
    all_homes = homes[["home_id"]].drop_duplicates()
    per_home = all_homes.merge(per_home, on="home_id", how="left").fillna({"tenant_count":0})
    per_home["tenant_count"] = per_home["tenant_count"].astype(int)
    dist = (per_home.value_counts(subset=["tenant_count"]).rename("n_homes")
                  .reset_index().sort_values("tenant_count").reset_index(drop=True))
    total = len(all_homes)
    dist["pct"] = (dist["n_homes"] / total * 100).round(1)
    dist["cum_pct"] = (dist["n_homes"].cumsum() / total * 100).round(1)
    return dist

def bucket_counts_one(dist: pd.DataFrame, cap: int = 5) -> pd.DataFrame:
    b = dist.copy()
    # ensure numeric
    b["tenant_count"] = pd.to_numeric(b["tenant_count"], errors="coerce").fillna(0).astype(int)

    # numeric bin used for grouping; anything >= cap goes to cap
    b["bucket_num"] = b["tenant_count"].clip(upper=cap)

    # human label (strings only)
    b["bucket"] = b["bucket_num"].astype(str)
    b.loc[b["bucket_num"] >= cap, "bucket"] = f"{cap}+"

    agg = (
        b.groupby(["bucket_num", "bucket"], as_index=False)["n_homes"]
         .sum()
         .sort_values("bucket_num")
    )
    total = agg["n_homes"].sum()
    agg["pct"] = (agg["n_homes"] / total * 100).round(1) if total else 0

    # return pretty columns only
    return agg[["bucket", "n_homes", "pct"]]

def kpis_for_case(dist: pd.DataFrame) -> pd.Series:
    total_homes = dist["n_homes"].sum()
    homes_with_tenants = dist.loc[dist["tenant_count"]>0,"n_homes"].sum()
    coverage_pct = round(homes_with_tenants/total_homes*100,1) if total_homes else 0.0
    avg_tenants = (dist["tenant_count"]*dist["n_homes"]).sum()/total_homes if total_homes else 0.0
    return pd.Series({
        "total_homes": int(total_homes or 0),
        "homes_with_tenants": int(homes_with_tenants or 0),
        "coverage_pct": coverage_pct,
        "avg_tenants_per_home": round(avg_tenants,2),
    })

def quantiles_for_case(dist: pd.DataFrame) -> pd.Series:
    s = []
    for _, r in dist.iterrows():
        s += [int(r["tenant_count"])] * int(r["n_homes"])
    if not s:
        return pd.Series({"P50":0,"P75":0,"P90":0,"P95":0,"P99":0})
    q = pd.Series(s).quantile([.5,.75,.9,.95,.99]).astype(int)
    q.index = ["P50","P75","P90","P95","P99"]
    return q

# ============================================================
# MAIN FORM — nothing executes until you click "Run analysis"
# ============================================================
# --- Controls (no form; live UI) ---
st.markdown("### Step 1 · Upload CSVs")
colA, colB = st.columns(2)
with colA:
    homes_file = st.file_uploader("Homes CSV (with GeoJSON geometry column, e.g. `geom_geojson`)", type=["csv"])
with colB:
    tenants_file = st.file_uploader("Tenant Areas CSV (with GeoJSON geometry column)", type=["csv"])

st.markdown("### Step 2 · Spatial & CRS")
colC, colD = st.columns(2)
with colC:
    predicate = st.selectbox("Spatial requirement", ["intersects", "within"], index=0,
                             help="All matches must satisfy this spatial relation.")
with colD:
    target_crs = st.selectbox("Coordinate system", ["EPSG:3857", "EPSG:4326"], index=0)

st.markdown("### Step 3 · Choose one scenario")
scenario = st.radio(
    "Scenario",
    [
        "Control · Only spatial overlap (home inside tenant search area)",
        "Case 1 · Shared homes only · No-pets homes only · Tenants must want shared · Tenants must not have pets · Rent within tenant budget",
        "Case 2 · Same as Case 1, but tenant budget can be 5% below rent",
        "Case 3 · Same as Case 2, plus move-in date within ±30 days (missing dates treated as today)",
        "Case 4 · Same as Case 1, plus move-in date within ±30 days (missing dates treated as today)",
        "Custom · Choose your own rules below",
    ],
    index=1,
)

# Custom filters only when Custom is chosen
if scenario.startswith("Custom"):
    st.markdown("##### Custom filter settings")
    colH, colI = st.columns(2)

    with colH:
        # HOME subset
        homes_shared_only = st.checkbox("Show shared homes only", value=True)
        homes_no_pets_only = st.checkbox("Show homes that do NOT allow pets only", value=True)

        rent_mode = st.selectbox("Rent rule", ["exact", "plus5"], index=0,
                                 help="Exact: tenant_max ≥ home_rent ·· Plus5: tenant_max × 1.05 ≥ home_rent")

    with colI:
        # TENANT subset
        tenants_require_shared = st.checkbox("Tenants must want shared", value=True)
        tenants_require_no_pets = st.checkbox("Tenants must NOT have pets", value=True)

        date_window = st.slider("Move-in window (±days)", 0, 90, 30, step=5,
                                help="Missing dates are treated as 'today' before comparing.")
elif not scenario.startswith("Custom"):
    # Defaults for non-custom scenarios; will be overridden per case mapping below
    homes_shared_only = False
    homes_no_pets_only = False
    tenants_require_shared = False
    tenants_require_no_pets = False
    rent_mode = "exact"
    date_window = 0

st.markdown("### Step 4 · Chart options")
colL, colM, colN = st.columns(3)
with colL:
    cap_bucket = st.number_input("Bucket cap (for 5+ group)", min_value=3, max_value=20, value=5, step=1)
with colM:
    pct_cap = st.slider("Capped distribution percentile", 80, 100, 95, step=1,
                        help="Tail above this percentile is grouped into an overflow bin.")
with colN:
    log_x = st.checkbox("Log-scale x-axis", value=False)

# The only trigger for heavy work
run_clicked = st.button("▶ Run analysis", use_container_width=True)

# ============================================================
# Run when button is clicked
# ============================================================
if not run_clicked:
    st.info("Set your inputs above, then click **Run analysis**.")
    st.stop()

# Validate files
if not (homes_file and tenants_file):
    st.error("Please upload both CSVs.")
    st.stop()

# Read CSVs
try:
    homes_df = pd.read_csv(io.BytesIO(homes_file.getvalue()))
    tenants_df = pd.read_csv(io.BytesIO(tenants_file.getvalue()))
    homes_geo_col = geojson_col(homes_df)
    tenants_geo_col = geojson_col(tenants_df)
except Exception as e:
    st.error(str(e))
    st.stop()

homes_gdf = df_geojson_to_gdf(homes_df, homes_geo_col, crs="EPSG:4326")
areas_gdf = df_geojson_to_gdf(tenants_df, tenants_geo_col, crs="EPSG:4326")

# Casting
if "pets" in areas_gdf: areas_gdf["pets"] = areas_gdf["pets"].astype(bool)
if "shared" in areas_gdf: areas_gdf["shared"] = areas_gdf["shared"].astype(bool)
if "max_monthly_cost" in areas_gdf:
    try:
        areas_gdf["max_monthly_cost"] = areas_gdf["max_monthly_cost"].astype(int)
    except Exception:
        pass

# Reproject
homes_gdf = homes_gdf.to_crs(target_crs)
areas_gdf = areas_gdf.to_crs(target_crs)

# Prep & spatial join
# Choose currency guardrail before heavy work
same_currency_only = st.checkbox(
    "Require same currency for rent comparison",
    value=True,
    help="Safer when datasets may mix SEK/EUR/NOK. Disable only if you’ve normalized rents."
)

with st.spinner("Checking spatial overlaps…"):
    homes, tenants = prep_frames(homes_gdf, areas_gdf)
    cand = spatial_candidates(homes, tenants, predicate=predicate)

# Apply currency filter after building cand
removed_pairs = 0
if same_currency_only and "home_currency" in cand and "t_currency" in cand:
    before = len(cand)
    cand = cand[cand["home_currency"] == cand["t_currency"]].copy()
    removed_pairs = before - len(cand)

msg = f"Spatial matches created: {len(cand):,} home×tenant pairs"
if same_currency_only and removed_pairs > 0:
    msg += f"  (removed {removed_pairs:,} cross-currency pairs)"
st.success(msg)

# Resolve scenario to filter params (matches new labels + new function signature)
if scenario.startswith("Control"):
    dist = distribution_control(cand, homes)
    scenario_label = scenario

elif scenario.startswith("Case 1"):
    dist = distribution_for_case(
        cand, homes,
        homes_shared_only=True,
        homes_no_pets_only=True,
        tenants_require_shared=True,
        tenants_require_no_pets=True,
        rent_mode="exact",
        date_within_days=None,
    )
    scenario_label = scenario

elif scenario.startswith("Case 2"):
    dist = distribution_for_case(
        cand, homes,
        homes_shared_only=True,
        homes_no_pets_only=True,
        tenants_require_shared=True,
        tenants_require_no_pets=True,
        rent_mode="plus5",
        date_within_days=None,
    )
    scenario_label = scenario

elif scenario.startswith("Case 3"):
    dist = distribution_for_case(
        cand, homes,
        homes_shared_only=True,
        homes_no_pets_only=True,
        tenants_require_shared=True,
        tenants_require_no_pets=True,
        rent_mode="plus5",
        date_within_days=30,
    )
    scenario_label = scenario

elif scenario.startswith("Case 4"):
    dist = distribution_for_case(
        cand, homes,
        homes_shared_only=True,
        homes_no_pets_only=True,
        tenants_require_shared=True,
        tenants_require_no_pets=True,
        rent_mode="exact",
        date_within_days=30,
    )
    scenario_label = scenario

else:  # Custom
    dist = distribution_for_case(
        cand, homes,
        homes_shared_only=homes_shared_only,
        homes_no_pets_only=homes_no_pets_only,
        tenants_require_shared=tenants_require_shared,
        tenants_require_no_pets=tenants_require_no_pets,
        rent_mode=rent_mode,
        date_within_days=(date_window if date_window > 0 else None),
    )
    scenario_label = "Custom · Your settings"

# ============================================================
# Overview
# ============================================================
st.markdown(f"## Results — {scenario_label}")

# KPI card
k = kpis_for_case(dist)
st.markdown(
    f"<div class='card'><b>{scenario_label}</b><br>"
    f"<span class='small-muted'>coverage</span><br>"
    f"<h3>{k['coverage_pct']}%</h3>"
    f"<span class='small-muted'>avg tenants/home</span><br>"
    f"<h4>{k['avg_tenants_per_home']}</h4>"
    f"<span class='small-muted'>homes</span><br>"
    f"{int(k['homes_with_tenants'])}/{int(k['total_homes'])}</div>",
    unsafe_allow_html=True,
)

# Quantiles
st.markdown("#### Quantiles (tenants per home)")
st.dataframe(quantiles_for_case(dist).to_frame(name="value").T)

# ============================================================
# Charts
# ============================================================
st.markdown("## Charts")

# ECDF
st.markdown("#### ECDF — cumulative distribution")
fig_ecdf, ax_ecdf = plt.subplots()
x, y, cum = [], [], 0
total = int(dist["n_homes"].sum())
for _, r in dist.sort_values("tenant_count").iterrows():
    cum += int(r["n_homes"])
    x.append(int(r["tenant_count"]))
    y.append(100.0 * cum / total if total else 0)
ax_ecdf.plot(x, y, marker="o", linewidth=1)
ax_ecdf.set_xlabel("tenants per home")
ax_ecdf.set_ylabel("cumulative % of homes")
ax_ecdf.set_ylim(0, 100)
if log_x: ax_ecdf.set_xscale("log")
ax_ecdf.grid(True, alpha=0.3)
st.pyplot(fig_ecdf)

# Capped distribution
st.markdown("#### Capped distribution (focus on the body; tail grouped)")
expanded = []
for _, r in dist.iterrows():
    expanded.extend([int(r["tenant_count"])] * int(r["n_homes"]))
if expanded:
    s = pd.Series(expanded)
    x_cap = int(s.quantile(pct_cap / 100.0))
    view = dist.copy()
    view["tc"] = view["tenant_count"].clip(upper=x_cap)
    view.loc[view["tenant_count"] > x_cap, "tc"] = x_cap + 1
    agg = view.groupby("tc", as_index=False)["n_homes"].sum().sort_values("tc")
    total = int(dist["n_homes"].sum())
    agg["pct"] = (agg["n_homes"]/total*100).round(2) if total else 0
    fig_h, ax_h = plt.subplots()
    ax_h.bar(agg["tc"].astype(int), agg["pct"])
    ax_h.set_xlabel(f"tenants/home (≤ {x_cap}; bar {x_cap+1}=overflow)")
    ax_h.set_ylabel("% of homes")
    if log_x: ax_h.set_xscale("log")
    ax_h.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig_h)
else:
    st.info("No data to plot.")

# Bucketed bars
st.markdown(f"#### Bucketed percent (0..{cap_bucket}+)")
buckets = bucket_counts_one(dist, cap=cap_bucket)
fig_b, ax_b = plt.subplots()
ax_b.bar(buckets["bucket"].astype(str), buckets["pct"])
ax_b.set_xlabel(f"bucket (0..{cap_bucket}+)")
ax_b.set_ylabel("% of homes")
ax_b.grid(True, axis="y", alpha=0.3)
st.pyplot(fig_b)

# ============================================================
# Tables & Downloads
# ============================================================
st.markdown("## Tables & Downloads")
st.markdown("#### Distribution table")
st.dataframe(dist)

st.markdown(f"#### Buckets (0..{cap_bucket}+)")
st.dataframe(buckets)

c1, c2 = st.columns(2)
with c1:
    st.download_button("Download distribution (CSV)",
        data=dist.to_csv(index=False).encode("utf-8"),
        file_name=f"distribution_{scenario_label.replace(' ','_')}.csv",
        mime="text/csv")
with c2:
    st.download_button("Download buckets (CSV)",
        data=buckets.to_csv(index=False).encode("utf-8"),
        file_name=f"buckets_{scenario_label.replace(' ','_')}.csv",
        mime="text/csv")

# ============================================================
# Glossary
# ============================================================
with st.expander("Glossary & assumptions", expanded=False):
    st.markdown(f"""
- **Spatial requirement**: A home must **{predicate}** a tenant's search area.
- **Cases 1–4 (home subset)**: Use **only** homes that are shared **and** do **not** allow pets.
- **Cases 1–4 (tenant subset)**: Include **only** tenants who want shared and **do not** have pets.
- **Rent exact**: tenant_max_monthly_cost ≥ home_rent.
- **Rent +5%**: tenant_max_monthly_cost × 1.05 ≥ home_rent.
- **Move-in window**: Missing dates are set to **today**, then require |tenant − home| ≤ N days.
- **Counts**: Unique tenants per home (a tenant with multiple polygons counts once).
""")
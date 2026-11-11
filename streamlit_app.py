import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

st.set_page_config(page_title="BookLens Analytics", page_icon="📚", layout="wide")

# Custom styles: fixed-height multiselects, less empty space at the top
st.markdown("""
<style>
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:first-child {
    max-height: 42px;
    overflow-y: auto;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 1) DATA LOADING
# ---------------------------------------------------------------
def load_data():
    return pd.read_parquet("B2B_Transaction_Data.parquet")


# ---------------------------------------------------------------
# 2) DATA PREPROCESSING
# ---------------------------------------------------------------
def preprocess(raw):
    df = raw.copy()
    report = {"Raw rows": len(df)}

    # Data type corrections
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])   # text -> datetime
    df["StockCode"] = df["StockCode"].astype(str)            # barcode is an ID, not a number

    # Duplicate records
    report["Duplicate rows removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Noisy data: missing/zero quantity, zero/negative price (returns/corrections)
    invalid = ~((df["Quantity"] > 0) & (df["NetPrice"] >= 1) & (df["UnitPrice"] >= 1))
    report["Invalid quantity/price rows removed"] = int(invalid.sum())
    df = df[~invalid]

    # Missing values: fill with a global constant
    report["Missing cities filled with 'Unknown'"] = int(df["City"].isna().sum())
    report["Missing categories filled with 'Unknown'"] = int(df["Category"].isna().sum())
    df["City"] = df["City"].fillna("Unknown")
    df["Category"] = df["Category"].fillna("Unknown")

    df["Quantity"] = df["Quantity"].astype(int)

    # Feature construction
    df["Revenue"] = df["Quantity"] * df["NetPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    report["Clean rows"] = len(df)
    return df.reset_index(drop=True), report


@st.cache_resource(show_spinner="Loading and cleaning the data...")
def get_data():
    """Load and clean the data once. The same objects are shared by all users,
    which keeps memory low on Streamlit Cloud (cache_data would make a copy per run)."""
    raw = load_data()
    df, report = preprocess(raw)
    return raw, df, report


raw, df, report = get_data()

# ---------------------------------------------------------------
# CHART HELPERS
# ---------------------------------------------------------------
BAR_SCALE = ["#38bdf8", "#7c3aed"]   # light blue -> purple, readable in light & dark themes
ABC_COLORS = {"A": "#7c3aed", "B": "#38bdf8", "C": "#94a3b8"}   # purple, blue, gray

def ranked_bar(data, label, title, value="Revenue", prefix="₺"):
    """Horizontal bar chart, largest on top, colored by value."""
    fig = px.bar(data, x=value, y=label, orientation="h", title=title,
                 color=value, color_continuous_scale=BAR_SCALE)
    unit = f" ({prefix})" if prefix else ""
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      xaxis_title=f"{value}{unit}", yaxis_title=None,
                      coloraxis_showscale=False)
    fig.update_traces(hovertemplate=f"{prefix}%{{x:,.0f}}<extra></extra>")
    return fig

def top_n_share(series, n=6):
    """Top n values by record count, the rest grouped as 'Other'."""
    counts = series.fillna("Missing").value_counts()
    top = counts.head(n)
    if len(counts) > n:
        top = pd.concat([top, pd.Series({"Other": counts.iloc[n:].sum()})])
    return top.rename_axis("Label").reset_index(name="Records")


def donut(data, title):
    """Donut chart with 'Other' shown in gray."""
    fig = px.pie(data, names="Label", values="Records", color="Label", hole=0.55, title=title,
                 color_discrete_map={"Other": "#94a3b8"})
    fig.update_traces(sort=False, textinfo="percent",
                      hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>")
    fig.update_layout(height=400, legend=dict(orientation="h", y=-0.05))
    return fig

# ---------------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------------
def rfm_segment(r, f):
    """Assign a segment from Recency and Frequency scores (1-5)."""
    if r >= 4 and f >= 4:
        return "Champions"
    if r <= 2 and f >= 4:
        return "At Risk"
    if f >= 3:
        return "Loyal"
    if r >= 4:
        return "Promising"
    if r <= 2:
        return "Hibernating"
    return "Needs Attention"

SEGMENT_INFO = {   # segment: (rule, recommended action) - order is used in cards and charts
    "Champions":       ("R ≥ 4 and F ≥ 4", "Bought recently and often. Reward them with loyalty discounts and early access to new titles."),
    "Loyal":           ("F ≥ 3", "Order regularly. Grow their basket with bundles and book series."),
    "Promising":       ("R ≥ 4 and F ≤ 2", "Recent but infrequent. Recommend related titles to build a buying habit."),
    "Needs Attention": ("R = 3 and F ≤ 2", "Average and slipping. Send targeted promotions before they go quiet."),
    "At Risk":         ("R ≤ 2 and F ≥ 4", "Used to order often but stopped. Contact them personally with a win-back offer."),
    "Hibernating":     ("R ≤ 2 and F ≤ 2", "Inactive and low value. Use low-cost reminders only."),
}
SEGMENT_COLORS = {"Champions": "#7c3aed", "Loyal": "#3b82f6", "Promising": "#14b8a6",
                  "Needs Attention": "#f59e0b", "At Risk": "#ef4444", "Hibernating": "#94a3b8"}

def build_rfm(data):
    """Recency, Frequency, Monetary values and scores per customer."""
    ref_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = data.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: (ref_date - d.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    ).reset_index()

    # Quintile scores; rank() avoids errors when many customers share the same value
    rfm["R"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["Segment"] = [rfm_segment(r, f) for r, f in zip(rfm["R"], rfm["F"])]
    return rfm

def build_abc(data):
    """ABC classification of products by cumulative revenue share."""
    prod = (data.groupby("StockCode")
                .agg(Description=("Description", "first"),
                     Category=("Category", "first"),
                     Units=("Quantity", "sum"),
                     Revenue=("Revenue", "sum"))
                .sort_values("Revenue", ascending=False)
                .reset_index())
    prod["CumShare"] = (prod["Revenue"].cumsum() / prod["Revenue"].sum()).clip(upper=1)
    prod["SKUShare"] = (prod.index + 1) / len(prod)
    prod["Class"] = pd.cut(prod["CumShare"], bins=[0, 0.80, 0.95, 1.0],
                           labels=["A", "B", "C"], include_lowest=True).astype(str)
    return prod

Z_SCORES = {80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}
MODELS = {   # name: function that creates a fresh, untrained model
    "Linear Regression": lambda: LinearRegression(),
    "SVR (RBF kernel)": lambda: TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1)),
        transformer=StandardScaler()),   # SVR needs scaled inputs and target
}


def build_forecast(data, metric, weeks, level=95, end_date=None, model_name="Linear Regression"):
    """Daily model (trend + day-of-week). All models are back-tested on the last 4 weeks."""
    daily = data.groupby(data["InvoiceDate"].dt.normalize())[metric].sum()
    end = end_date if end_date is not None else daily.index[-1]
    daily = daily.reindex(pd.date_range(daily.index[0], end, freq="D"), fill_value=0)  # no sales -> 0
    start = daily.index[0]

    def features(dates):
        X = pd.DataFrame({"trend": (dates - start).days}, index=dates)
        for d in range(1, 7):                          # Monday is the baseline
            X[f"dow_{d}"] = (dates.day_of_week == d).astype(int)
        return X

    X, y = features(daily.index), daily.values.astype(float)

    # 1) Back-test every model: train without the last 4 weeks, predict them, measure accuracy
    split = len(daily) - 28
    actual_weeks = daily.iloc[split:].resample("W").sum()
    accuracy = {}
    for name, make_model in MODELS.items():
        m = make_model().fit(X.iloc[:split], y[:split])
        pred = pd.Series(m.predict(X.iloc[split:]), index=daily.index[split:])
        mape = mean_absolute_percentage_error(actual_weeks, pred.resample("W").sum())
        accuracy[name] = max(0.0, 1 - mape)

    # 2) Final model (the one chosen by the user) on all data
    model = MODELS[model_name]().fit(X, y)
    resid_sd = (y - model.predict(X)).std()
    future = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=weeks * 7, freq="D")
    future_pred = pd.Series(model.predict(features(future)), index=future).clip(lower=0)
    monthly_trend = np.polyfit(np.arange(len(future_pred)), future_pred.values, 1)[0] * 30

    # 3) Weekly totals for the chart
    weekly = pd.concat([daily, future_pred]).resample("W").sum()
    if daily.index[0].day_of_week != 0:                  # drop incomplete first week
        weekly = weekly.iloc[1:]
    last_actual_week = daily.resample("W").sum().index[-1]
    history = weekly.loc[:last_actual_week]
    fc = weekly.loc[last_actual_week:].to_frame("value")
    if future[-1].day_of_week != 6:                      # drop incomplete last forecast week
        fc = fc.iloc[:-1]
    band = Z_SCORES[level] * resid_sd * np.sqrt(7)     # confidence interval for a weekly total
    fc["lower"] = (fc["value"] - band).clip(lower=0)
    fc["upper"] = fc["value"] + band
    fc.iloc[0, 1:] = fc.iloc[0, 0]                     # no band at the junction point

    return {
        "history": history, "forecast": fc,
        "total": future_pred.sum(), "previous": daily.iloc[-weeks * 7:].sum(),
        "accuracy": accuracy[model_name], "comparison": accuracy,
        "monthly_trend": monthly_trend,
    }


def iqr_summary(data, cols, k=1.5):
    """IQR outlier limits and counts for numeric columns (course method)."""
    rows = []
    for c in cols:
        s = data[c]
        q1, med, q3 = s.quantile([0.25, 0.50, 0.75])
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < low) | (s > high)).sum())
        rows.append({
            "Variable": c, "Min": s.min(), "Q1": q1, "Median": med, "Q3": q3, "Max": s.max(),
            "IQR": iqr, "Lower limit": low, "Upper limit": high,
            "Whisker low": s[s >= low].min(), "Whisker high": s[s <= high].max(),
            "Outliers": n_out, "Outlier %": n_out / len(s),
        })
    return pd.DataFrame(rows)

ATTRIBUTES = {   # column: (description, attribute type from the course)
    "InvoiceNo":   ("Invoice number (one invoice = several lines)", "Nominal"),
    "CustomerID":  ("Customer identifier (bookstore / reseller)", "Nominal"),
    "InvoiceDate": ("Invoice date", "Interval"),
    "StockCode":   ("Product barcode", "Nominal"),
    "Description": ("Product name", "Nominal"),
    "Category":    ("Product category", "Nominal"),
    "City":        ("Customer city", "Nominal"),
    "Quantity":    ("Units sold on the invoice line", "Numeric – discrete"),
    "NetPrice":    ("Net unit price (₺)", "Numeric – continuous"),
    "UnitPrice":   ("List unit price (₺)", "Numeric – continuous"),
}


@st.cache_resource
def describe_raw(_raw):
    """Attribute table of the raw dataset (computed once; '_' skips hashing the large DataFrame)."""
    raw = _raw
    return pd.DataFrame({
        "Attribute": raw.columns,
        "Description": [ATTRIBUTES.get(c, ("", ""))[0] for c in raw.columns],
        "Attribute type": [ATTRIBUTES.get(c, ("", ""))[1] for c in raw.columns],
        "Raw data type": raw.dtypes.astype(str).values,
        "Missing": raw.isna().sum().values,
        "Unique values": raw.nunique().values,
        "Example": raw.iloc[0].astype(str).values,
    })

# ---------------------------------------------------------------
# 4) FILTER BAR (used only on the analytics pages)
# ---------------------------------------------------------------
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


def filter_bar():
    """Show the filter bar and return the filtered data.
    Applied filters are stored in session_state, so they are kept when switching pages."""
    saved = st.session_state.setdefault(
        "filters", {"months": ("January", "December"), "cities": [], "categories": []})

    with st.form("filter_form"):
        c1, c2, c3, c4 = st.columns([3, 3, 3, 1], vertical_alignment="bottom")
        months = c1.select_slider("Month range (2020)", options=MONTHS, value=saved["months"])
        cities = c2.multiselect("City", sorted(df["City"].unique()),
                                default=saved["cities"], placeholder="All")
        categories = c3.multiselect("Category", sorted(df["Category"].unique()),
                                    default=saved["categories"], placeholder="All")
        if c4.form_submit_button("Apply", icon=":material/search:", type="primary", width="stretch"):
            saved.update(months=months, cities=cities, categories=categories)

    start_no = MONTHS.index(saved["months"][0]) + 1
    end_no = MONTHS.index(saved["months"][1]) + 1
    data = df[df["InvoiceDate"].dt.month.between(start_no, end_no)]
    if saved["cities"]:
        data = data[data["City"].isin(saved["cities"])]
    if saved["categories"]:
        data = data[data["Category"].isin(saved["categories"])]

    if data.empty:
        st.warning("No records match the selected filters.")
        st.stop()
    return data


# ---------------------------------------------------------------
# 5) PAGES
# ---------------------------------------------------------------
def page_overview():
    st.subheader("Dataset Overview",
                 help="Raw transaction data from a B2B book distributor, before any cleaning.")

    period = f"{df['InvoiceDate'].min():%d %b %Y} – {df['InvoiceDate'].max():%d %b %Y}"

    r1 = st.columns(3)
    r1[0].metric("Records", f"{len(raw):,}", border=True)
    r1[1].metric("Attributes", raw.shape[1], border=True)
    r1[2].metric("Period", "2020", help=period, border=True)

    r2 = st.columns(3)
    r2[0].metric("Customers", f"{raw['CustomerID'].nunique():,}", border=True)
    r2[1].metric("Products", f"{raw['StockCode'].nunique():,}", border=True)
    r2[2].metric("Cities", f"{raw['City'].nunique():,}", border=True)


    type_counts = (pd.Series([t for _, t in ATTRIBUTES.values()])
                     .value_counts().rename_axis("Label").reset_index(name="Records"))

    d1, d2, d3 = st.columns(3)
    d1.plotly_chart(donut(type_counts, "Attribute Types"))
    d2.plotly_chart(donut(top_n_share(raw["Category"]), "Records by Category"))
    d3.plotly_chart(donut(top_n_share(raw["City"]), "Records by City"))
    city_share = raw["City"].value_counts(normalize=True, dropna=False)
    cat_share = raw["Category"].value_counts(normalize=True, dropna=False)
    st.info(
        f"💡 **First observations:** **{city_share.index[0]}** accounts for {city_share.iloc[0]:.0%} of all "
        f"records, so overall results are strongly driven by one city. Sales are spread over "
        f"**{raw['Category'].nunique()} categories**, and even the top 6 cover only {cat_share.head(6).sum():.0%} "
        f"of records (a long tail). There are {raw['StockCode'].nunique():,} barcodes but only "
        f"{raw['Description'].nunique():,} product names, so some titles are sold under more than one "
        f"barcode (e.g. different editions).")
    st.subheader("Attributes")
    st.dataframe(describe_raw(raw), hide_index=True, column_config={
        "Missing": st.column_config.NumberColumn(alignment="center"),
        "Unique values": st.column_config.NumberColumn(alignment="center"),
    })

def page_sales():
    f = filter_bar()

    total_revenue = f["Revenue"].sum()
    invoices = f["InvoiceNo"].nunique()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"₺{total_revenue:,.0f}")
    k2.metric("Invoices", f"{invoices:,}")
    k3.metric("Customers", f"{f['CustomerID'].nunique():,}")
    k4.metric("Avg. Invoice Value", f"₺{total_revenue / invoices:,.0f}")

    # --- Aggregations ---
    monthly = f.groupby("Month", as_index=False)["Revenue"].sum()
    by_cat = (f.groupby("Category", as_index=False)["Revenue"].sum()
                .sort_values("Revenue", ascending=False))
    by_city = (f.groupby("City", as_index=False)["Revenue"].sum()
                 .sort_values("Revenue", ascending=False))

    # --- Automatic insights ---
    best_month = monthly.loc[monthly["Revenue"].idxmax()]
    top_cat = by_cat.iloc[0]
    top_city = by_city.iloc[0]
    insight = (f"💡 **Insights:** Peak month was **{best_month['Month']:%B}** "
               f"(₺{best_month['Revenue']:,.0f}).")
    if len(by_cat) > 1:
        insight += (f" **{top_cat['Category']}** is the leading category with "
                    f"{top_cat['Revenue'] / total_revenue:.1%} of revenue.")
    if len(by_city) > 1:
        insight += (f" **{top_city['City']}** generates "
                    f"{top_city['Revenue'] / total_revenue:.1%} of total revenue.")
    st.info(insight)

    # --- Monthly trend ---
    fig = px.line(monthly, x="Month", y="Revenue", markers=True, title="Monthly Revenue Trend")
    fig.update_layout(xaxis_title=None, yaxis_title="Revenue (₺)")
    fig.update_traces(hovertemplate="%{x|%B}: ₺%{y:,.0f}<extra></extra>")
    st.plotly_chart(fig)

    # --- Top categories & cities side by side ---
    left, right = st.columns(2)
    with left:
        st.plotly_chart(ranked_bar(by_cat.head(10), "Category", "Top 10 Categories by Revenue"))
    with right:
        st.plotly_chart(ranked_bar(by_city.head(10), "City", "Top 10 Cities by Revenue"))

def page_marketing():
    f = filter_bar()

    if f["CustomerID"].nunique() < 5:
        st.info("At least 5 customers are needed for segmentation. Please widen the filters.")
    else:
        rfm = build_rfm(f)

        k1, k2, k3 = st.columns(3)
        k1.metric("Customers", f"{len(rfm):,}")
        k2.metric("Avg. Revenue per Customer", f"₺{rfm['Monetary'].mean():,.0f}")
        k3.metric("Repeat Customers", f"{(rfm['Frequency'] > 1).mean():.0%}")

        seg = (rfm.groupby("Segment")
                  .agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum"))
                  .reset_index())

        # Insight: revenue at risk
        risk = seg.loc[seg["Segment"] == "At Risk"]
        if not risk.empty:
            st.warning(
                f"⚠️ **{int(risk['Customers'].iloc[0])} customers are At Risk.** They used to buy "
                f"frequently but have not ordered recently. Together they generated "
                f"₺{risk['Revenue'].iloc[0]:,.0f}. Consider a re-engagement campaign."
            )

        # --- RFM explanation and segment cards ---
        st.subheader("RFM Segmentation",
                     help="Each customer gets three scores from 1 to 5 by comparing them with the other "
                          "customers in the current selection (quintiles): R = how recently they ordered "
                          "(5 = most recent), F = number of invoices (5 = most frequent), M = total revenue. "
                          "Segments are assigned from the R and F scores; revenue is shown for context.")
        st.markdown(
            "Not every customer needs the same treatment. **RFM analysis** groups customers by "
            "**Recency** (days since last order), **Frequency** (number of invoices) and **Monetary** "
            "value (total revenue), so the marketing team can decide whom to reward, whom to win back "
            "and where to spend its budget.")

        stats = seg.set_index("Segment")
        total_rev = rfm["Monetary"].sum()
        names = list(SEGMENT_INFO)
        for row in (names[:3], names[3:]):
            for col, name in zip(st.columns(3), row):
                rule, action = SEGMENT_INFO[name]
                n = int(stats["Customers"].get(name, 0))
                share = stats["Revenue"].get(name, 0.0) / total_rev
                share_txt = f"{share:.0%}" if share >= 0.005 else "<1%"
                with col.container(border=True):
                    st.markdown(
                        f"<span style='color:{SEGMENT_COLORS[name]}'>●</span> **{name}** · "
                        f"{n:,} customers ({n / len(rfm):.0%})  \n"
                        f"{share_txt} of revenue · :gray[{rule}]\n\n{action}",
                        unsafe_allow_html=True)

        # --- Revenue by segment ---
        fig = px.bar(seg, x="Revenue", y="Segment", orientation="h", color="Segment",
                     color_discrete_map=SEGMENT_COLORS, title="Revenue by Customer Segment")
        fig.update_layout(showlegend=False, xaxis_title="Revenue (₺)", yaxis_title=None,
                          yaxis={"categoryorder": "total ascending"})
        fig.update_traces(hovertemplate="₺%{x:,.0f}<extra></extra>")
        st.plotly_chart(fig)

        # --- Customer map ---
        st.subheader("Customer Map",
                     help="Each bubble is one customer. Further left = ordered more recently, higher = "
                          "orders more often, bigger bubble = more revenue. Champions sit in the top-left "
                          "corner; At Risk customers are frequent buyers drifting to the right.")
        fig = px.scatter(rfm, x="Recency", y="Frequency", size="Monetary", color="Segment",
                         color_discrete_map=SEGMENT_COLORS, category_orders={"Segment": names},
                         hover_name="CustomerID", size_max=50)
        fig.update_traces(marker=dict(sizemin=4, line=dict(width=0)),
                          hovertemplate="<b>%{hovertext}</b><br>Days since last order: %{x}"
                                        "<br>Invoices: %{y}<br>Revenue: ₺%{marker.size:,.0f}<extra></extra>")
        fig.update_layout(xaxis_title="Days since last order", yaxis_title="Number of invoices",
                          height=550, legend=dict(title=None, orientation="h", x=0, y=1.02, yanchor="bottom"))
        st.plotly_chart(fig)

        st.subheader("Top 10 Customers")
        top = rfm.nlargest(10, "Monetary")[["CustomerID", "Segment", "Recency", "Frequency", "Monetary"]]
        st.dataframe(top, hide_index=True, column_config={
            "Recency": st.column_config.NumberColumn("Days Since Last Order", alignment="center"),
            "Frequency": st.column_config.NumberColumn("Invoices", alignment="center"),
            "Monetary": st.column_config.NumberColumn("Revenue", format="₺%,.0f", alignment="center"),
        })

def page_supply():
    f = filter_bar()

    abc = build_abc(f)
    class_a = abc[abc["Class"] == "A"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active Products (SKU)", f"{len(abc):,}")
    k2.metric("Units Sold", f"{f['Quantity'].sum():,}")
    k3.metric("Class A Products", f"{len(class_a):,}")
    k4.metric("Avg. Units per Invoice", f"{f.groupby('InvoiceNo')['Quantity'].sum().mean():,.1f}")

    st.info(
        f"💡 **Pareto insight:** Only **{len(class_a) / len(abc):.1%}** of products "
        f"({len(class_a):,} SKUs) generate **80%** of revenue. Keep these Class A items "
        f"in stock at all times. Class C items ({(abc['Class'] == 'C').sum():,} SKUs) "
        f"can be ordered on demand."
    )

    # --- ABC analysis ---
    st.subheader("ABC Analysis",
                 help="How to read the chart: products are sorted from highest to lowest revenue along "
                      "the x-axis, and the line shows their cumulative share of revenue. The steeper the "
                      "curve, the more revenue depends on a small number of titles.")
    st.markdown(
        "A distributor carries tens of thousands of titles but cannot give each one the same attention. "
        "**ABC analysis**, based on the Pareto (80/20) principle, ranks products by revenue and splits "
        "them into three classes, so purchasing and warehouse teams know where to focus.")

    class_info = {
        "A": ("Top 80% of revenue", "Never let these run out of stock; review weekly."),
        "B": ("Next 15% of revenue", "Keep in stock and review regularly."),
        "C": ("Last 5% of revenue", "Order on demand; candidates for delisting."),
    }
    for col, (cls, (share, action)) in zip(st.columns(3), class_info.items()):
        n = int((abc["Class"] == cls).sum())
        with col.container(border=True):
            st.markdown(f"**Class {cls}** · {n:,} SKUs ({n / len(abc):.1%} of products)\n\n"
                        f"{share}  \n:gray[{action}]")

    abc["Class label"] = abc["Class"].map({c: f"{c} – {v[0].lower()}" for c, v in class_info.items()})
    fig = px.line(abc, x="SKUShare", y="CumShare", color="Class label",
                  color_discrete_map={f"{c} – {v[0].lower()}": ABC_COLORS[c] for c, v in class_info.items()},
                  title="Cumulative Revenue Share by Product")
    fig.add_hline(y=0.80, line_dash="dash", annotation_text="80% of revenue",
                  annotation_position="bottom right")
    fig.add_hline(y=0.95, line_dash="dot", annotation_text="95% of revenue",
                  annotation_position="bottom right")
    fig.update_layout(xaxis_title="Share of products (sorted by revenue)",
                      yaxis_title="Cumulative revenue share",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=450,
                      legend=dict(title=None, orientation="h", x=0, y=1.02, yanchor="bottom"))
    fig.update_traces(hovertemplate="Top %{x:.1%} of products → %{y:.1%} of revenue<extra></extra>")
    st.plotly_chart(fig)

    # --- Best sellers by units ---
    top_units = abc.nlargest(10, "Units").copy()
    top_units["Product"] = top_units["Description"].str.slice(0, 45)
    same_name = top_units["Product"].duplicated(keep=False)        # same title, different barcode (edition)
    top_units.loc[same_name, "Product"] += " · " + top_units.loc[same_name, "StockCode"].str[-4:]
    st.plotly_chart(ranked_bar(top_units, "Product", "Top 10 Best-Selling Products (Units)",
                               value="Units", prefix=""))
    # --- Monthly demand ---
    monthly_units = f.groupby("Month", as_index=False)["Quantity"].sum()
    fig = px.line(monthly_units, x="Month", y="Quantity", markers=True,
                  title="Monthly Demand (Units)")
    fig.update_layout(xaxis_title=None, yaxis_title="Units")
    fig.update_traces(hovertemplate="%{x|%B}: %{y:,} units<extra></extra>")
    st.plotly_chart(fig)

    # --- Stock priority list ---
    st.subheader("Class A Products – Stock Priority List")
    st.caption(
        f"Only Class A products are listed, as they drive 80% of revenue and require "
        f"continuous stock availability. Class B ({(abc['Class'] == 'B').sum():,} SKUs) and "
        f"Class C ({(abc['Class'] == 'C').sum():,} SKUs) items are summarized in the chart above."
    )
    st.dataframe(class_a[["StockCode", "Description", "Category", "Units", "Revenue"]],
                 hide_index=True, column_config={
                     "Units": st.column_config.NumberColumn(format="%,.0f", alignment="center"),
                     "Revenue": st.column_config.NumberColumn(format="₺%,.0f", alignment="center"),
                 })

MODEL_COLORS = {"Linear Regression": "#7c3aed", "SVR (RBF kernel)": "#f97316"}   # purple, orange
ACTUAL_COLOR = "#3b82f6"                                                         # blue


def signed(value, prefix):
    """Format a number with its sign in front of the currency: +₺1,234 / −₺1,234."""
    return f"{'+' if value >= 0 else '−'}{prefix}{abs(value):,.0f}"


def change_vs_previous(res):
    """Forecast total compared with the same-length period before it (None if no sales)."""
    return res["total"] / res["previous"] - 1 if res["previous"] > 0 else None


def trend_direction(res, weeks):
    """'stable' if the monthly trend is below 1% of the monthly average."""
    monthly_avg = res["previous"] / weeks * 4.33
    pct = res["monthly_trend"] / monthly_avg if monthly_avg > 0 else 0
    if abs(pct) < 0.01:
        return "stable"
    return "growing" if pct > 0 else "declining"


def forecast_chart(res, metric_label, prefix, level, color, title):
    """Actual weekly values, forecast line and confidence band for one model."""
    hist, fc = res["history"], res["forecast"]
    hover = f"Week ending %{{x|%d %b}}: {prefix}%{{y:,.0f}}<extra></extra>"
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(fc.index) + list(fc.index[::-1]),
        y=list(fc["upper"]) + list(fc["lower"][::-1]),
        mode="lines", fill="toself", fillcolor=f"rgba({r}, {g}, {b}, 0.15)", line=dict(width=0),
        hoverinfo="skip", name=f"Expected range ({level}% confidence)", legendrank=3))
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, mode="lines+markers", name=f"Actual weekly {metric_label.lower()}",
        line=dict(color=ACTUAL_COLOR), hovertemplate=hover, legendrank=1))
    fig.add_trace(go.Scatter(
        x=fc.index, y=fc["value"], mode="lines+markers", name="Forecast",
        line=dict(color=color, dash="dash"), hovertemplate=hover, legendrank=2))
    fig.update_layout(title=title, xaxis_title="Week ending", yaxis_title=metric_label, height=450,
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=13)))
    return fig


def page_forecast():
    f = filter_bar()

    if (f["InvoiceDate"].max() - f["InvoiceDate"].min()).days < 56:
        st.info("At least 8 weeks of data are needed for forecasting. Please widen the month range.")
        return

    # --- Controls ---
    c1, c2, c3 = st.columns(3)
    metric_label = c1.radio(
        "Forecast target", ["Revenue", "Units"], horizontal=True,
        help="What to forecast. **Revenue** (₺) supports sales targets and budgeting; "
             "**Units** (books) supports inventory and purchasing decisions.")
    weeks = c2.slider(
        "Forecast horizon (weeks)", 4, 12, 8,
        help="How far into the future to forecast. Short horizons (4–6 weeks) are more "
             "reliable and suit operational planning such as restocking. Longer horizons "
             "suit budgeting but carry more uncertainty.")
    level = c3.select_slider(
        "Confidence level", options=[80, 90, 95, 99], value=95,
        format_func=lambda x: f"{x}%",
        help="Width of the shaded band on the chart. At 95%, actual weekly values are "
             "expected to fall inside the band 95 times out of 100. A higher level gives a "
             "wider, more cautious range, useful for worst-case stock planning.")
    metric = "Revenue" if metric_label == "Revenue" else "Quantity"
    prefix = "₺" if metric_label == "Revenue" else ""
    period_end = pd.Timestamp(2020, MONTHS.index(st.session_state["filters"]["months"][1]) + 1, 1) \
        + pd.offsets.MonthEnd(0)

    # --- Run both models ---
    results = {name: build_forecast(f, metric, weeks, level, end_date=period_end, model_name=name)
               for name in MODELS}
    best = max(results, key=lambda n: results[n]["accuracy"])

    # --- One section per model ---
    for name, res in results.items():
        with st.container(border=True):
            st.subheader(name + ("  🏆" if name == best else ""))
            change = change_vs_previous(res)
            k1, k2, k3 = st.columns(3)
            k1.metric(f"Forecast – next {weeks} weeks", f"{prefix}{res['total']:,.0f}",
                      delta=f"{change:+.1%} vs previous {weeks} weeks" if change is not None else None)
            k2.metric("Model accuracy (back-test)", f"{res['accuracy']:.0%}",
                      help="1 − MAPE on weekly totals of the last 4 weeks, which were hidden "
                           "from the model during training.")
            k3.metric("Underlying trend", f"{signed(res['monthly_trend'], prefix)} / month")
            st.plotly_chart(forecast_chart(res, metric_label, prefix, level, MODEL_COLORS[name],
                                           f"Weekly {metric_label}: Actual vs Forecast"))

    # --- Model comparison ---
    st.subheader("Model Comparison",
                 help="Both models are trained without the last 4 weeks and scored on those "
                      "hidden weeks (1 − MAPE on weekly totals).")

    hist = results[best]["history"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, mode="lines", name="Actual", line=dict(color=ACTUAL_COLOR),
        hovertemplate=f"Actual: {prefix}%{{y:,.0f}}<extra></extra>"))
    for name, res in results.items():
        fc = res["forecast"]
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc["value"], mode="lines+markers", name=name,
            line=dict(color=MODEL_COLORS[name], dash="dash"),
            hovertemplate=f"{name}: {prefix}%{{y:,.0f}}<extra></extra>"))
    fig.update_layout(title=f"Weekly {metric_label}: Linear Regression vs SVR",
                      xaxis_title="Week ending", yaxis_title=metric_label, height=450,
                      hovermode="x unified",
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=13)))
    st.plotly_chart(fig)

    table = pd.DataFrame([{
        "Model": name,
        "Back-test accuracy": res["accuracy"] * 100,
        f"Forecast – next {weeks} weeks": res["total"],
        "vs previous period": (f"{change_vs_previous(res):+.1%}"
                               if change_vs_previous(res) is not None else "n/a"),
        "Monthly trend": signed(res["monthly_trend"], prefix),
    } for name, res in results.items()])
    st.dataframe(table, hide_index=True, column_config={
        "Back-test accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        f"Forecast – next {weeks} weeks": st.column_config.NumberColumn(format=f"{prefix}%,.0f"),
    })

    res = results[best]
    other = next(n for n in results if n != best)
    st.success(
        f"🏆 **{best}** was more accurate on the hidden weeks "
        f"({res['accuracy']:.0%} vs {results[other]['accuracy']:.0%}). Based on this model, "
        f"{metric_label.lower()} is **{trend_direction(res, weeks)}** and the expected total for the next "
        f"{weeks} weeks is **{prefix}{res['total']:,.0f}**. Use it to plan inventory and sales targets.")

    st.caption("Both models use daily totals with a trend term and day-of-week effects. SVR (Support "
               "Vector Regression, RBF kernel) can capture non-linear patterns but tends to flatten "
               "the trend beyond the training period. With only one year of data (2020, including "
               "pandemic effects), yearly seasonality cannot be learned, so treat long horizons with caution.")


def page_data():
    raw_n = report["Raw rows"]
    dup_n = report["Duplicate rows removed"]
    inv_n = report["Invalid quantity/price rows removed"]
    clean_n = report["Clean rows"]
    city_fill = report["Missing cities filled with 'Unknown'"]
    cat_fill = report["Missing categories filled with 'Unknown'"]

    # --- Data Cleaning Pipeline card ---
    retained = clean_n / raw_n
    with st.container(border=True):
        st.subheader("Data Cleaning Pipeline")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Raw Data", f"{raw_n:,}", delta="Starting point", delta_color="off", delta_arrow="off")
        k2.metric("Duplicates Removed", f"{dup_n:,}", delta=f"-{dup_n / raw_n:.3%}")
        k3.metric("Invalid Records Removed", f"{inv_n:,}", delta=f"-{inv_n / raw_n:.4%}")
        k4.metric("Clean Data", f"{clean_n:,}", delta=f"{retained:.2%} retained")

        st.markdown("**Data Retention**")
        st.markdown(
            f"""<div style="display:flex; height:12px; border-radius:6px; overflow:hidden;">
                  <div style="width:{retained:.4%}; background:#10b981;"></div>
                  <div style="width:{1 - retained:.4%}; background:#ef4444;"></div>
                </div>""",
            unsafe_allow_html=True)
        st.markdown(
            f"""<div style="display:flex; justify-content:space-between; margin-top:4px;">
                  <span style="color:#059669;"><b>{retained:.2%}</b> retained</span>
                  <span style="color:#dc2626;"><b>{1 - retained:.2%}</b> removed</span>
                </div>""",
            unsafe_allow_html=True)

    # --- Step-by-step explanation ---
    st.subheader("Preprocessing Steps")

    with st.expander("1️⃣  Data type correction"):
        st.markdown(
            "- `InvoiceDate` was stored as text → converted to **datetime** to enable monthly and weekly analysis.\n"
            "- `StockCode` was stored as a number → converted to **text**, because a barcode is an identifier, "
            "not a quantity to sum or average.")

    with st.expander("2️⃣  Duplicate records"):
        st.markdown(
            f"**Result:** {dup_n:,} rows removed ({dup_n / raw_n:.2%} of the raw data)\n\n"
            "Rows identical in every column (same invoice, product, quantity and price) were treated as "
            "**duplicate entries** and removed, as they would inflate revenue and quantity figures.")

    with st.expander("3️⃣  Noisy data"):
        st.markdown(
            f"**Result:** {inv_n:,} rows removed ({inv_n / raw_n:.3%} of the raw data)\n\n"
            "Rows with **missing or zero quantity**, **zero or negative net price**, and "
            "**prices below ₺1** were removed. Zero-quantity rows are return/correction records "
            "(net price equal to the negative of the unit price), and near-zero prices are "
            "data-entry errors, not real sales.")

    with st.expander("4️⃣  Missing values"):
        st.markdown(
            f"**Result:** {city_fill:,} cities and {cat_fill:,} categories filled\n\n"
            "Missing `City` and `Category` values were filled with the **global constant 'Unknown'** "
            "instead of dropping the rows, because their quantity and revenue information is still valid.")
    with st.expander("5️⃣  Feature construction"):
        st.markdown(
            "- `Revenue` = `Quantity` × `NetPrice`\n"
            "- `Month` = first day of the invoice month, used for monthly aggregation")
    # --- Distribution analysis (histograms) ---
    st.subheader("Distribution Analysis",
                 help="Values above the 99th percentile are left out of the histograms for readability. "
                      "Mean and median are calculated on all values.")

    cols = st.columns(3)
    for col, c in zip(cols, ["Quantity", "NetPrice", "UnitPrice"]):
        s = df[c]
        p99 = s.quantile(0.99)
        bins = np.arange(0.5, p99 + 1.5) if c == "Quantity" else 30   # one bar per integer for Quantity
        counts, edges = np.histogram(s[s <= p99], bins=bins)
        fig = go.Figure(go.Bar(
            x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges),
            marker_color="#38bdf8",
            hovertemplate="%{x:,.1f}: %{y:,} rows<extra></extra>"))
        fig.add_vline(x=s.mean(), line_dash="dash", line_color="#7c3aed",
                      annotation_text=f"Mean {s.mean():.3g}", annotation_position="top right")
        fig.add_vline(x=s.median(), line_dash="dot", line_color="#ef4444",
                      annotation_text=f"Median {s.median():.3g}", annotation_position="top left")
        fig.update_layout(title=f"{c} (skewness {s.skew():.1f})", height=350, bargap=0.05,
                          xaxis_title=c, yaxis_title="Rows", showlegend=False)
        fig.update_yaxes(range=[0, counts.max() * 1.2])   # headroom for the labels
        col.plotly_chart(fig)

    st.info(
        "📊 **All three variables are positively skewed** (mean > median): most orders are small and "
        "low-priced, with a long tail of large orders. For this reason the median is a more reliable "
        "measure of a typical order than the mean, and the box plots below use a log scale.")
    # --- Outlier analysis (IQR) ---
    st.subheader("Outlier Analysis (IQR Method)",
             help="Box: middle 50% of values (Q1 to Q3), line inside = median. "
                  "Whiskers: lowest and highest values within the IQR limits. "
                  "Red diamond: maximum value. Values beyond the whiskers are outliers. "
                  "A log scale is used because of the strong right skew.")
    k = st.radio("Multiplier k", [1.5, 3.0], horizontal=True,
                 format_func=lambda x: f"k = {x}" + (" (standard)" if x == 1.5 else " (extreme outliers only)"),
                 help="Limits: Q1 − k·IQR and Q3 + k·IQR. Values outside these limits are flagged as outliers.")
    out = iqr_summary(df, ["Quantity", "NetPrice", "UnitPrice"], k)

    cols = st.columns(3)
    for col, (_, r) in zip(cols, out.iterrows()):
        fig = go.Figure()
        fig.add_trace(go.Box(
            x=[r["Variable"]], q1=[r["Q1"]], median=[r["Median"]], q3=[r["Q3"]],
            lowerfence=[r["Whisker low"]], upperfence=[r["Whisker high"]],
            marker_color="#7c3aed", name="Box", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[r["Variable"]], y=[r["Max"]], mode="markers", name="Max",
            marker=dict(color="#ef4444", size=9, symbol="diamond"),
            hovertemplate="Max: %{y:,.2f}<extra></extra>"))
        fig.update_layout(title=f"{r['Variable']}: {r['Outliers']:,} outliers ({r['Outlier %']:.1%})",
                          height=380, showlegend=False,
                          yaxis=dict(type="log", dtick=1, tickformat=",",
                                     title=f"{r['Variable']} (log scale)"))
        col.plotly_chart(fig)

    st.dataframe(
        out[["Variable", "Q1", "Median", "Q3", "IQR", "Lower limit", "Upper limit", "Max", "Outliers", "Outlier %"]],
        hide_index=True,
        column_config={"Outlier %": st.column_config.NumberColumn(format="percent")})

    q = out.set_index("Variable").loc["Quantity"]
    st.info(
        f"📌 **Decision: outliers were kept.** For Quantity, Q1 = {q['Q1']:.0f} and Q3 = {q['Q3']:.0f}, "
        f"so the upper limit is only {q['Upper limit']:.1f}: every order of {int(q['Upper limit']) + 1}+ "
        f"books is flagged. In B2B distribution these are genuine bulk orders, not errors, so the IQR rule "
        f"is too strict for this highly skewed variable. Removing them would understate revenue and demand. "
        f"Apart from duplicates, only clearly invalid records (zero quantity, prices below ₺1) were removed.")

    with st.expander("🔍  Sample of the cleaned data"):
        st.dataframe(df.head(20), hide_index=True, column_config={
            "InvoiceDate": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Month": st.column_config.DateColumn(format="MMM YYYY"),
        })


# ---------------------------------------------------------------
# 6) NAVIGATION
# ---------------------------------------------------------------
st.markdown("""
<div style="background: linear-gradient(120deg, #7c3aed 0%, #38bdf8 100%);
            padding: 26px 32px; border-radius: 14px; margin-bottom: 12px;">
  <div style="font-size: 2.2rem; font-weight: 700; color: white; line-height: 1.2;">
    📚 BookLens Analytics
  </div>
  <div style="font-size: 1rem; color: rgba(255, 255, 255, 0.9); margin-top: 6px;">
    Decision support for B2B book distribution · Sales, marketing and supply chain insights from 2020 transactions
  </div>
</div>
""", unsafe_allow_html=True)

nav = st.navigation({
    "Data": [
        st.Page(page_overview, url_path="overview", title="Dataset Overview", icon="📋", default=True),
        st.Page(page_data, url_path="preprocessing", title="Data Preprocessing", icon="🧹"),
    ],
    "Analytics": [
        st.Page(page_sales, url_path="sales", title="Sales Overview", icon="📈"),
        st.Page(page_forecast, url_path="forecast", title="Sales Forecast", icon="🔮"),
        st.Page(page_marketing, url_path="marketing", title="Customer Segmentation", icon="🎯"),
        st.Page(page_supply, url_path="supply-chain", title="ABC & Best Sellers", icon="📦"),
    ],
})
nav.run()

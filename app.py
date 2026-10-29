import warnings
warnings.filterwarnings("ignore")

from io import StringIO
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_URL = "https://gist.githubusercontent.com/rohithbollareddy/490136d929c54c0cc9d963a717ccb910/raw/2ae42c5c2901b14602af16944e4d84f48a86a544/flightRouteDomestic.csv"

st.set_page_config(page_title="Airfare Insight", page_icon="✈️", layout="wide")

@st.cache_data(show_spinner="Loading airfare dataset...")
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        response = requests.get(CSV_URL, timeout=30)
        response.raise_for_status()
        data = pd.read_csv(StringIO(response.text))
    data.columns = (data.columns.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False))
    data = data.loc[:, ~data.columns.str.contains("^unnamed", case=False)].copy()
    if "price" not in data.columns:
        raise ValueError("The dataset must contain a 'price' column.")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["price"]).drop_duplicates().reset_index(drop=True)
    return data

@st.cache_resource(show_spinner="Training airfare prediction model...")
def train_model(data):
    categorical_candidates = ["airline", "source_city", "destination_city", "class", "stops", "departure_time", "arrival_time"]
    numeric_candidates = ["days_left"]
    feature_cols = [c for c in categorical_candidates + numeric_candidates if c in data.columns]
    if not feature_cols:
        raise ValueError("No supported predictor columns were found.")
    cat_cols = [c for c in feature_cols if c in categorical_candidates]
    num_cols = [c for c in feature_cols if c in numeric_candidates]
    model_data = data.dropna(subset=["price"]).copy()
    if len(model_data) > 30000:
        model_data = model_data.sample(30000, random_state=42)
    X, y = model_data[feature_cols].copy(), model_data["price"].astype(float)
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols))
    preprocessor = ColumnTransformer(transformers)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=45, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R²": r2_score(y_test, pred),
    }
    return model, feature_cols, cat_cols, num_cols, metrics

def show_fig(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#102a43,#256d85); color:white; margin-bottom:1rem;}
.hero h1,.hero p {color:white;}
</style>
<div class="hero">
<h1>✈️ Airfare Insight</h1>
<p><b>Statistical Analysis of Airfare Prices & Travel Patterns</b></p>
<p>Explore fare patterns, compare airlines, run statistical tests, and estimate a ticket price using machine learning.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Optionally upload your own flight CSV", type=["csv"])
    st.caption("By default, the app loads the CSV used in your notebook from its public Gist URL.")
    st.divider()
    page = st.radio("Navigate", ["Overview", "Explore data", "Statistical analysis", "Price prediction"])

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Could not load the airfare dataset: {e}")
    st.info("Check your internet connection or upload the flight CSV using the sidebar.")
    st.stop()

st.caption(f"Dataset loaded: {len(df):,} rows × {len(df.columns):,} columns")

if page == "Overview":
    st.subheader("Dataset at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flight records", f"{len(df):,}")
    c2.metric("Average fare", f"₹{df['price'].mean():,.0f}")
    c3.metric("Median fare", f"₹{df['price'].median():,.0f}")
    c4.metric("Airlines", f"{df['airline'].nunique():,}" if "airline" in df else "N/A")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Fare distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["price"].dropna(), bins=30, edgecolor="white")
        ax.set_xlabel("Price (₹)"); ax.set_ylabel("Frequency"); ax.set_title("Distribution of Airfare Prices")
        show_fig(fig)
    with right:
        st.markdown("#### Fare box plot")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(df["price"].dropna(), vert=True)
        ax.set_ylabel("Price (₹)"); ax.set_title("Airfare Price Spread")
        show_fig(fig)
    if "airline" in df:
        st.markdown("#### Median fare by airline")
        airline_fares = df.groupby("airline")["price"].median().sort_values()
        fig, ax = plt.subplots(figsize=(10, max(4, len(airline_fares) * 0.35)))
        airline_fares.plot(kind="barh", ax=ax)
        ax.set_xlabel("Median price (₹)"); ax.set_ylabel("Airline")
        show_fig(fig)
        st.markdown("#### Number of flights by airline")
        st.bar_chart(df["airline"].value_counts())

elif page == "Explore data":
    st.subheader("Explore and filter flight records")
    with st.expander("Dataset structure", expanded=True):
        info = pd.DataFrame({"Column": df.columns, "Data type": [str(x) for x in df.dtypes], "Missing values": [int(x) for x in df.isna().sum()]})
        st.dataframe(info, use_container_width=True, hide_index=True)
        st.write("Descriptive statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)
    filtered = df.copy()
    filter_cols = [c for c in ["airline", "source_city", "destination_city", "class", "stops"] if c in filtered.columns]
    cols = st.columns(min(3, max(1, len(filter_cols))))
    for i, col in enumerate(filter_cols):
        vals = sorted(filtered[col].dropna().astype(str).unique().tolist())
        selected = cols[i % len(cols)].multiselect(f"Filter {col.replace('_',' ')}", vals, key=f"filter_{col}")
        if selected:
            filtered = filtered[filtered[col].astype(str).isin(selected)]
    if len(filtered):
        low, high = float(filtered["price"].min()), float(filtered["price"].max())
        price_range = st.slider("Price range (₹)", min_value=low, max_value=high if high > low else low + 1.0,
                                value=(low, high if high > low else low + 1.0))
        filtered = filtered[filtered["price"].between(*price_range)]
    st.write(f"Showing **{len(filtered):,}** matching records")
    st.dataframe(filtered, use_container_width=True, height=420)
    st.download_button("Download filtered data as CSV", filtered.to_csv(index=False).encode("utf-8"), "filtered_airfare_data.csv", "text/csv")

elif page == "Statistical analysis":
    st.subheader("Statistical analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Sampling & CLT", "Confidence interval & t-test", "Correlation", "ANOVA & Chi-square"])
    with tab1:
        st.markdown("#### Sampling")
        sample_size = st.slider("Sample size", min_value=5, max_value=min(500, len(df)), value=min(30, len(df)), step=1)
        sample = df.sample(sample_size, random_state=42)
        a, b = st.columns(2)
        a.metric("Population mean fare", f"₹{df['price'].mean():,.2f}")
        b.metric("Sample mean fare", f"₹{sample['price'].mean():,.2f}")
        st.dataframe(sample, use_container_width=True)
        st.markdown("#### Central Limit Theorem: distribution of sample means")
        reps = st.slider("Number of repeated samples", 100, 2000, 1000, 100)
        population = df["price"].dropna()
        means = [population.sample(sample_size, replace=True).mean() for _ in range(reps)]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(means, bins=25, edgecolor="white")
        ax.set_xlabel("Sample mean fare (₹)"); ax.set_ylabel("Frequency")
        ax.set_title("Central Limit Theorem — Sample Means")
        show_fig(fig)
        st.write(f"Mean of sample means: **₹{np.mean(means):,.2f}**")
    with tab2:
        st.markdown("#### 95% confidence interval and one-sample t-test")
        prices = pd.to_numeric(df["price"], errors="coerce").dropna()
        mean = prices.mean()
        ci = stats.t.interval(0.95, len(prices)-1, loc=mean, scale=stats.sem(prices))
        reference = st.number_input("Reference fare for one-sample t-test (₹)", min_value=0.0, value=5000.0, step=100.0)
        t_stat, p_value = stats.ttest_1samp(prices, reference)
        x, y, z = st.columns(3)
        x.metric("Mean airfare", f"₹{mean:,.2f}")
        y.metric("95% CI lower", f"₹{ci[0]:,.2f}")
        z.metric("95% CI upper", f"₹{ci[1]:,.2f}")
        st.write(f"**t-statistic:** {t_stat:.4f}")
        st.write(f"**p-value:** {p_value:.6g}")
        st.info("Reject H₀" if p_value < 0.05 else "Fail to reject H₀", icon="ℹ️")
    with tab3:
        numeric = df.select_dtypes(include="number").copy()
        if numeric.shape[1] < 2:
            st.warning("At least two numeric columns are needed.")
        else:
            st.markdown("#### Covariance matrix")
            st.dataframe(numeric.cov(), use_container_width=True)
            st.markdown("#### Correlation matrix")
            st.dataframe(numeric.corr(), use_container_width=True)
            candidates = [c for c in numeric.columns if c != "price"]
            if candidates:
                col = st.selectbox("Compare price with numeric column", candidates)
                pair = numeric[["price", col]].dropna()
                if len(pair) >= 3 and pair["price"].nunique() > 1 and pair[col].nunique() > 1:
                    r, p1 = pearsonr(pair["price"], pair[col])
                    rho, p2 = spearmanr(pair["price"], pair[col])
                    x, y = st.columns(2)
                    x.metric("Pearson correlation (r)", f"{r:.4f}")
                    y.metric("Spearman correlation (ρ)", f"{rho:.4f}")
                    st.write(f"Pearson p-value: `{p1:.6g}` · Spearman p-value: `{p2:.6g}`")
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.scatter(pair[col], pair["price"], alpha=0.5)
                    ax.set_xlabel(col); ax.set_ylabel("Price (₹)"); ax.set_title(f"Price vs {col}")
                    show_fig(fig)
                else:
                    st.warning("Not enough variation in this column to calculate correlation.")
    with tab4:
        if "airline" not in df:
            st.warning("The dataset does not contain an airline column.")
        else:
            groups = [pd.to_numeric(g["price"], errors="coerce").dropna() for _, g in df.groupby("airline")]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                F, p = f_oneway(*groups)
                st.markdown("#### One-way ANOVA: airfare across airlines")
                st.write(f"**F-statistic:** {F:.4f}")
                st.write(f"**p-value:** {p:.6g}")
                st.info("Reject H₀: at least one airline's mean fare differs." if p < 0.05 else "Fail to reject H₀: no significant mean difference found.")
            categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
            other_cat = next((c for c in categorical if c != "airline"), None)
            if other_cat:
                st.markdown(f"#### Chi-square: airline vs {other_cat}")
                table = pd.crosstab(df["airline"], df[other_cat])
                st.dataframe(table, use_container_width=True)
                chi2, pchi, dof, expected = chi2_contingency(table)
                st.write(f"**Chi-square:** {chi2:.4f} · **p-value:** {pchi:.6g} · **df:** {dof}")
                st.info("Reject H₀: variables are associated." if pchi < 0.05 else "Fail to reject H₀: no significant association found.")
            else:
                st.caption("No second categorical column found for a chi-square test.")

elif page == "Price prediction":
    st.subheader("Airfare Price Estimator")
    st.write("Choose trip details to estimate the fare. This is a model estimate, not a live ticket price.")
    try:
        model, feature_cols, cat_cols, num_cols, metrics = train_model(df)
    except Exception as e:
        st.error(f"Could not train the prediction model: {e}")
        st.stop()
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean absolute error", f"₹{metrics['MAE']:,.0f}")
    c2.metric("Root mean squared error", f"₹{metrics['RMSE']:,.0f}")
    c3.metric("Test R² score", f"{metrics['R²']:.3f}")
    categorical_candidates = ["airline", "source_city", "destination_city", "class", "stops", "departure_time", "arrival_time"]
    label_map = {"airline":"Airline", "source_city":"Source city", "destination_city":"Destination city", "class":"Travel class",
                 "stops":"Stops", "departure_time":"Departure time", "arrival_time":"Arrival time", "days_left":"Days before departure"}
    input_values = {}
    cols = st.columns(2)
    for i, col in enumerate(feature_cols):
        with cols[i % 2]:
            if col in cat_cols:
                values = sorted(df[col].dropna().astype(str).unique().tolist())
                if values:
                    input_values[col] = st.selectbox(label_map.get(col, col.replace("_"," ").title()), values, key=f"predict_{col}")
                else:
                    input_values[col] = ""
            else:
                default = float(pd.to_numeric(df[col], errors="coerce").median()) if df[col].notna().any() else 15.0
                minimum = float(pd.to_numeric(df[col], errors="coerce").min()) if df[col].notna().any() else 0.0
                maximum = float(pd.to_numeric(df[col], errors="coerce").max()) if df[col].notna().any() else max(default, 30.0)
                input_values[col] = st.number_input(label_map.get(col, col.replace("_"," ").title()), min_value=0.0,
                    max_value=max(maximum, 0.0), value=min(max(default, 0.0), max(maximum, 0.0)), step=1.0, key=f"predict_{col}")
    if st.button("Estimate airfare", type="primary", use_container_width=True):
        row = pd.DataFrame([{col: input_values[col] for col in feature_cols}], columns=feature_cols)
        prediction = float(model.predict(row)[0])
        st.success(f"Estimated airfare: ₹{prediction:,.0f}")
        st.caption("Estimate generated from the notebook's Random Forest model. Actual fares may differ.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import yfinance as yf

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(layout="wide")
st.title("📊 Time Series Harmonisation Toolkit")
st.caption("Upscale & downscale multiple time series to a common frequency")

# ==============================
# HELPER: Safe datetime conversion (pandas 3.x compatible)
# ==============================
def safe_to_datetime(series):
    """Convert a series to datetime safely, regardless of current dtype."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors='coerce')

def df_with_datetime_index(df):
    """Return a copy of df with the first column converted to a datetime index."""
    df = df.copy()
    date_col = safe_to_datetime(df.iloc[:, 0])
    data = df.iloc[:, 1:].copy()
    data.index = date_col.values
    data = data[date_col.notna().values].sort_index()
    return data

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel Workbook", type=["xlsx"])

# ==============================
# FREQUENCY DETECTION
# ==============================
def detect_frequency(df):
    col = safe_to_datetime(df.iloc[:, 0])
    col = col.dropna().sort_values().reset_index(drop=True)

    diffs = col.diff().dropna()
    if len(diffs) == 0:
        return "M"

    avg_days = diffs.dt.days.mean()

    if avg_days <= 2:
        return "D"
    elif avg_days <= 8:
        return "W"
    elif avg_days <= 35:
        return "M"
    elif avg_days <= 100:
        return "Q"
    else:
        return "A"

# ==============================
# FREQUENCY RANKING
# ==============================
freq_rank = {"D": 1, "W": 2, "M": 3, "Q": 4, "A": 5}

# ==============================
# USD TO ZAR CONVERSION
# ==============================
def fetch_usdzar(start_date, end_date):
    try:
        ticker = yf.download("USDZAR=X", start=start_date, end=end_date, progress=False)
        if ticker.empty:
            return None
        fx = ticker[["Close"]].copy()
        fx.columns = ["USDZAR"]
        fx.index = pd.to_datetime(fx.index)
        return fx
    except Exception as e:
        return None

# ==============================
# DENTON METHOD (PROPORTIONAL)
# ==============================
def denton_method(low_freq_series, high_freq_index):
    """
    Denton Proportional First Difference (PFD) method.

    Distributes each low-frequency aggregate value across the high-frequency
    sub-periods by minimising the sum of squared first differences of the
    ratio (adjusted / indicator), subject to the constraint that the
    high-frequency values sum to the low-frequency aggregate within each period.

    If no indicator series is available a uniform (flat) indicator of ones
    is used, which reduces to simple proportional distribution.

    Parameters
    ----------
    low_freq_series : pd.Series
        Low-frequency aggregates indexed by their period start dates.
    high_freq_index : pd.DatetimeIndex
        The target high-frequency date index.

    Returns
    -------
    pd.Series
        High-frequency series that aggregates back to the low-frequency totals.
    """
    n = len(high_freq_index)
    lf_dates  = low_freq_series.index.sort_values()
    lf_values = low_freq_series.reindex(lf_dates).values
    m = len(lf_dates)

    C = np.zeros((m, n))
    for i, lf_date in enumerate(lf_dates):
        if i + 1 < m:
            mask = (high_freq_index >= lf_date) & (high_freq_index < lf_dates[i + 1])
        else:
            mask = high_freq_index >= lf_date
        C[i, mask] = 1.0

    p = np.ones(n)

    D = np.zeros((n - 1, n))
    for t in range(n - 1):
        D[t, t]     = -1.0
        D[t, t + 1] =  1.0

    P_inv = np.diag(1.0 / p)
    D_p   = D @ P_inv
    Q     = D_p.T @ D_p

    try:
        Q_inv   = np.linalg.inv(Q + np.eye(n) * 1e-10)
        CQinv   = C @ Q_inv
        CQinvCt = CQinv @ C.T
        lambdas = np.linalg.solve(CQinvCt, lf_values)
        x       = Q_inv @ C.T @ lambdas
    except np.linalg.LinAlgError:
        x = np.zeros(n)
        for i in range(m):
            idx = np.where(C[i] == 1)[0]
            if len(idx) > 0 and not np.isnan(lf_values[i]):
                x[idx] = lf_values[i] / len(idx)

    return pd.Series(x, index=high_freq_index)

# ==============================
# METHODS
# ==============================
def downscale(df, freq, method):
    data = df_with_datetime_index(df)

    freq_map = {
        "D": "D",
        "W": "W",
        "M": "MS",
        "Q": "QS",
        "A": "YS"
    }
    pandas_freq = freq_map.get(freq, freq)

    if method == "First":
        return data.resample(pandas_freq).first()
    elif method == "Mean":
        return data.resample(pandas_freq).mean()
    elif method == "Median":
        return data.resample(pandas_freq).median()
    elif method == "Sum":
        return data.resample(pandas_freq).sum()
    else:
        return data.resample(pandas_freq).mean()


def upscale(df, freq, method):
    data = df_with_datetime_index(df)
    data = data.apply(pd.to_numeric, errors='coerce')

    freq_map = {
        "D": "D",
        "W": "W",
        "M": "MS",
        "Q": "QS",
        "A": "YS"
    }
    pandas_freq = freq_map.get(freq, freq)

    end_date = data.index.max()

    if freq == "Q":
        end_date = end_date + pd.DateOffset(months=9)
    elif freq == "M":
        end_date = end_date + pd.DateOffset(months=11)
    elif freq == "W":
        end_date = end_date + pd.DateOffset(weeks=52)

    full_index = pd.date_range(
        start=data.index.min(),
        end=end_date,
        freq=pandas_freq
    )

    if method == "Denton":
        result_cols = {}
        for col in data.columns:
            result_cols[col] = denton_method(data[col].dropna(), full_index)
        return pd.DataFrame(result_cols, index=full_index)

    data = data.reindex(full_index)

    if method == "Forward Fill":
        return data.ffill()
    elif method == "Linear":
        return data.interpolate(method="linear", limit_direction="both")
    elif method == "Spline":
        try:
            return data.interpolate(method="spline", order=2)
        except:
            return data.interpolate(method="linear", limit_direction="both")
    else:
        return data.ffill()

# ==============================
# MAIN APP
# ==============================
if uploaded_file:

    try:
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        st.sidebar.success("Workbook loaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    sheet_names = list(sheets.keys())

    # detect frequencies
    detected_freqs = {
        name: detect_frequency(sheets[name])
        for name in sheet_names
    }

    # TARGET SHEET
    target_name = st.sidebar.selectbox("Target Sheet", sheet_names)
    target_freq = detect_frequency(sheets[target_name])

    st.sidebar.subheader("Settings")
    st.sidebar.markdown(f"**Target Sheet:** {target_name}")
    st.sidebar.markdown(f"**Target Frequency:** {target_freq}")

    down_method = st.sidebar.selectbox(
        "Downscaling Method",
        ["First", "Mean", "Median", "Sum"]
    )

    up_method = st.sidebar.selectbox(
        "Upscaling Method",
        ["Forward Fill", "Linear", "Spline", "Denton"]
    )

    if up_method == "Denton":
        st.sidebar.caption(
            "ℹ️ **Denton:** Distributes each low-frequency "
            "aggregate across sub-periods by minimising first-difference "
            "revisions while preserving aggregation constraints."
        )

    st.sidebar.subheader("💱 Currency Conversion")
    enable_conversion = st.sidebar.checkbox("Convert USD → ZAR", value=False)

    usd_series = []
    if enable_conversion:
        st.sidebar.caption("ℹ️ USD/ZAR rate fetched automatically from Yahoo Finance. Coverage starts ~1994.")
        for name in sheet_names:
            if st.sidebar.checkbox(f"Convert {name}", value=False):
                usd_series.append(name)

    st.sidebar.subheader("Date Settings")
    intersect_dates = st.sidebar.checkbox("Intersect Dates (align start & end)", value=False)

    custom_dates = st.sidebar.checkbox("Set Custom Date Range", value=False)

    if custom_dates:
        all_dates = []
        for name in sheet_names:
            col = safe_to_datetime(sheets[name].iloc[:, 0])
            all_dates.extend(col.dropna().tolist())

        global_min = min(all_dates).date()
        global_max = max(all_dates).date()

        custom_start = st.sidebar.date_input("Start Date", value=global_min, min_value=global_min, max_value=global_max)
        custom_end   = st.sidebar.date_input("End Date",   value=global_max, min_value=global_min, max_value=global_max)

    # ==============================
    # SERIES SELECTION
    # ==============================
    st.sidebar.subheader("Series Selection")
    selected_series = []

    for name in sheet_names:
        freq = detected_freqs[name]
        label = f"{name} ({freq})"
        if st.sidebar.checkbox(label, value=True):
            selected_series.append(name)

    # ==============================
    # RUN / RESET BUTTONS
    # ==============================
    if "run_clicked" not in st.session_state:
        st.session_state.run_clicked = False

    run_btn = st.sidebar.button("▶ Run Harmonisation")
    reset_btn = st.sidebar.button("🔄 Reset")

    if run_btn:
        st.session_state.run_clicked = True

    if reset_btn:
        st.session_state.run_clicked = False
        st.rerun()

    # ==============================
    # INFO MESSAGE
    # ==============================
    if not st.session_state.run_clicked:
        st.info("👉 Select your settings and click 'Run Harmonisation' to generate output.")

    # ==============================
    # PROCESSING
    # ==============================
    if st.session_state.run_clicked:

        results = {}

        if custom_dates and intersect_dates:
            st.sidebar.caption("ℹ️ Intersection is applied first, then custom range.")

        # ==============================
        # CURRENCY CONVERSION (USD → ZAR)
        # ==============================
        if usd_series:
            all_dates = []
            for name in selected_series:
                col = safe_to_datetime(sheets[name].iloc[:, 0])
                all_dates.extend(col.dropna().tolist())

            fx = fetch_usdzar(
                start_date=min(all_dates).date(),
                end_date=max(all_dates).date()
            )

            if fx is not None:
                for name in usd_series:
                    data = df_with_datetime_index(sheets[name])
                    data = data.apply(pd.to_numeric, errors='coerce')

                    fx_aligned = fx.reindex(data.index, method="ffill").ffill().bfill()

                    for col in data.columns:
                        data[col] = data[col] * fx_aligned["USDZAR"].values

                    restored = data.reset_index()
                    restored.columns = [sheets[name].columns[0]] + list(data.columns)
                    sheets[name] = restored
                    st.sidebar.success(f"✅ {name} converted to ZAR")
            else:
                st.warning("⚠️ Could not fetch USD/ZAR rate from Yahoo Finance. Check your internet connection.")

        # ==============================
        # HARMONISATION LOOP
        # ==============================
        for name in selected_series:
            df = sheets[name].copy()

            if df.shape[1] < 2:
                st.warning(f"{name} skipped (not enough columns)")
                continue

            data = df_with_datetime_index(df)

            if data.empty:
                st.warning(f"{name} has no valid data")
                continue

            freq = detect_frequency(df)

            try:
                if name == target_name:
                    results[name] = data

                elif freq_rank[freq] < freq_rank[target_freq]:
                    results[name] = downscale(df, target_freq, down_method)

                elif freq_rank[freq] > freq_rank[target_freq]:
                    results[name] = upscale(df, target_freq, up_method)

                else:
                    results[name] = data

                results[name].columns = [
                    f"{name}_{col} ({target_freq})"
                    for col in results[name].columns
                ]

            except Exception as e:
                st.error(f"{name} failed: {e}")

        # Apply date filters AFTER all series are processed
        series_ranges = None

        if intersect_dates and results:
            series_ranges = {
                n: (d.index.min(), d.index.max())
                for n, d in results.items()
            }

            common_start = max(d.index.min() for d in results.values())
            common_end   = min(d.index.max() for d in results.values())

            if common_start <= common_end:
                results = {
                    n: d.loc[common_start:common_end]
                    for n, d in results.items()
                }
            else:
                series_ranges = None

        if custom_dates and results:
            if custom_start <= custom_end:
                results = {
                    n: d.loc[str(custom_start):str(custom_end)]
                    for n, d in results.items()
                }
            else:
                st.warning("⚠️ Custom start date is after end date — skipping custom range.")

        # ==============================
        # LAYOUT
        # ==============================
        col1, col2 = st.columns(2)

        # -------- ORIGINAL --------
        with col1:
            st.subheader("Original Series")
            fig, ax = plt.subplots()

            for name in selected_series:
                try:
                    data = df_with_datetime_index(sheets[name])
                    for col in data.columns:
                        ax.plot(data.index, data[col], label=name)
                except:
                    continue

            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # -------- HARMONISED --------
        with col2:
            st.subheader("Harmonised Series")
            fig2, ax2 = plt.subplots()

            for name, df in results.items():
                for col in df.columns:
                    ax2.plot(df.index, df[col], label=name)

            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

        # ==============================
        # SUMMARY
        # ==============================
        st.subheader("Data Summary")
        c1, c2, c3, c4 = st.columns(4)

        aligned_length = min(len(df) for df in results.values()) if results else 0

        c1.metric("Total Series", len(selected_series))
        c2.metric("Target Frequency", target_freq)
        c3.metric("Aligned Length", aligned_length)
        c4.metric("Sheets Loaded", len(sheet_names))

        if intersect_dates and series_ranges:
            st.markdown("**Series Date Ranges (before intersection)**")
            common_start = max(s for s, e in series_ranges.values())
            common_end   = min(e for s, e in series_ranges.values())

            for name, (s, e) in series_ranges.items():
                st.info(f"📅 **{name}**: {s.date()} → {e.date()}")

            st.success(f"✅ Intersected window applied: **{common_start.date()} → {common_end.date()}**")

        elif intersect_dates and series_ranges is None:
            st.warning("⚠️ No overlapping dates found across series.")

        if custom_dates:
            if custom_start <= custom_end:
                st.info(f"📅 **Custom range applied:** {custom_start} → {custom_end}")
            else:
                st.warning("⚠️ Custom start date is after end date.")

        # ==============================
        # QUALITY CHECKS
        # ==============================
        st.subheader("Quality Checks")
        qc1, qc2, qc3 = st.columns(3)

        if results:
            combined = pd.concat(results.values(), axis=1).sort_index()

            missing    = combined.isna().sum().sum() > 0
            duplicates = combined.index.duplicated().any()

            freq_map_qc = {"D": "D", "W": "W-MON", "M": "MS", "Q": "QS", "A": "YS"}
            full_range = pd.date_range(
                start=combined.index.min(),
                end=combined.index.max(),
                freq=freq_map_qc.get(target_freq, target_freq)
            )

            gaps = len(full_range.difference(combined.index)) > 0

            qc1.success("No Missing Values" if not missing else "Missing Values Found")
            qc2.success("No Duplicates" if not duplicates else "Duplicates Found")
            qc3.success("No Date Gaps" if not gaps else "Date Gaps Detected")

        else:
            qc1.warning("No data")
            qc2.warning("No data")
            qc3.warning("No data")

        # ==============================
        # DESCRIPTIVE STATISTICS
        # ==============================
        st.subheader("Descriptive Statistics")

        if results:
            combined = pd.concat(results.values(), axis=1).sort_index()

            stats = combined.describe().T
            stats.index = [idx.split("_")[0] if "_" in idx else idx for idx in stats.index]
            st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)

            st.subheader("Correlation Matrix")

            if combined.shape[1] > 1:
                corr = combined.corr()
                corr.columns = [c.split("_")[0] if "_" in c else c for c in corr.columns]
                corr.index   = corr.columns

                st.dataframe(corr.style.format("{:.4f}").background_gradient(
                    cmap="RdYlGn", vmin=-1, vmax=1
                ), use_container_width=True)

            else:
                st.info("Add more than one series to compute correlations.")

        # ==============================
        # EXPORT
        # ==============================
        st.subheader("Export")

        if results:
            try:
                buffer = io.BytesIO()

                with pd.ExcelWriter(buffer) as writer:
                    for name, df in results.items():
                        df.to_excel(writer, sheet_name=name)

                st.download_button(
                    label="📥 Download Excel File",
                    data=buffer.getvalue(),
                    file_name="harmonised_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Export failed: {e}")

        else:
            st.warning("No data available to export.")

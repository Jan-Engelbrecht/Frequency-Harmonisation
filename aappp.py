import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(layout="wide")
st.title("📊 Time Series Harmonisation Toolkit")
st.caption("Upscale & downscale multiple time series to a common frequency")

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel Workbook", type=["xlsx"])

# ==============================
# FREQUENCY DETECTION
# ==============================
def detect_frequency(df):
    df = df.copy()
    col = df.iloc[:, 0]
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(col):
        col = pd.to_datetime(col, errors='coerce')
    
    # Work directly with the series, don't assign back to df
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
# METHODS
# ==============================
def downscale(df, freq, method):
    df = df.copy()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.dropna().set_index(df.columns[0]).sort_index()

    freq_map = {
        "D": "D",
        "W": "W",   
        "M": "MS",       # month start
        "Q": "QS",       # quarter start
        "A": "YS"        # year start
    }
    pandas_freq = freq_map.get(freq, freq)

    if method == "First":
        return df.resample(pandas_freq).first()
    elif method == "Mean":
        return df.resample(pandas_freq).mean()
    elif method == "Median":
        return df.resample(pandas_freq).median()
    else:
        return df.resample(pandas_freq).mean()


def upscale(df, freq, method):
    df = df.copy()

    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.dropna().set_index(df.columns[0]).sort_index()

    df = df.apply(pd.to_numeric, errors='coerce')

    freq_map = {
        "D": "D",
        "W": "W",
        "M": "MS",
        "Q": "QS",
        "A": "YS"
    }
    pandas_freq = freq_map.get(freq, freq)

    end_date = df.index.max()

    if freq == "Q":
        end_date = end_date + pd.DateOffset(months=9)
    elif freq == "M":
        end_date = end_date + pd.DateOffset(months=11)
    elif freq == "W":
        end_date = end_date + pd.DateOffset(weeks=52)

    full_index = pd.date_range(
        start=df.index.min(),
        end=end_date,
        freq=pandas_freq
    )

    df = df.reindex(full_index)

    if method == "Forward Fill":
        return df.ffill()
    elif method == "Linear":
        return df.interpolate(method="linear", limit_direction="both")
    elif method == "Spline":
        try:
            return df.interpolate(method="spline", order=2)
        except:
            return df.interpolate(method="linear", limit_direction="both")
    else:
        return df.ffill()

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

    #detect frequencies
    detected_freqs = {
        name: detect_frequency(sheets[name])
        for name in sheet_names
    }

    # TARGET = FIRST SHEET    
    target_name = st.sidebar.selectbox("Target Sheet", sheet_names)
    target_freq = detect_frequency(sheets[target_name])
    
    st.sidebar.subheader("Settings")
    st.sidebar.markdown(f"**Target Sheet:** {target_name}")
    st.sidebar.markdown(f"**Target Frequency:** {target_freq}")

    down_method = st.sidebar.selectbox(
        "Downscaling Method",
        ["First", "Mean", "Median"]
    )

    up_method = st.sidebar.selectbox(
        "Upscaling Method",
        ["Forward Fill", "Linear", "Spline"]
    )

    intersect_dates = st.sidebar.checkbox("Intersect Dates (align start & end)", value=False)

    custom_dates = st.sidebar.checkbox("Set Custom Date Range", value=False)

    if custom_dates:
        all_dates = []
        for name in sheet_names:
            df = sheets[name].copy()
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
            all_dates.extend(df.iloc[:, 0].dropna().tolist())
    
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

        for name in selected_series:
            df = sheets[name].copy()

            if df.shape[1] < 2:
                st.warning(f"{name} skipped (not enough columns)")
                continue

            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
            df = df.dropna()

            if df.empty:
                st.warning(f"{name} has no valid data")
                continue

            freq = detect_frequency(df)

            try:
                if name == target_name:
                    df = df.set_index(df.columns[0])
                    results[name] = df

                elif freq_rank[freq] < freq_rank[target_freq]:
                    results[name] = downscale(df, target_freq, down_method)

                elif freq_rank[freq] > freq_rank[target_freq]:
                    results[name] = upscale(df, target_freq, up_method)

                else:
                    results[name] = df.set_index(df.columns[0])

                results[name].columns = [
                    f"{name}_{col} ({target_freq})"
                    for col in results[name].columns
                ]

            except Exception as e:
                st.error(f"{name} failed: {e}")

            if intersect_dates and results:
                series_ranges = {
                    name: (df.index.min(), df.index.max())
                    for name, df in results.items()
                }
            
                common_start = max(df.index.min() for df in results.values())
                common_end   = min(df.index.max() for df in results.values())
            
                if common_start <= common_end:
                    results = {
                        name: df.loc[common_start:common_end]
                        for name, df in results.items()
                    }
                else:
                    series_ranges = None  # flag: no overlap
            else:
                series_ranges = None

            if custom_dates and results:
                if custom_start <= custom_end:
                    results = {
                        name: df.loc[str(custom_start):str(custom_end)]
                        for name, df in results.items()
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
                    df = sheets[name].copy()
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                    df = df.dropna().set_index(df.columns[0])

                    for col in df.columns:
                        ax.plot(df.index, df[col], label=f"{name}_{col}")
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
                    ax2.plot(df.index, df[col], label=col)

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

            missing = combined.isna().sum().sum() > 0
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

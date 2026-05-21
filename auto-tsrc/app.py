import os
import json
import subprocess
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Auto-TSRC Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

OUTPUTS_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
DATA_DIR = os.path.join("data", "ucr")

def get_available_datasets():
    if not os.path.exists(REPORTS_DIR):
        return []
    datasets = []
    for f in os.listdir(REPORTS_DIR):
        if f.endswith("_best_params.json"):
            datasets.append(f.replace("_best_params.json", ""))
    return sorted(datasets)

def get_all_ucr_datasets():
    if not os.path.exists(DATA_DIR):
        return []
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

def main():
    st.title("📈 Auto-TSRC Research Dashboard")
    st.markdown("Explore the results of the **AutoML-Guided Robust Curriculum Distillation** pipeline.")

    # Sidebar: Run new experiment
    st.sidebar.header("🚀 Run Experiment")
    all_datasets = get_all_ucr_datasets()
    if all_datasets:
        target_ds = st.sidebar.selectbox("Select Dataset to Train", all_datasets)
        num_trials = st.sidebar.slider("Optuna Search Trials", 1, 20, 5)
        num_epochs = st.sidebar.slider("Epochs per Trial", 2, 20, 5)
        if st.sidebar.button("Train Auto-TSRC"):
            with st.spinner(f"Running Optuna search on {target_ds}... (check terminal)"):
                cmd = ["python", "main.py", "--datasets", target_ds, "--trials", str(num_trials), 
                       "--trial-epochs", str(num_epochs), "--final-epochs", "10", "--quick"]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    st.sidebar.success("Training Complete!")
                    st.rerun()
                else:
                    st.sidebar.error("Training Failed!")
                    st.sidebar.text(res.stderr)
    else:
        st.sidebar.warning("No UCR data found. Please run python main.py --download-data")

    # Dashboard display
    st.sidebar.markdown("---")
    st.sidebar.header("📊 View Results")
    processed_datasets = get_available_datasets()
    
    if not processed_datasets:
        st.info("No processed results found yet. Run an experiment from the sidebar!")
        return

    selected_dataset = st.sidebar.selectbox("Select Results to View", processed_datasets)
    view_mode = st.sidebar.radio("View Mode", ["Dashboard View", "Full Research Report"])

    if view_mode == "Dashboard View":
        render_dashboard(selected_dataset)
    else:
        render_report(selected_dataset)

def render_dashboard(dataset: str):
    st.header(f"Results for `{dataset}`")

    comp_path = os.path.join(TABLES_DIR, f"{dataset}_comparison_metrics.csv")
    if os.path.exists(comp_path):
        st.subheader("🏆 Model Comparison")
        df_comp = pd.read_csv(comp_path)
        def highlight_auto(row):
            if "Auto-TSRC" in str(row['model_name']):
                return ['background-color: rgba(76, 175, 80, 0.2)'] * len(row)
            return [''] * len(row)
        st.dataframe(df_comp.style.apply(highlight_auto, axis=1))

    rob_path = os.path.join(TABLES_DIR, f"{dataset}_robustness_metrics.csv")
    if os.path.exists(rob_path):
        st.subheader("🛡️ Robustness Evaluation")
        df_rob = pd.read_csv(rob_path)
        st.dataframe(df_rob)

    st.markdown("---")
    st.subheader("🔍 Explainability Outputs")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. Embedding PCA**")
        pca_path = os.path.join(PLOTS_DIR, f"{dataset}_embedding_pca.png")
        if os.path.exists(pca_path): st.image(pca_path)
        st.markdown("**3. Reconstruction Error Heatmap**")
        heatmap_path = os.path.join(PLOTS_DIR, f"{dataset}_error_heatmap.png")
        if os.path.exists(heatmap_path): st.image(heatmap_path)
    with col2:
        st.markdown("**2. Cluster Prototypes**")
        proto_path = os.path.join(PLOTS_DIR, f"{dataset}_cluster_prototypes.png")
        if os.path.exists(proto_path): st.image(proto_path)
        st.markdown("**4. Reconstruction Overlay**")
        overlay_path = os.path.join(PLOTS_DIR, f"{dataset}_reconstruction_overlay.png")
        if os.path.exists(overlay_path): st.image(overlay_path)
            
    st.markdown("---")
    st.subheader("⚙️ Optuna Search Parameters")
    params_path = os.path.join(REPORTS_DIR, f"{dataset}_best_params.json")
    if os.path.exists(params_path):
        with st.expander("View Best Hyperparameters JSON"):
            with open(params_path, "r") as f:
                params = json.load(f)
            st.json(params)

def render_report(dataset: str):
    st.header(f"Research Report: `{dataset}`")
    report_path = os.path.join(REPORTS_DIR, f"{dataset}_research_report.md")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

if __name__ == "__main__":
    main()

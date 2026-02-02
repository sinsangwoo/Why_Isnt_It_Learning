"""Real-time gradient monitoring dashboard using Streamlit."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from gradient_pathology.analyzer import GradientAnalyzer
from gradient_pathology.core import GradientPathology
from gradient_pathology.experiments import create_deep_network


def plot_gradient_distribution(report: any) -> plt.Figure:
    """Plot gradient distribution across layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Layer-wise mean gradients
    layer_indices = [s.layer_index for s in report.layer_stats]
    layer_means = [abs(s.mean) for s in report.layer_stats]
    colors = [
        "red" if s.diagnose() == GradientPathology.VANISHING
        else "orange" if s.diagnose() == GradientPathology.EXPLODING
        else "yellow" if s.diagnose() == GradientPathology.UNSTABLE
        else "green"
        for s in report.layer_stats
    ]

    ax1.bar(layer_indices, layer_means, color=colors, alpha=0.7)
    ax1.axhline(y=1e-7, color="red", linestyle="--", label="Vanishing threshold")
    ax1.axhline(y=1e2, color="orange", linestyle="--", label="Exploding threshold")
    ax1.set_yscale("log")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Mean Gradient (log scale)")
    ax1.set_title("Gradient Flow Across Layers")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Pathology distribution pie chart
    pathology_counts = {}
    for stats in report.layer_stats:
        p = stats.diagnose()
        pathology_counts[p.value] = pathology_counts.get(p.value, 0) + 1

    ax2.pie(
        pathology_counts.values(),
        labels=pathology_counts.keys(),
        autopct="%1.1f%%",
        colors=["green", "red", "orange", "gray", "yellow"],
    )
    ax2.set_title("Gradient Pathology Distribution")

    plt.tight_layout()
    return fig


def run_dashboard() -> None:
    """Launch Streamlit dashboard."""
    st.set_page_config(page_title="Gradient Pathology Dashboard", layout="wide")

    st.title("🔬 Gradient Pathology Monitor")
    st.markdown(
        """
    **Real-time diagnostic tool for deep learning training stability**  
    *From high school curiosity to production ML diagnostics*
    """
    )

    # Sidebar: Model configuration
    st.sidebar.header("Model Configuration")
    depth = st.sidebar.slider("Network Depth", 5, 100, 20)
    activation = st.sidebar.selectbox("Activation", ["relu", "sigmoid", "tanh", "gelu"])
    hidden_size = st.sidebar.slider("Hidden Size", 32, 512, 64)
    use_norm = st.sidebar.checkbox("Use LayerNorm", value=False)

    # Analysis parameters
    st.sidebar.header("Analysis Settings")
    num_steps = st.sidebar.slider("Gradient Samples", 10, 200, 50)

    # Run analysis button
    if st.sidebar.button("🚀 Analyze Gradients", type="primary"):
        with st.spinner("Building model and analyzing gradients..."):
            # Create model
            model = create_deep_network(
                depth=depth,
                activation=activation,
                hidden_size=hidden_size,
                use_norm=use_norm,
            )

            # Analyze
            analyzer = GradientAnalyzer(model, device="cpu")
            report = analyzer.diagnose(num_steps=num_steps)

            # Store in session state
            st.session_state["report"] = report
            st.session_state["model_config"] = {
                "depth": depth,
                "activation": activation,
                "hidden_size": hidden_size,
                "use_norm": use_norm,
            }

    # Display results
    if "report" in st.session_state:
        report = st.session_state["report"]
        config = st.session_state["model_config"]

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Global Mean Gradient", f"{report.global_mean:.2e}")
        with col2:
            st.metric("Global Std Gradient", f"{report.global_std:.2e}")
        with col3:
            problematic = len(report.get_problematic_layers())
            st.metric(
                "Problematic Layers",
                f"{problematic}/{len(report.layer_stats)}",
                delta=f"-{len(report.layer_stats) - problematic}" if problematic > 0 else None,
            )
        with col4:
            healthy_ratio = (
                len(report.layer_stats) - problematic
            ) / len(report.layer_stats) * 100
            st.metric("Health Score", f"{healthy_ratio:.1f}%")

        # Visualizations
        st.subheader("📊 Gradient Analysis")
        fig = plot_gradient_distribution(report)
        st.pyplot(fig)

        # Detailed report
        with st.expander("📋 Detailed Layer-by-Layer Report"):
            st.code(report.summary(), language="text")

        # Recommendations
        if report.get_problematic_layers():
            st.subheader("💡 Recommendations")
            st.warning(
                f"⚠️ Found {len(report.get_problematic_layers())} problematic layers. "
                "See detailed recommendations below."
            )

            for layer in report.get_problematic_layers():
                pathology = layer.diagnose()
                if pathology == GradientPathology.VANISHING:
                    st.error(
                        f"**{layer.layer_name}**: Vanishing gradients (mean={layer.mean:.2e})\n"
                        "- Try: ReLU/GELU activation\n"
                        "- Try: He/Xavier initialization\n"
                        "- Try: LayerNorm or BatchNorm"
                    )
                elif pathology == GradientPathology.EXPLODING:
                    st.error(
                        f"**{layer.layer_name}**: Exploding gradients (mean={layer.mean:.2e})\n"
                        "- Use gradient clipping\n"
                        "- Reduce learning rate\n"
                        "- Check weight initialization"
                    )
                elif pathology == GradientPathology.UNSTABLE:
                    st.warning(
                        f"**{layer.layer_name}**: Unstable gradients (std={layer.std:.2e})\n"
                        "- Consider: Gradient clipping\n"
                        "- Consider: LayerNorm"
                    )
        else:
            st.success("✅ All layers show healthy gradient flow!")

        # Model architecture
        with st.expander("🏗️ Model Architecture"):
            st.json(config)

    else:
        st.info("👈 Configure your model in the sidebar and click 'Analyze Gradients' to start.")


if __name__ == "__main__":
    run_dashboard()

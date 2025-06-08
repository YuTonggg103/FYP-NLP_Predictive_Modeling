import plotly.graph_objects as go
import numpy as np
import streamlit as st
import json
import joblib

def confusion_matrix(TN,TP,FN,FP,FPR,FNR):
    cm = np.array([[TN, TP],[FN, FP]])
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: Non-Hate', 'Predicted: Hate'],
        y=['Actual: Hate','Actual: Non-Hate'],
        colorscale='Purpor',
        showscale=True, # Color Bar
        opacity=0.8
    ))
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append(
                dict(
                    text=str(cm[i][j]),
                    x=['Predicted: Non-Hate', 'Predicted: Hate'][j],
                    y=['Actual: Hate','Actual: Non-Hate'][i],
                    font=dict(color='white', size=15, family='Arial Black'),
                    showarrow=False
                )
            )
    FPR = FPR
    FNR = FNR
    fig.update_layout(
        title=dict(
            text=f"FPR: {FPR:.2f}% | FNR: {FNR:.2f}%",
            font=dict(size=15, family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        annotations=annotations,
        paper_bgcolor='rgba(0,0,0,0.5)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)

def PR_curve(json_path):
    with open(json_path, "r", encoding="utf-8") as f: pr_data = json.load(f)

    precision = pr_data["precision"]
    recall = pr_data["recall"]
    pr_auc = pr_data["PR_AUC"]
    baseline= pr_data["baseline"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'PR Curve (AUC = {pr_auc:.4f})',
        line=dict(color='orange', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[baseline, baseline],
        mode='lines',
        name='Random Guess',
        line=dict(color='white', width=2, dash='dash')
    ))

    fig.update_layout(
        xaxis=dict(title='Recall', range=[0, 1]),
        yaxis=dict(title='Precision', range=[0, 1.02]),
        legend=dict(
            font=dict(size=12),
            orientation="h",
            x=0.5,
            xanchor='center',
            y=-0.2
        ),
        paper_bgcolor='rgba(0,0,0,0.5)',
        margin=dict(l=40, r=40, t=80, b=80),
        title=dict(
            text=f"PR-AUC Score: {pr_auc:.4f}",
            font=dict(size=15, family='Arial'),
            x=0.5,
            xanchor='center'
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def ROC_curve(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        roc_data = json.load(f)
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    roc_auc = roc_data["roc_auc"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='cyan', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Guess',
        line=dict(color='white', width=2, dash='dash')
    ))

    fig.update_layout(
        xaxis=dict(title='False Positive Rate', range=[0, 1]),
        yaxis=dict(title='True Positive Rate', range=[0, 1.02]),
        legend=dict(
            font=dict(size=12),
            orientation="h",
            x=0.5,
            xanchor='center',
            y=-0.2
        ),
        paper_bgcolor='rgba(0,0,0,0.5)',
        margin=dict(l=40, r=40, t=80, b=80),
        title=dict(
            text=f"AUC Score: {roc_auc:.4f}",
            font=dict(size=15, family='Arial'),
            x=0.5,
            xanchor='center'
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def learning_curve(history_path, patience):
    history = joblib.load(history_path)
    accuracy = history.get('accuracy', [])[:patience]
    val_accuracy = history.get('val_accuracy', [])[:patience]
    loss = history.get('loss', [])[:patience]
    val_loss = history.get('val_loss', [])[:patience]
    epochs = list(range(1, len(accuracy) + 1))

    col1, col2 = st.columns(2)
    # Accuracy Curve
    with col1:
        st.subheader("👾 Accuracy Curve")
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs, y=accuracy, mode='lines+markers', name='Train Accuracy',
                                     line=dict(color='lightgreen', width=3)))
        fig_acc.add_trace(go.Scatter(x=epochs, y=val_accuracy, mode='lines+markers', name='Test Accuracy',
                                     line=dict(color='red', width=3)))

        for i in range(len(epochs)):
            fig_acc.add_annotation(x=epochs[i], y=accuracy[i], text=f"{accuracy[i]:.4f}", showarrow=False,
                                   font=dict(size=10, color="lightgreen"), yshift=10)
            fig_acc.add_annotation(x=epochs[i], y=val_accuracy[i], text=f"{val_accuracy[i]:.4f}", showarrow=False,
                                   font=dict(size=10, color="red"), yshift=-10)

        fig_acc.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            xaxis=dict(tickmode='linear'),
            # yaxis=dict(range=[0.6, 0.9]),
            legend=dict(orientation="h", x=0.5, xanchor='center', y=-0.2),
            paper_bgcolor='rgba(0,0,0,0.5)',
            margin=dict(l=40, r=40, t=80, b=80)
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    # Loss Curve
    with col2:
        st.subheader("👾 Loss Curve")

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=loss, mode='lines+markers', name='Train Loss',
                                      line=dict(color='lightblue', width=3)))
        fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Test Loss',
                                      line=dict(color='red', width=3)))

        for i in range(len(epochs)):
            fig_loss.add_annotation(x=epochs[i], y=loss[i], text=f"{loss[i]:.4f}", showarrow=False,
                                    font=dict(size=10, color="lightblue"), yshift=10)
            fig_loss.add_annotation(x=epochs[i], y=val_loss[i], text=f"{val_loss[i]:.4f}", showarrow=False,
                                    font=dict(size=10, color="red"), yshift=-10)

        fig_loss.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss",
            xaxis=dict(tickmode='linear'),
            # yaxis=dict(range=[0.04, 0.08]),
            legend=dict(orientation="h", x=0.5, xanchor='center', y=-0.2),
            paper_bgcolor='rgba(0,0,0,0.5)',
            margin=dict(l=40, r=40, t=80, b=80)
        )

        st.plotly_chart(fig_loss, use_container_width=True)



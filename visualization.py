import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Set the style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'

# Define custom color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Define the data for each model
model1_data = pd.DataFrame({
    'Epoch': [0, 1, 2],
    'Training Loss': [0.1873, 0.0798, 0.0664],
    'Validation Loss': [0.11899, 0.127992, 0.135917],
    'Accuracy': [0.962074, 0.962074, 0.963889],
    'F1': [0.977112, 0.977328, 0.978358],
    'Precision': [0.981174, 0.972169, 0.975645],
    'Recall': [0.973084, 0.982541, 0.981086],
    'True Positives': [4013, 4052, 4046],
    'False Negatives': [111, 72, 78],
    'False Positives': [77, 116, 101],
    'True Negatives': [756, 717, 732],
    'Roc Auc': [0.98676, 0.988772, 0.987982],
    'Model': ['Model 1'] * 3
})

model2_data = pd.DataFrame({
    'Epoch': [1, 2, 3],
    'Training Loss': [None, 0.1716, 0.1716],
    'Validation Loss': [0.111312, 0.122438, 0.117192],
    'Accuracy': [0.961872, 0.958846, 0.959451],
    'F1': [0.976926, 0.975439, 0.975681],
    'Precision': [0.983772, 0.968675, 0.973678],
    'Recall': [0.970175, 0.982299, 0.977692],
    'True Positives': [4001, 4051, 4032],
    'False Negatives': [123, 73, 92],
    'False Positives': [66, 131, 109],
    'True Negatives': [767, 702, 724],
    'Roc Auc': [0.986523, 0.987618, 0.987771],
    'Model': ['Model 2'] * 3
})

model3_data = pd.DataFrame({
    'Epoch': [1, 2, 3],
    'Training Loss': [None, 0.1824, 0.1824],
    'Validation Loss': [0.109733, 0.129349, 0.11514],
    'Accuracy': [0.962679, 0.957837, 0.96167],
    'F1': [0.977425, 0.974907, 0.977009],
    'Precision': [0.983788, 0.965517, 0.975121],
    'Recall': [0.971145, 0.984481, 0.978904],
    'True Positives': [4005, 4060, 4037],
    'False Negatives': [119, 64, 87],
    'False Positives': [66, 145, 103],
    'True Negatives': [767, 688, 730],
    'Roc Auc': [0.986077, 0.987285, 0.987655],
    'Model': ['Model 3'] * 3
})

model4_data = pd.DataFrame({
    'Epoch': [1, 2, 3],
    'Training Loss': [None, 0.1669, 0.1669],
    'Validation Loss': [0.109293, 0.124124, 0.110447],
    'Accuracy': [0.96046, 0.957232, 0.96167],
    'F1': [0.976179, 0.974556, 0.977036],
    'Precision': [0.978558, 0.964829, 0.973976],
    'Recall': [0.973812, 0.984481, 0.980116],
    'True Positives': [4016, 4060, 4042],
    'False Negatives': [108, 64, 82],
    'False Positives': [88, 148, 108],
    'True Negatives': [745, 685, 725],
    'Roc Auc': [0.986124, 0.987145, 0.98752],
    'Model': ['Model 4'] * 3
})

# Combine all data
all_models = pd.concat([model1_data, model2_data, model3_data, model4_data])

# VISUALIZATION 1: Precision-Recall Trade-off Scatter Plot
def create_precision_recall_plot():
    # Get the data for each epoch of each model
    fig, ax = plt.subplots(figsize=(12, 10))
    
    markers = ['o', 's', '^', 'D']
    sizes = [100, 200, 300]  # Different sizes for different epochs
    
    for i, model_name in enumerate(['Model 1', 'Model 2', 'Model 3', 'Model 4']):
        model_data = all_models[all_models['Model'] == model_name]
        
        # Plot each epoch as a different size
        for j, epoch in enumerate(model_data['Epoch'].unique()):
            epoch_data = model_data[model_data['Epoch'] == epoch]
            ax.scatter(
                epoch_data['Recall'], 
                epoch_data['Precision'],
                s=sizes[j % len(sizes)],
                color=colors[i],
                marker=markers[i],
                alpha=0.7,
                edgecolors='black',
                linewidths=1,
                label=f"{model_name} (Epoch {epoch})"
            )
    
    # Add text annotations for each point
    for _, row in all_models.iterrows():
        ax.annotate(
            f"E{int(row['Epoch'])}",
            (row['Recall'], row['Precision']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Plot iso-f1 curves
    f1_scores = [0.975, 0.976, 0.977, 0.978]
    x = np.linspace(0.96, 0.99, 100)
    
    for f1 in f1_scores:
        y = (f1 * x) / (2 * x - f1)
        valid_idx = ~np.isnan(y) & ~np.isinf(y) & (y >= 0) & (y <= 1)
        ax.plot(x[valid_idx], y[valid_idx], '--', color='gray', alpha=0.5)
        # Label the F1 curve at a suitable position
        middle_idx = len(x[valid_idx]) // 2
        if middle_idx > 0:
            ax.annotate(
                f"F1={f1}",
                (x[valid_idx][middle_idx], y[valid_idx][middle_idx]),
                xytext=(0, -10),
                textcoords='offset points',
                fontsize=8,
                color='gray',
                ha='center'
            )
    
    # Set axis limits to focus on the relevant area
    ax.set_xlim(0.965, 0.99)
    ax.set_ylim(0.96, 0.99)
    
    # Add labels and title
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Trade-off Across Models and Epochs', fontsize=16)
    
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return plt

# VISUALIZATION 2: Confusion Matrix Heatmaps with Performance Metrics
def create_enhanced_confusion_matrices():
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Get the final epoch for each model
    final_epochs = all_models.groupby('Model').tail(1).reset_index()
    
    # Create a custom colormap that goes from white to blue
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ffffff', '#0571b0'])
    
    # Create confusion matrix visualizations for each model
    models = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
    
    for i, model_name in enumerate(models):
        row, col = i // 2, i % 2
        model_data = final_epochs[final_epochs['Model'] == model_name].iloc[0]
        
        # Extract confusion matrix values
        tp = model_data['True Positives']
        fn = model_data['False Negatives']
        fp = model_data['False Positives']
        tn = model_data['True Negatives']
        
        # Create confusion matrix
        cm = np.array([[tp, fn], [fp, tn]])
        
        # Calculate percentages for annotations
        total = np.sum(cm)
        cm_percentages = cm / total * 100
        
        # Create a heatmap with both counts and percentages
        sns.heatmap(
            cm, 
            annot=np.array([[f"{tp}\n({cm_percentages[0,0]:.1f}%)", f"{fn}\n({cm_percentages[0,1]:.1f}%)"],
                           [f"{fp}\n({cm_percentages[1,0]:.1f}%)", f"{tn}\n({cm_percentages[1,1]:.1f}%)"]]),
            fmt="", 
            cmap=cmap, 
            ax=axes[row][col], 
            cbar=False,
            annot_kws={"size": 14, "ha": 'center', "va": 'center'}
        )
        
        # Add model name and performance metrics as title
        axes[row][col].set_title(
            f"{model_name} - Final Epoch\n" + 
            f"Accuracy: {model_data['Accuracy']:.4f}, F1: {model_data['F1']:.4f}\n" +
            f"Precision: {model_data['Precision']:.4f}, Recall: {model_data['Recall']:.4f}",
            fontsize=14, 
            pad=10
        )
        
        # Set labels
        axes[row][col].set_xlabel('Predicted', fontsize=12)
        axes[row][col].set_ylabel('Actual', fontsize=12)
        axes[row][col].set_xticklabels(['Positive', 'Negative'], fontsize=12)
        axes[row][col].set_yticklabels(['Positive', 'Negative'], fontsize=12, rotation=0)
    
    plt.tight_layout()
    plt.suptitle('Confusion Matrix Comparison (Final Epoch)', fontsize=18, y=1.02)
    
    return plt

# VISUALIZATION 3: Radar Chart of Final Metrics
def create_radar_chart():
    # Get the final epoch for each model
    final_epochs = all_models.groupby('Model').tail(1).reset_index()
    
    # Metrics to plot (excluding Validation Loss as it's a "lower is better" metric)
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'Roc Auc']
    
    # Number of variables
    N = len(metrics)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metrics, size=14)
    
    # Draw the y-axis labels (0.95 to 1)
    ax.set_rlabel_position(30)
    plt.yticks([0.96, 0.97, 0.98, 0.99, 1.0], ["0.96", "0.97", "0.98", "0.99", "1.0"], 
               color="grey", size=12)
    plt.ylim(0.95, 1.0)
    
    # Plot each model
    for i, model_name in enumerate(['Model 1', 'Model 2', 'Model 3', 'Model 4']):
        model_data = final_epochs[final_epochs['Model'] == model_name]
        
        # Get the values for this model
        values = model_data[metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.2, color=colors[i])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)
    
    # Add some grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a title
    plt.title('Performance Metrics Comparison (Final Epoch)', size=18, y=1.05)
    
    return plt

# Execute the visualizations
plt.close('all')  # Close any existing plots

# Create and save the visualizations
fig1 = create_precision_recall_plot()
fig1.savefig('precision_recall_plot.png', dpi=300, bbox_inches='tight')

fig2 = create_enhanced_confusion_matrices()
fig2.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')

fig3 = create_radar_chart()
fig3.savefig('radar_chart.png', dpi=300, bbox_inches='tight')

# Display the images
from IPython.display import Image, display

print("Visualization 1: Precision-Recall Trade-off Plot")
display(Image('precision_recall_plot.png'))

print("\nVisualization 2: Enhanced Confusion Matrices")
display(Image('confusion_matrices.png'))

print("\nVisualization 3: Radar Chart of Performance Metrics")
display(Image('radar_chart.png'))
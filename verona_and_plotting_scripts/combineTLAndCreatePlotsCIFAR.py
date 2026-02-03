import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_style("darkgrid")
sns.set_theme(rc={"figure.figsize": (11.7, 8.27)}, font_scale=1.5)
sns.set_palette(sns.color_palette("Paired"))

def create_hist_figure(df) -> plt.Figure:
        hist_plot = sns.histplot(data=df, x="epsilon_value", hue="network", multiple="stack")
        figure = hist_plot.get_figure()

        plt.close()

        return figure

def create_box_figure_TL(df, network_name, accuracies=None, median_epsilon_by_network=None) -> plt.Figure:
    """
    Create boxplot for transfer learning results with optional accuracy overlay.
    
    Args:
        df: DataFrame with 'network' and 'epsilon_value' columns
        network_name: Title for the plot
        accuracies: Optional list of accuracy values corresponding to each network/layer
                   Should be in same order as networks appear in df
    """
    if accuracies is None or median_epsilon_by_network is None:
        box_plot = sns.boxplot(data=df, x="network", y="epsilon_value")
        
        current_labels = [label.get_text() for label in box_plot.get_xticklabels()]
        new_labels = [label.replace('TL', '') if label.startswith('TL') else label for label in current_labels]
        
        box_plot.set_xticklabels(new_labels)
        box_plot.set_xlabel("Retrained layers")
        box_plot.set_ylabel("Minimum adversarial perturbation")
        box_plot.set_title(network_name)

        figure = box_plot.get_figure()
    else:
        # Dual y-axis plot with robustness and accuracy
        fig, ax1 = plt.subplots()
        
        # Plot robustness (epsilon values) on left y-axis
        box_plot = sns.boxplot(data=df, x="network", y="epsilon_value", ax=ax1, showfliers=False)
        paired_palette = sns.color_palette("Paired")

        # Add median epsilon values as text annotations on the boxplot
        unique_networks = df['network'].unique()
        x_positions_box = range(len(unique_networks))
        for x, network in zip(x_positions_box, unique_networks):
            median_val = median_epsilon_by_network[network]
            # Position text below the mean value
            ax1.text(x, median_val-0.0005, f'{median_val:.3f}', ha='center', va='bottom', fontsize=16, 
                    color="darkblue")
        
        current_labels = [label.get_text() for label in box_plot.get_xticklabels()]
        new_labels = [label.replace('TL', '') if label.startswith('TL') else label for label in current_labels]
        
        ax1.set_xticklabels(new_labels)
        ax1.set_xlabel("Retrained blocks")
        ax1.set_ylabel("Minimum adversarial perturbation")
        ax1.tick_params(axis='y')
        ax1.set_title(network_name)
        # Create second y-axis for accuracy
        ax2 = ax1.twinx()
        
        # Plot accuracy as a line with markers
        ax2.plot(x_positions_box, accuracies, color=paired_palette[5], marker='o', linewidth=2, 
                markersize=8, label='Clean Accuracy')
        
        # Add accuracy values as text annotations
        for x, acc in zip(x_positions_box, accuracies):
            ax2.text(x, acc+0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=16, color=paired_palette[5])
        ax2.set_ylabel("Clean Accuracy")
        ax2.tick_params(axis='y')
        ax2.grid(False)
        ax2.set_ylim([0, 1.0])  # Assuming accuracy is between 0 and 1
        
        figure = fig

    plt.close()

    return figure

def create_box_figure(df, network_name) -> plt.Figure:
    box_plot = sns.boxplot(data=df, x="network", y="epsilon_value")
    box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=45)
    box_plot.set_xlabel("Training method")
    box_plot.set_ylabel("Minimum adversarial perturbation")
    box_plot.set_title(network_name)

    figure = box_plot.get_figure()

    plt.close()

    return figure

def create_kde_figure(df) -> plt.Figure:
    kde_plot = sns.kdeplot(data=df, x="epsilon_value", hue="network", multiple="stack")

    figure = kde_plot.get_figure()

    plt.close()

    return figure

def create_ecdf_figure(df) -> plt.Figure:
    # Create a copy with renamed network labels
    df_plot = df.copy()
    df_plot['Retrained blocks'] = df_plot['network'].str.replace('TL', '')
    
    ecdf_plot = sns.ecdfplot(data=df_plot, x="epsilon_value", hue="Retrained blocks")
    ecdf_plot.set_xlabel("Minimum adversarial perturbation")

    figure = ecdf_plot.get_figure()

    plt.close()

    return figure

def create_anneplot(df):
    for network in df.network.unique():
        df = df.sort_values(by="epsilon_value")
        cdf_x = np.linspace(0, 1, len(df))
        plt.plot(df.epsilon_value, cdf_x, label=network)
        plt.fill_betweenx(cdf_x, df.epsilon_value, df.smallest_sat_value, alpha=0.3)
        plt.xlim(0, 0.35)
        plt.xlabel("Epsilon values")
        plt.ylabel("Fraction critical epsilon values found")
        plt.legend()

    return plt.gca()

def create_scatterplots(accuracies, median_epsilon_by_network, network_name):
    # Prepare data for seaborn
    data = pd.DataFrame({
        'Clean Accuracy': accuracies,
        'Median minimum adversarial perturbation': list(median_epsilon_by_network.values),
        'Retrained blocks': [label.replace('TL', '') for label in median_epsilon_by_network.index]
    })
    
    # Create figure and axis
    fig, ax = plt.subplots()
        
    # Create scatter plot using seaborn
    paired_palette = sns.color_palette("Paired")
    sns.scatterplot(data=data, x='Clean Accuracy', y='Median minimum adversarial perturbation',
                   color=paired_palette[1], marker='x', s=130, ax=ax, legend=False)
    
    # Add labels for each point
    for idx in range(len(data)):
        ax.annotate(data.iloc[idx]['Retrained blocks'], 
                   (data.iloc[idx]['Clean Accuracy'], data.iloc[idx]['Median minimum adversarial perturbation']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=16)
    
    ax.set_title(network_name)
    ax.grid(True, alpha=0.3)

    plt.close()

    return fig


def find_result_files(base_path, dataset_name):
    """Find result_df.csv files directly in dataset/results directory"""
    results_path = base_path / dataset_name / "results"
    result_files = []
    
    if results_path.exists():
        # Look for CSV files directly in the results directory
        for csv_file in results_path.glob("*.csv"):
            if "result_df" in csv_file.name:
                # Extract experiment type from filename
                filename = csv_file.name.lower()
                if "pgd" in filename:
                    exp_type = "pgd"
                else:
                    exp_type = "conventional"
                
                result_files.append({
                    'path': csv_file,
                    'dataset': dataset_name.lower(),
                    'experiment_type': exp_type,
                    'order_key': f"{exp_type}_{dataset_name.lower()}"
                })
    
    return result_files

def combine_results(base_path, layers):
    # Base path where MNIST and EMNIST directories are located    
    # Find all result files
    all_files = []
    for layer in range(1, layers + 1):
        all_files.extend(find_result_files(base_path, f"TL{layer}"))
    
    if not all_files:
        print("No result files found!")
        return
    
    # Combine all dataframes
    combined_dfs = []
    
    for file_info in all_files:
        print(f"Loading: {file_info['path']}")
        df = pd.read_csv(file_info['path'])
        
        # Add metadata columns
        df['dataset'] = file_info['dataset']
        df['experiment_type'] = file_info['experiment_type']
        df['source'] = f"{file_info['experiment_type']}_{file_info['dataset']}"
        
        combined_dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    
    # Save combined results
    output_path = base_path / "TL_df.csv"
    combined_df.to_csv(output_path, index=False)
    
    print(f"\nCombined results saved to: {output_path}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Sources: {combined_df['source'].value_counts().to_dict()}")
    print(f"Datasets: {combined_df['dataset'].value_counts().to_dict()}")
    print(f"Experiment types: {combined_df['experiment_type'].value_counts().to_dict()}")

def sort_scratch_df(scratch_df_path, layers):
    # Read the CSV file
    df = pd.read_csv(scratch_df_path)#'/home/s3665534/VERONA/examples/Resnet34/scratch_df.csv')

    # Map source to network names for all layers
    for layer in range(1, layers + 1):
        df.loc[df['source'] == f'conventional_tl{layer}', 'network'] = f'TL{layer}'

    # Sort by network column with custom order
    network_order = [f"TL{layer}" for layer in range(1, layers + 1)]
    df['network'] = pd.Categorical(df['network'], categories=network_order, ordered=True)
    df_sorted = df.sort_values('network')

    # Save the sorted dataframe
    df_sorted.to_csv(scratch_df_path, index=False)

    # Display first few rows to verify
    print(df_sorted.head())
    print(f"\nSorted by network. Shape: {df_sorted.shape}")
    print(f"Unique networks: {df_sorted['network'].unique()}")


def main():
    accuraciesTL = {
        'ResNet-18' : [0.1843, 0.1877, 0.3866, 0.46, 0.5858, 0.613, 0.6129],
        'ResNet-34' : [0.1967,0.1925,0.3954,0.4652,0.5863,0.6154,0.6042],
        'ResNet-34 Batch 128' : [0.1932, 0.1856,0.3665, 0.4482, 0.534, 0.57, 0.5686],
        'ResNet-50' : [0.0995,0.1032,0.2888,0.4747,0.5696,0.6002,0.6061],
        'WideResNet-28-10' : [0.0948,0.1601,0.4025,0.5092,0.5987],
        'WideResNet-34-10' : [0.1052,0.1733,0.3895,0.5093,0.5901],
        'WideResNet-34-10 Batch 128' : [0.4371,0.4409,0.4347,0.3961,0.3967]
    }

    # accuraciesTLMLP = {
    #     'MLP1' : [0.2579,0.2508,0.3933,0.4501,0.5723,0.596,0.6029],
    #     'MLP2' : [0.2995, 0.3069, 0.4423, 0.5491, 0.5831, 0.6195, 0.5974],
    #     'MLP3' : [0.3096, 0.301, 0.4189, 0.5427, 0.5664, 0.6121, 0.6128]
    # }
    # accuraciesTLAdversarial = {
    #     'ResNet-34 Adversarial Retraining' : [0.4473, 0.4422, 0.4528,0.4229,0.3806,0.3712,0.3548], 
    #     'WideResNet-34-10 Adversarial Retraining' : [0.4526,0.4782,0.4617,0.4269,0.4164]
    # }


    base_path = Path("/home/s3665534/VERONA/examples/Networks/CIFAR/WideResnet34-10/TransferLearnedAdversarialB128")
    network_name = "WideResNet-34-10 Batch 128"
    accuracies = accuraciesTL[network_name]
    retrained_layers = len(accuracies)

    combine_results(base_path, retrained_layers)

    result_df_path = base_path /  "TL_df.csv"

    sort_scratch_df(result_df_path, retrained_layers)

    if result_df_path.exists():
        df = pd.read_csv(result_df_path, index_col=0)
    else:
        raise Exception(f"Error, no result file found at {result_df_path}")

    # Calculate median epsilon values for each network
    median_epsilon_by_network = df.groupby('network')['epsilon_value'].median()
    print("\nMean epsilon values by network:")
    print(median_epsilon_by_network)

    hist_figure = create_hist_figure(df)
    hist_figure.savefig(base_path / "hist_figure_TL.pdf",format="pdf", bbox_inches="tight")

    boxplot = create_box_figure_TL(df, network_name, accuracies, median_epsilon_by_network)
    boxplot.savefig(base_path / "boxplot_TL.pdf", format="pdf", bbox_inches="tight")

    kde_figure = create_kde_figure(df)
    kde_figure.savefig(base_path / "kde_plot_TL.pdf", format="pdf", bbox_inches="tight")

    ecdf_figure = create_ecdf_figure(df)
    ecdf_figure.savefig(base_path / "ecdf_plot_TL.pdf",format="pdf", bbox_inches="tight")

    scatterplot = create_scatterplots(accuracies, median_epsilon_by_network, network_name)
    scatterplot.savefig(base_path / "scatter_accuracy.pdf",format="pdf", bbox_inches="tight")



if __name__ == "__main__":
    main()
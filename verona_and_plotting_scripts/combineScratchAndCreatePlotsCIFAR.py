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

def create_box_figure_TL(df, network_name) -> plt.Figure:
    box_plot = sns.boxplot(data=df, x="network", y="epsilon_value")
    
    # Extract numbers from TL labels (TL1 -> 1, TL2 -> 2, etc.)
    current_labels = [label.get_text() for label in box_plot.get_xticklabels()]
    new_labels = [label.replace('TL', '') if label.startswith('TL') else label for label in current_labels]
    
    box_plot.set_xticklabels(new_labels)
    box_plot.set_xlabel("Retrained blocks")
    box_plot.set_ylabel("Minimum adversarial perturbation")
    box_plot.set_title(network_name)

    figure = box_plot.get_figure()

    plt.close()

    return figure

def create_box_figure(df, network_name, accuracies=None, median_epsilon_by_network=None) -> plt.Figure:
    if accuracies is None or median_epsilon_by_network is None:
        box_plot = sns.boxplot(data=df, x="network", y="epsilon_value")
        box_plot.set_xticklabels(box_plot.get_xticklabels())
        box_plot.set_xlabel("Training method")
        box_plot.set_ylabel("Minimum adversarial perturbation")
        box_plot.set_title(network_name)

        figure = box_plot.get_figure()

    else: 
        # Dual y-axis plot with robustness and accuracy
        fig, ax1 = plt.subplots()#figsize=(10, 6))
        
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
        
        ax1.set_xticklabels(box_plot.get_xticklabels())
        ax1.set_xlabel("Training method")
        ax1.set_ylabel("Minimum adversarial perturbation")
        ax1.tick_params(axis='y')
        ax1.set_title(network_name)

        # Create second y-axis for accuracy
        ax2 = ax1.twinx()
        # Plot accuracy as scatter points
        ax2.scatter(x_positions_box, accuracies, color=paired_palette[5], marker='x',s=100, label='Clean Accuracy')
        
        # Add accuracy values as text annotations
        for x, acc in zip(x_positions_box, accuracies):
            ax2.text(x, acc - 0.02, f'{acc:.3f}', ha='center', va='top', fontsize=16, color=paired_palette[5])
        ax2.set_ylabel("Clean Accuracy")
        ax2.tick_params(axis='y')
        ax2.grid(False)
        ax2.set_ylim([0, 1.05])
        
        figure = fig

    plt.close()

    return figure

def create_kde_figure(df) -> plt.Figure:
    kde_plot = sns.kdeplot(data=df, x="epsilon_value", hue="network", multiple="stack")

    figure = kde_plot.get_figure()

    plt.close()

    return figure

def create_ecdf_figure(df) -> plt.Figure:
    df_plot = df.copy()
    df_plot['Training method'] = df_plot['network']
    ecdf_plot = sns.ecdfplot(data=df_plot, x="epsilon_value", hue="Training method")

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

def combine_results(base_path):
    # Base path where CIFAR and CIFAR100 directories are located    
    # Find all result files
    all_files = []
    all_files.extend(find_result_files(base_path, "CIFAR10STANDARD"))
    all_files.extend(find_result_files(base_path, "CIFAR100STANDARD"))
    all_files.extend(find_result_files(base_path, "CIFAR10PGD"))
    all_files.extend(find_result_files(base_path, "CIFAR100PGD"))
    
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
    output_path = base_path / "scratch_df.csv"
    combined_df.to_csv(output_path, index=False)
    
    print(f"\nCombined results saved to: {output_path}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Sources: {combined_df['source'].value_counts().to_dict()}")
    print(f"Datasets: {combined_df['dataset'].value_counts().to_dict()}")
    print(f"Experiment types: {combined_df['experiment_type'].value_counts().to_dict()}")

def sort_scratch_df(scratch_df_path):
    # Read the CSV file
    df = pd.read_csv(scratch_df_path)

    # Map source to network names
    df.loc[df['source'] == 'conventional_cifar10standard', 'network'] = 'standard_cifar10'
    df.loc[df['source'] == 'conventional_cifar100standard', 'network'] = 'standard_cifar100'
    df.loc[df['source'] == 'conventional_cifar10pgd', 'network'] = 'pgd_cifar10'
    df.loc[df['source'] == 'conventional_cifar100pgd', 'network'] = 'pgd_cifar100'

    # Sort by network column with custom order
    network_order = ["standard_cifar10", "standard_cifar100", "pgd_cifar10","pgd_cifar100"]
    df['network'] = pd.Categorical(df['network'], categories=network_order, ordered=True)
    df_sorted = df.sort_values('network')

    # Save the sorted dataframe
    df_sorted.to_csv(scratch_df_path, index=False)

    # Display first few rows to verify
    print(df_sorted.head())
    print(f"\nSorted by network. Shape: {df_sorted.shape}")
    print(f"Unique networks: {df_sorted['network'].unique()}")


def main():
    accuracies_scratch = {
        'ResNet-18' : [0.946,0.7691,0.8018,0.4803],
        'ResNet-34' : [0.8392,0.5506,0.7844,0.4708],
        'ResNet-50' : [0.8104,0.5504,0.5391,0.2849],
        'WideResNet-28-10' :  [0.8422,0.6558,0.8195,0.5378],
        'WideResNet-34-10' : [0.8465,0.6461,0.8219,0.5283]
    }

    base_path = Path("/home/s3665534/VERONA/examples/Networks/CIFAR/WideResnet28-10")
    network_name = "WideResNet-28-10"
    accuracies = accuracies_scratch[network_name]

    combine_results(base_path)

    result_df_path = base_path /  "scratch_df.csv"

    sort_scratch_df(result_df_path)

    if result_df_path.exists():
        df = pd.read_csv(result_df_path, index_col=0)
    else:
        raise Exception(f"Error, no result file found at {result_df_path}")

    # Calculate median epsilon values for each network
    median_epsilon_by_network = df.groupby('network')['epsilon_value'].median()
    print("\nMedian epsilon values by network:")
    print(median_epsilon_by_network)

    hist_figure = create_hist_figure(df)
    hist_figure.savefig(base_path / "hist_figure.pdf", format="pdf",bbox_inches="tight")

    boxplot = create_box_figure(df, network_name, accuracies, median_epsilon_by_network)
    boxplot.savefig(base_path / "boxplot.pdf", format="pdf",bbox_inches="tight")

    kde_figure = create_kde_figure(df)
    kde_figure.savefig(base_path / "kde_plot.pdf",format="pdf", bbox_inches="tight")

    ecdf_figure = create_ecdf_figure(df)
    ecdf_figure.savefig(base_path / "ecdf_plot.pdf",format="pdf", bbox_inches="tight")



if __name__ == "__main__":
    main()
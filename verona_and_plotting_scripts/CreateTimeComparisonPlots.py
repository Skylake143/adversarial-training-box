import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
sns.set_style("darkgrid")
sns.set_theme(rc={"figure.figsize": (12, 12)}, font_scale=2.0)
sns.set_palette(sns.color_palette("Paired"))

def create_training_time_plot(base_path, special_path):
    csv_path = base_path + special_path

    # Load the data
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=['Unnamed: 0']).copy()
    df.rename(columns={'Unnamed: 0': 'Model', 'Training Time': 'Time_Str'}, inplace=True)
    df['Model'] = df['Model'].str.replace('Conv', 'Standard', regex=False)

    # Convert Training Time to numeric (e.g., "52s" -> 52)
    df['Training Time (s)'] = df['Time_Str'].str.extract('(\d+)').astype(float)

    # 2. Categorize Methods for grouping/hue
    def categorize(model_name):
        if 'PGD' in model_name:
            return 'PGD (Adversarial)'
        elif 'TL' in model_name:
            if 'Adversarial' in model_name:
                return 'TL Adversarial'
            return 'TL (Standard)'
        elif 'Standard' in model_name:
            return 'Standard'
        return 'Other'

    df['Method'] = df['Model'].apply(categorize)

    ax = sns.barplot(data=df, x='Model', y='Training Time (s)', hue='Method', dodge=False)

    ax.set_yscale('log')
    ax.set_ylim(bottom=30)

    # Add training time labels on top of bars (vertical)
    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 0.97,
                f'{int(height)}s',
                ha='center', va='top', rotation=90, fontsize=22, color="white")

    plt.xticks(rotation=45, ha='right')
    plt.title('Training times for RELU_4_1024')
    plt.xlabel('Training method')
    plt.ylabel('Training Time (s) [log]')
    
    ax.get_legend().remove()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.22, right=0.95)

    # Save the figure as PDF at the same location as the CSV
    output_filename = csv_path.replace('.csv', '.pdf')
    plt.savefig(output_filename, format='pdf')
    print(f"Plot saved as {output_filename}")

base_path = '/Users/davidwunsch/Desktop/TimeComparison/'
special_path = 'ResultsCIFARAdversarialRetraining/WideResnet34-10-WideResnet34-10.csv'
create_training_time_plot(base_path, special_path)
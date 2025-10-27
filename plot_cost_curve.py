from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns

PREDICTIONS_PATH = Path('model_predictions.jsonl')
MODEL_COSTS: Dict[str, float] = {
    'anthropic/claude-haiku-4.5': 0.00785714285714286,
    'anthropic/claude-sonnet-4.5': 0.0219642857142857,
    'openai/gpt-5': 0.0139644970414201,
    'openai/gpt-5-mini': 0.00204241071428571,
    'openai/gpt-5-nano': 0.000641176470588235,
    'openai/gpt-oss-120b': 0.000651162790697675,
    'openai/gpt-oss-20b': 0.000327564102564103,
}


def load_predictions(file_path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f'Predictions file not found at {file_path}')
    df = pd.read_json(file_path, lines=True)
    if 'problem' in df.columns:
        problem_df = pd.json_normalize(df['problem'])
        problem_df = problem_df.rename(
            columns={
                'benchmark': 'benchmark',
                'example_id': 'example_id',
                'prompt': 'prompt',
            }
        )
        df = pd.concat([df.drop(columns=['problem']), problem_df], axis=1)
    return df


def compute_model_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    accuracy = (
        predictions.groupby('model')['true_label']
        .mean()
        .reset_index(name='accuracy')
    )
    accuracy['cost_per_task'] = accuracy['model'].map(MODEL_COSTS)
    missing_costs = accuracy[accuracy['cost_per_task'].isna()]['model'].tolist()
    if missing_costs:
        raise ValueError(f'Missing cost entries for models: {missing_costs}')
    return accuracy


def compute_pareto_frontier(
    predictions: pd.DataFrame,
    lambda_values: int | List[float] = 80,
) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions['cost_per_task'] = predictions['model'].map(MODEL_COSTS)
    if predictions['cost_per_task'].isna().any():
        missing = predictions[predictions['cost_per_task'].isna()]['model'].unique().tolist()
        raise ValueError(f'Missing cost entries for models: {missing}')

    if predictions.empty:
        return pd.DataFrame(columns=['lambda', 'mean_cost', 'mean_predicted_accuracy', 'mean_true_accuracy'])

    grouped_keys = ['benchmark', 'example_id']
    total_tasks = predictions.groupby(grouped_keys).ngroups

    if isinstance(lambda_values, int):
        lambda_grid = np.logspace(-3, 3, num=lambda_values)
    else:
        lambda_grid = np.asarray(lambda_values)

    frontier_rows: List[Dict[str, float]] = []

    for lam in lambda_grid:
        predictions['score'] = predictions['predicted_likelihood'] - lam * predictions['cost_per_task']
        idx = predictions.groupby(grouped_keys)['score'].idxmax()
        chosen = predictions.loc[idx].copy()

        mean_predicted_accuracy = chosen['predicted_likelihood'].mean()
        mean_true_accuracy = chosen['true_label'].mean()
        mean_cost = chosen['cost_per_task'].mean()

        frontier_rows.append(
            {
                'lambda': float(lam),
                'mean_cost': float(mean_cost),
                'mean_predicted_accuracy': float(mean_predicted_accuracy),
                'mean_true_accuracy': float(mean_true_accuracy),
                'tasks_covered': int(len(chosen)),
                'coverage_pct': float(len(chosen) / total_tasks) if total_tasks else float('nan'),
            }
        )

    return pd.DataFrame(frontier_rows).sort_values('mean_cost').reset_index(drop=True)


def plot_cost_curve(accuracy_df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    sns.set_theme(style='whitegrid')
    ax = sns.scatterplot(
        data=accuracy_df,
        x='cost_per_task',
        y='accuracy',
        hue='model',
        palette='Set2',
        s=100,
        edgecolor='black',
    )

    pareto_curve = compute_pareto_frontier(predictions)
    if not pareto_curve.empty:
        sns.lineplot(
            data=pareto_curve,
            x='mean_cost',
            y='mean_true_accuracy',
            ax=ax,
            color='darkorange',
            linewidth=2,
            marker='o',
            markersize=6,
            label='Dynamic selection (true accuracy)',
        )
        sns.lineplot(
            data=pareto_curve,
            x='mean_cost',
            y='mean_predicted_accuracy',
            ax=ax,
            color='midnightblue',
            linewidth=2,
            marker='s',
            markersize=5,
            label='Dynamic selection (predicted accuracy)',
        )

    for _, row in accuracy_df.iterrows():
        ax.annotate(
            row['model'],
            (row['cost_per_task'], row['accuracy']),
            textcoords='offset points',
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xscale('log')
    ax.set_xlabel('Cost per Task (log scale)')
    ax.set_ylabel('True Accuracy')
    ax.set_title('Model Accuracy vs Cost per Task with Dynamic Selection')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()

    output_path = Path('model_cost_accuracy.png')
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=300)
    print(f'Saved plot to {output_path.resolve()}')


def main() -> None:
    predictions = load_predictions()
    accuracy_df = compute_model_accuracy(predictions)
    plot_cost_curve(accuracy_df, predictions)


if __name__ == '__main__':
    main()

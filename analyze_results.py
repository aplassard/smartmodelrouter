import pandas as pd

DEFAULT_MODEL_NAME = 'openai/gpt-5-mini'


def analyze_model_results(
    results_df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
) -> pd.DataFrame:
    """Bucket predicted likelihoods and compute outcome counts for the specified model."""
    model_df = results_df[results_df['model'] == model_name].copy()
    if model_df.empty:
        return pd.DataFrame(columns=['bucket', 'count', 'positives', 'negatives'])

    bins = [i / 10 for i in range(11)]
    labels = [f'{bins[i]:.1f} to {bins[i+1]:.1f}' for i in range(len(bins) - 1)]

    model_df['bucket'] = pd.cut(
        model_df['predicted_likelihood'],
        bins=bins,
        include_lowest=True,
        right=True,
        labels=labels,
    )

    grouped = model_df.groupby('bucket')
    summary = grouped['true_label'].agg(
        count='count',
        positives='sum',
    ).reset_index()
    summary['negatives'] = summary['count'] - summary['positives']

    return summary[['bucket', 'count', 'positives', 'negatives']]

def load_analysis_results(file_path: str) -> pd.DataFrame:
    """Load analysis results from a JSON file."""
    return pd.read_json(file_path, lines=True)

def main():
    results_df = load_analysis_results('model_predictions.jsonl')
    print(results_df.head())
    print(analyze_model_results(results_df, model_name=DEFAULT_MODEL_NAME))

if __name__ == '__main__':
    main()

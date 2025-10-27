import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm
from xgboost import XGBClassifier

INPUT_DIR = Path('/Users/andrewplassard/src/successat/model-router')
MISSING_LOG_FILE = Path('missing_combinations.log')
DEFAULT_MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
PREDICTIONS_OUTPUT_FILE = Path('model_predictions.jsonl')

logger = logging.getLogger('build_model')

LOOCV_COLUMNS = ['model', 'benchmark', 'example_id', 'correct', 'predicted_prob']


def load_df_from_dir(input_dir: Union[Path, str]) -> pd.DataFrame:
    directory = Path(input_dir).expanduser()
    if not directory.is_dir():
        logger.error('Provided input directory does not exist: %s', directory)
        raise NotADirectoryError(f'Provided input directory does not exist: {directory}')

    records: List[Dict[str, Any]] = []
    json_files = sorted(directory.glob('*.json'))
    for json_file in tqdm(json_files, desc='Loading JSON files', unit='file'):
        with json_file.open('r', encoding='utf-8') as file:
            payload = json.load(file)

        if isinstance(payload, dict):
            row = payload.copy()
        else:
            row = {'data': payload}

        row['source_file'] = json_file.name
        records.append(row)

    logger.info('Loaded %d JSON files from %s', len(records), directory)
    return pd.DataFrame.from_records(records)


def write_missing_log(missing_df: pd.DataFrame, log_path: Path = MISSING_LOG_FILE) -> None:
    if missing_df.empty:
        log_path.write_text('No missing model/problem combinations found.\n', encoding='utf-8')
    else:
        missing_df.to_csv(log_path, index=False)
    logger.info('Missing combinations written to %s', log_path.resolve())


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    logger.info('Loading sentence transformer model: %s', model_name)
    model = SentenceTransformer(model_name)
    logger.info('Model loaded successfully: %s', model_name)
    return model


def create_embeddings(model: SentenceTransformer, problems_df: pd.DataFrame) -> pd.DataFrame:
    prompts = problems_df[['example_id', 'prompt']].dropna(subset=['prompt']).reset_index(drop=True)
    if prompts.empty:
        logger.warning('No prompts available for embedding generation.')
        return pd.DataFrame(columns=['example_id', 'embedding'])

    logger.info('Generating embeddings for %d prompts', len(prompts))
    embeddings = model.encode(prompts['prompt'].tolist(), show_progress_bar=True, batch_size=8)
    embedding_df = prompts[['example_id']].copy()
    embedding_df['embedding'] = list(embeddings)
    logger.debug('Generated embeddings dataframe:\n%s', embedding_df.head().to_string(index=False))
    logger.info('Embeddings generated successfully')
    return embedding_df


def _run_loocv_for_model(model_name: str, model_df: pd.DataFrame, xgb_jobs: int) -> pd.DataFrame:
    model_df = model_df.dropna(subset=['embedding']).copy()
    if model_df.empty:
        logger.warning('Skipping model %s due to missing embeddings', model_name)
        return pd.DataFrame(columns=LOOCV_COLUMNS)

    embeddings = np.stack(model_df['embedding'].to_numpy())
    targets = model_df['correct'].astype(int).to_numpy()
    benchmarks = model_df['benchmark'].to_numpy()
    example_ids = model_df['example_id'].to_numpy()

    results: List[Dict[str, Any]] = []

    for idx in range(len(model_df)):
        mask = np.ones(len(model_df), dtype=bool)
        mask[idx] = False

        X_train = embeddings[mask]
        y_train = targets[mask]

        if X_train.size == 0 or len(np.unique(y_train)) < 2:
            prob = float(y_train.mean()) if y_train.size else float(targets.mean())
        else:
            clf = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                n_jobs=xgb_jobs,
            )
            clf.fit(X_train, y_train)
            prob = float(clf.predict_proba(embeddings[idx].reshape(1, -1))[0, 1])

        results.append(
            {
                'model': model_name,
                'benchmark': benchmarks[idx],
                'example_id': example_ids[idx],
                'correct': bool(targets[idx]),
                'predicted_prob': prob,
            }
        )

    return pd.DataFrame(results, columns=LOOCV_COLUMNS)


def _loocv_worker(args: Tuple[str, pd.DataFrame, int]) -> pd.DataFrame:
    model_name, model_df, xgb_jobs = args
    return _run_loocv_for_model(model_name, model_df, xgb_jobs)


def run_leave_one_out_cv(
    model_problem_df: pd.DataFrame,
    problems_with_embeddings: pd.DataFrame,
    num_workers: int = 4,
    xgb_jobs: int = 4,
) -> pd.DataFrame:
    merged = model_problem_df.merge(
        problems_with_embeddings[['example_id', 'benchmark', 'embedding']],
        on=['example_id', 'benchmark'],
        how='left',
    )

    model_groups = [
        (model_name, group.reset_index(drop=True))
        for model_name, group in merged.groupby('model')
        if not group['embedding'].isna().all()
    ]

    logger.info('Running leave-one-out cross validation across %d models', len(model_groups))

    results: List[pd.DataFrame] = []
    worker_args: Iterable[Tuple[str, pd.DataFrame, int]] = (
        (model_name, group, xgb_jobs) for model_name, group in model_groups
    )

    if num_workers <= 1:
        for args in worker_args:
            results.append(_loocv_worker(args))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_model = {
                executor.submit(_loocv_worker, args): args[0] for args in worker_args
            }
            for future in tqdm(as_completed(future_to_model), total=len(future_to_model), desc='LOO CV'):
                model_name = future_to_model[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.exception('LOO CV failed for model %s: %s', model_name, exc)

    if not results:
        return pd.DataFrame(columns=LOOCV_COLUMNS)

    predictions = pd.concat(results, ignore_index=True)
    logger.info('Generated leave-one-out predictions for %d rows', len(predictions))
    return predictions


def summarize_model_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        logger.warning('No predictions available to summarize.')
        return pd.DataFrame(
            columns=[
                'model',
                'num_samples',
                'baseline_accuracy',
                'predicted_accuracy',
                'true_negatives',
                'false_positives',
                'false_negatives',
                'true_positives',
                'roc_auc',
                'pr_auc',
            ]
        )

    records: List[Dict[str, Any]] = []
    for model_name, group in predictions.groupby('model'):
        y_true = group['correct'].astype(int).to_numpy()
        y_pred_prob = group['predicted_prob'].to_numpy()
        y_pred = (y_pred_prob >= 0.5).astype(int)

        baseline_accuracy = float(y_true.mean()) if len(y_true) else float('nan')
        predicted_accuracy = float((y_pred == y_true).mean()) if len(y_true) else float('nan')

        try:
            roc_auc = float(roc_auc_score(y_true, y_pred_prob))
        except ValueError:
            roc_auc = float('nan')

        try:
            pr_auc = float(average_precision_score(y_true, y_pred_prob))
        except ValueError:
            pr_auc = float('nan')

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        records.append(
            {
                'model': model_name,
                'num_samples': int(len(group)),
                'baseline_accuracy': baseline_accuracy,
                'predicted_accuracy': predicted_accuracy,
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
            }
        )

    summary_df = pd.DataFrame(records)
    logger.debug('Model performance summary:\n%s', summary_df.to_string(index=False))
    return summary_df


def save_predictions_jsonl(
    predictions: pd.DataFrame,
    problems_df: pd.DataFrame,
    output_path: Path = PREDICTIONS_OUTPUT_FILE,
) -> None:
    if predictions.empty:
        output_path.write_text('', encoding='utf-8')
        logger.info('No predictions to save; created empty file at %s', output_path.resolve())
        return

    problem_details = problems_df[['benchmark', 'example_id', 'prompt']].drop_duplicates()
    merged = predictions.merge(problem_details, on=['benchmark', 'example_id'], how='left')

    with output_path.open('w', encoding='utf-8') as handle:
        for row in merged.itertuples(index=False):
            record = {
                'model': row.model,
                'problem': {
                    'benchmark': row.benchmark,
                    'example_id': row.example_id,
                    'prompt': row.prompt,
                },
                'true_label': bool(row.correct),
                'predicted_likelihood': float(row.predicted_prob),
            }
            handle.write(json.dumps(record) + '\n')

    logger.info('Saved predictions to %s', output_path.resolve())


def analyze_dataframe(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df_clean = df.copy()
    df_clean['correct_flag'] = df_clean['correct'].fillna(False).astype(bool)

    per_model_problem = (
        df_clean.groupby(['model', 'benchmark', 'example_id'], as_index=False)['correct_flag']
        .any()
        .rename(columns={'correct_flag': 'correct'})
    )

    model_benchmark_summary = (
        per_model_problem.groupby(['model', 'benchmark'])
        .agg(
            correct_problems=('correct', lambda s: int(s.sum())),
            total_problems=('correct', 'size'),
        )
        .reset_index()
    )
    model_benchmark_summary['accuracy_pct'] = (
        model_benchmark_summary['correct_problems'] / model_benchmark_summary['total_problems'] * 100
    )

    problem_correct_counts = (
        per_model_problem[per_model_problem['correct']]
        .groupby(['benchmark', 'example_id'])
        .size()
        .reset_index(name='models_correct')
    )

    examples = df_clean[['benchmark', 'example_id']].drop_duplicates().assign(key=1)
    models = pd.DataFrame({'model': df_clean['model'].drop_duplicates(), 'key': 1})
    all_combinations = examples.merge(models, on='key').drop(columns='key')
    observed = df_clean[['benchmark', 'example_id', 'model']].drop_duplicates()
    missing = (
        all_combinations.merge(observed, on=['benchmark', 'example_id', 'model'], how='left', indicator=True)
        .loc[lambda df_: df_['_merge'] == 'left_only']
        .drop(columns=['_merge'])
        .sort_values(['benchmark', 'example_id', 'model'])
        .reset_index(drop=True)
    )

    logger.info('Computed accuracy for %d model/benchmark pairs', len(model_benchmark_summary))
    logger.info('Identified %d problems lacking responses for at least one model', len(missing))
    logger.debug('Model/benchmark accuracy table:\n%s', model_benchmark_summary.to_string(index=False))
    logger.debug('Problem correct counts:\n%s', problem_correct_counts.to_string(index=False))
    logger.debug('Missing model responses:\n%s', missing.to_string(index=False))
    write_missing_log(missing)

    return {
        'model_benchmark_accuracy': model_benchmark_summary,
        'problem_correct_counts': problem_correct_counts,
        'missing_model_responses': missing,
    }


def build_problem_model_views(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    problems = df[['example_id', 'benchmark', 'prompt']].drop_duplicates().reset_index(drop=True)

    per_model_problem = (
        df.assign(correct_flag=df['correct'].fillna(False).astype(bool))
        .groupby(['model', 'benchmark', 'example_id'], as_index=False)['correct_flag']
        .any()
        .rename(columns={'correct_flag': 'correct'})
    )

    return problems, per_model_problem


def main(input_dir: Path = INPUT_DIR) -> None:
    df = load_df_from_dir(input_dir)
    logger.info('Resulting dataframe shape: %s', df.shape)
    analysis = analyze_dataframe(df)
    problems, model_problem_results = build_problem_model_views(df)
    model = load_model()
    embeddings_df = create_embeddings(model, problems)
    problems_with_embeddings = problems.merge(embeddings_df, on='example_id', how='left')
    predictions = run_leave_one_out_cv(model_problem_results, problems_with_embeddings)
    performance_summary = summarize_model_predictions(predictions)
    save_predictions_jsonl(predictions, problems)
    logger.info(
        'Model/benchmark accuracy rows: %d | Problems summarized: %d | Missing combos: %d',
        len(analysis['model_benchmark_accuracy']),
        len(problems),
        len(analysis['missing_model_responses']),
    )
    logger.info('Model-problem correctness rows: %d', len(model_problem_results))
    logger.info('Problems with embeddings shape: %s', problems_with_embeddings.shape)
    logger.info('LOOCV predictions shape: %s', predictions.shape)
    logger.info('Performance summary rows: %d', len(performance_summary))

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] %(message)s',
    )
    main()

import os

pipeline_dir = os.path.dirname(os.path.abspath(__file__))

scripts = [
    "pipelines/fetch_news.py",
    "pipelines/generate_stock_prices.py",
    "pipelines/process_rhetoric.py",
    "pipelines/merge_market_data.py",
    "utils/finance_metrics.py",
    "models/signal_generator/train_models.py",
    "pipelines/generate_signals.py"
]

for script in scripts:
    print(f"Running: {script}")
    os.system(f"python {script}")
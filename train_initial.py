"""
Train and persist initial models for SmartGrid AI.
"""
import numpy as np
import pandas as pd
from src.data_processing import load_and_preprocess_data
from src.forecasting import split_data, train_xgboost, train_sarimax, get_xgboost_metrics, get_sarimax_metrics
from src.q_learning import QLearningAgent
from src.model_manager import save_models, load_models

def main():
    print("Loading dataset...")
    df = load_and_preprocess_data()
    train_split, test_split = split_data(df)

    print("Training XGBoost...")
    xgb_model = train_xgboost(train_split)
    xgb_metrics, xgb_pred = get_xgboost_metrics(xgb_model, test_split)
    print(f"XGBoost metrics: MAE={xgb_metrics[0]:.2f}, RMSE={xgb_metrics[1]:.2f}, R2={xgb_metrics[2]:.4f}")

    print("Training SARIMAX...")
    sari_tuple = None
    sari_metrics = (float("nan"), float("nan"), float("nan"))
    try:
        sari_model, sari_exog = train_sarimax(train_split)
        sari_metrics, _, _ = get_sarimax_metrics(sari_model, sari_exog, test_split)
        sari_tuple = (sari_model, sari_exog)
        print(f"SARIMAX metrics: MAE={sari_metrics[0]:.2f}, RMSE={sari_metrics[1]:.2f}, R2={sari_metrics[2]:.4f}")
    except Exception as e:
        print(f"SARIMAX failed: {e}")

    print("Training Q-Learning Agent...")
    agent = QLearningAgent()
    agent.train(episodes=3000)
    print("Q-Learning training complete.")

    metrics_bundle = {
        "xgb": xgb_metrics,
        "sarimax": sari_metrics,
        "xgb_actual": test_split["energy_consumption"].values[:120],
        "xgb_pred": xgb_pred[:120],
        "train_rows": len(train_split),
        "test_rows": len(test_split),
    }

    print("Saving models to disk...")
    save_models(xgb_model, sari_tuple, agent, metrics_bundle)

    print("Verifying model loading...")
    loaded = load_models()
    assert loaded is not None, "Failed to load saved models!"
    xgb_l, sari_l, ql_l, meta_l = loaded
    print("Verification successful!")
    print("Metadata:", meta_l)

if __name__ == "__main__":
    main()

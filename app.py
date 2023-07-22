import os

import numpy as np
import psycopg2

from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

COL_NAMES = ["home_team", "away_team", "stage", "score_home", "score_away", "status", "timestamp"]


def connect_to_db(url: str) -> psycopg2.extensions.connection:
    return psycopg2.connect(url)


def log_loss(prediction: np.ndarray, outcome: int):
    return -np.log(prediction[outcome])


def get_results(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    sql = """SELECT * FROM results"""
    cur = conn.cursor()
    cur.execute(sql)
    results = cur.fetchall()
    results_df = pd.DataFrame(results, columns=COL_NAMES)
    results_df["timestamp"] = pd.to_datetime(results_df["timestamp"])
    return results_df.sort_values("timestamp").reset_index(drop=True)


def read_predictions(predictions_csv: Path) -> pd.DataFrame:
    return pd.read_csv(predictions_csv)


def get_stage(group: str) -> str:
    if group == "Knockout":
        return "KNOCKOUT"
    else:
        return "GROUP"


def make_match_keys(df: pd.DataFrame) -> pd.Series:
    keys = []
    for idx, row in df.iterrows():
        keys.append("_".join([row["team1"], row["team2"], row["stage"]]))

    return pd.Series(keys)


def get_prediction(result: pd.Series, predictions: pd.DataFrame) -> np.ndarray:
    pass


def calculate_outcome(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if away_score > home_score:
        return 1
    if home_score == away_score:
        return 2


def main():
    conn = connect_to_db(os.environ["DATABASE_URL"])
    results_df = get_results(conn)

    tom_df = read_predictions(Path("tghaynes_predictions.csv"))
    ben_df = read_predictions(Path("bpowell_predictions.csv"))

    tom_df["stage"] = tom_df["group"].apply(get_stage)
    ben_df["stage"] = tom_df["group"].apply(get_stage)

    ben_df["key"] = make_match_keys(ben_df)
    tom_df["key"] = make_match_keys(tom_df)

    st.title("RSS Prediction Results")

    ben_score = [0]
    tom_score = [0]

    display_df = pd.DataFrame(columns=["time", "home", "away", "result", "ben_loss", "tom_loss", "status"])

    for idx, row in results_df.iterrows():
        key = "_".join([row["home_team"], row["away_team"], row["stage"]])
        ben_pred = ben_df[ben_df["key"] == key].copy()
        tom_pred = tom_df[tom_df["key"] == key].copy()

        ben_y = np.array(ben_pred[["p_team1_win", "p_team2_win", "p_draw"]]).reshape(-1)
        tom_y = np.array(tom_pred[["p_team1_win", "p_team2_win", "p_draw"]]).reshape(-1)

        outcome = calculate_outcome(row["score_home"], row["score_away"])

        ben_score.append(log_loss(ben_y, outcome) + ben_score[-1])
        tom_score.append(log_loss(tom_y, outcome) + tom_score[-1])

        display_df.loc[len(display_df)] = [
            row["timestamp"],
            row["home_team"],
            row["away_team"],
            f"{row['score_home']}-{row['score_away']}",
            log_loss(ben_y, outcome),
            log_loss(tom_y, outcome),
            row["status"]
        ]

    fig, ax = plt.subplots()
    plt.plot(ben_score, label="ben")
    plt.plot(tom_score, label="tom")
    plt.xlabel("Number of results")
    plt.ylabel("Score")
    plt.legend()
    plt.grid("on")

    latest_result = results_df[results_df["timestamp"] == results_df["timestamp"].max()]

    st.text(f"Latest update: {latest_result.home_team.to_string(index=False)} vs {latest_result.away_team.to_string(index=False)} {latest_result.timestamp.to_string(index=False)}")

    st.text(f"Ben: {ben_score[-1]:.3f}")
    st.text(f"Tom: {tom_score[-1]:.3f}")

    st.pyplot(fig=fig)

    st.dataframe(display_df)

    print("Done!")


if __name__ == "__main__":
    main()

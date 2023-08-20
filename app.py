import os

import numpy as np
import psycopg2

from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

COL_NAMES = ["home_team", "away_team", "stage", "score_home", "score_away", "status", "timestamp"]
name_map = {
    "Ireland": "Republic of Ireland"
}


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


def remap_name(team_name: str) -> str:
    if name_map.get(team_name):
        return name_map[team_name]
    else:
        return team_name


def main():
    conn = connect_to_db(os.environ["DATABASE_URL"])
    results_df = get_results(conn)

    tom_df = read_predictions(Path("tghaynes_predictions.csv"))
    ben_df = read_predictions(Path("bpowell_predictions.csv"))
    tom_cpu_df = read_predictions(Path("cpu_predictions.csv"))

    tom_df["stage"] = tom_df["group"].apply(get_stage)
    ben_df["stage"] = tom_df["group"].apply(get_stage)
    tom_cpu_df["stage"] = tom_df["group"].apply(get_stage)

    ben_df["key"] = make_match_keys(ben_df)
    tom_df["key"] = make_match_keys(tom_df)
    tom_cpu_df["key"] = make_match_keys(tom_cpu_df)

    st.title("RSS Prediction Results")

    ben_score = [0]
    tom_score = [0]
    tom_cpu_score = [0]

    display_df = pd.DataFrame(columns=["time", "home", "away", "stage", "result", "ben_loss", "tom_loss", "status"])

    for idx, row in results_df.iterrows():
        if row["status"] != "FINISHED":
            continue

        home_team = remap_name(row["home_team"])
        away_team = remap_name(row["away_team"])

        key = "_".join([home_team, away_team, row["stage"]])
        if len(ben_df[ben_df["key"] == key]) == 0:
            key = "_".join([away_team, home_team, row["stage"]])
            away_score, home_score = (row["score_away"], row["score_home"])
            row["score_home"] = away_score
            row["score_away"] = home_score
            home_team, away_team = (away_team, home_team)
            key = "_".join([home_team, away_team, row["stage"]])

        ben_pred = ben_df[ben_df["key"] == key].copy()
        tom_pred = tom_df[tom_df["key"] == key].copy()
        tom_cpu_pred = tom_cpu_df[tom_cpu_df["key"] == key].copy()

        ben_y = np.array(ben_pred[["p_team1_win", "p_team2_win", "p_draw"]]).reshape(-1)
        tom_y = np.array(tom_pred[["p_team1_win", "p_team2_win", "p_draw"]]).reshape(-1)
        tom_cpu_y = np.array(tom_cpu_pred[["p_team1_win", "p_team2_win", "p_draw"]]).reshape(-1)

        outcome = calculate_outcome(row["score_home"], row["score_away"])

        ben_score.append(log_loss(ben_y, outcome) + ben_score[-1])
        tom_score.append(log_loss(tom_y, outcome) + tom_score[-1])
        tom_cpu_score.append(log_loss(tom_cpu_y, outcome) + tom_score[-1])

        display_df.loc[len(display_df)] = [
            row["timestamp"],
            home_team,
            away_team,
            row["stage"],
            f"{row['score_home']}-{row['score_away']}",
            log_loss(ben_y, outcome),
            log_loss(tom_y, outcome),
            row["status"]
        ]

    fig, ax = plt.subplots()
    plt.plot(ben_score, label="ben")
    plt.plot(tom_score, label="tom")
    plt.plot(tom_cpu_score, label="tom-cpu")
    plt.xlabel("Number of results")
    plt.ylabel("Score")
    plt.legend()
    plt.grid("on")

    latest_result = results_df[results_df["timestamp"] == results_df["timestamp"].max()]

    st.text(
        f"Latest update: {latest_result.home_team.to_string(index=False)} vs {latest_result.away_team.to_string(index=False)} {latest_result.timestamp.to_string(index=False)}"
    )

    st.text(f"Ben: {ben_score[-1]:.3f}")
    st.text(f"Tom: {tom_score[-1]:.3f}")

    display_df = display_df.sort_values("time", ascending=False)
    display_df.set_index("time", inplace=True)

    st.pyplot(fig=fig)
    st.dataframe(display_df)

    print("Done!")


if __name__ == "__main__":
    main()

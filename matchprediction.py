import sqlite3
import pandas as pd
import numpy as np

from collections import defaultdict, deque

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# ============================================================
# 1. LOAD DATA FROM SQLITE
# ============================================================

DB_PATH = "database.sqlite"   # CHANGE THIS TO YOUR DATABASE FILE

conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    home_team_api_id,
    away_team_api_id,
    home_team_goal,
    away_team_goal,
    date,
    league_id,
    season,
    stage
FROM Match_Optimized
"""

df = pd.read_sql_query(query, conn)

conn.close()


# ============================================================
# 2. BASIC PREPROCESSING
# ============================================================

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Remove rows with missing important values
df = df.dropna(
    subset=[
        "home_team_api_id",
        "away_team_api_id",
        "home_team_goal",
        "away_team_goal",
        "date"
    ]
)

# Convert goals to integers
df["home_team_goal"] = df["home_team_goal"].astype(int)
df["away_team_goal"] = df["away_team_goal"].astype(int)

# Sort matches chronologically
df = df.sort_values("date").reset_index(drop=True)

print("Dataset loaded successfully!")
print("Number of matches:", len(df))


# ============================================================
# 3. CREATE TARGET VARIABLE
# ============================================================

def get_match_result(home_goals, away_goals):

    if home_goals > away_goals:
        return "Home Win"

    elif home_goals < away_goals:
        return "Away Win"

    else:
        return "Draw"


df["result"] = df.apply(
    lambda row: get_match_result(
        row["home_team_goal"],
        row["away_team_goal"]
    ),
    axis=1
)


# ============================================================
# 4. CREATE HISTORICAL TEAM STATISTICS
# ============================================================

# Dictionary to store historical statistics for every team

team_stats = defaultdict(
    lambda: {
        "matches": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_scored": 0,
        "goals_conceded": 0,
        "recent_points": deque(maxlen=5)
    }
)


def get_team_features(team_id):

    stats = team_stats[team_id]

    matches = stats["matches"]

    # Avoid division by zero for teams with no previous matches
    if matches == 0:

        return {
            "matches_played": 0,
            "win_rate": 0,
            "draw_rate": 0,
            "avg_goals_scored": 0,
            "avg_goals_conceded": 0,
            "points_per_game": 0,
            "recent_form": 0
        }

    total_points = (
        stats["wins"] * 3
        + stats["draws"]
    )

    return {

        "matches_played":
            matches,

        "win_rate":
            stats["wins"] / matches,

        "draw_rate":
            stats["draws"] / matches,

        "avg_goals_scored":
            stats["goals_scored"] / matches,

        "avg_goals_conceded":
            stats["goals_conceded"] / matches,

        "points_per_game":
            total_points / matches,

        "recent_form":
            np.mean(stats["recent_points"])
            if len(stats["recent_points"]) > 0
            else 0
    }


def update_team_stats(
    team_id,
    goals_scored,
    goals_conceded
):

    stats = team_stats[team_id]

    stats["matches"] += 1

    stats["goals_scored"] += goals_scored
    stats["goals_conceded"] += goals_conceded

    # Win
    if goals_scored > goals_conceded:

        stats["wins"] += 1

        # 3 points for a win
        stats["recent_points"].append(3)

    # Loss
    elif goals_scored < goals_conceded:

        stats["losses"] += 1

        # 0 points for a loss
        stats["recent_points"].append(0)

    # Draw
    else:

        stats["draws"] += 1

        # 1 point for a draw
        stats["recent_points"].append(1)


# ============================================================
# 5. CREATE PRE-MATCH FEATURES
# ============================================================

features = []

for index, row in df.iterrows():

    home_team = row["home_team_api_id"]
    away_team = row["away_team_api_id"]

    # --------------------------------------------------------
    # IMPORTANT:
    # Get team statistics BEFORE updating with current match.
    # This prevents data leakage.
    # --------------------------------------------------------

    home_features = get_team_features(home_team)
    away_features = get_team_features(away_team)

    match_features = {

        # Home team features

        "home_matches_played":
            home_features["matches_played"],

        "home_win_rate":
            home_features["win_rate"],

        "home_draw_rate":
            home_features["draw_rate"],

        "home_avg_goals_scored":
            home_features["avg_goals_scored"],

        "home_avg_goals_conceded":
            home_features["avg_goals_conceded"],

        "home_points_per_game":
            home_features["points_per_game"],

        "home_recent_form":
            home_features["recent_form"],


        # Away team features

        "away_matches_played":
            away_features["matches_played"],

        "away_win_rate":
            away_features["win_rate"],

        "away_draw_rate":
            away_features["draw_rate"],

        "away_avg_goals_scored":
            away_features["avg_goals_scored"],

        "away_avg_goals_conceded":
            away_features["avg_goals_conceded"],

        "away_points_per_game":
            away_features["points_per_game"],

        "away_recent_form":
            away_features["recent_form"],


        # League information

        "league_id":
            row["league_id"],

        "stage":
            row["stage"]
    }

    features.append(match_features)


    # --------------------------------------------------------
    # NOW update statistics using current match result
    # --------------------------------------------------------

    update_team_stats(
        home_team,
        row["home_team_goal"],
        row["away_team_goal"]
    )

    update_team_stats(
        away_team,
        row["away_team_goal"],
        row["home_team_goal"]
    )


# Convert feature list to DataFrame
X = pd.DataFrame(features)

# Target
y = df["result"]

print("\nFeatures created successfully!")
print("Feature shape:", X.shape)


# ============================================================
# 6. REMOVE VERY EARLY MATCHES
# ============================================================

# Early matches have little or no historical information.
# We keep matches where both teams have played at least 3 matches.

valid_rows = (
    (X["home_matches_played"] >= 3)
    &
    (X["away_matches_played"] >= 3)
)

X = X[valid_rows]
y = y[valid_rows]

print("Matches after filtering:", len(X))


# ============================================================
# 7. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

# We do NOT randomly shuffle matches.
# Earlier matches are used for training.
# Later matches are used for testing.

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. TRAIN RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=12,

    min_samples_split=10,

    min_samples_leaf=5,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# 10. EVALUATE MODEL
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(
    feature_importance.head(10)
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "match_prediction_model.pkl"
)

print("\nModel saved successfully!")


# ============================================================
# 13. SAVE PROCESSED DATA
# ============================================================

processed_data = X.copy()

processed_data["result"] = y.values

processed_data.to_csv(
    "processed_match_data.csv",
    index=False
)

print("Processed data saved successfully!")

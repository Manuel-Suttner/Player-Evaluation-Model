import ast
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", message="An input array is constant")

# Load Data
features = pd.read_csv(r"C:\Users\manue\Desktop\Soccer-logs Thesis\features.csv", encoding="utf-8")
actions = pd.read_csv(r"C:\Users\manue\Desktop\Soccer-logs Thesis\actions.csv", encoding="utf-8")
players = pd.read_csv(r"C:\Users\manue\Desktop\Soccer-logs Thesis\players.csv", encoding="utf-8")
player_games = pd.read_csv(r"C:\Users\manue\Desktop\Soccer-logs Thesis\player_games.csv", encoding="utf-8")
games = pd.read_csv(r"C:\Users\manue\Desktop\Soccer-logs Thesis\games.csv", encoding="utf-8")

# Competition Names
try:
    competitions = pd.read_csv(r"C:\Users\manue\Desktop\Soccer-logs Thesis\competitions.csv", encoding="utf-8")
    games = games.merge(
        competitions[["wyId", "name"]].rename(columns={"wyId": "competition_id", "name": "competition_name"}),
        on="competition_id", how="left")
    print("✅ competitions merged")
except:
    games["competition_name"] = "Unknown"
    print("⚠️ competitions.csv missing")

# Merge Player & Game Info
for col in ["shortName", "firstName", "lastName", "middleName"]:
    if col in players.columns:
        players[col] = players[col].astype(str).str.encode("latin1", "ignore").str.decode("utf-8", "ignore")
if "wyId" in players.columns:
    players = players.rename(columns={'wyId': 'player_id'})

player_game_stats = player_games[['player_id', 'game_id', 'minutes_played', 'team_id']].merge(players,
                                                                                              on="player_id")
player_game_stats = player_game_stats.merge(
    games[['game_id', 'competition_id', 'season_id', 'game_date', 'competition_name']],
    on="game_id", how="left")


# Helper
def safe_merge(df1, df2, on, fill):
    return df1.merge(df2, on=on, how='left').fillna(fill)


def count_actions(cond, name):
    temp = actions[cond].groupby(['game_id', 'player_id']).size().reset_index(name=name)
    return temp


# Compute Statistics (base event counts)
stats = {
    "total_actions": actions.groupby(['game_id', 'player_id']).size().reset_index(name="total_actions"),
    "successful_actions": actions[actions['result_name'] == "success"].groupby(
        ['game_id', 'player_id']).size().reset_index(name="successful_actions"),
    "goals": count_actions((actions['type_name'] == "shot") & (actions['result_name'] == "success"), "goals"),
    "shots_on_target": count_actions(actions['type_name'] == "shot", "shots_on_target"),
    "total_passes": count_actions(actions['type_name'] == "pass", "total_passes"),
    "successful_passes": count_actions((actions['type_name'] == "pass") &
                                       (actions['result_name'] == "success"), "successful_passes"),
    "dribbles_completed": count_actions((actions['type_name'].isin(["dribble", "take_on"])) &
                                        (actions['result_name'] == "success"), "dribbles_completed"),
    "clearances": count_actions(actions['type_name'] == "clearance", "clearances"),
    "crosses": count_actions(actions['type_name'] == "cross", "crosses"),
    "fouls_committed": count_actions(actions['type_name'] == "foul", "fouls_committed")
}

for name, d in stats.items():
    player_game_stats = safe_merge(player_game_stats, d, ["game_id", "player_id"], {name: 0})

# Remove any old incorrect merges
for col in ["tackles", "interceptions"]:
    if col in player_game_stats.columns:
        player_game_stats.drop(columns=[col], inplace=True)

# Tackles
tackle_events = (
        ((actions['type_name'] == 'tackle') & (actions['result_name'] == 'success')) |
        ((actions['type_name'] == 'dribble') & (actions['result_name'] == 'fail')) |
        ((actions['type_name'] == 'take_on') & (actions['result_name'] == 'fail'))
)
tackles = actions[tackle_events].groupby(['game_id', 'player_id']).size().reset_index(name="tackles")
player_game_stats = safe_merge(player_game_stats, tackles, ['game_id', 'player_id'], {'tackles': 0})

# Interceptions
interception_events = actions[
    (actions['type_name'] == 'interception') &
    ((actions['end_x'] - actions['start_x']) > 0)
    ]
interceptions = interception_events.groupby(['game_id', 'player_id']).size().reset_index(name="interceptions")
player_game_stats = safe_merge(player_game_stats, interceptions, ['game_id', 'player_id'], {'interceptions': 0})

# Progressive actions
progressive_passes = actions[(actions['type_name'] == "pass") & ((actions['end_x'] - actions['start_x']) > 15)]
prog_send = progressive_passes.groupby(['game_id', 'player_id']).size().reset_index(name='progressive_passes')
player_game_stats = safe_merge(player_game_stats, prog_send, ['game_id', 'player_id'], {'progressive_passes': 0})

# Compute xG
shots = actions[actions['type_name'] == "shot"].copy()
shots['is_goal'] = (shots['result_name'] == "success").astype(int)
shots['dx'] = 105 - shots['start_x'];
shots['dy'] = 34 - shots['start_y']
shots['shot_distance'] = np.sqrt(shots['dx'] ** 2 + shots['dy'] ** 2)
shots['shot_angle'] = np.arctan2(7.32 * shots['dx'], (shots['dx'] ** 2 + shots['dy'] ** 2 - (7.32 / 2) ** 2))
shots = shots.dropna(subset=['shot_distance', 'shot_angle'])

if shots['is_goal'].nunique() > 1:
    model = LogisticRegression(max_iter=1000).fit(shots[['shot_distance', 'shot_angle']], shots['is_goal'])
    shots['xG'] = model.predict_proba(shots[['shot_distance', 'shot_angle']])[:, 1]
else:
    shots['xG'] = 0.1

xg = shots.groupby(['game_id', 'player_id'])['xG'].sum().reset_index()
player_game_stats = safe_merge(player_game_stats, xg, ['game_id', 'player_id'], {'xG': 0})

# Compute xA using xG on the receiving shot
passes = actions[actions['type_name'] == 'pass'].copy()

# Merge passes with shots using matching game + spatial coordinates
merged = passes.merge(
    shots[['game_id', 'start_x', 'start_y', 'xG', 'player_id']],
    left_on=['game_id', 'end_x', 'end_y'],  # pass end point = shot start point
    right_on=['game_id', 'start_x', 'start_y'],  # shooter location
    suffixes=('_pass', '_shot')
)

# Sum xG contribution for the passer = xA
xA = merged.groupby(['game_id', 'player_id_pass'])['xG'].sum().reset_index()
xA = xA.rename(columns={'player_id_pass': 'player_id', 'xG': 'xA'})

player_game_stats = safe_merge(player_game_stats, xA, ['game_id', 'player_id'], {'xA': 0})


# Per-90 Metrics
def per90(n, m):
    return (n / m * 90 if m > 0 else 0)


metrics = [
    'goals', 'shots_on_target', 'total_passes', 'successful_passes',
    'progressive_passes', 'dribbles_completed', 'tackles',
    'interceptions', 'clearances', 'crosses', 'xG', 'xA',
    'fouls_committed'
]

for c in metrics:
    player_game_stats[f"{c}_per90"] = player_game_stats.apply(lambda r: per90(r[c], r['minutes_played']), axis=1)

# Defensive Duels (Ground)
# Duel Won:
duel_won_mask = (
        ((actions['type_name'] == 'tackle') & (actions['result_name'] == 'success')) |
        ((actions['type_name'].isin(['dribble', 'take_on'])) & (actions['result_name'] == 'fail'))
)

duels_won = actions[duel_won_mask].groupby(
    ['game_id', 'player_id']
).size().reset_index(name='duels_won')
# Duel Lost:
# Use fouls committed as stable proxy
duels_lost = actions[actions['type_name'] == 'foul'].groupby(
    ['game_id', 'player_id']
).size().reset_index(name='duels_lost')

player_game_stats = safe_merge(player_game_stats, duels_won,
                               ['game_id', 'player_id'], {'duels_won': 0})
player_game_stats = safe_merge(player_game_stats, duels_lost,
                               ['game_id', 'player_id'], {'duels_lost': 0})
# Duel Ratio Metric
player_game_stats['duel_balance'] = (
        player_game_stats['duels_won'] /
        (player_game_stats['duels_won'] + player_game_stats['duels_lost'] + 1e-6)
)

# Cleaning & Remove Goalkeepers
df = player_game_stats.copy()
df['game_date'] = pd.to_datetime(df['game_date'])
df['role_clean'] = df['role'].apply(lambda x: ast.literal_eval(x).get("name") if isinstance(x, str) else None)
df = df[df['role_clean'].str.lower() != "goalkeeper"].copy()

# Reliability Filters
min_games = 8;
min_minutes = 400
df = df.groupby('player_id').filter(lambda x: len(x) >= min_games and x['minutes_played'].sum() >= min_minutes)
print("✅ Filtered unreliable players")

# Normalize within competition
from sklearn.preprocessing import StandardScaler

core_features = [c for c in df.columns if c.endswith("_per90")]


def scale_group(g):
    if len(g) > 1:
        return pd.DataFrame(StandardScaler().fit_transform(g), index=g.index, columns=g.columns)
    return g  # leave unchanged if only 1 sample in competition


df[core_features] = (
    df.groupby("competition_name", group_keys=False)[core_features]
    .apply(scale_group)
)

print("✅ Competition normalization applied safely")

# Weighted Performance Score
import numpy as np
import pandas as pd

for base in ["xG", "xA", "goals", "shots_on_target", "progressive_passes",
             "dribbles_completed", "successful_passes", "crosses",
             "tackles", "interceptions", "clearances"]:
    per90 = f"{base}_per90"
    if per90 not in df.columns:
        df[per90] = 0.0

role_weights = {
    "forward": {
        "goals_per90": 10.0,
        "xG_per90": 9.0,
        "xA_per90": 7.0,
        "shots_on_target_per90": 5.0,
        "progressive_passes_per90": 3.0,
        "dribbles_completed_per90": 2.5,
        "successful_passes_per90": 1.0,
        "crosses_per90": 1.0,
        "tackles_per90": 0.5,
        "interceptions_per90": 0.3,
        "clearances_per90": 0.1,
        "duel_balance": 3.0
    },
    "midfielder": {
        "goals_per90": 5.0,
        "xG_per90": 4.5,
        "xA_per90": 6.0,
        "shots_on_target_per90": 3.0,
        "progressive_passes_per90": 5.0,
        "dribbles_completed_per90": 2.5,
        "successful_passes_per90": 2.0,
        "crosses_per90": 1.5,
        "tackles_per90": 2.0,
        "interceptions_per90": 2.0,
        "clearances_per90": 0.5,
        "duel_balance": 4.0
    },
    "defender": {
        "goals_per90": 1.0,
        "xG_per90": 1.0,
        "xA_per90": 1.5,
        "shots_on_target_per90": 0.5,
        "progressive_passes_per90": 3.0,
        "dribbles_completed_per90": 1.5,
        "successful_passes_per90": 2.0,
        "crosses_per90": 1.5,
        "tackles_per90": 3.5,
        "interceptions_per90": 3.0,
        "clearances_per90": 2.5,
        "duel_balance": 5.0
    }
}


# Map role_clean to simple buckets
def _role_bucket(r):
    if pd.isna(r):
        return "midfielder"
    rlow = str(r).lower()
    if "forward" in rlow or "striker" in rlow or "wing" in rlow:
        return "forward"
    if "defender" in rlow or "fullback" in rlow or "centre back" in rlow or "center back" in rlow:
        return "defender"
    return "midfielder"


df["role_bucket"] = df["role_clean"].apply(_role_bucket)

# Percentile scaling inside competition+role for each per90 feature
per90_cols = [c for c in df.columns if c.endswith("_per90")]


def pct_rank(s: pd.Series):
    # percentile rank in [0,1]; constant groups become 0.5
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=s.index)
    return s.rank(pct=True, method="average")


for col in per90_cols:
    df[f"{col}_pct"] = (
        df.groupby(["competition_name", "role_bucket"], observed=False)[col]
        .transform(pct_rank)
        .astype(float)
    )
# Ensure duel metric percentiles exist
if "duel_balance_pct" not in df.columns:
    df["duel_balance_pct"] = (
        df.groupby(["competition_name", "role_bucket"], observed=False)["duel_balance"]
        .transform(pct_rank)
        .astype(float)
    )

# Minutes reliability factor per player-season
# Use season_id already merged in earlier
df["season_minutes"] = df.groupby(["player_id", "season_id"], observed=False)["minutes_played"].transform("sum")
# Stronger penalty for small samples; saturate near 1800 minutes
df["minutes_factor"] = np.sqrt(np.clip(df["season_minutes"] / 1800.0, 0.0, 1.0))
# Slight bonus for sustained availability beyond 2700 minutes (capped)
df["minutes_factor"] = np.clip(
    df["minutes_factor"] * (1.0 + 0.10 * np.clip((df["season_minutes"] - 2700) / 900, 0, 1)),
    0.0, 1.2)


# Outcome-heavy game score with team-strength correction already computed earlier
def score_row(r):
    bucket = r["role_bucket"]
    weights = role_weights[bucket]
    s = 0.0
    for k, w in weights.items():
        # use percentile version of each metric
        k_pct = f"{k}_pct"
        s += w * float(r.get(k_pct, 0.5))  # default to middle value if missing

    # include duel balance explicitly if not included yet via weights
    s += 4.0 * float(r.get("duel_balance_pct", 0.5))  # weight adaptable by role if needed

    # reliability adjustment based on playing time
    s *= float(r["minutes_factor"])
    return s


df["season17_score_game"] = df.apply(score_row, axis=1)
# Maintain compatibility with Part 2 Supervised model block
df["performance_score"] = df["season17_score_game"]

# ollapse to season-level ranking
season_window = (df["game_date"] >= pd.Timestamp("2017-07-01")) & (df["game_date"] <= pd.Timestamp("2018-06-30"))
season_ids_in_window = df.loc[season_window, "season_id"]
if not season_ids_in_window.empty:
    season_1718 = int(season_ids_in_window.value_counts().idxmax())
else:
    # fallback to most common season overall
    season_1718 = int(df["season_id"].value_counts().idxmax())

df["is_1718"] = (df["season_id"] == season_1718).astype(int)

season_scores = (
    df[df["is_1718"] == 1]
    .groupby(["role_bucket", "player_id", "shortName"], as_index=False)
    .agg(season17_score=("season17_score_game", "mean"),
         season_minutes=("season_minutes", "max"))
    .sort_values(["role_bucket", "season17_score"], ascending=[True, False])
)

# Save nice tables
season_scores.to_csv("season17_outcome_heavy_scores.csv", index=False)

print("✅ Built season-17/18 outcome-heavy scores with minutes reliability.")
print("\n🏆 Top 15 per role, season 2017/18:")
for bucket in ["forward", "midfielder", "defender"]:
    topk = (season_scores[season_scores["role_bucket"] == bucket]
    .head(15)[["shortName", "season17_score", "season_minutes"]])
    print(f"\n{bucket.upper()} — Top 15")
    print(topk.to_string(index=False))

# Role-wise supervised scoring (predict next-match performance)
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define feature set
role_features = [
    "xG_per90", "goals_per90", "xA_per90", "shots_on_target_per90",
    "progressive_passes_per90", "dribbles_completed_per90",
    "successful_passes_per90", "crosses_per90",
    "tackles_per90", "interceptions_per90", "clearances_per90",
    "duel_balance"
]

role_features = [f for f in role_features if f in df.columns]  # guard

# Create next-match target per player (temporal shift)
df_sorted = df.sort_values(["player_id", "game_date"]).copy()
df_sorted["next_perf"] = (
    df_sorted
    .groupby("player_id", group_keys=False)["performance_score"]
    .shift(-1)  # next match performance of the same player
)

# Remove rows without a valid next match target
df_supervised = df_sorted.dropna(subset=["next_perf"]).copy()

# Train per-role models (temporal split)
results_rows = []
coef_rows = []
predictions_all = []


# LassoCV first, fall back to RidgeCV if Lasso is too sparse for a role
def make_lasso():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LassoCV(alphas=None, cv=5, max_iter=20000, n_alphas=50, random_state=42))
    ])


def make_ridge():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 31), cv=5))
    ])


roles = df_supervised["role_clean"].dropna().unique()
print(f"🔧 Training supervised role models for next-match performance on {len(roles)} roles...")

for role in roles:
    role_df = df_supervised[df_supervised["role_clean"] == role].copy()

    # Temporal split: last 20% by date is test
    role_df = role_df.sort_values("game_date")
    n = len(role_df)
    if n < 200:
        # if very small, still proceed but warn
        print(f"⚠️ Role '{role}': only {n} samples. Results may be unstable.")
    split_idx = int(n * 0.8)
    train_df, test_df = role_df.iloc[:split_idx], role_df.iloc[split_idx:]

    X_train = train_df[role_features].values
    y_train = train_df["next_perf"].values
    X_test = test_df[role_features].values
    y_test = test_df["next_perf"].values

    # Try LassoCV first, fallback to RidgeCV if fails or is degenerate
    pipe = make_lasso()
    try:
        pipe.fit(X_train, y_train)
        # If Lasso zeroes everything, fallback to Ridge
        nonzero = np.sum(np.abs(pipe.named_steps["model"].coef_) > 1e-8)
        if nonzero == 0:
            print(f"ℹ️ Role '{role}': Lasso too sparse, switching to Ridge.")
            pipe = make_ridge()
            pipe.fit(X_train, y_train)
            model_name = "RidgeCV"
            coefs = pipe.named_steps["model"].coef_
        else:
            model_name = "LassoCV"
            coefs = pipe.named_steps["model"].coef_
    except Exception as e:
        print(f"ℹ️ Role '{role}': Lasso failed ({e}), switching to Ridge.")
        pipe = make_ridge()
        pipe.fit(X_train, y_train)
        model_name = "RidgeCV"
        coefs = pipe.named_steps["model"].coef_

    # Evaluate
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results_rows.append({
        "role": role,
        "model": model_name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    # Save coefficients in original feature space order
    for f, c in zip(role_features, coefs):
        coef_rows.append({"role": role, "feature": f, "coef": float(c), "model": model_name})

    # Store predictions for analysis
    tmp_pred = test_df[["player_id", "shortName", "role_clean", "game_id", "game_date"]].copy()
    tmp_pred["y_true_next_perf"] = y_test
    tmp_pred["y_pred_next_perf"] = y_pred
    predictions_all.append(tmp_pred.assign(model=model_name))

# Consolidate and persist
eval_df = pd.DataFrame(results_rows).sort_values(["role", "RMSE"])
coef_df = pd.DataFrame(coef_rows)
pred_df = pd.concat(predictions_all, ignore_index=True) if predictions_all else pd.DataFrame()

eval_df.to_csv("role_nextperf_eval.csv", index=False)
coef_df.to_csv("role_nextperf_coefficients.csv", index=False)
pred_df.to_csv("role_nextperf_predictions.csv", index=False)

print("\n✅ Supervised role models trained.")
print("📄 Saved: role_nextperf_eval.csv, role_nextperf_coefficients.csv, role_nextperf_predictions.csv")
print(eval_df.to_string(index=False))

# Create a per-match learned role score
learned_scores_list = []

for role in roles:
    role_df_all = df_supervised[df_supervised["role_clean"] == role].copy().sort_values("game_date")
    if len(role_df_all) < 50:
        # Very small roles will still be processed, but warn
        print(f"ℹ️ Role '{role}': small sample size ({len(role_df_all)}).")

    X_all = role_df_all[role_features].values
    y_all = role_df_all["next_perf"].values

    # Use the better model discovered above
    role_eval = eval_df[eval_df["role"] == role].sort_values("RMSE").head(1)
    preferred = role_eval["model"].values[0] if len(role_eval) > 0 else "RidgeCV"
    if preferred == "LassoCV" and role_eval["R2"].values[0] >= 0.1:
        final_pipe = make_lasso()
    else:
        final_pipe = make_ridge()

    final_pipe.fit(X_all, y_all)
    role_df_all["learned_role_score"] = final_pipe.predict(X_all)

    # Keep only columns we need for downstream steps
    learned_scores_list.append(
        role_df_all[["player_id", "shortName", "role_clean", "game_id", "game_date", "competition_name",
                     "team_id", "performance_score", "learned_role_score"] + role_features]
    )

learned_df = pd.concat(learned_scores_list, ignore_index=True) if learned_scores_list else pd.DataFrame()

# Save for downstream modeling
learned_df.to_csv("role_scored_timeseries.csv", index=False)
print("✅ Saved per-match learned_role_score to role_scored_timeseries.csv")


# Quick sanity check rankings
def top_k_by_role_score(frame, score_col="learned_role_score", k=15):
    tops = (
        frame.sort_values(["role_clean", score_col], ascending=[True, False])
        .groupby("role_clean")
        .head(k)
        .loc[:, ["role_clean", "shortName", score_col]]
    )
    return tops


print("\n🏆 Top 15 by learned_role_score (all matches, may repeat players):")
print(top_k_by_role_score(learned_df, "learned_role_score", 15).to_string(index=False))

# Aggregated per-player view including player_id
agg_player = (
    learned_df.groupby(["player_id", "role_clean", "shortName"], as_index=False)["learned_role_score"]
    .mean()
    .sort_values(["role_clean", "learned_role_score"], ascending=[True, False])
)

agg_player.to_csv("role_scored_players_mean.csv", index=False)
print("✅ Updated aggregated file includes player_id")

print("\n👉 Use 'role_scored_timeseries.csv' instead of raw df for your ARIMA/GRU steps.")
print("   Target to forecast for next match can be 'learned_role_score' or the original 'performance_score',")
print("   but learned_role_score is aligned with predicting the very next match.")

# ARIMA + GRU hybrid forecasting (per player, per role)
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Get the time series data
if os.path.exists("role_scored_timeseries.csv"):
    ts_df = pd.read_csv("role_scored_timeseries.csv")
else:
    ts_df = df[[
        "player_id", "shortName", "role_clean", "game_date", "performance_score"
    ]].copy()

# Basic cleanup
ts_df["game_date"] = pd.to_datetime(ts_df["game_date"], errors="coerce")
ts_df = ts_df.dropna(subset=["game_date", "performance_score", "player_id", "role_clean"])
ts_df = ts_df.sort_values(["role_clean", "player_id", "game_date"])

target_col = "performance_score"

# Small helpers
def to_weekly_series(frame: pd.DataFrame, target="performance_score"):
    """turn player’s rows into a regular weekly series for ARIMA"""
    s = (
        frame.set_index("game_date")[target]
        .asfreq("7D")
        .interpolate()
    )
    return s


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


# GRU on residuals
class ResidualDataset(Dataset):
    def __init__(self, values: np.ndarray, seq_len=6):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.values) - self.seq_len)

    def __getitem__(self, idx):
        x = self.values[idx:idx + self.seq_len].unsqueeze(-1)  # (seq_len, 1)
        y = self.values[idx + self.seq_len]
        return x, y


class GRUResidual(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        y = self.fc(out[:, -1, :])
        return y


def train_gru_on_residuals(residuals: np.ndarray, seq_len=6, epochs=20, lr=0.01, device="cpu"):
    """fit tiny GRU to predict next residual"""
    if len(residuals) <= seq_len + 2:
        return None  # not enough history
    ds = ResidualDataset(residuals, seq_len=seq_len)
    if len(ds) == 0:
        return None
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    model = GRUResidual(hidden=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict_next_residual(model, residuals: np.ndarray, seq_len=6, device="cpu"):
    if model is None or len(residuals) < seq_len:
        return 0.0
    x = torch.tensor(residuals[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        y = model(x).cpu().numpy().flatten()[0]
    return float(y)


# Main loop
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Using device: {device}")

CANDIDATE_ORDERS = [(0, 1, 1), (1, 1, 0), (1, 0, 0)]  # small, stable
min_points = 12  # need at least this much history

eval_rows = []
forecast_rows = []

for role in ts_df["role_clean"].dropna().unique():
    role_df = ts_df[ts_df["role_clean"] == role]
    players = role_df["player_id"].unique()

    for pid in tqdm(players, desc=f"{role} players"):
        p = role_df[role_df["player_id"] == pid][["shortName", "game_date", target_col]].dropna()
        if len(p) < min_points:
            continue

        s = to_weekly_series(p, target=target_col)
        if s.std() == 0 or len(s) < min_points:
            continue

        n = len(s)
        split = int(n * 0.8)
        train = s.iloc[:split]
        test = s.iloc[split:]

        # Pick best ARIMA on train
        best_order = None
        best_rmse = np.inf
        best_fit = None

        for order in CANDIDATE_ORDERS:
            try:
                m = ARIMA(train, order=order).fit()
                fc = m.forecast(steps=len(test))
                rmse, mae = evaluate(test.values, fc.values)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = order
                    best_fit = m
            except Exception:
                continue

        if best_fit is None:
            # Fallback: just repeat last value
            next_arima = float(train.values[-1])
            eval_rows.append({
                "role": role,
                "player_id": pid,
                "player": p["shortName"].iloc[-1],
                "n_obs": n,
                "arima_order": None,
                "arima_RMSE": None,
                "arima_MAE": None,
            })
            forecast_rows.append({
                "role": role,
                "player_id": pid,
                "player": p["shortName"].iloc[-1],
                "n_obs": n,
                "arima_next_forecast": next_arima,
                "hybrid_next_forecast": next_arima
            })
            continue

        # Evaluate ARIMA on test
        arima_preds = best_fit.forecast(steps=len(test)).values
        arima_rmse, arima_mae = evaluate(test.values, arima_preds)

        # Get residuals from train fit
        fitted_in_train = best_fit.fittedvalues.values
        # align with train
        residuals = train.values[-len(fitted_in_train):] - fitted_in_train

        # Train small GRU on residuals
        if len(residuals) >= 20:
            gru_model = train_gru_on_residuals(residuals, seq_len=6, epochs=15, lr=0.01, device=device)
            next_resid = predict_next_residual(gru_model, residuals, seq_len=6, device=device)
        else:
            gru_model = None
            next_resid = 0.0

        # Refit ARIMA on full series to predict next game
        try:
            full_fit = ARIMA(s, order=best_order).fit()
            next_arima = float(full_fit.forecast(steps=1).values[0])
        except Exception:
            next_arima = float(s.values[-1])

        hybrid_next = next_arima + next_resid

        eval_rows.append({
            "role": role,
            "player_id": pid,
            "player": p["shortName"].iloc[-1],
            "n_obs": n,
            "arima_order": str(best_order),
            "arima_RMSE": arima_rmse,
            "arima_MAE": arima_mae,
        })

        forecast_rows.append({
            "role": role,
            "player_id": pid,
            "player": p["shortName"].iloc[-1],
            "n_obs": n,
            "arima_order": str(best_order),
            "arima_next_forecast": next_arima,
            "hybrid_next_forecast": hybrid_next
        })

# Save outputs
eval_df = pd.DataFrame(eval_rows)
fc_df = pd.DataFrame(forecast_rows)

eval_df.to_csv("hybrid_arima_gru_eval.csv", index=False)
fc_df.to_csv("hybrid_arima_gru_forecasts.csv", index=False)

# --- Compute average RMSE and MAE per competition and role for Hybrid model ---
print("\n📊 Computing RMSE and MAE averages by competition and role (Hybrid ARIMA–GRU)...")

eval_hybrid = eval_df.copy()
eval_hybrid["model_name"] = "Hybrid_ARIMA_GRU"

try:
    comp_map = ts_df[["player_id", "role_clean", "competition_name"]].drop_duplicates()
    eval_hybrid = eval_hybrid.merge(comp_map, on="player_id", how="left")
except Exception as e:
    print("⚠️ Could not merge competition info:", e)
    eval_hybrid["competition_name"] = "Unknown"

eval_by_comp_role = (
    eval_hybrid.groupby(["model_name", "role", "competition_name"], as_index=False)
    .agg(avg_RMSE=("arima_RMSE", "mean"),
         avg_MAE=("arima_MAE", "mean"))
)

eval_by_comp_role.to_csv("hybrid_eval_by_competition_and_role.csv", index=False)
print("✅ Saved Hybrid evaluation summary to 'hybrid_eval_by_competition_and_role.csv'")
print(eval_by_comp_role.head(10))

# Top 15 per role
if not fc_df.empty:
    top15 = (fc_df.sort_values(["role", "hybrid_next_forecast"], ascending=[True, False])
             .groupby("role")
             .head(15)
             .reset_index(drop=True))
    top15.to_csv("hybrid_arima_gru_top15.csv", index=False)

    print("\n🏆 Top 15 Players per Role by ARIMA+GRU Hybrid:\n")
    for role in top15["role"].unique():
        sub = top15[top15["role"] == role][["player", "hybrid_next_forecast"]]
        print(f"⚽ {role.upper()} — Top 15")
        print(sub.to_string(index=False))
        print("-" * 40)

print("\n✅ ARIMA+GRU hybrid forecasting complete.")
print(f"Models evaluated: {len(eval_df)} | Forecasts generated: {len(fc_df)}")

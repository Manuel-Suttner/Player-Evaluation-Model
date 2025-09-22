import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Loading files
features = pd.read_csv(r"C:\Users\manue\Desktop\Soccer-logs Thesis\features.csv", encoding="utf-8")
actions = pd.read_csv(r"C:\Users\manue\Desktop\Soccer-logs Thesis\actions.csv", encoding="utf-8")
players = pd.read_csv(r"C:\Users\manue\Desktop\Soccer-logs Thesis\players.csv", encoding="utf-8")
player_games = pd.read_csv(r"C:\Users\manue\Desktop\Soccer-logs Thesis\player_games.csv", encoding="utf-8")
games = pd.read_csv(r"C:\Users\manue\Desktop\Soccer-logs Thesis\games.csv", encoding="utf-8")

# Normalize player names
for col in ["shortName", "firstName", "lastName", "middleName"]:
    if col in players.columns:
        players[col] = players[col].astype(str).apply(
            lambda x: x.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        )

# Build player-game dataset
player_game_stats = player_games[['player_id', 'game_id', 'minutes_played', 'is_starter']].copy()

# Merge player info
if 'wyId' in players.columns:
    players = players.rename(columns={'wyId': 'player_id'})
player_game_stats = player_game_stats.merge(players, on='player_id', how='left')

# Merge game info
player_game_stats = player_game_stats.merge(
    games[['game_id', 'competition_id', 'season_id', 'game_date']],
    on='game_id', how='left'
)

# Compute stats from actions.csv
# Basic actions
player_actions = actions.groupby(['game_id', 'player_id']).size().reset_index(name='total_actions')
player_game_stats = player_game_stats.merge(player_actions, on=['game_id', 'player_id'], how='left').fillna(
    {'total_actions': 0})

successful_actions = actions[actions['result_name'] == 'success'] \
    .groupby(['game_id', 'player_id']).size().reset_index(name='successful_actions')
player_game_stats = player_game_stats.merge(successful_actions, on=['game_id', 'player_id'], how='left').fillna(
    {'successful_actions': 0})

# Success rate
player_game_stats['action_success_rate'] = (
        player_game_stats['successful_actions'] / player_game_stats['total_actions'].replace(0, 1)
)

# Passing
total_passes = actions[actions['type_name'] == 'pass'] \
    .groupby(['game_id', 'player_id']).size().reset_index(name='total_passes')
player_game_stats = player_game_stats.merge(total_passes, on=['game_id', 'player_id'], how='left').fillna(
    {'total_passes': 0})

successful_passes = actions[(actions['type_name'] == 'pass') & (actions['result_name'] == 'success')] \
    .groupby(['game_id', 'player_id']).size().reset_index(name='successful_passes')
player_game_stats = player_game_stats.merge(successful_passes, on=['game_id', 'player_id'], how='left').fillna(
    {'successful_passes': 0})

# Pass success rate
player_game_stats['pass_success_rate'] = (
        player_game_stats['successful_passes'] / player_game_stats['total_passes'].replace(0, 1)
)

# Goals
goals = actions[(actions['type_name'] == 'shot') & (actions['result_name'] == 'success')] \
    .groupby(['game_id', 'player_id']).size().reset_index(name='goals')
player_game_stats = player_game_stats.merge(goals, on=['game_id', 'player_id'], how='left').fillna({'goals': 0})

# Shots on target
shots_on_target = actions[(actions['type_name'] == 'shot') & (actions['result_name'] == 'success')] \
    .groupby(['game_id', 'player_id']).size().reset_index(name='shots_on_target')
player_game_stats = player_game_stats.merge(shots_on_target, on=['game_id', 'player_id'], how='left').fillna(
    {'shots_on_target': 0})

# Dribbles completed
dribbles = actions[((actions['type_name'] == 'dribble') | (actions['type_name'] == 'take_on')) &
                   (actions['result_name'] == 'success')] \
    .groupby(['game_id', 'player_id']).size().reset_index(name='dribbles_completed')
player_game_stats = player_game_stats.merge(dribbles, on=['game_id', 'player_id'], how='left').fillna(
    {'dribbles_completed': 0})

# Tackles
tackles = actions[actions['type_name'] == 'tackle'].groupby(['game_id', 'player_id']).size().reset_index(name='tackles')
player_game_stats = player_game_stats.merge(tackles, on=['game_id', 'player_id'], how='left').fillna({'tackles': 0})

# Fouls committed
fouls = actions[actions['type_name'] == 'foul'].groupby(['game_id', 'player_id']).size().reset_index(
    name='fouls_committed')
player_game_stats = player_game_stats.merge(fouls, on=['game_id', 'player_id'], how='left').fillna(
    {'fouls_committed': 0})

# Clearances
clearances = actions[actions['type_name'] == 'clearance'].groupby(['game_id', 'player_id']).size().reset_index(
    name='clearances')
player_game_stats = player_game_stats.merge(clearances, on=['game_id', 'player_id'], how='left').fillna(
    {'clearances': 0})

# Crosses
crosses = actions[actions['type_name'] == 'cross'].groupby(['game_id', 'player_id']).size().reset_index(name='crosses')
player_game_stats = player_game_stats.merge(crosses, on=['game_id', 'player_id'], how='left').fillna({'crosses': 0})

# Interceptions
interceptions = actions[actions['type_name'] == 'interception'].groupby(['game_id', 'player_id']).size().reset_index(
    name='interceptions')
player_game_stats = player_game_stats.merge(interceptions, on=['game_id', 'player_id'], how='left').fillna(
    {'interceptions': 0})

# Duels won
duels = actions[(actions['type_name'] == 'duel') & (actions['result_name'] == 'success')] \
    .groupby(['game_id', 'player_id']).size().reset_index(name='duels_won')
player_game_stats = player_game_stats.merge(duels, on=['game_id', 'player_id'], how='left').fillna({'duels_won': 0})

# Progressive passes
progressive_passes = actions[(actions['type_name'] == 'pass') & ((actions['end_x'] - actions['start_x']) > 15)]
prog_pass_counts = progressive_passes.groupby(['game_id', 'player_id']).size().reset_index(name='progressive_passes')
player_game_stats = player_game_stats.merge(prog_pass_counts, on=['game_id', 'player_id'], how='left').fillna(
    {'progressive_passes': 0})

# Progressive carries
progressive_carries = actions[(actions['type_name'] == 'carry') & ((actions['end_x'] - actions['start_x']) > 15)]
prog_carry_counts = progressive_carries.groupby(['game_id', 'player_id']).size().reset_index(name='progressive_carries')
player_game_stats = player_game_stats.merge(prog_carry_counts, on=['game_id', 'player_id'], how='left').fillna(
    {'progressive_carries': 0})

# xG model
shots = actions[actions['type_name'] == 'shot'].copy()
shots['is_goal'] = (shots['result_name'] == 'success').astype(int)

shots['dx'] = 105 - shots['start_x']
shots['dy'] = 34 - shots['start_y']
shots['shot_distance'] = np.sqrt(shots['dx'] ** 2 + shots['dy'] ** 2)
shots['shot_angle'] = np.arctan2(7.32 * shots['dx'], (shots['dx'] ** 2 + shots['dy'] ** 2 - (7.32 / 2) ** 2))

shots = shots.replace([np.inf, -np.inf], np.nan).dropna(subset=['shot_distance', 'shot_angle'])
X = shots[['shot_distance', 'shot_angle']]
y = shots['is_goal']

if len(y.unique()) > 1:
    xg_model = LogisticRegression(max_iter=1000)
    xg_model.fit(X, y)
    shots['xG'] = xg_model.predict_proba(X)[:, 1]
else:
    shots['xG'] = 0.1  # fallback baseline

xg_per_game = shots.groupby(['game_id', 'player_id'])['xG'].sum().reset_index()
player_game_stats = player_game_stats.merge(xg_per_game, on=['game_id', 'player_id'], how='left').fillna({'xG': 0})

# xA model
passes = actions[actions['type_name'] == 'pass'].copy()
merged = passes.merge(
    shots[['game_id', 'start_x', 'start_y', 'xG', 'player_id']],
    left_on=['game_id', 'end_x', 'end_y'],
    right_on=['game_id', 'start_x', 'start_y'],
    suffixes=('_pass', '_shot')
)

xA_per_game = merged.groupby(['game_id', 'player_id_pass'])['xG'].sum().reset_index().rename(
    columns={'player_id_pass': 'player_id', 'xG': 'xA'}
)
player_game_stats = player_game_stats.merge(xA_per_game, on=['game_id', 'player_id'], how='left').fillna({'xA': 0})


# Compute per 90 minutes stats
def per90(numerator, minutes):
    return (numerator / minutes * 90) if minutes > 0 else 0


cols_for_per90 = [
    'goals', 'shots_on_target', 'total_passes', 'successful_passes',
    'dribbles_completed', 'tackles', 'fouls_committed', 'clearances', 'crosses',
    'interceptions', 'duels_won', 'progressive_passes', 'progressive_carries',
    'xG', 'xA'
]

for col in cols_for_per90:
    player_game_stats[f"{col}_per90"] = player_game_stats.apply(lambda r: per90(r[col], r['minutes_played']), axis=1)

# Save final dataset
player_game_stats.to_csv("player_game_stats.csv", index=False, encoding="utf-8-sig")

print(player_game_stats.head(20))
print("✅ Player-game time-series stats saved to player_game_stats.csv")

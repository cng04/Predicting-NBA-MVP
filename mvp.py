import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Constants
TEAM_FULL_TO_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK", "New Jersey Nets": "BRK",
    "Charlotte Hornets": "CHO", "Charlotte Bobcats": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU", "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New Orleans Hornets": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

TEAM_ABBREV_MAPPING = {
    "NOH": "NOP",
    "CHA": "CHO",
}

STATS_COLUMNS = ["G", "MP", "PTS", "TRB", "AST", "STL", "BLK", "FG%", "3P%", "FT%", "WS", "W/L%"]

# Helper Functions
def get_season_games(season):
    """Returns the number of games in a specific season to handle shortened seasons."""
    year = int(season.split("-")[1])
    if year == 12: return 66
    elif year == 20: return 67
    elif year == 21: return 72
    else: return 82

def ridge_cv_tuning(X, y, alphas, k=10):
    """
    Performs K-Fold Cross-Validation to find the best alpha for Ridge Regression.
    """
    # Ensure inputs are numpy arrays
    X_data = X.values if isinstance(X, pd.DataFrame) else np.array(X)
    y_data = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.array(y)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []
    
    print(f"Starting {k}-Fold Cross-Validation on {len(alphas)} hyperparameters...")
    
    for alpha in alphas:
        fold_mses = []
        for train_index, val_index in kf.split(X_data):
            X_train_fold, X_val_fold = X_data[train_index], X_data[val_index]
            y_train_fold, y_val_fold = y_data[train_index], y_data[val_index]
            
            model = Ridge(alpha=alpha)
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            fold_mses.append(mean_squared_error(y_val_fold, y_pred))
        
        avg_mse = np.mean(fold_mses)
        mse_scores.append(avg_mse)
        # print(f"Alpha: {alpha} | MSE: {avg_mse:.6f}")

    best_idx = np.argmin(mse_scores)
    return alphas[best_idx], mse_scores[best_idx], mse_scores

# Data Preprocessing Functions
def preprocess_historical_data(mvp_path, standings_path):
    """Loads and cleans historical training data (2010-2025)."""
    print("Loading historical data...")
    mvp_df = pd.read_csv(mvp_path)
    standings_df = pd.read_csv(standings_path)

    # 1. Clean MVP Data
    mvp_df = mvp_df.drop(columns=["Player", "Rank", "Age", "First", "Share", "Pts Max", "WS/48"], errors='ignore')
    mvp_df = mvp_df.rename(columns={"Tm": "Team"})
    mvp_df["Team"] = mvp_df["Team"].replace(TEAM_ABBREV_MAPPING)
    
    # Normalize Games Played
    mvp_df["G"] = mvp_df["G"] / mvp_df["Season"].apply(get_season_games)

    # Normalize stats by season (Z-Score)
    cols_to_normalize = ["PTS", "TRB", "AST", "STL", "BLK", "FG%", "3P%", "FT%"]
    for col in cols_to_normalize:
        mvp_df[col] = mvp_df.groupby("Season")[col].transform(lambda x: (x - x.mean()) / x.std())

    # Create Target Variable: Pts Won % (Share of points in that season)
    mvp_df["Pts Won %"] = mvp_df.groupby("Season")["Pts Won"].transform(lambda x: x / x.sum())
    mvp_df = mvp_df.drop(columns=["Pts Won"])

    # 2. Clean Standings Data
    standings_df = standings_df.drop(columns=["W", "L", "PS/G", "PA/G", "SRS", "GB", "Conference"], errors='ignore')
    standings_df['Team'] = standings_df['Team'].map(TEAM_FULL_TO_ABBREV)

    # 3. Merge Datasets
    final_df = pd.merge(mvp_df, standings_df, on=["Season", "Team"], how='left')
    
    # Prepare X and Y
    # Ensure we strictly select the columns we expect to use as features
    X = final_df[STATS_COLUMNS].fillna(0)
    y = final_df["Pts Won %"]
    
    return X, y

def preprocess_current_season_data(per_game_path, advanced_path, standings_path):
    """Loads and cleans the test data (Current Season)."""
    print("Loading current season data...")
    test_df = pd.read_csv(per_game_path)
    test_df = test_df[test_df["Player"] != "League Average"]
    
    # Merge with Advanced Stats (for Win Shares)
    advanced_stats_df = pd.read_csv(advanced_path)
    test_df = pd.merge(test_df, advanced_stats_df[["Player", "WS"]], on="Player", how="left")
    
    # Merge with Standings (for W/L%)
    standings_df = pd.read_csv(standings_path)
    standings_df = standings_df[["Team", "W/L%"]]
    standings_df['Team'] = standings_df['Team'].map(TEAM_FULL_TO_ABBREV)
    test_df = pd.merge(test_df, standings_df, on="Team", how="left")
    
    # Keep identification columns for final output, but separate features
    player_info = test_df[["Player", "Team"]]
    
    # Select Features
    X_test = test_df[STATS_COLUMNS].fillna(0)
    
    # Normalize features (Z-Score) 
    # Since this is a single season, we normalize the whole dataframe column-wise
    X_test = X_test.transform(lambda x: (x - x.mean()) / x.std())
    
    return X_test, player_info

# Main Execution
def main():
    # 1. Load Data
    X_train, y_train = preprocess_historical_data(
        "data/nba_mvp_voting_2010_to_2025.csv",
        "data/nba_standings_2010_to_2025.csv"
    )
    
    X_test, player_info = preprocess_current_season_data(
        "data/nba_per_game_stats_2025_to_2026.csv",
        "data/nba_advanced_stats_2025_to_2026.csv",
        "data/nba_standings_2025_to_2026.csv"
    )
    
    # 2. Tune Hyperparameters
    alpha_list = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    best_alpha, best_mse, _ = ridge_cv_tuning(X_train, y_train, alpha_list, k=10)
    
    print(f"\nBest Alpha Found: {best_alpha}")
    print(f"Lowest MSE: {best_mse:.6f}")
    
    # 3. Train Final Model
    print(f"Training final model with alpha={best_alpha}...")
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_train, y_train)
    
    # 4. Predict
    raw_predictions = final_model.predict(X_test)
    clipped_predictions = np.clip(raw_predictions, 0, None)

    total = np.sum(clipped_predictions)
    if total > 0:
        predictions = clipped_predictions / total
    else:
        predictions = clipped_predictions
    
    # 5. Display Results
    results = player_info.drop(columns=["Team"]).copy()
    results["Probability"] = predictions
    results = results.sort_values(by="Probability", ascending=False).reset_index(drop=True)
    
    print("\n--- 2025-26 MVP Predictions (Top 10) ---")
    print(results.head(10))
    
    # Save to CSV
    results.to_csv("mvp_predictions_2026.csv", index=False)
    print("\nFull predictions saved to 'mvp_predictions_2026.csv'")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
# coding: utf-8

# In[23]:


import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
import io


# # Step 1: Data Acquisition

# In[24]:


def scrape_mvp_voting(season: int = 2025) -> pd.DataFrame:
    """
    Scrape MVP Award Voting table from Basketball Reference.

    Args:
        season: Year of the award (e.g., 2000 for the 1999-00 season)
    """
    url = f"https://www.basketball-reference.com/awards/awards_{season}.html"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    # The MVP table usually has id="mvp"
    # It might be commented out to save bandwidth, so we check both
    table_html = None
    table = soup.find("table", {"id": "mvp"})

    if table:
        table_html = str(table)
    else:
        # Check inside comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            if 'id="mvp"' in comment:
                table_html = comment
                break

    if not table_html:
        raise ValueError(f"Could not find MVP table for season {season}")

    # Parse with Pandas
    # header=1 tells pandas to use the second row as the main header 
    # (ignoring the top 'Voting', 'Per Game' grouping row)
    df = pd.read_html(io.StringIO(table_html), header=1)[0]

    # Clean up columns
    # Sometimes read_html with header=1 leaves "Unnamed" columns or duplicates
    # We explicitly rename key columns to be safe

    # 1. Remove repeated header rows (scraped data often has headers repeated every 20 rows)
    df = df[df['Player'] != 'Player']

    # 2. Handle the "Team" column issue
    # The column is usually "Tm", but might have spaces. 
    # We rename columns to be standard.
    df.columns = [c.strip() for c in df.columns]

    # 3. Convert Numeric Columns
    # We leave 'Player', 'Tm', 'Pos' as strings, convert rest to numeric
    text_cols = ['Player', 'Tm', 'Pos', 'Rank']
    for col in df.columns:
        if col not in text_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaN in 'First' (First Place Votes) with 0, as empty usually means 0 votes
    if 'First' in df.columns:
        df['First'] = df['First'].fillna(0)

    # Add Season column
    start_year = season - 1
    season_str = f"{start_year}-{season % 100:02d}"    
    df.insert(0, 'Season', season_str)

    return df.reset_index(drop=True)


# In[25]:


df = scrape_mvp_voting(2025)
df


# In[26]:


import time

def scrape_nba_standings(season):
    # Construct URL (2026 is valid for the current 2025-26 season)
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_standings.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Retry logic to handle 429 errors
    max_retries = 3
    wait_time = 65 # Start with > 1 minute to be safe

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)

            # CHECK 429 BEFORE RAISING STATUS
            if response.status_code == 429:
                print(f"⚠️ Rate limited (429). Pausing for {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2 # Wait longer next time if it fails again
                continue

            response.raise_for_status() # Check for other errors

            # If we get here, the request worked!
            soup = BeautifulSoup(response.content, "html.parser")

            # --- PARSING LOGIC START ---
            dfs = []

            # Helper function to process a raw html table into a clean df
            def process_table(table_html):
                if not table_html: return None
                df = pd.read_html(str(table_html))[0]
                # Rename first column to "Team" (it handles "Eastern Conference", "Atlantic Division", etc.)
                df.rename(columns={df.columns[0]: "Team"}, inplace=True)
                return df

            # STRATEGY 1: Look for Conference Standings (Modern format, preferred)
            # We explicitly check for both E and W.
            found_conference = False
            for region in ['E', 'W']:
                table = soup.find('table', id=f'confs_standings_{region}')
                if table:
                    dfs.append(process_table(table))
                    found_conference = True

            # STRATEGY 2: If Conference tables aren't found, look for Division Standings
            # (Common in older years or if the view defaults to divisions)
            if not found_conference:
                print(f"   Note: Conference tables missing for {season}, trying Divisions...")
                for region in ['E', 'W']:
                    table = soup.find('table', id=f'divs_standings_{region}')
                    if table:
                        dfs.append(process_table(table))

            # STRATEGY 3: Catch-all for very old years (e.g. 1950s)
            # These sometimes don't have E/W regions in the ID, just one big table
            if not dfs:
                print(f"   Note: Standard E/W tables missing for {season}, grabbing generic standings...")
                # Find any table with 'standings' in the ID (e.g. "BFS_standings_NBA")
                tables = soup.find_all('table', attrs={'id': lambda x: x and 'standings' in x})
                for table in tables:
                    dfs.append(process_table(table))

            # If we still have nothing, we can't proceed for this year
            if not dfs:
                print(f"❌ Could not find any standings data for {season}")
                return None

            # Combine all found tables
            standings_df = pd.concat(dfs, ignore_index=True)

            # --- CLEANUP ---

            # 1. Filter out header rows found INSIDE the data
            # Conference tables have "Western Conference" headers; Division tables have "Atlantic Division" headers
            # We filter out any row where the Team name contains "Conference" OR "Division"
            standings_df = standings_df[~standings_df['Team'].str.contains("Conference|Division", case=False, na=False)]

            # 2. Add Season
            standings_df['Season'] = season

            # 3. Handle the asterisk (*) often found in Division tables for playoff teams
            # e.g. "Boston Celtics*" -> "Boston Celtics"
            standings_df['Team'] = standings_df['Team'].str.replace('*', '', regex=False)

            return standings_df

            # --- PARSING LOGIC END ---

            # return standing_df (Move this here)
            return soup # For now returning soup so you can verify it works

        except Exception as e:
            print(f"Error: {e}")
            if attempt == max_retries - 1:
                raise # Give up if retries fail


# In[27]:


mvp_dfs = []
standing_dfs = []

for season in range(1981, 2026):
    time.sleep(4)
    mvp_df = scrape_mvp_voting(season)
    mvp_dfs.append(mvp_df)
    time.sleep(4)
    standing_df = scrape_nba_standings(season)
    standing_dfs.append(standing_df)


all_mvp_dfs = pd.concat(mvp_dfs, ignore_index=True)
all_mvp_dfs.to_csv("data/nba_mvp_voting_1980_to_2025.csv", index=False)

all_standing_dfs = pd.concat(standing_dfs, ignore_index=True)
all_standing_dfs.to_csv("data/nba_standings_1980_to_2025.csv", index=False)


# In[ ]:


def scrape_per_game_stats(season: int = 2026) -> pd.DataFrame:
   """
   Scrape per-game stats from Basketball Reference.

   Args:
       season: NBA season year (e.g., 2025 for 2024-25 season)

   Returns:
       DataFrame with player per-game stats
   """
   url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"

   headers = {
       "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
       "Accept-Language": "en-US,en;q=0.5",
       "Referer": "https://www.basketball-reference.com/",
   }

   print(f"Fetching {url}...")
   response = requests.get(url, headers=headers, timeout=30)
   response.raise_for_status()

   soup = BeautifulSoup(response.content, "html.parser")

   # Find the per_game table (may be in comments)
   table = soup.find("table", {"id": "per_game_stats"})

   if table is None:
       comments = soup.find_all(string=lambda text: isinstance(text, Comment))
       for comment in comments:
           if "per_game_stats" in comment:
               comment_soup = BeautifulSoup(comment, "html.parser")
               table = comment_soup.find("table", {"id": "per_game_stats"})
               if table:
                   break

   if table is None:
       raise ValueError("Could not find per_game_stats table")

   df = pd.read_html(io.StringIO(str(table)))[0]

   # Clean up the dataframe
   # 1. Remove repeating header rows (where Rk is 'Rk')
   df = df[df['Rk'] != 'Rk']

   # 2. Handle Numeric Conversion safely
   # We infer objects first, then force numeric on everything except the known string columns
   df = df.infer_objects()

   cols_to_ignore = ['Player', 'Pos', 'Team', 'Awards']
   for col in df.columns:
       if col not in cols_to_ignore:
           df[col] = pd.to_numeric(df[col], errors='coerce')


   return df


# In[ ]:


df = scrape_per_game_stats(2026)
df.to_csv("data/nba_per_game_stats_2025_to_2026.csv", index=False)


# In[ ]:


df = scrape_nba_standings(2026)
df.to_csv("data/nba_standings_2025_to_2026.csv", index=False)


# In[ ]:





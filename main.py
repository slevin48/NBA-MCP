"""NBA-focused MCP server.

This server exposes tools, resources, and prompts inspired by the
https://github.com/slevin48/NBA project. It provides live scoreboard
information, recent game summaries, and simple win probability
calculations derived from Elo ratings.
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguegamefinder
from pydantic import Field

mcp = FastMCP("NBA MCP Server", stateless_http=True)


def _serialize_game_datetime(value: str) -> str:
    """Return an ISO 8601 timestamp for values provided by the NBA API."""

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        # Some endpoints return dates such as "2024-10-24T19:30:00" without
        # a timezone. In that case we return the original value.
        return value


def _determine_winner(
    home_team: Dict[str, Any],
    away_team: Dict[str, Any],
    status_text: str,
) -> str:
    home_score = int(home_team.get("score", 0) or 0)
    away_score = int(away_team.get("score", 0) or 0)

    if home_score > away_score:
        return home_team.get("teamName", "")
    if away_score > home_score:
        return away_team.get("teamName", "")
    if status_text.lower() == "final":
        return "Tie"
    return "Undefined"


def _format_live_games(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for game in games:
        home_team = game.get("homeTeam", {})
        away_team = game.get("awayTeam", {})
        status_text = game.get("gameStatusText", "")

        formatted.append(
            {
                "gameId": game.get("gameId"),
                "gameDate": _serialize_game_datetime(game.get("gameEt", "")),
                "gameStatusText": status_text,
                "homeTeamId": home_team.get("teamId"),
                "homeTeamName": home_team.get("teamName"),
                "homeTeamScore": int(home_team.get("score", 0) or 0),
                "awayTeamId": away_team.get("teamId"),
                "awayTeamName": away_team.get("teamName"),
                "awayTeamScore": int(away_team.get("score", 0) or 0),
                "winningTeam": _determine_winner(home_team, away_team, status_text),
            }
        )

    return formatted


def _load_league_games(season: str) -> pd.DataFrame:
    """Fetch the LeagueGameFinder dataset for a season with caching."""

    try:
        endpoint = leaguegamefinder.LeagueGameFinder(
            league_id_nullable="00", season_nullable=season
        )
    except Exception as exc:  # noqa: BLE001 - surface readable error message
        raise RuntimeError(
            "Unable to retrieve league games from the NBA stats API."
        ) from exc

    data_frame = endpoint.get_data_frames()[0]
    return data_frame


@lru_cache(maxsize=6)
def _season_game_table(season: str) -> pd.DataFrame:
    """Return a normalized table of games for the requested season."""

    games = _load_league_games(season)

    # Separate home and away entries using the MATCHUP column semantics.
    home_games = games[games["MATCHUP"].str.contains("vs.")].copy()
    away_games = games[games["MATCHUP"].str.contains("@")].copy()

    home_games = home_games.rename(
        columns={"TEAM_ID": "homeTeamId", "TEAM_NAME": "homeTeamName", "PTS": "homeTeamScore"}
    )
    away_games = away_games.rename(
        columns={"TEAM_ID": "awayTeamId", "TEAM_NAME": "awayTeamName", "PTS": "awayTeamScore"}
    )

    home_games["homeTeamName"] = home_games["homeTeamName"].str.split().str[-1]
    away_games["awayTeamName"] = away_games["awayTeamName"].str.split().str[-1]

    merged = pd.merge(
        home_games[["GAME_ID", "GAME_DATE", "homeTeamId", "homeTeamName", "homeTeamScore"]],
        away_games[["GAME_ID", "awayTeamId", "awayTeamName", "awayTeamScore"]],
        on="GAME_ID",
    )

    merged = merged.rename(columns={"GAME_ID": "gameId", "GAME_DATE": "gameDate"})
    merged["gameStatusText"] = "Final"
    merged["winningTeam"] = np.where(
        merged["homeTeamScore"] > merged["awayTeamScore"],
        merged["homeTeamName"],
        merged["awayTeamName"],
    )

    merged["gameDate"] = pd.to_datetime(merged["gameDate"]).dt.strftime("%Y-%m-%d")

    columns = [
        "gameId",
        "gameStatusText",
        "gameDate",
        "homeTeamId",
        "homeTeamName",
        "homeTeamScore",
        "awayTeamId",
        "awayTeamName",
        "awayTeamScore",
        "winningTeam",
    ]

    return merged[columns]


def _games_for_date(season: str, date_str: str) -> pd.DataFrame:
    table = _season_game_table(season)
    return table[table["gameDate"].str.startswith(date_str)]


def _game_by_id(season: str, game_id: str) -> pd.DataFrame:
    table = _season_game_table(season)
    return table[table["gameId"] == game_id]


def _calculate_win_probability(elo_team: float, elo_opponent: float) -> float:
    return 1 / (1 + 10 ** ((elo_opponent - elo_team) / 400))


@mcp.tool(
    title="List Live NBA Games",
    description="Fetch the current NBA scoreboard with scores, status, and winners.",
)
def list_live_games() -> List[Dict[str, Any]]:
    try:
        board = scoreboard.ScoreBoard()
    except Exception as exc:  # noqa: BLE001 - provide simple failure notice
        raise RuntimeError("Unable to retrieve the live NBA scoreboard.") from exc

    games = board.games.get_dict()
    return _format_live_games(games)


@mcp.tool(
    title="Find Games by Date",
    description="List NBA games for a given date within a season (YYYY-MM-DD).",
)
def find_games_by_date(
    date: str = Field(description="Date in YYYY-MM-DD format", examples=["2024-10-24"]),
    season: str = Field(
        description="NBA season in YYYY-YY format", default="2024-25", examples=["2024-25"]
    ),
) -> List[Dict[str, Any]]:
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("date must be provided in YYYY-MM-DD format") from exc

    games = _games_for_date(season, date)
    return games.to_dict(orient="records")


@mcp.tool(
    title="Lookup Game by ID",
    description="Retrieve a single NBA game for the provided gameId within a season.",
)
def lookup_game_by_id(
    game_id: str = Field(description="The NBA game identifier"),
    season: str = Field(description="NBA season in YYYY-YY format", default="2024-25"),
) -> List[Dict[str, Any]]:
    games = _game_by_id(season, game_id)
    return games.to_dict(orient="records")


@mcp.tool(
    title="Calculate Win Probability",
    description="Estimate win probability and implied odds from Elo ratings.",
)
def calculate_win_probability(
    team_elo: float = Field(description="Elo rating for the primary team"),
    opponent_elo: float = Field(description="Elo rating for the opposing team"),
) -> Dict[str, float]:
    win_probability = _calculate_win_probability(team_elo, opponent_elo)
    loss_probability = 1 - win_probability
    return {
        "winProbability": win_probability,
        "lossProbability": loss_probability,
        "impliedWinOdds": 1 / win_probability if win_probability else float("inf"),
        "impliedLossOdds": 1 / loss_probability if loss_probability else float("inf"),
    }


@mcp.resource(
    uri="nba://games/live",
    name="NBA Live Scoreboard",
    description="JSON payload summarizing the current NBA scoreboard.",
)
def live_games_resource() -> str:
    games = list_live_games()
    return json.dumps({"games": games}, indent=2)


@mcp.prompt("nba_matchup_report")
def matchup_report_prompt(
    home_team: str = Field(description="Home team name"),
    away_team: str = Field(description="Away team name"),
    home_elo: float = Field(description="Home team Elo rating"),
    away_elo: float = Field(description="Away team Elo rating"),
) -> str:
    """Create a structured analysis prompt for an NBA matchup."""

    home_prob = _calculate_win_probability(home_elo, away_elo)
    away_prob = 1 - home_prob

    return (
        "You are preparing a scouting report for tonight's NBA game between "
        f"{home_team} (home) and {away_team} (away).\n"
        "Provide:\n"
        "1. A quick overview of both teams' recent form and key players.\n"
        "2. Tactical matchups or trends to watch.\n"
        "3. A probabilistic prediction based on Elo ratings given below.\n\n"
        f"Elo ratings: {home_team} = {home_elo}, {away_team} = {away_elo}.\n"
        f"Implied win probabilities: {home_team} {home_prob:.1%}, "
        f"{away_team} {away_prob:.1%}."
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

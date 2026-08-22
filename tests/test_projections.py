"""Unit tests for the Marcel / Marcel-X projection math (pure functions)."""
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import projections as P


def league_avg_season(pa=180000):
    """A league whose per-PA rates are round numbers, plus totals."""
    return {
        "pa": pa, "single": int(pa * 0.14), "double": int(pa * 0.042),
        "triple": int(pa * 0.003), "hr": int(pa * 0.031), "ubb": int(pa * 0.08),
        "hbp": int(pa * 0.01), "so": int(pa * 0.22), "sf": int(pa * 0.007),
        "sb": int(pa * 0.018), "cs": int(pa * 0.005), "runs": int(pa * 0.118),
    }


def scaled(league, pa):
    """A player who is EXACTLY league average over `pa` plate appearances."""
    f = pa / league["pa"]
    return {k: (v * f if k != "pa" else pa) for k, v in league.items() if k != "runs"}


def test_age_factor_peaks_at_29():
    assert P.age_factor(29) == 1.0
    assert math.isclose(P.age_factor(25), 1.024)
    assert math.isclose(P.age_factor(33), 0.988)
    assert P.age_factor(None) == 1.0


def test_regressed_rate_pulls_toward_league():
    # 30 HR in 600 PA (.050) regressed with 1200 PA at .030
    r = P.regressed_rate([30], [600], (5, 4, 3), 0.03, 1200)
    assert math.isclose(r, (5 * 30 + 1200 * 0.03) / (5 * 600 + 1200))
    assert 0.03 < r < 0.05
    # no seasons at all → league rate
    assert P.regressed_rate([], [], (5, 4, 3), 0.03, 1200) == 0.03


def test_weighted_league_rate_blends_by_exposure():
    assert math.isclose(P.weighted_league_rate([0.03, 0.02], [600, 600], (5, 4)),
                        (5 * 600 * 0.03 + 4 * 600 * 0.02) / (5 * 600 + 4 * 600))
    assert P.weighted_league_rate([0.03], [0], (5,)) == 0.03


def test_league_average_hitter_projects_to_100_wrc_plus():
    lg = league_avg_season()
    seasons = [scaled(lg, 500), scaled(lg, 600), scaled(lg, 600)]
    out = P.project_hitter(seasons, [lg, lg, lg], age=29)
    assert out["pa"] == 600
    assert out["wrc_plus"] == 100
    # rates reproduce the league line
    assert math.isclose(out["avg"], lg_avg(lg), abs_tol=0.002)


def lg_avg(lg):
    h = lg["single"] + lg["double"] + lg["triple"] + lg["hr"]
    ab = lg["pa"] - lg["ubb"] - lg["hbp"] - lg["sf"]
    return h / ab


def test_power_hitter_regresses_but_stays_above_average():
    lg = league_avg_season()
    slugger = scaled(lg, 600)
    slugger["hr"] = 45
    out = P.project_hitter([slugger], [lg], age=27)
    lg_hr_600 = lg["hr"] / lg["pa"] * 600
    assert lg_hr_600 < out["hr"] < 45          # regressed, not copied
    assert out["wrc_plus"] > 100


def test_young_player_gets_aging_bonus():
    lg = league_avg_season()
    s = scaled(lg, 600)
    young = P.project_hitter([s], [lg], age=23)
    old = P.project_hitter([s], [lg], age=35)
    assert young["hr"] > old["hr"]
    assert young["so"] < old["so"]


def test_marcel_x_overrides_contact_quality_only():
    lg = league_avg_season()
    base = P.project_hitter([scaled(lg, 600)], [lg], age=29)
    xs = [{"pa": 600, "xba": 0.300, "xslg": 0.520, "xwoba": 0.380}]
    lgx = [{"xba": 0.245, "xslg": 0.400, "xwoba": 0.315, "lg_totals": lg}]
    out = P.project_hitter_x(xs, lgx, age=29, base=base)
    assert out["avg"] > base["avg"] and out["slg"] > base["slg"]
    assert out["hr"] == base["hr"] and out["sb"] == base["sb"]   # untouched
    assert out["wrc_plus"] > base["wrc_plus"]
    # regressed: below the raw expected line
    assert out["avg"] < 0.300


def test_marcel_x_requires_expected_inputs():
    lg = league_avg_season()
    base = P.project_hitter([scaled(lg, 600)], [lg], age=29)
    assert P.project_hitter_x([{"pa": 600}], [{}], age=29, base=base) == {}


def pitching_league(bf=180000):
    return {"bf": bf, "so": int(bf * 0.22), "bb": int(bf * 0.08), "hbp": int(bf * 0.01),
            "hr": int(bf * 0.031), "h": int(bf * 0.22), "ip": bf * 0.256,
            "er": bf * 0.256 * 4.1 / 9}


def test_league_average_pitcher_fip_matches_league_era():
    lg = pitching_league()
    f = 600 / lg["bf"]
    avg = {k: v * f for k, v in lg.items() if k != "er"}
    avg["bf"] = 600
    out = P.project_pitcher([avg], [lg], age=29)
    assert math.isclose(out["fip"], 4.1, abs_tol=0.05)
    assert out["bf"] == 600


def test_components_and_merge():
    stat = {"plateAppearances": 100, "hits": 30, "doubles": 5, "triples": 1,
            "homeRuns": 4, "baseOnBalls": 10, "intentionalWalks": 2}
    c = P.hitter_components(stat)
    assert c["single"] == 20 and c["ubb"] == 8
    splits = [{"season": "2025", "stat": stat}, {"season": "2025", "stat": stat},
              {"season": "bad", "stat": stat}]
    merged = P.merge_seasons(splits, P.hitter_components)
    assert list(merged) == [2025] and merged[2025]["pa"] == 200
    pc = P.pitcher_components({"inningsPitched": "10.2", "battersFaced": 45})
    assert math.isclose(pc["ip"], 10 + 2 / 3)

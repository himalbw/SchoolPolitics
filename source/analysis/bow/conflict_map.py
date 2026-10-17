"""Where is there the most in-meeting conflict? District index + maps.

Conflict = two word-list components (lexicon.py), mentions per 10,000 words, pooled over a district's meetings:
  meeting_conflict  decorum / disruption / hostility ("out of order", "point of order", "decorum", "shouting", "liar", ...)
  conflict_legal    lawsuits, recall, censure, resignation, misconduct, "public comment", "first amendment", ...
Each component is converted to a percentile among eligible districts (>= MIN_WORDS words) and the index is their mean (0-100).
Auto-captions have no tone or volume, so this measures conflict that is SAID (procedural/hostile vocabulary), not heard.
Writes output/tables/conflict_{district,state,period_national}.csv and output/figures/conflict_map.png, conflict_state_by_period.png.
No boundary files are available offline, so maps are (1) district points on lon/lat and (2) a state tile grid.
Run from the repo root (after count_phrases.py).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bow_data import INDIR_STATIC, LoadDQ, LoadPhraseCounts, OUTDIR_FIGURES, OUTDIR_TABLES, PERIODS
from lexicon import LEXICON

COMPONENTS = ["meeting_conflict", "conflict_legal"]
# Routine agenda/speech words inside conflict_legal that are NOT conflict ("public comment" is ~40% of its mentions).
EXCLUDE = {"public comment", "transparency", "recall", "open meetings act", "first amendment"}
MIN_WORDS = 100_000          # about 8 meetings
MIN_DISTRICTS_STATE = 5
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E4E4E4"
CMAP = "YlOrRd"

# State tile grid (col, row): rough geography, one square per state.
TILES = {"ME": (10, 0), "WI": (5, 1), "VT": (9, 1), "NH": (10, 1), "WA": (0, 2), "ID": (1, 2), "MT": (2, 2), "ND": (3, 2),
         "MN": (4, 2), "IL": (5, 2), "MI": (6, 2), "NY": (8, 2), "MA": (9, 2), "OR": (0, 3), "NV": (1, 3), "WY": (2, 3),
         "SD": (3, 3), "IA": (4, 3), "IN": (5, 3), "OH": (6, 3), "PA": (7, 3), "NJ": (8, 3), "CT": (9, 3), "RI": (10, 3),
         "CA": (0, 4), "UT": (1, 4), "CO": (2, 4), "NE": (3, 4), "MO": (4, 4), "KY": (5, 4), "WV": (6, 4), "VA": (7, 4),
         "MD": (8, 4), "DE": (9, 4), "AZ": (1, 5), "NM": (2, 5), "KS": (3, 5), "AR": (4, 5), "TN": (5, 5), "NC": (6, 5),
         "SC": (7, 5), "DC": (8, 5), "OK": (3, 6), "LA": (4, 6), "MS": (5, 6), "AL": (6, 6), "GA": (7, 6), "HI": (0, 7),
         "TX": (3, 7), "FL": (8, 7), "AK": (0, 0)}

def Main():
    OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    OUTDIR_FIGURES.mkdir(parents=True, exist_ok=True)
    dtm, vocab, units = LoadDQ()
    df_phrases = LoadPhraseCounts(units)
    units = AddConflictCounts(dtm, vocab, units, df_phrases)
    df_district = DistrictIndex(units)
    df_district.to_csv(OUTDIR_TABLES / "conflict_district.csv", index=False)
    df_state = StateTable(df_district, units)
    df_state.to_csv(OUTDIR_TABLES / "conflict_state.csv", index=False)
    National(units).to_csv(OUTDIR_TABLES / "conflict_period_national.csv", index=False)
    print("Top 15 districts:\n", df_district.sort_values("index", ascending=False).head(15)
          [["lea_name", "state", "n_words", "meeting_conflict", "conflict_legal", "index"]].round(2).to_string(index=False))
    print("\nTop 12 states (>= %d districts):\n" % MIN_DISTRICTS_STATE, df_state.dropna(subset=["index"]).sort_values("index", ascending=False).head(12).round(2).to_string(index=False))
    PlotMap(df_district, df_state)
    PlotStateByPeriod(df_state)
    return

def AddConflictCounts(dtm, vocab, units, df_phrases):
    """Add per-unit counts of each component's terms (matrix) and phrases (text) to `units`."""
    vocab_pos = {t: i for i, t in enumerate(vocab)}
    units = units.copy()
    for c in COMPONENTS:
        idx = [vocab_pos[t] for t in LEXICON[c]["terms"] if t in vocab_pos and t not in EXCLUDE]
        phrases = [p for p in LEXICON[c]["phrases"] if p in df_phrases.columns and p not in EXCLUDE]
        units[c] = np.asarray(dtm[:, idx].sum(axis=1)).ravel() + df_phrases[phrases].sum(axis=1).to_numpy()
    return units

def Rate(g):
    out = {"n_words": g.n_words.sum(), "n_meetings": g.n_docs.sum()}
    for c in COMPONENTS:
        out[c] = g[c].sum() / g.n_words.sum() * 1e4
    return pd.Series(out)

def DistrictIndex(units):
    df = units.groupby("leaid").apply(Rate, include_groups=False).reset_index()
    static = pd.read_parquet(INDIR_STATIC, columns=["leaid", "lea_name", "state_location", "latitude", "longitude"])
    static["leaid"] = static.leaid.astype(str).str.zfill(7)
    df = df.merge(static.rename(columns={"state_location": "state"}), on="leaid", how="left")
    eligible = df.n_words >= MIN_WORDS
    for c in COMPONENTS:
        df[f"{c}_pct"] = df.loc[eligible, c].rank(pct=True) * 100
    df["index"] = df[[f"{c}_pct" for c in COMPONENTS]].mean(axis=1)
    print(f"{eligible.sum()} of {len(df)} districts have >= {MIN_WORDS:,} words and get an index")
    return df

def StateTable(df_district, units):
    elig = df_district[df_district["index"].notna()]
    state = elig.groupby("state").agg(n_districts=("leaid", "size"), index=("index", "mean"),
                                      meeting_conflict=("meeting_conflict", "mean"), conflict_legal=("conflict_legal", "mean")).reset_index()
    state.loc[state.n_districts < MIN_DISTRICTS_STATE, "index"] = np.nan
    # by period: district-pooled rates within eligible districts, then mean across districts of the state
    u = units[units.leaid.isin(elig.leaid)].merge(elig[["leaid", "state"]], on="leaid")
    parts = []
    for period, g in u.groupby("period"):
        r = g.groupby("leaid").apply(Rate, include_groups=False).reset_index().merge(elig[["leaid", "state"]], on="leaid")
        r = r[r.n_words >= MIN_WORDS / 3]
        s = r.groupby("state")[COMPONENTS].mean().add_suffix(f"_{period}")
        parts.append(s)
    return state.merge(pd.concat(parts, axis=1).reset_index(), on="state", how="left")

def National(units):
    g = units.groupby("quarter")[COMPONENTS + ["n_words"]].sum()
    for c in COMPONENTS:
        g[c] = g[c] / g.n_words * 1e4
    return g.reset_index()

def PlotMap(df_district, df_state):
    fig = plt.figure(figsize=(14, 6.4))
    ax = fig.add_axes([0.02, 0.05, 0.56, 0.86])
    d = df_district[df_district["index"].notna() & df_district.longitude.between(-125, -66) & df_district.latitude.between(24, 50)]
    d = d.sort_values("index")
    sc = ax.scatter(d.longitude, d.latitude, c=d["index"], cmap=CMAP, s=18, vmin=0, vmax=100, edgecolor="white", linewidth=0.3)
    ax.set_aspect(1.25)
    ax.set_axis_off()
    ax.set_title("Districts: conflict index (percentile; contiguous US)", loc="left", fontsize=11)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("conflict index (0-100)")
    ax2 = fig.add_axes([0.62, 0.05, 0.37, 0.86])
    DrawTiles(ax2, df_state.set_index("state")["index"], "State mean index (>= %d districts; grey = fewer)" % MIN_DISTRICTS_STATE, vmin=0, vmax=100)
    fig.savefig(OUTDIR_FIGURES / "conflict_map.png", dpi=160)
    plt.close(fig)

def PlotStateByPeriod(df_state):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    s = df_state.set_index("state")
    both = s[[f"{c}_{p}" for c in COMPONENTS for p in ("pre", "covid", "post")]].copy()
    vmax = np.nanpercentile(both.to_numpy(), 95)
    for ax, p in zip(axes, ["pre", "covid", "post"]):
        total = s[f"meeting_conflict_{p}"] + s[f"conflict_legal_{p}"]
        total[s.n_districts < MIN_DISTRICTS_STATE] = np.nan
        DrawTiles(ax, total, {"pre": "2018-19", "covid": "2020-21", "post": "2022-23"}[p], vmin=0, vmax=2 * vmax, fmt="{:.1f}")
    fig.suptitle("Conflict-word mentions per 10k words (meeting_conflict + conflict_legal), state mean of districts", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR_FIGURES / "conflict_state_by_period.png", dpi=160)
    plt.close(fig)

def DrawTiles(ax, values, title, vmin, vmax, fmt="{:.0f}"):
    cmap = plt.get_cmap(CMAP)
    for st, (x, y) in TILES.items():
        v = values.get(st, np.nan)
        color = "#DDDDDD" if pd.isna(v) else cmap((v - vmin) / (vmax - vmin))
        ax.add_patch(plt.Rectangle((x, -y), 0.94, 0.94, color=color, lw=0))
        dark = (not pd.isna(v)) and (v - vmin) / (vmax - vmin) > 0.6
        ax.text(x + 0.47, -y + 0.6, st, ha="center", va="center", fontsize=8, color="white" if dark else INK)
        if not pd.isna(v):
            ax.text(x + 0.47, -y + 0.25, fmt.format(v), ha="center", va="center", fontsize=7, color="white" if dark else MUTED)
    ax.set_xlim(-0.2, 11)
    ax.set_ylim(-7.3, 1.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=11)

if __name__ == "__main__":
    Main()

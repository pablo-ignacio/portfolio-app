TICKERS = [
    # US Equities — broad & factor
    "SPY", "QQQ", "IWM", "MDY", "VTV", "VUG",
    # International Equities
    "EFA", "EEM", "VEA", "VWO", "EWJ", "FXI",
    # Fixed Income
    "TLT", "IEF", "SHY", "HYG", "LQD", "TIP", "EMB",
    # Real Assets & REITs
    "VNQ", "VNQI",
    # Broad Commodities
    "DBC", "PDBC", "COMT",
    # Copper
    "CPER", "COPX",
    # Energy
    "USO", "BNO", "UNG", "XLE",
    # Precious Metals
    "GLD", "IAU", "SLV", "PPLT", "PALL",
    # Agriculture
    "DBA", "CORN", "WEAT", "SOYB",
    # Broad Commodities (additional)
    "GSG",
    # Sector Equities
    "XLF", "XLK", "XLV", "XLE", "XLI", "XLB", "XLU", "XLRE",
    # International Single-Country / Regional
    "EWZ", "EWG", "EWU", "INDA",
    # Thematic / Real Assets
    "WOOD", "MOO",
]

# Deduplicate while preserving order
_seen = set()
TICKERS = [t for t in TICKERS if not (t in _seen or _seen.add(t))]

START_DATE = "2015-01-01"
END_DATE = None

MOMENTUM_LOOKBACK_DAYS = 126
VOL_LOOKBACK_DAYS = 20
TOP_N = 10
REBALANCE_FREQ = "ME"

ANNUAL_TARGET_VOL = 0.12
MAX_LEVERAGE = 1.5
TRANSACTION_COST_BPS = 5.0
WEIGHT_CHANGE_THRESHOLD = 0.02

USE_SPY_FILTER = True
REGIME_FILTER_TICKER = "SPY"
REGIME_FILTER_MA_DAYS = 200

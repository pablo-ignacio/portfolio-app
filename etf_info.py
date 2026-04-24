# etf_info.py — short descriptions for every ETF in the universe
# Used to annotate ticker cells with "TICKER — description" so
# double-clicking a cell in Streamlit reveals the full context.

ETF_INFO = {
    # US Equities — broad & factor
    "SPY":  "SPDR S&P 500 ETF — tracks the 500 largest US companies",
    "QQQ":  "Invesco Nasdaq-100 ETF — 100 largest non-financial Nasdaq stocks, tech-heavy",
    "IWM":  "iShares Russell 2000 ETF — ~2,000 small-cap US stocks",
    "MDY":  "SPDR S&P MidCap 400 ETF — mid-sized US companies",
    "VTV":  "Vanguard Value ETF — large-cap US stocks trading at low valuations",
    "VUG":  "Vanguard Growth ETF — large-cap US stocks with high growth characteristics",
    # International Equities
    "EFA":  "iShares MSCI EAFE ETF — developed-market equities ex-US/Canada",
    "EEM":  "iShares MSCI Emerging Markets ETF — broad EM equity exposure",
    "VEA":  "Vanguard Developed Markets ETF — Europe, Asia-Pacific large/mid caps",
    "VWO":  "Vanguard Emerging Markets ETF — broad EM equities, low cost",
    "EWJ":  "iShares MSCI Japan ETF — Japanese equities",
    "FXI":  "iShares China Large-Cap ETF — 50 largest Hong Kong-listed Chinese stocks",
    "EWZ":  "iShares MSCI Brazil ETF — Brazilian large/mid-cap equities",
    "EWG":  "iShares MSCI Germany ETF — German equities",
    "EWU":  "iShares MSCI United Kingdom ETF — UK large/mid-cap equities",
    "INDA": "iShares MSCI India ETF — Indian large/mid-cap equities",
    # Fixed Income
    "TLT":  "iShares 20+ Year Treasury Bond ETF — long-duration US Treasuries",
    "IEF":  "iShares 7-10 Year Treasury Bond ETF — intermediate US Treasuries",
    "SHY":  "iShares 1-3 Year Treasury Bond ETF — short-duration US Treasuries",
    "HYG":  "iShares iBoxx $ High Yield Corporate Bond ETF — US junk bonds",
    "LQD":  "iShares iBoxx $ Investment Grade Corporate Bond ETF — US IG corporates",
    "TIP":  "iShares TIPS Bond ETF — US Treasury Inflation-Protected Securities",
    "EMB":  "iShares JP Morgan EM Bond ETF — USD-denominated emerging market bonds",
    "AGG":  "iShares Core US Aggregate Bond ETF — broad US investment-grade bond market",
    # Real Assets & REITs
    "VNQ":  "Vanguard Real Estate ETF — US REITs and real estate companies",
    "VNQI": "Vanguard Global ex-US Real Estate ETF — international REITs",
    # Broad Commodities
    "DBC":  "Invesco DB Commodity Index ETF — diversified commodity futures basket",
    "PDBC": "Invesco Optimum Yield Diversified Commodity ETF — commodity futures, no K-1",
    "COMT": "iShares GSCI Commodity Dynamic Roll ETF — broad commodity index with roll optimisation",
    "GSG":  "iShares S&P GSCI Commodity-Indexed Trust — broad commodity index, ~55% energy",
    # Copper
    "CPER": "United States Copper Index Fund — copper futures ETF",
    "COPX": "Global X Copper Miners ETF — stocks of copper mining companies",
    # Energy
    "USO":  "United States Oil Fund — front-month WTI crude oil futures",
    "BNO":  "United States Brent Oil Fund — front-month Brent crude oil futures",
    "UNG":  "United States Natural Gas Fund — front-month Henry Hub natural gas futures",
    "XLE":  "SPDR Energy Select Sector ETF — S&P 500 energy companies",
    # Precious Metals
    "GLD":  "SPDR Gold Shares — physical gold bullion ETF",
    "IAU":  "iShares Gold Trust — physical gold, lower expense ratio than GLD",
    "SLV":  "iShares Silver Trust — physical silver bullion ETF",
    "PPLT": "Aberdeen Physical Platinum Shares ETF — physical platinum bullion",
    "PALL": "Aberdeen Physical Palladium Shares ETF — physical palladium bullion",
    # Agriculture
    "DBA":  "Invesco DB Agriculture Fund — diversified agriculture commodity futures",
    "CORN": "Teucrium Corn Fund — corn futures ETF",
    "WEAT": "Teucrium Wheat Fund — wheat futures ETF",
    "SOYB": "Teucrium Soybean Fund — soybean futures ETF",
    # US Sectors
    "XLF":  "SPDR Financial Select Sector ETF — US banks, insurers, and financial services",
    "XLK":  "SPDR Technology Select Sector ETF — US tech hardware, software, and semiconductors",
    "XLV":  "SPDR Health Care Select Sector ETF — US pharma, biotech, and health services",
    "XLI":  "SPDR Industrial Select Sector ETF — US aerospace, machinery, and transports",
    "XLB":  "SPDR Materials Select Sector ETF — US chemicals, metals, and mining",
    "XLU":  "SPDR Utilities Select Sector ETF — US electric, gas, and water utilities",
    "XLRE": "SPDR Real Estate Select Sector ETF — S&P 500 real estate companies and REITs",
    # Thematic / Real Assets
    "WOOD": "iShares Global Timber & Forestry ETF — timber, paper, and packaging companies",
    "MOO":  "VanEck Agribusiness ETF — global companies in farm equipment, seeds, and fertilisers",
}


def ticker_label(ticker: str) -> str:
    """Return 'TICKER — description' if known, else just the ticker."""
    desc = ETF_INFO.get(ticker.upper())
    if desc:
        return f"{ticker} — {desc}"
    return ticker

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Advanced Signal Scanner", layout="wide")
st.title("📊 Advanced Signal Scanner (34 strategies)")
st.caption("Educational tool. No signal is guaranteed. Use a demo account only.")

PAIRS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X", "AUD/JPY": "AUDJPY=X", "EUR/AUD": "EURAUD=X",
    "EUR/CAD": "EURCAD=X", "GBP/AUD": "GBPAUD=X", "CAD/JPY": "CADJPY=X",
    "CHF/JPY": "CHFJPY=X", "AUD/CAD": "AUDCAD=X", "NZD/JPY": "NZDJPY=X",
}
PERIODS = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "5y"}


@st.cache_data(ttl=60)
def get_data(symbol, interval):
    df = yf.download(symbol, interval=interval, period=PERIODS[interval],
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


# ---------- helpers ----------
def cu(a, b):
    bs = b.shift() if hasattr(b, "shift") else b
    return (a > b) & (a.shift() <= bs)


def cd(a, b):
    bs = b.shift() if hasattr(b, "shift") else b
    return (a < b) & (a.shift() >= bs)


def S(call, put):
    s = pd.Series(0, index=call.index)
    s[call.fillna(False)] = 1
    s[put.fillna(False)] = -1
    return s


def rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l)


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def cci(df, n):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md)


def heikin_ashi_bull(df):
    o, h, l, c = (df[x].values for x in ["Open", "High", "Low", "Close"])
    hc = (o + h + l + c) / 4
    ho = np.zeros(len(df))
    ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(df)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    return pd.Series((hc > ho).astype(int), index=df.index)


# ---------- 34 strategies ----------
def strategies(df):
    c, h, l, o = df["Close"], df["High"], df["Low"], df["Open"]
    out = {}

    r14, r7 = rsi(c, 14), rsi(c, 7)
    out["RSI14 30/70"] = S(r14 < 30, r14 > 70)
    out["RSI7 20/80"] = S(r7 < 20, r7 > 80)
    out["RSI14 re-cross"] = S(cu(r14, 30), cd(r14, 70))

    lowk, highk = l.rolling(14).min(), h.rolling(14).max()
    k = 100 * (c - lowk) / (highk - lowk)
    d = k.rolling(3).mean()
    out["Stoch 20/80"] = S(k < 20, k > 80)
    out["Stoch cross in zone"] = S(cu(k, d) & (k < 30), cd(k, d) & (k > 70))
    wr = -100 * (highk - c) / (highk - lowk)
    out["Williams %R"] = S(wr < -80, wr > -20)

    c20, c14 = cci(df, 20), cci(df, 14)
    out["CCI20 +-100"] = S(c20 < -100, c20 > 100)
    out["CCI14 +-200"] = S(c14 < -200, c14 > 200)

    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sg = macd.ewm(span=9).mean()
    hist = macd - sg
    out["MACD cross"] = S(cu(macd, sg), cd(macd, sg))
    out["MACD zero cross"] = S(cu(macd, 0), cd(macd, 0))
    out["MACD hist reversal"] = S(
        (hist > hist.shift()) & (hist.shift() > hist.shift(2)) & (hist < 0),
        (hist < hist.shift()) & (hist.shift() < hist.shift(2)) & (hist > 0))

    for f, s in [(5, 13), (9, 21), (20, 50)]:
        ef, es = c.ewm(span=f).mean(), c.ewm(span=s).mean()
        out[f"EMA {f}/{s} cross"] = S(cu(ef, es), cd(ef, es))
    s10, s30 = c.rolling(10).mean(), c.rolling(30).mean()
    out["SMA 10/30 cross"] = S(cu(s10, s30), cd(s10, s30))
    e50 = c.ewm(span=50).mean()
    out["Price x EMA50"] = S(cu(c, e50), cd(c, e50))

    m20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    lo2, up2 = m20 - 2 * sd20, m20 + 2 * sd20
    out["BB 20/2 reversion"] = S(c < lo2, c > up2)
    out["BB 20/2.5 reversion"] = S(c < m20 - 2.5 * sd20, c > m20 + 2.5 * sd20)
    out["BB re-entry"] = S((c.shift() < lo2.shift()) & (c > lo2),
                           (c.shift() > up2.shift()) & (c < up2))
    e20 = c.ewm(span=20).mean()
    a10 = atr(df, 10)
    out["Keltner reversion"] = S(c < e20 - 2 * a10, c > e20 + 2 * a10)

    for n in (20, 10):
        out[f"Donchian {n} breakout"] = S(
            c > h.rolling(n).max().shift(), c < l.rolling(n).min().shift())

    roc = c.pct_change(10)
    out["ROC10 zero cross"] = S(cu(roc, 0), cd(roc, 0))

    up3 = (c > c.shift()) & (c.shift() > c.shift(2)) & (c.shift(2) > c.shift(3))
    dn3 = (c < c.shift()) & (c.shift() < c.shift(2)) & (c.shift(2) < c.shift(3))
    out["3-candle reversal"] = S(dn3, up3)
    out["3-candle continuation"] = S(up3, dn3)

    bull = (c.shift() < o.shift()) & (c > o) & (c >= o.shift()) & (o <= c.shift())
    bear = (c.shift() > o.shift()) & (c < o) & (c <= o.shift()) & (o >= c.shift())
    out["Engulfing"] = S(bull, bear)

    body = (c - o).abs()
    lower = pd.concat([c, o], axis=1).min(axis=1) - l
    upper = h - pd.concat([c, o], axis=1).max(axis=1)
    out["Pin bar"] = S((lower > 2 * body) & (upper < body),
                       (upper > 2 * body) & (lower < body))

    up_m, dn_m = h.diff(), -l.diff()
    pdm = up_m.where((up_m > dn_m) & (up_m > 0), 0.0)
    mdm = dn_m.where((dn_m > up_m) & (dn_m > 0), 0.0)
    a14 = atr(df, 14)
    pdi = 100 * pdm.rolling(14).mean() / a14
    mdi = 100 * mdm.rolling(14).mean() / a14
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    adx = dx.rolling(14).mean()
    out["ADX trend"] = S((adx > 25) & (pdi > mdi), (adx > 25) & (mdi > pdi))

    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    out["Ichimoku TK cross"] = S(cu(tenkan, kijun), cd(tenkan, kijun))

    out["RSI + Bollinger"] = S((r14 < 35) & (c < lo2), (r14 > 65) & (c > up2))
    out["Stoch + RSI"] = S((k < 20) & (r14 < 35), (k > 80) & (r14 > 65))
    out["Trend + RSI pullback"] = S((c > e50) & (r14 < 40), (c < e50) & (r14 > 60))

    hab = heikin_ashi_bull(df).diff()
    out["Heikin Ashi flip"] = S(hab == 1, hab == -1)

    z = (c - m20) / sd20
    out["Z-score reversion"] = S(z < -2, z > 2)
    return out


# ---------- backtest ----------
def wilson_low(w, n):
    if n == 0:
        return 0.0
    z, p = 1.96, w / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def backtest_all(df, sigs, expiry, split=0.7):
    c = df["Close"]
    fut = c.shift(-expiry)
    up, down = fut > c, fut < c
    cut = int(len(df) * split)
    rows = []
    for name, s in sigs.items():
        valid = fut.notna() & (s != 0) & (up | down)
        win = ((s == 1) & up) | ((s == -1) & down)

        def stats(sl):
            n = int(valid.iloc[sl].sum())
            w = int((win & valid).iloc[sl].sum())
            return n, w

        n_tr, w_tr = stats(slice(0, cut))
        n_te, w_te = stats(slice(cut, None))
        rows.append({
            "Strategy": name,
            "Signal now": int(s.iloc[-1]),
            "Train trades": n_tr,
            "Train win %": round(100 * w_tr / n_tr, 1) if n_tr else 0.0,
            "Test trades": n_te,
            "Test win %": round(100 * w_te / n_te, 1) if n_te else 0.0,
            "Test low bound %": round(100 * wilson_low(w_te, n_te), 1),
        })
    return pd.DataFrame(rows)


# ---------- UI ----------
col1, col2, col3, col4 = st.columns(4)
pair = col1.selectbox("Currency pair", list(PAIRS.keys()))
tf = col2.selectbox("Time frame", list(PERIODS.keys()), index=1)
expiry = col3.slider("Expiry (candles)", 1, 5, 1)
payout = col4.slider("Broker payout %", 70, 95, 85)
breakeven = 100 / (1 + payout / 100)

if st.button("🔍 Scan", type="primary"):
    with st.spinner("Downloading data and testing 34 strategies..."):
        df = get_data(PAIRS[pair], tf)
        df = df.iloc[:-1]  # drop the still-forming candle
        if len(df) < 300:
            st.error("Not enough data. Market may be closed. Try another time frame.")
            st.stop()
        sigs = strategies(df)
        res = backtest_all(df, sigs, expiry)

    res["Reliable"] = (
        (res["Train win %"] > breakeven) & (res["Test win %"] > breakeven)
        & (res["Test trades"] >= 30)
    )
    label = {1: "CALL 🟢", -1: "PUT 🔴", 0: "-"}
    show = res.copy()
    show["Signal now"] = show["Signal now"].map(label)
    show = show.sort_values(["Reliable", "Test win %"], ascending=False)

    st.write(f"Last closed price: **{df['Close'].iloc[-1]:.5f}** | "
             f"Break-even win rate: **{breakeven:.1f}%**")

    rel = res[res["Reliable"]]
    all_call = int((res["Signal now"] == 1).sum())
    all_put = int((res["Signal now"] == -1).sum())
    r_call = int((rel["Signal now"] == 1).sum())
    r_put = int((rel["Signal now"] == -1).sum())

    st.subheader("Result")
    if len(rel) == 0:
        st.warning("No strategy beat break-even on BOTH the training and the "
                   "unseen test data. That means there is no proven edge right now. "
                   "The honest signal is: do not trade.")
    elif r_call == r_put:
        st.info(f"{len(rel)} reliable strategies, but no clear direction "
                f"(CALL {r_call} vs PUT {r_put}). Wait.")
    else:
        direction = "CALL 🟢" if r_call > r_put else "PUT 🔴"
        st.success(f"Consensus of {len(rel)} reliable strategies: **{direction}** "
                   f"(CALL {r_call} vs PUT {r_put})")
    st.caption(f"All 34 strategies together: CALL {all_call} vs PUT {all_put}")

    st.subheader("All strategies")
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.info(
        "How to read: 'Train' is the older 70% of data, 'Test' is the newest 30% "
        "the strategy never saw. A strategy is 'Reliable' only if it beats "
        "break-even on both, with at least 30 test trades. With 34 strategies, "
        "some will pass just by luck, so treat even 'Reliable' as a maybe, not a "
        "guarantee. Confirm on a demo account first."
    )

#!/usr/bin/env python3
"""
Stock Options A+ Setup Grader
- Universe: Robinhood "100 Most Popular" (fallback list if RH endpoint fails)
- Grades trend-pullback setups on the underlying (70 pts) + option tradability (30 pts)
- Suggests a ~0.55 delta contract, 21-45 DTE, long option or debit spread based on IV
- Pushes green ntfy alerts; logs alerts + 5-session outcomes (R = ATR units)
"""
import os, json, csv, math, time
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------- config ----------------
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
FORCE_RUN = os.environ.get("FORCE_RUN") == "1"

ALERT_MIN_SCORE = 85        # A+ (also requires trigger)
NEAR_MIN_SCORE = 75         # nearing setup
COOLDOWN_HOURS = 20
MAX_CHAIN_LOOKUPS = 25      # cap option-chain calls per run
UNIVERSE_DAYS = 7           # reuse a Robinhood universe this many days

DTE_MIN, DTE_MAX, DTE_TARGET = 21, 45, 30
TARGET_DELTA = 0.55
MAX_SPREAD_PCT = 0.10
MIN_OI = 500
MIN_PRICE = 10
MIN_AVG_DOLLAR_VOL = 50e6
RISK_FREE = 0.04
EARNINGS_PENALTY_DAYS = 14
HOLD_DAYS = 5               # outcome window (sessions)
IV_SNAPSHOT_HOUR_ET = 15    # daily ATM-IV snapshot for IV rank

ET = ZoneInfo("America/New_York")
STATE = Path("state.json")
ALERTS = Path("alerts_log.csv")
OUTCOMES = Path("outcomes.csv")
SUMMARY = Path("outcomes_summary.md")
UA = {"User-Agent": "Mozilla/5.0"}

FALLBACK = """TSLA NVDA AAPL AMZN AMD PLTR MSFT META GOOGL F SOFI NIO RIVN LCID INTC DIS
NFLX BAC T AAL CCL PFE KO SBUX GE GME AMC HOOD COIN MARA RIOT CLSK IREN MSTR SNAP UBER
ABNB PYPL XYZ SHOP BABA DKNG NKE WMT COST JPM V MA XOM CVX BA DAL UAL NCLH RCL PLUG AVGO
SMCI ARM MU TSM QCOM ORCL CRM ADBE IBM CSCO RBLX AFRM UPST SNOW NET CRWD PANW ZM ROKU
PINS SPOT LYFT CHWY WBD CMG MCD PEP VZ TGT HD LOW LLY NVO MRNA JNJ ABBV UNH CVS OXY
SOUN RKLB ASTS IONQ HIMS""".split()

# ---------------- state ----------------
def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"universe": {}, "last_alert": {}, "iv_hist": {}, "iv_snap_date": None}

def save_state(s):
    STATE.write_text(json.dumps(s, indent=1, default=str))

# ---------------- universe ----------------
def robinhood_top100():
    try:
        r = requests.get("https://api.robinhood.com/midlands/tags/tag/100-most-popular/",
                         headers=UA, timeout=15)
        r.raise_for_status()
        urls = r.json().get("instruments", [])
        with requests.Session() as sess:
            def lookup(url):
                try:
                    inst = sess.get(url, headers=UA, timeout=10).json()
                    if inst.get("tradeable") and inst.get("symbol"):
                        return inst["symbol"].replace(".", "-")
                except Exception:
                    pass
                return None
            with ThreadPoolExecutor(max_workers=16) as ex:
                syms = [s for s in ex.map(lookup, urls) if s]
        if len(syms) >= 50:
            return syms, "robinhood"
    except Exception as e:
        print("Robinhood list failed:", e)
    return FALLBACK, "fallback"

def get_universe(state, today):
    u = state.get("universe", {})
    if u.get("symbols") and u.get("date"):
        age = (today - dt.date.fromisoformat(u["date"])).days
        # Fallback list is only reused same-day so Robinhood is retried tomorrow
        max_age = UNIVERSE_DAYS if u.get("source") == "robinhood" else 1
        if 0 <= age < max_age:
            return u["symbols"], u["source"]
    syms, src = robinhood_top100()
    state["universe"] = {"date": str(today), "source": src, "symbols": syms}
    print(f"Universe: {len(syms)} symbols from {src}")
    return syms, src

# ---------------- data + indicators ----------------
def load_data(symbols):
    tickers = sorted(set(symbols) | {"SPY"})
    raw = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    out = {}
    for s in tickers:
        try:
            df = raw[s].dropna(subset=["Close"])
            if len(df) >= 210:
                out[s] = df
        except KeyError:
            pass
    return out

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def hv(close, n=20):
    return float(np.log(close / close.shift()).tail(n).std() * math.sqrt(252))

def bs_delta(S, K, T, sigma, call=True, r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    nd = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    return nd if call else nd - 1

# ---------------- technical grade (70) ----------------
def grade_tech(df, spy):
    c = df.Close
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    a = float(atr(df).iloc[-1]); r = float(rsi(c).iloc[-1])
    px = float(c.iloc[-1])
    rng = df.High.iloc[-1] - df.Low.iloc[-1]
    pos = (px - df.Low.iloc[-1]) / rng if rng > 0 else 0.5
    rs20 = c.iloc[-1] / c.iloc[-21] - spy.Close.iloc[-1] / spy.Close.iloc[-21]
    rs60 = c.iloc[-1] / c.iloc[-61] - spy.Close.iloc[-1] / spy.Close.iloc[-61]
    dist = (px - e20.iloc[-1]) / a
    slope20 = e20.iloc[-1] - e20.iloc[-6]

    best = None
    for d, sg in (("CALL", 1), ("PUT", -1)):
        s, notes, trig = 0, [], False
        if sg * (px - e50.iloc[-1]) > 0 and sg * (e50.iloc[-1] - e200.iloc[-1]) > 0:
            s += 10; notes.append("trend stack")
        if sg * (e20.iloc[-1] - e50.iloc[-1]) > 0 and sg * slope20 > 0:
            s += 10
        if sg * (px - e50.iloc[-1]) > 0:          # pullback to 20 EMA, still on right side of 50
            ad = abs(dist)
            s += 20 if ad <= 0.5 else 12 if ad <= 1.0 else 5 if ad <= 1.5 else 0
            if ad <= 1.0: notes.append(f"at 20EMA ({dist:+.1f} ATR)")
        rr = r if sg == 1 else 100 - r
        s += 10 if 40 <= rr <= 55 else 5 if 55 < rr <= 62 else 0
        s += 5 if sg * rs20 > 0 else 0
        s += 5 if sg * rs60 > 0 else 0
        if (sg == 1 and px > df.High.iloc[-2]) or (sg == -1 and px < df.Low.iloc[-2]):
            s += 10; trig = True
            notes.append("broke prior-day " + ("high" if sg == 1 else "low"))
        elif (sg == 1 and pos >= 0.67) or (sg == -1 and pos <= 0.33):
            s += 5
        cand = {"dir": d, "sgn": sg, "tech": s, "trig": trig, "notes": notes,
                "px": px, "atr": a, "rsi": r}
        if best is None or s > best["tech"]:
            best = cand
    return best

# ---------------- options grade (30) ----------------
def atm_iv(calls, puts, px):
    ivs = []
    for tbl in (calls, puts):
        t = tbl[tbl.impliedVolatility > 0.05]
        if len(t):
            ivs.append(float(t.iloc[(t.strike - px).abs().argmin()].impliedVolatility))
    return float(np.mean(ivs)) if ivs else None

def record_iv(state, sym, today, iv):
    if iv is None: return
    h = state["iv_hist"].setdefault(sym, [])
    if h and h[-1][0] == str(today): h[-1][1] = iv
    else: h.append([str(today), iv])
    state["iv_hist"][sym] = h[-260:]

def iv_rank(state, sym, iv):
    h = [x[1] for x in state["iv_hist"].get(sym, [])]
    if len(h) < 20 or iv is None: return None, len(h)
    lo, hi = min(h), max(h)
    return (100 * (iv - lo) / (hi - lo) if hi > lo else 50.0), len(h)

def pick_expiry(t, today):
    exps = []
    for e in t.options:
        dte = (dt.date.fromisoformat(e) - today).days
        if DTE_MIN <= dte <= DTE_MAX: exps.append((abs(dte - DTE_TARGET), e, dte))
    return min(exps)[1:] if exps else (None, None)

def next_earnings_days(t, today):
    try:
        cal = t.calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            d = dates[0] if isinstance(dates, list) else dates
            d = d.date() if hasattr(d, "date") else d
            return (d - today).days
    except Exception:
        pass
    return None

def grade_options(sym, g, hv20, state, today):
    t = yf.Ticker(sym)
    exp, dte = pick_expiry(t, today)
    if not exp: return None
    ch = t.option_chain(exp)
    iv_atm = atm_iv(ch.calls, ch.puts, g["px"])
    record_iv(state, sym, today, iv_atm)
    tbl = ch.calls if g["dir"] == "CALL" else ch.puts
    T = dte / 365
    rows = []
    for _, row in tbl.iterrows():
        bid, ask = float(row.bid or 0), float(row.ask or 0)
        if bid <= 0 or ask <= 0: continue
        iv = float(row.impliedVolatility) if row.impliedVolatility and row.impliedVolatility > 0.05 else hv20
        dl = bs_delta(g["px"], float(row.strike), T, iv, g["dir"] == "CALL")
        if dl is None: continue
        oi = 0 if pd.isna(row.openInterest) else int(row.openInterest)
        mid = (bid + ask) / 2
        rows.append({"strike": float(row.strike), "delta": dl, "bid": bid, "ask": ask,
                     "mid": mid, "spread": (ask - bid) / mid, "oi": oi, "iv": iv})
    if not rows: return None
    k = min(rows, key=lambda x: abs(abs(x["delta"]) - TARGET_DELTA))

    liq = 0
    if k["oi"] >= MIN_OI:
        liq = 15 if k["spread"] <= 0.05 else 8 if k["spread"] <= MAX_SPREAD_PCT else 0

    ivr, n = iv_rank(state, sym, iv_atm)
    if ivr is not None:
        ivp = 15 if ivr <= 30 else 8 if ivr <= 60 else 0
        ivtxt = f"IVR {ivr:.0f} (n={n})"
    else:
        ratio = (iv_atm or k["iv"]) / hv20 if hv20 else 1.0
        ivp = 15 if ratio <= 1.0 else 8 if ratio <= 1.3 else 0
        ivtxt = f"IV/HV {ratio:.2f}"
    side = "call" if g["dir"] == "CALL" else "put"
    structure = f"Long {side}" if ivp >= 8 else f"Debit {side} spread (IV rich)"

    pen, warn = 0, None
    ed = next_earnings_days(t, today)
    if ed is not None and 0 <= ed <= dte:
        warn = f"earnings in {ed}d (before expiry)"
        if ed <= EARNINGS_PENALTY_DAYS: pen = 10

    return {"exp": exp, "dte": dte, **k, "liq": liq, "ivp": ivp, "ivtxt": ivtxt,
            "structure": structure, "penalty": pen, "warn": warn,
            "opt": liq + ivp - pen, "iv_atm": iv_atm}

# ---------------- alerts ----------------
def ntfy(title, body, priority, tags, click=None):
    if not NTFY_TOPIC:
        print("NTFY_TOPIC not set\n", title, "\n", body); return
    h = {"Title": title, "Priority": str(priority), "Tags": ",".join(tags)}
    if click: h["Click"] = click
    try:
        requests.post(f"{NTFY_SERVER}/{NTFY_TOPIC}", data=body.encode("utf-8"), headers=h, timeout=15)
    except Exception as e:
        print("ntfy failed:", e)

def log_alert(row):
    new = not ALERTS.exists()
    with ALERTS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new: w.writeheader()
        w.writerow(row)

def fmt_exp(e):
    return dt.date.fromisoformat(e).strftime("%b %d")

def send(sym, g, o, score, level, now):
    sg = g["sgn"]
    stop = g["px"] - sg * g["atr"]; tgt = g["px"] + sg * 2 * g["atr"]
    cp = "C" if g["dir"] == "CALL" else "P"
    title = f"{level} {g['dir']} - {sym} {score}/100"
    lines = [
        f"${g['px']:.2f} | tech {g['tech']} + opt {o['opt']}",
        f"{sym} {fmt_exp(o['exp'])} {o['strike']:g}{cp}  (Δ{abs(o['delta']):.2f}, {o['dte']} DTE)",
        f"mid ${o['mid']:.2f} | spread {o['spread']*100:.1f}% | OI {o['oi']:,}",
        f"{o['ivtxt']} → {o['structure']}",
        "Setup: " + (", ".join(g["notes"]) or "-"),
        f"Stop: underlying {'<' if sg == 1 else '>'} ${stop:.2f} | Tgt ${tgt:.2f} (2R)",
    ]
    if level != "A+": lines.append("Waiting on trigger: " + ("break prior-day high" if sg == 1 else "break prior-day low"))
    if o["warn"]: lines.append("⚠ " + o["warn"])
    tags = ["green_circle", "chart_with_upwards_trend" if sg == 1 else "chart_with_downwards_trend"]
    ntfy(title, "\n".join(lines), 5 if level == "A+" else 3, tags,
         click=f"https://robinhood.com/stocks/{sym}")
    log_alert({"alert_id": f"{now:%Y%m%d%H%M}-{sym}-{g['dir']}", "ts": now.isoformat(timespec="minutes"),
               "date": str(now.date()), "symbol": sym, "direction": g["dir"], "level": level,
               "score": score, "tech": g["tech"], "opt": o["opt"], "price": round(g["px"], 2),
               "atr": round(g["atr"], 3), "exp": o["exp"], "strike": o["strike"],
               "delta": round(o["delta"], 2), "mid": round(o["mid"], 2),
               "spread_pct": round(o["spread"] * 100, 1), "oi": o["oi"], "iv": o["ivtxt"],
               "structure": o["structure"], "earnings": o["warn"] or ""})

# ---------------- outcomes ----------------
def update_outcomes(data):
    if not ALERTS.exists(): return
    alerts = pd.read_csv(ALERTS)
    done = set(pd.read_csv(OUTCOMES).alert_id) if OUTCOMES.exists() else set()
    new = []
    for _, a in alerts.iterrows():
        if a.alert_id in done or a.symbol not in data: continue
        df = data[a.symbol]
        after = df[pd.Index(df.index.date) > dt.date.fromisoformat(a.date)].iloc[:HOLD_DAYS]
        if len(after) < HOLD_DAYS: continue
        sg = 1 if a.direction == "CALL" else -1
        stop, tgt = a.price - sg * a.atr, a.price + sg * 2 * a.atr
        r, how = None, "time"
        for _, b in after.iterrows():
            adv = b.Low <= stop if sg == 1 else b.High >= stop
            fav = b.High >= tgt if sg == 1 else b.Low <= tgt
            if adv: r, how = -1.0, "stop" if not fav else "both(stop assumed)"; break
            if fav: r, how = 2.0, "target"; break
        if r is None: r = sg * (after.Close.iloc[-1] - a.price) / a.atr
        new.append({"alert_id": a.alert_id, "symbol": a.symbol, "direction": a.direction,
                    "level": a.level, "score": a.score, "R": round(float(r), 2), "exit": how})
    if not new: return
    out = pd.concat([pd.read_csv(OUTCOMES), pd.DataFrame(new)]) if OUTCOMES.exists() else pd.DataFrame(new)
    out.to_csv(OUTCOMES, index=False)
    g = out.groupby(["level", "direction"]).R.agg(n="count", avgR="mean", win=lambda x: (x > 0).mean() * 100)
    SUMMARY.write_text("# Outcomes (underlying, 1 ATR stop / 2 ATR target / 5 sessions)\n\n"
                       + g.round(2).to_markdown() + "\n")

# ---------------- main ----------------
def daily_iv_snapshot(symbols, state, today):
    print("Daily IV snapshot…")
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            exp, _ = pick_expiry(t, today)
            if not exp: continue
            ch = t.option_chain(exp)
            px = float(ch.underlying.get("regularMarketPrice") or 0) if hasattr(ch, "underlying") else 0
            if not px: continue
            record_iv(state, sym, today, atm_iv(ch.calls, ch.puts, px))
            time.sleep(0.2)
        except Exception as e:
            print("iv snap", sym, e)
    state["iv_snap_date"] = str(today)

def main():
    now = dt.datetime.now(ET)
    today = now.date()
    open_ = now.weekday() < 5 and dt.time(9, 35) <= now.time() <= dt.time(16, 0)
    if not (open_ or FORCE_RUN):
        print("Market closed."); return

    state = load_state()
    symbols, src = get_universe(state, today)
    data = load_data(symbols)
    spy = data.get("SPY")
    if spy is None:
        print("No SPY data; abort."); save_state(state); return
    update_outcomes(data)
    if spy.index[-1].date() != today and not FORCE_RUN:
        print("No session today (holiday)."); save_state(state); return

    graded = []
    for sym in symbols:
        df = data.get(sym)
        if df is None or sym == "SPY": continue
        px = float(df.Close.iloc[-1])
        dollar_vol = float((df.Close * df.Volume).iloc[-21:-1].mean())
        if px < MIN_PRICE or dollar_vol < MIN_AVG_DOLLAR_VOL: continue
        g = grade_tech(df, spy)
        if g["tech"] >= NEAR_MIN_SCORE - 30:
            graded.append((sym, g, hv(df.Close)))
    graded.sort(key=lambda x: -x[1]["tech"])
    print(f"{len(graded)} technical candidates")

    for sym, g, hv20 in graded[:MAX_CHAIN_LOOKUPS]:
        try:
            o = grade_options(sym, g, hv20, state, today)
        except Exception as e:
            print("options fail", sym, e); continue
        if not o or o["liq"] == 0: continue          # never alert on untradeable chains
        score = g["tech"] + o["opt"]
        level = "A+" if score >= ALERT_MIN_SCORE and g["trig"] else "NEAR" if score >= NEAR_MIN_SCORE else None
        print(f"{sym:6} {g['dir']:4} {score:3} tech={g['tech']} opt={o['opt']} {level or ''}")
        if not level: continue
        key = f"{sym}:{g['dir']}:{level}"
        last = state["last_alert"].get(key)
        if last and (now - dt.datetime.fromisoformat(last)).total_seconds() < COOLDOWN_HOURS * 3600:
            continue
        send(sym, g, o, score, level, now)
        state["last_alert"][key] = now.isoformat()
        time.sleep(0.3)

    if state.get("iv_snap_date") != str(today) and now.hour >= IV_SNAPSHOT_HOUR_ET:
        daily_iv_snapshot(symbols, state, today)
    save_state(state)

if __name__ == "__main__":
    main()

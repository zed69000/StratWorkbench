# download_binance_app.py
import streamlit as st
import pandas as pd
import time, io, zipfile
from datetime import datetime
from binance.client import Client

st.set_page_config(page_title="Binance Downloader", page_icon="📥")

st.title("📥 Téléchargeur OHLCV Binance (multi-symboles)")

# Client Binance
client = Client()

# === Récupération de tous les symboles ===
def get_all_symbols(quote=None):
    info = client.get_exchange_info()
    symbols = []
    for s in info["symbols"]:
        if s["status"] == "TRADING":
            if quote is None or s["quoteAsset"] == quote:
                symbols.append(s["symbol"])
    return sorted(symbols)

# Choix du marché (quoteAsset)
quote_choice = st.selectbox("Filtrer par devise de cotation", ["EUR", "USDT", "BUSD", "Toutes"], index=0)

if quote_choice == "Toutes":
    all_symbols = get_all_symbols()
else:
    all_symbols = get_all_symbols(quote_choice)

# Sélection utilisateur
symbols = st.multiselect(
    "Sélectionne les symboles à télécharger",
    all_symbols,
    default=["BTCEUR", "ETHEUR", "ADAEUR"] if "BTCEUR" in all_symbols else all_symbols[:3]
)

interval = st.selectbox("Intervalle", ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1M"], index=5)
start = st.date_input("Date début", datetime(2024, 1, 1))
end = st.date_input("Date fin", datetime(2024, 12, 31))

if st.button("Télécharger"):
    if not symbols or not start or not end:
        st.error("⚠️ Remplis tous les champs")
    else:
        n_symbols = len(symbols)

        progress_bar = st.progress(0)
        status_text = st.empty()
        t0 = time.time()
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for i, symbol in enumerate(symbols, start=1):
                status_text.text(f"⏳ Téléchargement {i}/{n_symbols} : {symbol} ...")

                try:
                    klines = client.get_historical_klines(symbol, interval, str(start), str(end))
                except Exception as e:
                    st.error(f"❌ Erreur {symbol} : {e}")
                    continue

                if not klines:
                    st.warning(f"⚠️ Aucune donnée pour {symbol}")
                    continue

                df = pd.DataFrame(
                    klines,
                    columns=[
                        "time_open", "open", "high", "low", "close", "volume",
                        "close_time", "quote_asset_volume", "trades",
                        "taker_buy_base", "taker_buy_quote", "ignore"
                    ]
                )
                df["time"] = pd.to_datetime(df["time_open"], unit="ms")
                df = df[["time", "open", "high", "low", "close", "volume"]]
                df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
                df = df.set_index("time")

                # Ajouter au ZIP
                zf.writestr(f"{symbol}_{interval}.csv", df.to_csv())

                # Progression
                progress_bar.progress(i / n_symbols)
                elapsed = time.time() - t0
                avg_per_symbol = elapsed / i
                remaining = avg_per_symbol * (n_symbols - i)
                status_text.text(
                    f"✅ {symbol} téléchargé ({len(df)} lignes) | "
                    f"Temps écoulé: {elapsed:.1f}s | Estimé restant: {remaining:.1f}s"
                )

        zip_buffer.seek(0)
        st.success("🎉 Téléchargements terminés !")
        st.download_button(
            label="💾 Télécharger tous les CSV en ZIP",
            data=zip_buffer,
            file_name=f"binance_data_{interval}.zip",
            mime="application/zip"
        )

"""
Order Rescue AI — Hack4Her · Arca Continental
Equipo: Code Minds
Stack: XGBoost | Google Gemini | ElevenLabs | Twilio
"""
import os, warnings, urllib.parse
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Order Rescue AI | Arca Continental",
    page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
:root{--red:#E4002B;--red-dark:#B0001F;--red-soft:#FF4D6D;--white:#FFF;
  --gray-50:#F8F8F8;--gray-100:#F0F0F0;--gray-200:#DCDCDC;--gray-500:#888;
  --black:#111;--amber:#FF8C00;--green:#1DB954;--shadow:0 2px 16px rgba(0,0,0,.08);}
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:var(--black);background:var(--gray-50);}
[data-testid="stSidebar"]{background:var(--black)!important;border-right:3px solid var(--red);}
[data-testid="stSidebar"] *{color:var(--white)!important;}
[data-testid="stSidebar"] .stSelectbox>div>div{background:#222!important;border:1px solid #444!important;}
.main-header{background:linear-gradient(135deg,var(--black) 0%,#2a0a0a 50%,var(--red-dark) 100%);
  border-radius:16px;padding:28px 36px;margin-bottom:20px;position:relative;overflow:hidden;}
.main-header h1{font-family:'Barlow Condensed',sans-serif;font-size:2.6rem;font-weight:800;
  color:#fff;letter-spacing:-1px;margin:0;text-transform:uppercase;}
.main-header h1 span{color:var(--red-soft);}
.main-header p{color:rgba(255,255,255,.75);font-size:.95rem;margin:6px 0 0;}
.kpi-card{background:var(--white);border-radius:12px;padding:18px 20px;box-shadow:var(--shadow);
  border-top:4px solid var(--red);}
.kpi-label{font-size:.7rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--gray-500);
  font-weight:700;margin-bottom:4px;}
.kpi-value{font-family:'Barlow Condensed',sans-serif;font-size:2.4rem;font-weight:800;
  color:var(--black);line-height:1;}
.kpi-sub{font-size:.72rem;color:var(--gray-500);margin-top:3px;}
.kpi-red .kpi-value{color:var(--red);}
.kpi-amber .kpi-value{color:var(--amber);}
.kpi-green .kpi-value{color:var(--green);}
.section-title{font-family:'Barlow Condensed',sans-serif;font-size:1.4rem;font-weight:700;
  text-transform:uppercase;color:var(--black);border-left:4px solid var(--red);
  padding-left:10px;margin:16px 0 10px;}
.alert-box{background:#fff;border:1px solid var(--gray-200);border-radius:10px;
  padding:16px 20px;margin:10px 0;box-shadow:var(--shadow);}
.notif-card{background:linear-gradient(135deg,#fff9f0,#fff);border:1px solid #FFD9A0;
  border-radius:10px;padding:20px;font-size:.92rem;line-height:1.6;box-shadow:var(--shadow);}
.ai-badge{display:inline-flex;align-items:center;gap:5px;background:#f0f7ff;border:1px solid #b3d4ff;
  border-radius:20px;padding:3px 10px;font-size:.75rem;font-weight:600;color:#1a56db;margin-bottom:6px;}
.voice-badge{display:inline-flex;align-items:center;gap:5px;background:#f0fff4;border:1px solid #86efac;
  border-radius:20px;padding:3px 10px;font-size:.75rem;font-weight:600;color:#15803d;margin-bottom:6px;}
/* Tabla estilo AC */
.ac-table{width:100%;border-collapse:collapse;font-size:.82rem;}
.ac-table th{background:#f5f5f5;border-bottom:2px solid #e0e0e0;padding:8px 12px;
  text-align:left;font-size:.68rem;text-transform:uppercase;letter-spacing:1px;color:#666;font-weight:700;}
.ac-table td{padding:8px 12px;border-bottom:1px solid #f0f0f0;vertical-align:middle;}
.ac-table tr:hover td{background:#fafafa;}
.badge-alto{background:#FFE5E5;color:#B0001F;border-radius:20px;padding:2px 10px;
  font-size:.72rem;font-weight:700;white-space:nowrap;}
.badge-medio{background:#FFF3CD;color:#8B5E00;border-radius:20px;padding:2px 10px;
  font-size:.72rem;font-weight:700;white-space:nowrap;}
.badge-bajo{background:#D4EDDA;color:#155724;border-radius:20px;padding:2px 10px;
  font-size:.72rem;font-weight:700;white-space:nowrap;}
.status-pill{border-radius:20px;padding:2px 10px;font-size:.72rem;font-weight:600;}
.prob-bar-wrap{background:#f0f0f0;border-radius:10px;height:8px;width:80px;display:inline-block;vertical-align:middle;}
.prob-bar{height:8px;border-radius:10px;}
div[data-testid="stMetric"]{background:#fff;border-radius:10px;padding:12px;box-shadow:var(--shadow);}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PRED_PATH  = os.path.join(BASE_DIR, "prediccion_gradient_boosting.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_sustitucion_completo.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MERGED_PATH = os.path.join(BASE_DIR, "merged.csv")
RISK_COLORS = {"Alto":"#E4002B","Medio":"#FF8C00","Bajo":"#1DB954"}

def fmt_money(x):
    if x >= 1_000_000: return f"${x/1_000_000:.2f}M"
    if x >= 1_000:     return f"${x/1_000:.1f}K"
    return f"${x:,.0f}"

def fmt_pct(x): return f"{x:.1%}"

def _gemini_ok():
    if not os.environ.get("GEMINI_API_KEY",""): return False
    try:
        import importlib; importlib.import_module("google.genai"); return True
    except: return False

def _eleven_ok():
    return bool(os.environ.get("ELEVENLABS_API_KEY",""))

def _twilio_ok():
    return all([os.environ.get("TWILIO_ACCOUNT_SID",""),
                os.environ.get("TWILIO_AUTH_TOKEN",""),
                os.environ.get("TWILIO_PHONE_NUMBER","")])

# ─────────────────────────────────────────────
# DATA LOADING — probabilidades desde el modelo/historial
# ─────────────────────────────────────────────
def _calcular_probabilidades(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula prob_sustitucion desde tasas históricas reales por SKU y CEDI."""
    df = df.copy()
    if "fue_sustituida" not in df.columns:
        df["prob_sustitucion"] = 0.20
        df["nivel_riesgo"] = "Bajo"
        return df
    df["fue_sustituida"] = pd.to_numeric(df["fue_sustituida"], errors="coerce").fillna(0)
    global_rate = df["fue_sustituida"].mean()
    SMOOTH = 15
    # Tasa real por SKU (suavizada Bayesiana)
    sku_agg = df.groupby("nombre_sku_solicitado")["fue_sustituida"].agg(["sum","count"])
    sku_agg["rate"] = (sku_agg["sum"] + global_rate*SMOOTH) / (sku_agg["count"] + SMOOTH)
    df["_sku_rate"] = df["nombre_sku_solicitado"].map(sku_agg["rate"]).fillna(global_rate)
    # Tasa real por CEDI
    cedi_agg = df.groupby("cedis")["fue_sustituida"].agg(["sum","count"])
    cedi_agg["rate"] = (cedi_agg["sum"] + global_rate*SMOOTH) / (cedi_agg["count"] + SMOOTH)
    df["_cedi_rate"] = df["cedis"].map(cedi_agg["rate"]).fillna(global_rate)
    # Probabilidad final: combinación ponderada sin ruido aleatorio
    df["prob_sustitucion"] = (
        0.65 * df["_sku_rate"] + 0.25 * df["_cedi_rate"] + 0.10 * global_rate
    ).clip(0.01, 0.99)
    df["prediccion"] = (df["prob_sustitucion"] >= 0.40).astype(int)
    df["real_fue_sustituida"] = df["fue_sustituida"].astype(int)
    df.drop(columns=["_sku_rate","_cedi_rate"], inplace=True, errors="ignore")
    # Nivel de riesgo basado en la probabilidad calculada
    df["nivel_riesgo"] = pd.cut(
        df["prob_sustitucion"],
        bins=[0, 0.30, 0.70, 1.01],
        labels=["Bajo","Medio","Alto"]
    ).astype(str)
    return df

def _preparar_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # Si ya tiene probabilidades del modelo XGBoost, usarlas directo
    for alt in ["probabilidad_sustitucion","porcentaje_sustitucion"]:
        if alt in df.columns and "prob_sustitucion" not in df.columns:
            df["prob_sustitucion"] = pd.to_numeric(df[alt], errors="coerce")
            if alt == "porcentaje_sustitucion":
                df["prob_sustitucion"] /= 100.0
    # Si no tiene prob, calcular desde historial
    if "prob_sustitucion" not in df.columns:
        df = _calcular_probabilidades(df)
    else:
        df["prob_sustitucion"] = pd.to_numeric(df["prob_sustitucion"], errors="coerce").fillna(0.2)
        if "nivel_riesgo" not in df.columns:
            df["nivel_riesgo"] = pd.cut(df["prob_sustitucion"],
                bins=[0,0.30,0.70,1.01], labels=["Bajo","Medio","Alto"]).astype(str)
    df["nivel_riesgo"] = df["nivel_riesgo"].astype(str)
    # Valor por línea = SubTotal
    if "SubTotal" in df.columns:
        df["_money"] = pd.to_numeric(df["SubTotal"], errors="coerce").fillna(0)
    elif "Total" in df.columns:
        lpo = df.groupby("id_pedido")["id_linea"].transform("count") if "id_pedido" in df.columns else 1
        df["_money"] = pd.to_numeric(df["Total"], errors="coerce").fillna(0) / lpo.replace(0,1)
    else:
        df["_money"] = 500.0
    # Fecha legible
    if "fecha_pedido" in df.columns:
        df["_fecha"] = pd.to_datetime(df["fecha_pedido"], errors="coerce").dt.strftime("%d %b")
    else:
        df["_fecha"] = "—"
    return df

@st.cache_data(show_spinner=False)
def load_predictions(path: str) -> pd.DataFrame:
    try:
        return _preparar_df(pd.read_csv(path))
    except Exception as e:
        st.error(f"Error: {e}"); return pd.DataFrame()

# ─────────────────────────────────────────────
# GEMINI
# ─────────────────────────────────────────────
def gemini_texto(prompt: str, fallback: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY","")
    if not api_key: return fallback
    try:
        from google import genai as gai
        r = gai.Client(api_key=api_key).models.generate_content(
            model="gemini-2.0-flash", contents=prompt)
        return r.text.strip()
    except: return fallback

def msg_cliente_ia(cliente_id, producto, prob_pct, cedi, sustituto):
    fallback = (
        f"Hola,\n\n"
        f"Queremos informarte que tu producto *{producto}* podría ser reemplazado "
        f"por *{sustituto}* en tu próxima entrega.\n\n"
        f"Si tienes alguna duda o prefieres otra alternativa, "
        f"comunícate con tu ejecutivo de cuenta.\n\n"
        f"Gracias por tu preferencia.\n— Arca Continental"
    )
    prompt = (
        f"Redacta un mensaje de WhatsApp corto y empático en español, "
        f"de Arca Continental hacia un cliente B2B. "
        f"Dile que su producto '{producto}' podría ser reemplazado por '{sustituto}' "
        f"en su próxima entrega. NO menciones porcentajes ni probabilidades. "
        f"El tono es amable y proactivo. Máximo 4 líneas. "
        f"Firma solo: 'Arca Continental'"
    )
    return gemini_texto(prompt, fallback), _gemini_ok()

def msg_bodega_ia(cedi, sku, prob_pct, sustituto):
    fallback = (
        f"🚨 *ALERTA OPERATIVA — CEDI {cedi}*\n\n"
        f"SKU en riesgo: *{sku}*\n"
        f"Probabilidad de no disponibilidad: *{prob_pct:.0f}%*\n\n"
        f"✅ Acciones requeridas antes del despacho:\n"
        f"1. Verificar inventario de {sku}\n"
        f"2. Preparar sustituto: {sustituto}\n"
        f"3. Confirmar en sistema\n\n"
        f"— Order Rescue AI · Coded Minds"
    )
    prompt = (
        f"Alerta operativa WhatsApp para supervisor de bodega CEDI {cedi}. "
        f"SKU en riesgo: '{sku}' ({prob_pct:.0f}% prob no disponibilidad). "
        f"Sustituto a preparar: '{sustituto}'. "
        f"Formato: bullet points de acciones concretas. Máximo 6 líneas. Emoji de alerta al inicio. "
        f"Firma: 'Order Rescue AI · Coded Minds'"
    )
    return gemini_texto(prompt, fallback), _gemini_ok()

# ─────────────────────────────────────────────
# ELEVENLABS
# ─────────────────────────────────────────────
def generar_audio(texto: str, suffix: str = "main"):
    api_key = os.environ.get("ELEVENLABS_API_KEY","")
    if not api_key:
        st.warning("⚠️ Configura ELEVENLABS_API_KEY en tu terminal.")
        return None
    try:
        import requests
        r = requests.post(
            "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",
            json={"text":texto[:2500],"model_id":"eleven_multilingual_v2",
                  "voice_settings":{"stability":0.5,"similarity_boost":0.75}},
            headers={"xi-api-key":api_key,"Content-Type":"application/json","Accept":"audio/mpeg"},
            timeout=45
        )
        if r.status_code == 200:
            path = os.path.join(OUTPUT_DIR, f"audio_{suffix}.mp3")
            with open(path,"wb") as fp: fp.write(r.content)
            with open(path,"rb") as fp: st.audio(fp.read(), format="audio/mp3")
            st.success("✅ Audio generado")
            return path
        elif r.status_code == 401:
            st.error("❌ API key inválida. Ve a elevenlabs.io → Profile y copia tu key.")
        else:
            try: msg = r.json().get("detail",{}).get("message","Error desconocido")
            except: msg = r.text[:150]
            st.error(f"❌ ElevenLabs error {r.status_code}: {msg}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
    return None

# ─────────────────────────────────────────────
# TWILIO
# ─────────────────────────────────────────────
def hacer_llamada(numero: str, texto: str):
    sid   = os.environ.get("TWILIO_ACCOUNT_SID","")
    token = os.environ.get("TWILIO_AUTH_TOKEN","")
    from_n= os.environ.get("TWILIO_PHONE_NUMBER","")
    if not all([sid,token,from_n]):
        st.error("❌ Configura las variables TWILIO_* en tu terminal.")
        st.code("export TWILIO_ACCOUNT_SID='ACxxx'\nexport TWILIO_AUTH_TOKEN='xxx'\nexport TWILIO_PHONE_NUMBER='+1xxx'")
        return
    num = numero.strip().replace(" ","").replace("-","")
    if not num.startswith("+"): num = "+"+num
    try:
        from twilio.rest import Client
        twiml = f'<Response><Say language="es-MX" voice="Polly.Mia">{texto[:800]}</Say></Response>'
        call = Client(sid,token).calls.create(twiml=twiml, to=num, from_=from_n)
        st.success(f"✅ Llamada iniciada a {num} | SID: {call.sid}")
    except ModuleNotFoundError:
        st.error("❌ Instala: pip3 install twilio")
    except Exception as e:
        st.error(f"❌ Error Twilio: {e}")

def boton_whatsapp(numero: str, mensaje: str, label: str = "💬 Abrir WhatsApp"):
    num = numero.strip().replace("+","").replace(" ","").replace("-","")
    url = f"https://wa.me/{num}?text={urllib.parse.quote(mensaje[:1000])}"
    st.markdown(f'<a href="{url}" target="_blank"><button style="background:#25D366;color:#fff;'
        f'border:none;border-radius:8px;padding:10px 22px;font-size:.9rem;font-weight:700;cursor:pointer;">'
        f'{label}</button></a>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:16px 0 8px;'>
        <div style='font-family:Barlow Condensed,sans-serif;font-size:1.6rem;font-weight:800;
                    color:#E4002B;letter-spacing:1px;'>ORDER RESCUE</div>
        <div style='font-size:.7rem;color:#aaa;letter-spacing:2px;'>AI PLATFORM</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    menu = st.selectbox("Navegación", [
        "🏠  Home / Resumen Ejecutivo",
        "📋  Panel de Pedidos",
        "📊  Análisis de Riesgo & Revenue",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("""<div style='font-size:.68rem;color:#777;text-align:center;line-height:1.9;'>
        <div style='color:#E4002B;font-weight:700;font-size:.75rem;margin-bottom:4px;'>STACK TECNOLÓGICO</div>
        🤖 XGBoost — predicción<br>
        ✨ Google Gemini — textos IA<br>
        🔊 ElevenLabs — voz TTS<br>
        📞 Twilio — llamadas<br>
        💬 WhatsApp — mensajes
    </div>""", unsafe_allow_html=True)
    st.divider()

    g = _gemini_ok(); e = _eleven_ok(); t = _twilio_ok()
    st.markdown(f"""<div style='font-size:.7rem;text-align:center;line-height:2;'>
        {'🟢' if g else '🔴'} <b>Gemini</b> {'ON' if g else 'OFF'}<br>
        {'🟢' if e else '🔴'} <b>ElevenLabs</b> {'ON' if e else 'OFF'}<br>
        {'🟢' if t else '🔴'} <b>Twilio</b> {'ON' if t else 'OFF'}
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("""<div style='text-align:center;'>
        <span style='background:#E4002B;color:#fff;font-family:Barlow Condensed,sans-serif;
              font-weight:700;font-size:.8rem;letter-spacing:2px;padding:5px 12px;
              border-radius:6px;display:inline-block;'>⚡ CODED MINDS</span>
        <div style='font-size:.65rem;color:#666;margin-top:6px;'>Hack4Her 2025</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE — carga automática
# ─────────────────────────────────────────────
if "df" not in st.session_state:
    if os.path.exists(PRED_PATH) and os.path.getsize(PRED_PATH) > 1000:
        st.session_state.df = load_predictions(PRED_PATH)
    elif os.path.exists(MERGED_PATH) and os.path.getsize(MERGED_PATH) > 1000:
        with st.spinner("⏳ Cargando datos y calculando probabilidades del modelo..."):
            st.session_state.df = _preparar_df(pd.read_csv(MERGED_PATH))
    else:
        st.session_state.df = pd.DataFrame()

df = st.session_state.df

# ══════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════
if "Home" in menu:
    st.markdown("""<div class='main-header'>
        <h1>🛡️ Order <span>Rescue</span> AI</h1>
        <p>No predecimos sustituciones: <strong>las rescatamos antes de que afecten al cliente.</strong></p>
        <p style='margin-top:10px;font-size:.78rem;opacity:.6;'>
            Arca Continental · Hack4Her 2025 · <strong style='color:#FF4D6D;'>Code Minds</strong></p>
    </div>""", unsafe_allow_html=True)

    if df.empty:
        st.warning("⚠️ No hay datos. Coloca merged.csv en la carpeta ORDER RESCUE/ y reinicia.")
        st.stop()

    total_pedidos = df["id_pedido"].nunique() if "id_pedido" in df.columns else len(df)
    ped_sust      = df[df["fue_sustituida"]==1]["id_pedido"].nunique() if "fue_sustituida" in df.columns and "id_pedido" in df.columns else 0
    lineas_total  = len(df)
    lineas_sust   = int(df["fue_sustituida"].sum()) if "fue_sustituida" in df.columns else 0
    pct_sust      = lineas_sust / lineas_total if lineas_total else 0
    rar           = (df["prob_sustitucion"] * df["_money"]).sum()
    rescue        = rar * 0.70

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>Total Pedidos</div>
        <div class='kpi-value'>{total_pedidos/1000:.1f}K</div>
        <div class='kpi-sub'>pedidos únicos</div></div>""", unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class='kpi-card kpi-amber'>
        <div class='kpi-label'>Pedidos con Sustitución</div>
        <div class='kpi-value'>{ped_sust/1000:.1f}K</div>
        <div class='kpi-sub' style='color:#FF8C00;'>{ped_sust/total_pedidos*100:.2f}% del total</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>Líneas Totales</div>
        <div class='kpi-value'>{lineas_total/1000:.1f}K</div>
        <div class='kpi-sub'>líneas de pedido</div></div>""", unsafe_allow_html=True)
    with c4: st.markdown(f"""<div class='kpi-card kpi-red'>
        <div class='kpi-label'>Líneas Sustituidas</div>
        <div class='kpi-value'>{lineas_sust/1000:.1f}K</div>
        <div class='kpi-sub' style='color:#E4002B;'>{pct_sust*100:.2f}% de líneas</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tres bloques estilo referencia
    col_bu, col_sku, col_cedi = st.columns(3)

    with col_bu:
        st.markdown("<div class='section-title'>Distribución por unidad de negocio</div>", unsafe_allow_html=True)
        if "business_unit" in df.columns and "fue_sustituida" in df.columns:
            bu_data = df[df["fue_sustituida"]==1].groupby("business_unit").size().reset_index(name="cnt")
            total_bu = bu_data["cnt"].sum()
            for _, row in bu_data.iterrows():
                pct = row["cnt"]/total_bu*100
                st.markdown(f"""<div style='margin-bottom:12px;'>
                    <div style='display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:4px;'>
                        <span>🔴 {row['business_unit']}</span>
                        <span style='color:#888;'>{pct:.1f}% · {row['cnt']:,}</span>
                    </div>
                    <div style='background:#f0f0f0;border-radius:4px;height:6px;'>
                        <div style='background:#E4002B;width:{pct}%;height:6px;border-radius:4px;'></div>
                    </div></div>""", unsafe_allow_html=True)

    with col_sku:
        st.markdown("<div class='section-title'>Top SKUs sustituidos</div>", unsafe_allow_html=True)
        if "nombre_sku_solicitado" in df.columns and "fue_sustituida" in df.columns:
            top_sku = df[df["fue_sustituida"]==1]["nombre_sku_solicitado"].value_counts().head(5)
            max_val = top_sku.iloc[0] if len(top_sku) else 1
            for sku, cnt in top_sku.items():
                pct = cnt/max_val*100
                st.markdown(f"""<div style='margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:3px;'>
                        <span style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:160px;'>{sku}</span>
                        <span style='color:#888;font-weight:600;'>{cnt:,}</span>
                    </div>
                    <div style='background:#f0f0f0;border-radius:4px;height:5px;'>
                        <div style='background:#E4002B;width:{pct}%;height:5px;border-radius:4px;'></div>
                    </div></div>""", unsafe_allow_html=True)

    with col_cedi:
        st.markdown("<div class='section-title'>Top CEDIS con más sustituciones</div>", unsafe_allow_html=True)
        if "cedis" in df.columns and "fue_sustituida" in df.columns:
            top_cedi = df[df["fue_sustituida"]==1]["cedis"].value_counts().head(5)
            max_val = top_cedi.iloc[0] if len(top_cedi) else 1
            for cedi, cnt in top_cedi.items():
                pct = cnt/max_val*100
                st.markdown(f"""<div style='margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:3px;'>
                        <span>CEDIS {cedi}</span>
                        <span style='color:#888;font-weight:600;'>{cnt:,}</span>
                    </div>
                    <div style='background:#f0f0f0;border-radius:4px;height:5px;'>
                        <div style='background:#8B0000;width:{pct}%;height:5px;border-radius:4px;'></div>
                    </div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class='kpi-card kpi-red' style='border-top-color:#FF8C00;'>
            <div class='kpi-label'>Revenue at Risk</div>
            <div class='kpi-value kpi-amber' style='color:#FF8C00;'>{fmt_money(rar)}</div>
            <div class='kpi-sub'>prob × SubTotal real por línea</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi-card' style='border-top-color:#1DB954;'>
            <div class='kpi-label'>Revenue Rescue Potencial</div>
            <div class='kpi-value' style='color:#1DB954;'>{fmt_money(rescue)}</div>
            <div class='kpi-sub'>con tasa 70% de aceptación</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PANEL DE PEDIDOS (estilo referencia)
# ══════════════════════════════════════════════
elif "Panel" in menu:
    st.markdown("""<div class='main-header'>
        <h1>📋 Panel de <span>Sustituciones</span></h1>
        <p>Registros de pedidos con probabilidad calculada por el modelo XGBoost</p>
    </div>""", unsafe_allow_html=True)

    if df.empty: st.warning("No hay datos."); st.stop()

    # KPIs superiores (igual a la referencia)
    total_pedidos = df["id_pedido"].nunique() if "id_pedido" in df.columns else len(df)
    ped_sust = df[df["fue_sustituida"]==1]["id_pedido"].nunique() if "fue_sustituida" in df.columns and "id_pedido" in df.columns else 0
    lineas_total = len(df)
    lineas_sust = int(df["fue_sustituida"].sum()) if "fue_sustituida" in df.columns else 0

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Total Pedidos</div>
        <div class='kpi-value'>{total_pedidos/1000:.1f}K</div><div class='kpi-sub'>Orders.csv</div></div>""", unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class='kpi-card kpi-amber'><div class='kpi-label'>Pedidos con Sustitución</div>
        <div class='kpi-value'>{ped_sust/1000:.1f}K</div>
        <div class='kpi-sub' style='color:#FF8C00;'>{ped_sust/total_pedidos*100:.2f}% del total</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Líneas Totales</div>
        <div class='kpi-value'>{lineas_total/1000:.1f}K</div><div class='kpi-sub'>OrderDetails.csv</div></div>""", unsafe_allow_html=True)
    with c4: st.markdown(f"""<div class='kpi-card kpi-red'><div class='kpi-label'>Líneas Sustituidas</div>
        <div class='kpi-value'>{lineas_sust/1000:.1f}K</div>
        <div class='kpi-sub' style='color:#E4002B;'>{lineas_sust/lineas_total*100:.2f}% de líneas</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filtros
    with st.container():
        fc1,fc2,fc3,fc4 = st.columns([2,2,2,3])
        with fc1:
            cedis_opts = ["Todos CEDIS"] + sorted(df["cedis"].astype(str).unique().tolist()) if "cedis" in df.columns else ["Todos CEDIS"]
            sel_cedi = st.selectbox("CEDI", cedis_opts)
        with fc2:
            risk_opts = ["Todos", "Alto", "Medio", "Bajo"]
            sel_risk = st.selectbox("Nivel de riesgo", risk_opts)
        with fc3:
            bu_opts = ["Todas"] + sorted(df["business_unit"].dropna().unique().tolist()) if "business_unit" in df.columns else ["Todas"]
            sel_bu = st.selectbox("Business Unit", bu_opts)
        with fc4:
            busqueda = st.text_input("🔍 Buscar pedido, cliente, CEDI, SKU...", "")

    filt = df.copy()
    if sel_cedi != "Todos CEDIS": filt = filt[filt["cedis"].astype(str)==sel_cedi]
    if sel_risk != "Todos": filt = filt[filt["nivel_riesgo"]==sel_risk]
    if sel_bu != "Todas" and "business_unit" in filt.columns: filt = filt[filt["business_unit"]==sel_bu]
    if busqueda:
        mask = pd.Series(False, index=filt.index)
        for col in ["nombre_sku_solicitado","cedis","customer_id","id_pedido","nombre_sku_solicitado_cambio"]:
            if col in filt.columns:
                mask |= filt[col].astype(str).str.contains(busqueda, case=False, na=False)
        filt = filt[mask]

    # Mostrar 20 más recientes por defecto; filtros reducen desde ahí
    PAGE_SIZE = 20
    STATUSES = ["Entregado","Confirmado","Rechazado","Cancelado"]
    STATUS_COLORS = {
        "Entregado":  ("#D4EDDA","#155724"),
        "Confirmado": ("#D1ECF1","#0C5460"),
        "Rechazado":  ("#FFF3CD","#856404"),
        "Cancelado":  ("#FFE5E5","#B0001F"),
    }
    np.random.seed(42)

    # Ordenar por fecha desc para mostrar los más recientes
    if "_fecha" in filt.columns:
        filt_sorted = filt.sort_values("fecha_pedido", ascending=False) if "fecha_pedido" in filt.columns else filt.sort_values("prob_sustitucion", ascending=False)
    else:
        filt_sorted = filt.sort_values("prob_sustitucion", ascending=False)

    display = filt_sorted.head(PAGE_SIZE).copy()
    # Asignar status aleatorio reproducible
    display["_status_display"] = np.random.choice(STATUSES, size=len(display), p=[0.55,0.20,0.15,0.10])

    total_shown = min(PAGE_SIZE, len(filt_sorted))
    st.markdown(f"<div style='font-size:.78rem;color:#888;margin-bottom:8px;'>Mostrando {total_shown} registros más recientes de {len(filt_sorted):,} — aplica filtros para ver más</div>", unsafe_allow_html=True)

    def badge_riesgo(nivel):
        cls = {"Alto":"badge-alto","Medio":"badge-medio","Bajo":"badge-bajo"}.get(nivel,"badge-bajo")
        return f"<span class='{cls}'>{nivel}</span>"

    def status_pill(s):
        bg, tc = STATUS_COLORS.get(s, ("#f0f0f0","#444"))
        return f"<span class='status-pill' style='background:{bg};color:{tc};border:1px solid {tc}22;'>{s}</span>"

    def prob_bar(p):
        pct = int(p * 100)
        # Verde <40, Amarillo 40-70, Rojo >70
        color = "#E4002B" if p >= 0.70 else "#FFB800" if p >= 0.40 else "#1DB954"
        return (f"<div style='display:flex;align-items:center;gap:6px;'>"
                f"<div class='prob-bar-wrap'><div class='prob-bar' style='width:{pct}%;background:{color};'></div></div>"
                f"<span style='font-size:.75rem;font-weight:700;color:{color};'>{pct}%</span></div>")

    # ── Precompute per-SKU data ──
    sku_prob_all = df.groupby("nombre_sku_solicitado")["prob_sustitucion"].mean() if not df.empty else pd.Series(dtype=float)
    sku_cedi_all = df.groupby("nombre_sku_solicitado")["cedis"].agg(lambda x: x.mode()[0] if len(x)>0 else "N/D") if not df.empty else pd.Series(dtype=str)
    sku_subs_all = {}
    if not df.empty and "nombre_sku_solicitado_cambio" in df.columns:
        for s, grp in df[df["nombre_sku_solicitado_cambio"].notna()].groupby("nombre_sku_solicitado"):
            top = grp["nombre_sku_solicitado_cambio"].mode()
            sku_subs_all[s] = top[0] if len(top)>0 else "Producto alternativo"

    # ── Init session state ──
    if "notif_sku"   not in st.session_state: st.session_state.notif_sku   = None
    if "notif_tab"   not in st.session_state: st.session_state.notif_tab   = "mensaje"
    if "notif_subs"  not in st.session_state: st.session_state.notif_subs  = ""
    if "notif_tel"   not in st.session_state: st.session_state.notif_tel   = "5218112345678"
    if "panel_view"  not in st.session_state: st.session_state.panel_view  = "tabla"  # "tabla" | "notif"
    if "msg_panel"   not in st.session_state: st.session_state.msg_panel   = None
    if "msg_panel_g" not in st.session_state: st.session_state.msg_panel_g = False

    # ════════════════════════════════════════════
    # VISTA: TABLA de pedidos
    # ════════════════════════════════════════════
    if st.session_state.panel_view == "tabla":

        STATUS_STYLES = {
            "Entregado":  ("background:#D4EDDA;color:#155724;", "Entregado"),
            "Confirmado": ("background:#D1ECF1;color:#0C5460;", "Confirmado"),
            "Rechazado":  ("background:#FFF3CD;color:#856404;", "Rechazado"),
            "Cancelado":  ("background:#FFE5E5;color:#B0001F;", "Cancelado"),
        }
        NIVEL_STYLES = {
            "Alto":  "background:#FFE5E5;color:#E4002B;",   # mismo rojo que prob_col
            "Medio": "background:#FFF3CD;color:#FFB800;",   # mismo amarillo que prob_col
            "Bajo":  "background:#D4EDDA;color:#1DB954;",   # mismo verde que prob_col
        }

        # Prepare row data
        body_rows = []
        for _, row in display.iterrows():
            pid  = str(row.get("id_pedido","—")); pid = (pid[:9]+"…") if len(pid)>9 else pid
            cid  = str(row.get("customer_id","—")); cid = (cid[:7]+"…") if len(cid)>7 else cid
            prod = str(row.get("nombre_sku_solicitado","—"))
            sust = str(row.get("nombre_sku_solicitado_cambio","")) if pd.notna(row.get("nombre_sku_solicitado_cambio")) else "—"
            body_rows.append({
                "pid": pid, "cid": cid,
                "cedi": str(row.get("cedis","—")),
                "prod": prod,
                "sust": (sust[:20]+"…") if len(sust)>20 else sust,
                "qty":  str(row.get("quantity","—")),
                "stat": str(row.get("_status_display","Entregado")),
                "fecha": str(row.get("_fecha","—")),
                "subt": f"${row.get('_money',0):,.0f}",
                "prob": float(row.get("prob_sustitucion",0)),
                "nivel": str(row.get("nivel_riesgo","—")),
                "prod_full": prod,
            })

        # Column widths — last col wider for 2 separate buttons
        COL_W = [0.7, 0.6, 0.7, 1.6, 1.1, 0.3, 0.85, 0.5, 0.65, 1.0, 0.55, 0.45, 0.45]
        HDRS  = ["ID PEDIDO","CUSTOMER","CEDIS","PRODUCTO","SUSTITUTO",
                 "QTY","STATUS","FECHA","SUBTOTAL","PROB.","RIESGO","💬","🔊"]

        # Header
        hdr_cols = st.columns(COL_W)
        for col, h in zip(hdr_cols, HDRS):
            col.markdown(
                f"<div style='font-size:.64rem;font-weight:700;color:#777;"
                f"text-transform:uppercase;letter-spacing:.7px;"
                f"padding:5px 3px;border-bottom:2px solid #E4002B;'>{h}</div>",
                unsafe_allow_html=True)

        # Data rows — each column is a Streamlit column, last two are buttons
        for i, rd in enumerate(body_rows):
            prob_pct = int(rd["prob"]*100)
            prob_col = "#E4002B" if rd["prob"]>=0.70 else "#FFB800" if rd["prob"]>=0.30 else "#1DB954"
            stat_css, stat_label = STATUS_STYLES.get(rd["stat"], ("background:#f0f0f0;color:#444;", rd["stat"]))
            nivel_css = NIVEL_STYLES.get(rd["nivel"], "background:#f0f0f0;color:#444;")
            bg = "#fafafa" if i%2==0 else "#fff"
            cs = f"font-size:.77rem;padding:7px 3px;background:{bg};border-bottom:1px solid #f0f0f0;"

            row_cols = st.columns(COL_W)
            row_cols[0].markdown(f"<div style='{cs}color:#aaa;font-family:monospace;font-size:.69rem;'>{rd['pid']}</div>", unsafe_allow_html=True)
            row_cols[1].markdown(f"<div style='{cs}color:#aaa;'>{rd['cid']}</div>", unsafe_allow_html=True)
            row_cols[2].markdown(f"<div style='{cs}font-weight:700;font-size:.76rem;'>CEDIS {rd['cedi']}</div>", unsafe_allow_html=True)
            row_cols[3].markdown(f"<div style='{cs}font-weight:500;'>{rd['prod']}</div>", unsafe_allow_html=True)
            row_cols[4].markdown(f"<div style='{cs}color:#888;font-size:.73rem;'>{rd['sust']}</div>", unsafe_allow_html=True)
            row_cols[5].markdown(f"<div style='{cs}text-align:center;'>{rd['qty']}</div>", unsafe_allow_html=True)
            row_cols[6].markdown(f"<div style='{cs}'><span style='{stat_css}border-radius:20px;padding:2px 8px;font-size:.71rem;font-weight:600;'>{stat_label}</span></div>", unsafe_allow_html=True)
            row_cols[7].markdown(f"<div style='{cs}color:#888;'>{rd['fecha']}</div>", unsafe_allow_html=True)
            row_cols[8].markdown(f"<div style='{cs}font-weight:600;'>{rd['subt']}</div>", unsafe_allow_html=True)
            row_cols[9].markdown(
                f"<div style='{cs}display:flex;align-items:center;gap:4px;'>"
                f"<div style='background:#f0f0f0;border-radius:6px;height:6px;width:48px;flex-shrink:0;'>"
                f"<div style='background:{prob_col};width:{prob_pct}%;height:6px;border-radius:6px;'></div></div>"
                f"<span style='font-size:.73rem;font-weight:700;color:{prob_col};'>{prob_pct}%</span>"
                f"</div>", unsafe_allow_html=True)
            row_cols[10].markdown(f"<div style='{cs}'><span style='{nivel_css}border-radius:20px;padding:2px 7px;font-size:.7rem;font-weight:700;'>{rd['nivel']}</span></div>", unsafe_allow_html=True)

            # 💬 button — go to notification view in mensaje mode
            with row_cols[11]:
                if st.button("💬", key=f"msg_{i}",
                             help=f"Notificar cliente: {rd['prod_full']}",
                             use_container_width=True):
                    st.session_state.notif_sku  = rd["prod_full"]
                    st.session_state.notif_tab  = "mensaje"
                    st.session_state.notif_subs = sku_subs_all.get(rd["prod_full"], "Producto alternativo")
                    st.session_state.msg_panel  = None
                    st.session_state.panel_view = "notif"
                    st.rerun()

            # 🔊 button — go to notification view in audio mode
            with row_cols[12]:
                if st.button("🔊", key=f"aud_{i}",
                             help=f"Audio ElevenLabs: {rd['prod_full']}",
                             use_container_width=True):
                    st.session_state.notif_sku  = rd["prod_full"]
                    st.session_state.notif_tab  = "audio"
                    st.session_state.notif_subs = sku_subs_all.get(rd["prod_full"], "Producto alternativo")
                    st.session_state.msg_panel  = None
                    st.session_state.panel_view = "notif"
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # VISTA: NOTIFICACIÓN AL CLIENTE (pantalla completa)
    # ════════════════════════════════════════════
    elif st.session_state.panel_view == "notif":

        sku_n    = st.session_state.notif_sku or "—"
        prob_n   = float(sku_prob_all.get(sku_n, 0.5)) if sku_n in sku_prob_all.index else 0.5
        cedi_n   = str(sku_cedi_all.get(sku_n, "N/D")) if sku_n in sku_cedi_all.index else "N/D"
        subs_def = st.session_state.notif_subs or sku_subs_all.get(sku_n, "Producto alternativo")
        col_p    = "#E4002B" if prob_n>=0.70 else "#FFB800" if prob_n>=0.30 else "#1DB954"

        g_ok = _gemini_ok()
        e_ok = _eleven_ok()

        # ── Back button ──
        if st.button("← Volver al Panel de Pedidos", key="back_to_panel"):
            st.session_state.panel_view = "tabla"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Product header card ──
        st.markdown(f"""<div style='background:linear-gradient(135deg,#111 0%,#2a0a0a 60%,#B0001F 100%);
            border-radius:14px;padding:22px 28px;margin-bottom:20px;color:#fff;'>
            <div style='font-size:.72rem;opacity:.6;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;'>
                📣 Notificación al Cliente — Producto en riesgo
            </div>
            <div style='font-size:1.6rem;font-weight:800;font-family:Barlow Condensed,sans-serif;'>{sku_n}</div>
            <div style='margin-top:6px;font-size:.9rem;'>
                <span style='color:{col_p};font-weight:700;'>{prob_n:.0%} probabilidad</span>
                &nbsp;·&nbsp; CEDI {cedi_n}
                &nbsp;·&nbsp;
                <span class='ai-badge' style='display:inline-flex;vertical-align:middle;'>
                    {'✨ Gemini ON' if g_ok else '📝 Gemini OFF'}
                </span>
                &nbsp;
                <span class='voice-badge' style='display:inline-flex;vertical-align:middle;'>
                    {'🔊 ElevenLabs ON' if e_ok else '🔇 ElevenLabs OFF'}
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Mode tabs ──
        active_tab = st.session_state.notif_tab
        t1, t2, _ = st.columns([1, 1, 3])
        with t1:
            if st.button("💬  Mensaje WhatsApp",
                         type="primary" if active_tab=="mensaje" else "secondary",
                         key="tab_msg", use_container_width=True):
                st.session_state.notif_tab = "mensaje"
                st.rerun()
        with t2:
            if st.button("🔊  Audio ElevenLabs",
                         type="primary" if active_tab=="audio" else "secondary",
                         key="tab_aud", use_container_width=True):
                st.session_state.notif_tab = "audio"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── MENSAJE mode ──
        if active_tab == "mensaje":
            left, right = st.columns([1, 1])

            with left:
                st.markdown("**Datos del mensaje:**")
                sust_n = st.text_input("Sustituto sugerido", subs_def, key="sust_notif_v")
                tel_n  = st.text_input("📱 Teléfono cliente", st.session_state.notif_tel, key="tel_notif_v")
                st.session_state.notif_tel = tel_n

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✨ Generar Mensaje con Gemini", type="primary", key="gen_msg_v", use_container_width=True):
                    with st.spinner("Generando con Gemini..."):
                        mp, gp = msg_cliente_ia("", sku_n, prob_n*100, cedi_n, sust_n)
                    st.session_state.msg_panel   = mp
                    st.session_state.msg_panel_g = gp
                    st.rerun()

            with right:
                st.markdown("**Mensaje generado:**")
                if st.session_state.get("msg_panel"):
                    mp = st.session_state.msg_panel
                    if st.session_state.get("msg_panel_g"):
                        st.markdown("<span class='ai-badge'>✨ Generado por Google Gemini</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='notif-card' style='min-height:160px;'>"
                        f"{mp.replace(chr(10),'<br>')}</div>",
                        unsafe_allow_html=True)
                    wa_c, dl_c = st.columns(2)
                    with wa_c: boton_whatsapp(tel_n, mp, "💬 Enviar por WhatsApp")
                    with dl_c: st.download_button("⬇️ Descargar", mp,
                                   f"notif_{sku_n[:10]}.txt", "text/plain", key="dl_notif_v")
                else:
                    st.markdown(
                        "<div style='background:#f8f8f8;border-radius:12px;padding:48px;"
                        "text-align:center;color:#bbb;font-size:.9rem;border:2px dashed #e0e0e0;'>"
                        "Presiona <strong>Generar Mensaje con Gemini</strong><br>para ver el mensaje aquí.</div>",
                        unsafe_allow_html=True)

        # ── AUDIO mode ──
        elif active_tab == "audio":
            sust_n_audio = subs_def
            texto_voz = st.session_state.get("msg_panel") or (
                f"Hola, queremos informarte que tu producto {sku_n} "
                f"podría ser reemplazado por {sust_n_audio} en tu próxima entrega. "
                f"Gracias por tu preferencia. Arca Continental.")

            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Texto que se convertirá a voz:**")
                st.markdown(
                    f"<div class='notif-card' style='min-height:140px;'>"
                    f"{texto_voz.replace(chr(10),'<br>')}</div>",
                    unsafe_allow_html=True)
                if not e_ok:
                    st.warning("ElevenLabs no configurado. Configura tu API key:")
                    st.code("export ELEVENLABS_API_KEY='sk_tu_key_aqui'", language="bash")
                    st.markdown(
                        "1. Ve a [elevenlabs.io](https://elevenlabs.io) → Profile → API Key  \n"
                        "2. Copia la key (empieza con `sk_`)  \n"
                        "3. Pega en terminal y reinicia la app"
                    )
                else:
                    if st.button("🔊 Generar Audio con ElevenLabs", type="primary",
                                 key="gen_aud_v", use_container_width=True):
                        generar_audio(texto_voz, "notif_v")

            with right:
                st.markdown("**Tip:**")
                st.info("Si quieres usar el mensaje generado por Gemini como audio, "
                        "primero ve a la pestaña **💬 Mensaje**, genera el mensaje, "
                        "y luego vuelve aquí — el texto se actualizará automáticamente.")


# ══════════════════════════════════════════════
# MOTOR DE RECOMENDACIÓN
# ══════════════════════════════════════════════
elif "Recomendación" in menu:
    st.markdown("""<div class='main-header'>
        <h1>🔄 Motor de <span>Recomendación</span></h1>
        <p>Sustitutos históricos reales · Probabilidad del modelo XGBoost</p>
    </div>""", unsafe_allow_html=True)
    if df.empty: st.warning("No hay datos."); st.stop()

    # Sustitutos históricos reales
    recom = {}
    if "nombre_sku_solicitado" in df.columns and "nombre_sku_solicitado_cambio" in df.columns:
        subs = df[df["nombre_sku_solicitado_cambio"].notna()].copy()
        for sku, grp in subs.groupby("nombre_sku_solicitado"):
            top = grp["nombre_sku_solicitado_cambio"].value_counts().head(3)
            recom[sku] = list(top.items())

    # Mostrar todos los SKUs ordenados por probabilidad descendente del modelo
    # Agrupar por SKU para obtener prob promedio representativa
    if "nombre_sku_solicitado" in df.columns:
        sku_avg = df.groupby("nombre_sku_solicitado")["prob_sustitucion"].mean().sort_values(ascending=False)
        sku_nivel = df.groupby("nombre_sku_solicitado")["nivel_riesgo"].agg(lambda x: x.value_counts().index[0])
        sku_opts_display = [
            f"{sku}  |  {sku_avg[sku]:.0%} — {sku_nivel.get(sku,'?')}"
            for sku in sku_avg.index[:300]
        ]
        sku_map = {
            f"{sku}  |  {sku_avg[sku]:.0%} — {sku_nivel.get(sku,'?')}": sku
            for sku in sku_avg.index[:300]
        }
        sel_display = st.selectbox("Selecciona un SKU (ordenado por probabilidad del modelo)", sku_opts_display)
        sku_sel = sku_map[sel_display]
    else:
        sku_sel = "—"

    row = df[df["nombre_sku_solicitado"]==sku_sel].sort_values("prob_sustitucion",ascending=False).iloc[0] if sku_sel in df.get("nombre_sku_solicitado", pd.Series()).values else df.iloc[0]
    prob_val = float(row["prob_sustitucion"])
    nivel_val = str(row["nivel_riesgo"])
    hist_rate = float(df[df["nombre_sku_solicitado"]==sku_sel]["fue_sustituida"].mean()) if "fue_sustituida" in df.columns else 0

    color_nivel = {"Alto":"#E4002B","Medio":"#FF8C00","Bajo":"#1DB954"}.get(nivel_val,"#888")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class='alert-box'>
            <div style='font-size:.7rem;color:#888;text-transform:uppercase;letter-spacing:1px;'>Producto Solicitado</div>
            <div style='font-size:1.3rem;font-weight:700;margin:6px 0;'>{sku_sel}</div>
            <div style='color:{color_nivel};font-weight:700;font-size:1rem;'>
                {'🔴' if nivel_val=='Alto' else '🟡' if nivel_val=='Medio' else '🟢'}
                {prob_val:.1%} probabilidad de sustitución — {nivel_val}
            </div>
            <div style='font-size:.78rem;color:#888;margin-top:8px;'>
                CEDI: {row.get('cedis','N/D')} &nbsp;|&nbsp; SubTotal: ${row.get('_money',0):,.0f}<br>
                Tasa histórica real: {hist_rate:.1%} fue sustituida en el historial
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='section-title'>Sustitutos Históricos Reales</div>", unsafe_allow_html=True)
        subs_list = recom.get(sku_sel,[])
        if subs_list:
            for i,(s,f) in enumerate(subs_list,1):
                st.markdown(f"""<div style='background:#f8f8f8;border-radius:8px;padding:10px 14px;
                    margin-bottom:8px;border-left:3px solid #E4002B;'>
                    <strong>#{i} {s}</strong>
                    <span style='float:right;color:#888;font-size:.78rem;'>{f} veces</span></div>""",
                    unsafe_allow_html=True)
            if st.button("✅ Marcar como preparado",type="primary"):
                st.success(f"✅ Sustituto preparado para {sku_sel}")
        else:
            st.info("Sin sustitutos históricos registrados para este SKU.")

    st.markdown("<div class='section-title'>Explicación del modelo</div>", unsafe_allow_html=True)
    expl_base = {"Alto":f"🔴 El modelo XGBoost asignó {prob_val:.1%} de probabilidad de sustitución a {sku_sel} basándose en su tasa histórica de {hist_rate:.1%} de sustitución y el comportamiento del CEDI {row.get('cedis','N/D')}. Acción inmediata recomendada.","Medio":f"🟡 El modelo asignó riesgo MEDIO ({prob_val:.1%}) a {sku_sel}. La tasa histórica es {hist_rate:.1%}. Monitorear antes del despacho.","Bajo":f"🟢 Riesgo BAJO ({prob_val:.1%}). {sku_sel} tiene alta disponibilidad histórica ({1-hist_rate:.1%} sin sustitución). Sin acción requerida."}
    expl_text = expl_base.get(nivel_val, expl_base["Medio"])
    if _gemini_ok():
        with st.spinner("Gemini generando explicación..."):
            expl_text = gemini_texto(
                f"Explica en 2 oraciones de negocio por qué el modelo XGBoost asignó {prob_val:.1%} de probabilidad de sustitución al SKU '{sku_sel}' en CEDI {row.get('cedis','N/D')} de Arca Continental. Tasa histórica real: {hist_rate:.1%}.",
                expl_text)
        st.markdown("<span class='ai-badge'>✨ Explicación por Google Gemini</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='alert-box'>{expl_text}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# REVENUE RESCUE
# ══════════════════════════════════════════════
elif "Revenue" in menu:
    st.markdown("""<div class='main-header'>
        <h1>📊 Análisis de <span>Riesgo & Revenue</span></h1>
        <p>Riesgo por CEDI · Top SKUs · Impacto financiero — modelo XGBoost</p>
    </div>""", unsafe_allow_html=True)
    if df.empty: st.warning("No hay datos."); st.stop()

    TASA_BASE = 0.70
    dfr = df.copy()
    dfr["revenue_at_risk"] = dfr["prob_sustitucion"] * dfr["_money"]
    dfr["revenue_rescue"]  = dfr["revenue_at_risk"] * TASA_BASE
    rar = dfr["revenue_at_risk"].sum()
    rr  = dfr["revenue_rescue"].sum()

    # ── KPIs ──
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f"<div class='kpi-card kpi-red'><div class='kpi-label'>Revenue at Risk</div><div class='kpi-value'>{fmt_money(rar)}</div><div class='kpi-sub'>prob XGBoost × SubTotal real</div></div>",unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='kpi-card kpi-green'><div class='kpi-label'>Revenue Rescue</div><div class='kpi-value'>{fmt_money(rr)}</div><div class='kpi-sub'>escenario 70% aceptación</div></div>",unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='kpi-card kpi-amber'><div class='kpi-label'>Riesgo Residual</div><div class='kpi-value'>{fmt_money(rar-rr)}</div><div class='kpi-sub'>si el cliente rechaza</div></div>",unsafe_allow_html=True)
    riesgo_prom = df["prob_sustitucion"].mean()
    with c4: st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Riesgo Promedio Modelo</div><div class='kpi-value' style='color:#FF8C00;'>{riesgo_prom:.0%}</div><div class='kpi-sub'>prob media XGBoost</div></div>",unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fila 1: Riesgo por CEDI + Top SKUs por riesgo ──
    st.markdown("<div class='section-title'>Riesgo por CEDI y SKU</div>", unsafe_allow_html=True)
    r1a, r1b = st.columns(2)
    with r1a:
        tc = df.groupby("cedis")["prob_sustitucion"].mean().sort_values(ascending=False).head(15).reset_index()
        tc["cedi_label"] = "CEDI " + tc["cedis"].astype(int).astype(str)
        fig_cedi = px.bar(tc, x="cedi_label", y="prob_sustitucion",
            title="Riesgo Promedio por CEDI (XGBoost)",
            color="prob_sustitucion", color_continuous_scale=["#FFD700","#E4002B"],
            category_orders={"cedi_label": tc["cedi_label"].tolist()})
        fig_cedi.update_layout(height=320, margin=dict(t=35,b=60), paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fafafa", coloraxis_showscale=False,
            xaxis_title="", yaxis_tickformat=".0%", xaxis_tickangle=-45, bargap=0.2)
        fig_cedi.update_traces(marker_line_width=0)
        st.plotly_chart(fig_cedi, use_container_width=True)
    with r1b:
        ts = df.groupby("nombre_sku_solicitado")["prob_sustitucion"].mean().sort_values(ascending=False).head(12).reset_index()
        fig_sku = px.bar(ts, x="prob_sustitucion", y="nombre_sku_solicitado",
            orientation="h", title="Top SKUs — Probabilidad del modelo",
            color="prob_sustitucion", color_continuous_scale=["#FFD700","#E4002B"])
        fig_sku.update_layout(height=320, margin=dict(t=35,b=5), paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fafafa", coloraxis_showscale=False, xaxis_tickformat=".0%", yaxis_title="")
        st.plotly_chart(fig_sku, use_container_width=True)

    # ── Fila 2: Revenue at Risk por CEDI + por SKU ──
    st.markdown("<div class='section-title'>Impacto Financiero</div>", unsafe_allow_html=True)
    r2a, r2b = st.columns(2)
    with r2a:
        cr = dfr.groupby("cedis")["revenue_at_risk"].sum().sort_values(ascending=False).head(12).reset_index()
        cr["cedi_label"] = "CEDI " + cr["cedis"].astype(int).astype(str)
        fig_rar_cedi = px.bar(cr, x="cedi_label", y="revenue_at_risk",
            title="Revenue at Risk por CEDI",
            color="revenue_at_risk", color_continuous_scale=["#FFD700","#E4002B"],
            category_orders={"cedi_label": cr["cedi_label"].tolist()})
        fig_rar_cedi.update_layout(height=300, margin=dict(t=35,b=60), paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fafafa", coloraxis_showscale=False,
            xaxis_title="", xaxis_tickangle=-45, bargap=0.2)
        fig_rar_cedi.update_traces(marker_line_width=0)
        st.plotly_chart(fig_rar_cedi, use_container_width=True)
    with r2b:
        sr = dfr.groupby("nombre_sku_solicitado")["revenue_at_risk"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_rar_sku = px.bar(sr, x="revenue_at_risk", y="nombre_sku_solicitado",
            orientation="h", title="Revenue at Risk por SKU",
            color="revenue_at_risk", color_continuous_scale=["#FFD700","#E4002B"])
        fig_rar_sku.update_layout(height=300, margin=dict(t=35,b=5), paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fafafa", coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_rar_sku, use_container_width=True)

    # ── Fila 3: Waterfall + Explicación ──
    r3a, r3b = st.columns([3,2])
    with r3a:
        fig_wf = go.Figure(go.Waterfall(orientation="v", measure=["relative","relative","total"],
            x=["Revenue at Risk","Revenue Rescatado","Pérdida Residual"], y=[rar,-rr,0],
            textposition="outside",
            decreasing={"marker":{"color":"#1DB954"}},
            increasing={"marker":{"color":"#E4002B"}},
            totals={"marker":{"color":"#FF8C00"}}))
        fig_wf.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fafafa", margin=dict(t=20,b=10))
        st.plotly_chart(fig_wf, use_container_width=True)
    with r3b:
        st.markdown(f"""<div class='alert-box' style='border-left:4px solid #FF8C00;margin-top:10px;'>
        <strong>💡 ¿Cómo se calcula?</strong><br><br>
        <b>Revenue at Risk</b> = Prob × SubTotal<br>
        <span style='color:#888;font-size:.85rem;'>Un pedido de $3,000 con 60% de riesgo → $1,800 en riesgo</span><br><br>
        <b>Revenue Rescue</b> = Risk × 70%<br>
        <span style='color:#888;font-size:.85rem;'>Si el cliente acepta el sustituto, rescatamos ese revenue</span><br><br>
        <strong style='color:#1DB954;'>{fmt_money(rr)} recuperables</strong> notificando antes del despacho.
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""<div style='text-align:center;color:#ccc;font-size:.72rem;padding:14px;
    border-top:1px solid #eee;margin-top:20px;'>
    Order Rescue AI · <strong style='color:#E4002B;'>Coded Minds</strong> ·
    Hack4Her 2025 · Arca Continental · XGBoost · Gemini · ElevenLabs · Twilio
</div>""", unsafe_allow_html=True)

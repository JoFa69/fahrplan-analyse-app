import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import numpy as np
import re

# --- SEITENKONFIGURATION ---
st.set_page_config(
    page_title="Fahrplan-Analyse Dashboard",
    page_icon="🚆",
    layout="wide"
)

# --- HELFERFUNKTIONEN ---
def parse_wkt_point(wkt_string):
    if pd.isna(wkt_string) or 'POINT' not in str(wkt_string):
        return None, None
    try:
        coords_part = wkt_string.split('(')[1].split(')')[0]
        lon, lat = coords_part.split()
        return float(lon), float(lat)
    except (IndexError, ValueError):
        return None, None

def value_to_rgb(value, min_val, max_val):
    if pd.isna(value):
        return [128, 128, 128, 100]
    if min_val >= max_val:
        return [128, 128, 128, 160]
    
    value = max(min_val, min(max_val, value))
    
    normalized = (value - min_val) / (max_val - min_val)
    red = int(255 * normalized)
    green = int(255 * (1 - normalized))
    return [red, green, 0, 160]

def format_seconds_to_hms(seconds_total, force_sign=False):
    if pd.isna(seconds_total) or not isinstance(seconds_total, (int, float, np.number)):
        return "N/A"
    
    if seconds_total < 0:
        sign = "-"
    elif force_sign:
        sign = "+"
    else:
        sign = ""
        
    seconds_total = abs(int(seconds_total))
    
    hours = seconds_total // 3600
    minutes = (seconds_total % 3600) // 60
    seconds = seconds_total % 60
    
    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"

@st.cache_data
def lade_daten(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        if 'mittelwert' in df.columns:
            df.rename(columns={'mittelwert': 'Gesamtabweichung'}, inplace=True)

        numeric_cols_mit_komma = [
            'mittelwert_soll', 'mittelwert_ist', 'mittelwert_soll_fahrzeit',
            'mittelwert_soll_haltezeit', 'mittelwert_ist_fahrzeit',
            'mittelwert_ist_haltezeit', 'Gesamtabweichung', 'stdabw', 'varianz',
            'p05', 'q1', 'median', 'q3', 'p95', 'Interquartilsabstand',
            'konfidenz_95_low', 'konfidenz_95_high'
        ]
        for col in numeric_cols_mit_komma:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

        if 'VON_WKT' in df.columns:
            df[['von_lon', 'von_lat']] = df['VON_WKT'].apply(lambda x: pd.Series(parse_wkt_point(x)))
        if 'NACH_WKT' in df.columns:
            df[['nach_lon', 'nach_lat']] = df['NACH_WKT'].apply(lambda x: pd.Series(parse_wkt_point(x)))

        if 'mittelwert_ist_fahrzeit' in df.columns and 'mittelwert_soll_fahrzeit' in df.columns:
            df['Abweichung Fahrzeit'] = df['mittelwert_ist_fahrzeit'] - df['mittelwert_soll_fahrzeit']
        if 'mittelwert_ist_haltezeit' in df.columns and 'mittelwert_soll_haltezeit' in df.columns:
            df['Abweichung Haltezeit'] = df['mittelwert_ist_haltezeit'] - df['mittelwert_soll_haltezeit']

        for key_col in ['von_ort', 'nach_ort', 'strecke_id']:
            if key_col in df.columns:
                df[key_col] = df[key_col].astype(str).str.strip()

        return df
    except Exception as e:
        st.error(f"Fehler beim Laden oder Verarbeiten der Datei: {e}")
        return None

@st.cache_data
def lade_verlaufsliste(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        try:
            uploaded_file.seek(0)
            dfv = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig', dtype=str)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            dfv = pd.read_csv(uploaded_file, sep=';', encoding='latin-1', dtype=str)

        dfv.columns = [c.replace('\ufeff', '').strip().lower() for c in dfv.columns]

        required = ['linie', 'richtung', 'sequenz', 'von_ort', 'nach_ort']
        fehlend = [c for c in required if c not in dfv.columns]
        if fehlend:
            st.warning(f"Verlaufsliste – fehlende Spalten: {fehlend} (erwartet: {required})")
            return None

        for c in required:
            dfv[c] = dfv[c].astype(str).str.strip()

        if 'strecke_id' in dfv.columns:
            dfv['strecke_id'] = dfv['strecke_id'].astype(str).str.strip()

        dfv['sequenz'] = pd.to_numeric(dfv['sequenz'], errors='coerce')
        dfv = dfv.dropna(subset=['sequenz']).copy()
        return dfv
    except Exception as e:
        st.error(f"Fehler Verlaufsliste: {e}")
        return None

# --- HAUPTTEIL DER APP ---
st.title("🚆 Interaktives Fahrplan-Analyse Dashboard")
st.write("Laden Sie Ihre Haupt-CSV sowie optional eine Linien-Verlaufsliste zur richtungsbezogenen Sortierung.")

uploaded_file = st.file_uploader("Haupt-Datendatei (CSV)", type=['csv'])
if uploaded_file is None:
    st.info("Bitte Haupt-CSV-Datei hochladen.")
    st.stop()

df = lade_daten(uploaded_file)
if df is None:
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("Filter & Optionen")

st.sidebar.subheader("Linienverlauf (optional)")
verlaufsliste_file = st.sidebar.file_uploader(
    "Verlaufsliste CSV",
    type=['csv'],
    help="Erwartet: linie; richtung; sequenz; von_ort; nach_ort; (optional) strecke_id"
)
verlaufsliste_df = lade_verlaufsliste(verlaufsliste_file)

st.sidebar.subheader("Hauptfilter")
analyse_typen = df['Analyse_Typ'].unique()
analyse_typen_list = list(analyse_typen)
default_index_analyse = analyse_typen_list.index('Mittel') if 'Mittel' in analyse_typen_list else 0
selected_analyse_typ = st.sidebar.selectbox("Analyse-Typ:", analyse_typen, index=default_index_analyse)
df = df[df['Analyse_Typ'] == selected_analyse_typ]

tag_typen = sorted(df['tagtyp'].unique())
default_tag = ['Mo-Fr'] if 'Mo-Fr' in tag_typen else []
selected_tag = st.sidebar.multiselect("Tag-Typ:", tag_typen, default=default_tag)

if selected_tag:
    zeitschicht_optionen = sorted(df[df['tagtyp'].isin(selected_tag)]['zeitschicht'].unique())
else:
    zeitschicht_optionen = sorted(df['zeitschicht'].unique())

default_zeit = ['16:00-19:00'] if '16:00-19:00' in zeitschicht_optionen else []
selected_zeitschicht = st.sidebar.multiselect("Zeitschicht:", zeitschicht_optionen, default=default_zeit)

if 'linien' in df.columns:
    all_lines = set()
    for line_list in df['linien'].dropna():
        parts = [p.strip() for p in str(line_list).split(',') if p.strip() != ""]
        all_lines.update(parts)
    sorted_lines = sorted(list(all_lines), key=lambda x: int(x) if x.isdigit() else x)
    default_linien = ['1'] if '1' in sorted_lines else []
    selected_linien = st.sidebar.multiselect("Linien:", sorted_lines, default=default_linien)
else:
    selected_linien = []

selected_richtung_label = None
selected_richtung_code = None
if verlaufsliste_df is not None and len(selected_linien) == 1:
    lin = selected_linien[0]
    verl_lin = verlaufsliste_df[verlaufsliste_df['linie'].astype(str) == str(lin)]
    if not verl_lin.empty:
        richtung_optionen = []
        mapping_label_to_code = {}
        for richtung_code, gruppe in verl_lin.groupby('richtung'):
            gruppe_sorted = gruppe.sort_values('sequenz')
            start = gruppe_sorted.iloc[0]['von_ort']
            ende = gruppe_sorted.iloc[-1]['nach_ort']
            label = f"{start} → {ende} ({richtung_code})"
            richtung_optionen.append(label)
            mapping_label_to_code[label] = richtung_code
        richtung_optionen = sorted(richtung_optionen)
        if richtung_optionen:
            selected_richtung_label = st.sidebar.selectbox("Linienrichtung:", richtung_optionen)
            if selected_richtung_label:
                selected_richtung_code = mapping_label_to_code[selected_richtung_label]

with st.sidebar.expander("Weitere Filteroptionen"):
    if 'anzahl' in df.columns and not df['anzahl'].empty:
        min_fahrten = int(df['anzahl'].min())
        max_fahrten = int(df['anzahl'].max())
        min_anzahl_fahrten = st.number_input(
            "Minimale Anzahl Fahrten:",
            min_value=min_fahrten,
            max_value=max_fahrten,
            value=min_fahrten,
            step=10
        )
    else:
        min_anzahl_fahrten = None

    if 'mittelwert_soll_fahrzeit' in df.columns and 'mittelwert_soll_haltezeit' in df.columns:
        ausschluss_soll_zeit_null = st.checkbox("Strecken mit Soll-Zeit 0 ausschliessen", value=True)
    else:
        ausschluss_soll_zeit_null = False

    if 'abweichung_signifikant' in df.columns:
        nur_signifikante = st.checkbox("Nur signifikante Abweichungen anzeigen")
    else:
        nur_signifikante = False

# --- KARTEN-FILTER ---
st.sidebar.subheader("Karten-Darstellung")

stat_options = {'Mittelwert': 'mean', 'Perzentil 75': 'q3', 'Perzentil 95': 'p95'}
verfuegbare_stat_options = {k: v for k, v in stat_options.items() if v == 'mean' or v in df.columns}
selected_stat_label = st.sidebar.selectbox(
    "Statistischer Kennwert:",
    options=list(verfuegbare_stat_options.keys())
)
selected_stat = verfuegbare_stat_options.get(selected_stat_label)

if selected_stat == 'mean':
    abw_options = {
        'Gesamtabweichung': 'Gesamtabweichung',
        'Abweichung Fahrzeit': 'Abweichung Fahrzeit',
        'Abweichung Haltezeit': 'Abweichung Haltezeit'
    }
    verfuegbare_abw_options = {k: v for k, v in abw_options.items() if v in df.columns}
    selected_abw_label = st.sidebar.selectbox(
        "Abweichungs-Typ:",
        options=list(verfuegbare_abw_options.keys())
    )
    map_color_column = verfuegbare_abw_options.get(selected_abw_label)
else:
    selected_abw_label = 'Gesamtabweichung'
    map_color_column = selected_stat
    st.sidebar.selectbox(
        "Abweichungs-Typ:",
        options=[selected_abw_label],
        disabled=True
    )

# --- DATENFILTERUNG ---
df_filtered = df.copy()
if selected_tag:
    df_filtered = df_filtered[df_filtered['tagtyp'].isin(selected_tag)]
if selected_zeitschicht:
    df_filtered = df_filtered[df_filtered['zeitschicht'].isin(selected_zeitschicht)]
if 'linien' in df_filtered.columns and selected_linien:
    pattern = r'\b(' + '|'.join(map(re.escape, selected_linien)) + r')\b'
    df_filtered = df_filtered[df_filtered['linien'].astype(str).str.contains(pattern, regex=True, na=False)]
if 'anzahl' in df_filtered.columns and min_anzahl_fahrten is not None:
    df_filtered = df_filtered[df_filtered['anzahl'] >= min_anzahl_fahrten]
if ausschluss_soll_zeit_null:
    if 'mittelwert_soll_fahrzeit' in df_filtered.columns and 'mittelwert_soll_haltezeit' in df_filtered.columns:
        df_filtered = df_filtered[~((df_filtered['mittelwert_soll_fahrzeit'] == 0) & (df_filtered['mittelwert_soll_haltezeit'] == 0))]
if nur_signifikante and 'abweichung_signifikant' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['abweichung_signifikant'] == 'ja']

# --- SEQUENZ / RICHTUNG INTEGRATION ---
if (verlaufsliste_df is not None and
        len(selected_linien) == 1 and
        selected_richtung_code is not None and
        not df_filtered.empty):
    lin = selected_linien[0]
    verl_subset = verlaufsliste_df[
        (verlaufsliste_df['linie'].astype(str) == str(lin)) &
        (verlaufsliste_df['richtung'] == selected_richtung_code)
    ].copy()

    if not verl_subset.empty:
        merge_keys = ['von_ort', 'nach_ort']
        if 'strecke_id' in verl_subset.columns and 'strecke_id' in df_filtered.columns:
            df_filtered['strecke_id'] = df_filtered['strecke_id'].astype(str).str.strip()
            verl_subset['strecke_id'] = verl_subset['strecke_id'].astype(str).str.strip()
            merge_keys = ['strecke_id']
        
        df_filtered = df_filtered.merge(
            verl_subset[merge_keys + ['sequenz']],
            on=merge_keys,
            how='left'
        )
        df_filtered = df_filtered.dropna(subset=['sequenz'])

        if 'sequenz' in df_filtered.columns and df_filtered['sequenz'].notna().any():
            df_filtered = df_filtered.sort_values('sequenz')
        else:
            st.warning("Keine Sequenz-Zuordnung für diese Richtung gefunden.")
else:
    if 'sequenz' not in df_filtered.columns:
        df_filtered['sequenz'] = np.nan

st.header("Analyse der ausgewählten Daten")
if df_filtered.empty:
    st.warning("Keine Daten für die ausgewählten Filter (nach Richtungs-/Sequenzfilterung).")
    st.stop()

# --- VISUALISIERUNGEN ---
st.header("Visualisierungen")
st.subheader(f"Geografische Darstellung (Farbe nach '{selected_stat_label} / {selected_abw_label}')")

if selected_richtung_label:
    st.info(f"Angezeigte Linienrichtung: **{selected_richtung_label}**")

map_data_lines = df_filtered.copy()
map_data_points = pd.DataFrame()

for col in ['Gesamtabweichung', 'Abweichung Fahrzeit', 'Abweichung Haltezeit', 'q3', 'p95']:
    if col in map_data_lines.columns:
        map_data_lines[f'{col}_str'] = map_data_lines[col].apply(lambda x: f"{x:.1f}s" if pd.notna(x) else "N/A")

color_scale_columns = ['Gesamtabweichung', 'Abweichung Fahrzeit', 'Abweichung Haltezeit', 'q3', 'p95']
vorhandene_spalten = [col for col in color_scale_columns if col in df_filtered.columns]
global_min_val, global_max_val = 0.0, 1.0

if vorhandene_spalten and not df_filtered.empty:
    all_values = pd.concat([df_filtered[col] for col in vorhandene_spalten]).dropna()
    if not all_values.empty:
        global_min_val = all_values.min()
        global_max_val = all_values.max()
        default_min_cap = all_values.quantile(0.25)
        default_max_cap = all_values.quantile(0.75)
    else:
        default_min_cap, default_max_cap = global_min_val, global_max_val
else:
    default_min_cap, default_max_cap = 0.0, 1.0

selected_min_val, selected_max_val = st.slider(
    "Farbskala-Bereich anpassen:",
    min_value=float(global_min_val),
    max_value=float(global_max_val),
    value=(float(default_min_cap), float(default_max_cap)),
    help="Passen Sie den unteren und oberen Grenzwert der Farbskala an, um Details besser sichtbar zu machen."
)

if selected_abw_label == 'Abweichung Haltezeit':
    if map_color_column in df_filtered.columns:
        map_data_points = df_filtered.groupby('nach_ort').agg(
            lon=('nach_lon', 'mean'), lat=('nach_lat', 'mean'), value=(map_color_column, 'mean')
        ).dropna().reset_index()
        if not map_data_points.empty:
            map_data_points['farbcode'] = map_data_points['value'].apply(lambda x: value_to_rgb(x, selected_min_val, selected_max_val))
    map_data_lines['farbcode'] = [[128, 128, 128, 70]] * len(map_data_lines)
else:
    if map_color_column and map_color_column in map_data_lines.columns and not map_data_lines[map_color_column].empty:
        map_data_lines['farbcode'] = map_data_lines[map_color_column].apply(lambda x: value_to_rgb(x, selected_min_val, selected_max_val))
        map_data_points = df_filtered.groupby('nach_ort').agg(
            lon=('nach_lon', 'mean'), lat=('nach_lat', 'mean')
        ).dropna().reset_index()
        map_data_points['farbcode'] = [[100, 100, 100, 100]] * len(map_data_points)
    else:
        map_data_lines['farbcode'] = [[128, 128, 128, 70]] * len(map_data_lines)

st.write("Aktuelle Farblegende:")
legend_html = f"""
<div style="
    background: linear-gradient(to right, rgb(0,255,0), rgb(255,255,0), rgb(255,0,0));
    padding: 10px; border-radius: 5px; color: black;
    display: flex; justify-content: space-between; font-weight: bold;
">
    <span><= {selected_min_val:.1f}s</span>
    <span>{((selected_min_val+selected_max_val)/2):.1f}s</span>
    <span>>= {selected_max_val:.1f}s</span>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)

if not map_data_lines.empty:
    all_stops = pd.concat([
        df_filtered[['von_ort', 'von_lon', 'von_lat']].rename(columns={'von_ort': 'ort', 'von_lon': 'lon', 'von_lat': 'lat'}),
        df_filtered[['nach_ort', 'nach_lon', 'nach_lat']].rename(columns={'nach_ort': 'ort', 'nach_lon': 'lon', 'nach_lat': 'lat'})
    ]).drop_duplicates(subset=['ort']).dropna()

    # --- HIER KOMMT DER OFFSET-CODE HIN ---
    # 1. Hinzufügen eines Indexes für jede eindeutige Linie pro Strecke
    map_data_lines['direction_index'] = map_data_lines.groupby(['von_ort', 'nach_ort']).cumcount()

    # 2. Anwenden des Offsets basierend auf dem Index
    offset = 0.0001
    map_data_lines['von_lon_offset'] = map_data_lines['von_lon'] + offset * map_data_lines['direction_index']
    map_data_lines['von_lat_offset'] = map_data_lines['von_lat'] + offset * map_data_lines['direction_index']
    map_data_lines['nach_lon_offset'] = map_data_lines['nach_lon'] + offset * map_data_lines['direction_index']
    map_data_lines['nach_lat_offset'] = map_data_lines['nach_lat'] + offset * map_data_lines['direction_index']
    # --- ENDE OFFSET-CODE ---

    layers = [
        # 3. LineLayer anpassen, um die Offset-Koordinaten zu verwenden
        pdk.Layer('LineLayer', data=map_data_lines, 
                  get_source_position='[von_lon_offset, von_lat_offset]', 
                  get_target_position='[nach_lon_offset, nach_lat_offset]',
                  get_color='farbcode', get_width=5, auto_highlight=True, pickable=True),
    ]
    if not map_data_points.empty:
        layers.append(pdk.Layer('ScatterplotLayer', data=map_data_points, get_position='[lon, lat]',
                                get_fill_color='farbcode', get_radius=40, pickable=True, auto_highlight=True))
    if not all_stops.empty:
        layers.append(pdk.Layer('TextLayer', data=all_stops, get_position='[lon, lat]', get_text='ort',
                                get_size=12, get_color=[0, 0, 0, 200], get_angle=0,
                                get_text_anchor="'middle'", get_alignment_baseline="'bottom'"))

    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=map_data_lines['von_lat'].mean(),
            longitude=map_data_lines['von_lon'].mean(),
            zoom=11
        ),
        layers=layers,
        tooltip={
            "html": "<b>{von_ort} → {nach_ort}</b><br/><hr>"
                    "Sequenz: {sequenz}<br/>"
                    "Gesamtabweichung (Mittelwert): {Gesamtabweichung_str}<br/>"
                    "Abweichung P75: {q3_str}<br/>"
                    "Abweichung P95: {p95_str}<br/>"
                    "<hr>"
                    "Fahrzeit-Abw. (Mittelwert): {Abweichung Fahrzeit_str}<br/>"
                    "Haltezeit-Abw. (Mittelwert): {Abweichung Haltezeit_str}<br/>"
                    "Anzahl Fahrten (Gesamt): {anzahl}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    )
    st.pydeck_chart(deck, height=900)
else:
    st.info("🗺️ Keine Linien für die Karte (nach Filter).")

# --- DATEN-AGGREGATION & ANZEIGE ---
def create_display_label(row):
    label = f"{row['von_ort']} → {row['nach_ort']}"
    if 'sequenz' in row and pd.notna(row['sequenz']):
        label = f"[{int(row['sequenz'])}] " + label
    return label

if 'sequenz' in df_filtered.columns and df_filtered['sequenz'].notna().any():
    df_filtered_sorted = df_filtered.sort_values('sequenz').copy()
else:
    sort_col = 'Gesamtabweichung' if 'Gesamtabweichung' in df_filtered.columns else df_filtered.columns[0]
    df_filtered_sorted = df_filtered.sort_values(sort_col, ascending=False).copy()

df_filtered_sorted['Strecke'] = df_filtered_sorted.apply(create_display_label, axis=1)
stats_tabelle = df_filtered_sorted

st.markdown("<hr style='border:1px solid #e74c3c; margin-top: 1.2rem; margin-bottom: 1.2rem;'>", unsafe_allow_html=True)
tab_analyse, tab_fahrplan = st.tabs(["📊 Übersicht & Analyse", "📈 Fahrplan-Vorschläge (Export)"])

with tab_analyse:
    st.subheader("Statistische Kennwerte im Überblick")
    cols = st.columns(6)
    if not stats_tabelle.empty:
        cols[0].metric("Datensätze", f"{stats_tabelle.shape[0]:,}".replace(",", "'"))
        if 'anzahl' in stats_tabelle.columns:
            cols[1].metric("Gesamtzahl Messungen", f"{int(stats_tabelle['anzahl'].sum()):,}".replace(",", "'"))
            cols[2].metric("Ø Messungen / Strecke", f"{stats_tabelle['anzahl'].mean():,.0f}".replace(",", "'"))
        if 'abweichung_signifikant' in stats_tabelle.columns:
            cols[3].metric("Signifikante Abweichungen", f"{stats_tabelle[stats_tabelle['abweichung_signifikant'] == 'ja'].shape[0]:,}".replace(",", "'"))
        if 'sequenz' in stats_tabelle.columns:
            cols[4].metric("Gematchte Sequenzen", f"{stats_tabelle['sequenz'].notna().sum():,}".replace(",", "'"))

    st.markdown("---")
    col_stats, col_boxplot = st.columns([0.55, 0.45])

    with col_stats:
        st.subheader("Detailübersicht der Abweichungen")
        if not stats_tabelle.empty:
            cols_to_display = ['Strecke', 'sequenz', 'linien', 'anzahl', 'mittelwert_soll', 'mittelwert_ist', 'Gesamtabweichung']
            display_cols = [c for c in cols_to_display if c in stats_tabelle.columns]
            df_display = stats_tabelle[display_cols].copy().rename(columns={
                'sequenz': 'Sequenz', 'linien': 'Linien', 'anzahl': 'Anzahl',
                'mittelwert_soll': 'Soll-Zeit', 'mittelwert_ist': 'Ø Ist-Zeit', 'Gesamtabweichung': 'Ø Abw.'
            })
            if 'Sequenz' in df_display.columns:
                df_display['Sequenz'] = df_display['Sequenz'].astype('Int64')

            format_dict = {col: '{:.1f}' for col in df_display.columns if col not in ['Strecke', 'Anzahl', 'Linien', 'Sequenz']}
            if 'Anzahl' in df_display.columns:
                df_display['Anzahl'] = df_display['Anzahl'].astype(int)
                format_dict['Anzahl'] = '{:d}'

            st.dataframe(df_display.style.format(format_dict).background_gradient(cmap='Reds', axis=0, subset=['Ø Abw.']),
                         use_container_width=True, height=(len(df_display) + 1) * 35 + 3)
        else:
            st.warning("Keine Daten für Anzeige vorhanden.")

    with col_boxplot:
        st.subheader("Verteilung der Gesamtabweichung")
        if not stats_tabelle.empty:
            # Boxplot benötigt einen eindeutigen Index
            df_for_plot = stats_tabelle.reset_index()
            spalten_zu_pruefen = ['Gesamtabweichung', 'q1', 'median', 'q3', 'p05', 'p95']
            if all(col in df_for_plot.columns for col in spalten_zu_pruefen):
                fig = go.Figure()
                for index, row in df_for_plot.iterrows():
                    fig.add_trace(go.Box(
                        y=[row['Strecke']], q1=[row['q1']], median=[row['median']], q3=[row['q3']],
                        lowerfence=[row['p05']], upperfence=[row['p95']], name=row['Strecke'],
                        boxpoints=False, orientation='h', line_color='royalblue', fillcolor='lightsteelblue'
                    ))
                    fig.add_trace(go.Scatter(y=[row['Strecke']], x=[row['Gesamtabweichung']], mode='markers',
                                           marker=dict(color='crimson', size=8, symbol='circle'), name='Mittelwert'))
                fig.update_layout(xaxis_title="Gesamtabweichung (Sekunden)", yaxis_title=None, showlegend=False,
                                  height=max(400, 30 * len(df_for_plot)), yaxis=dict(autorange="reversed"),
                                  template="plotly_white", margin=dict(t=10, b=20, l=20, r=50))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Nicht alle Statistik-Spalten (Median, Quartile etc.) vorhanden.")
        else:
            st.warning("Keine Daten für Boxplot vorhanden.")

with tab_fahrplan:
    st.header("Detaillierte Fahrplan-Vorschläge für den Export")
    vorschlags_df = stats_tabelle.copy()
    vorschlag_berechnet = False
    if 'mittelwert_soll_fahrzeit' in vorschlags_df.columns and 'mittelwert_soll_haltezeit' in vorschlags_df.columns:
        for p_val, p_col in zip([50, 75, 95], ['median', 'q3', 'p95']):
            if p_col in vorschlags_df.columns:
                vorschlag_berechnet = True
                perzentil_abw = vorschlags_df[p_col]
                anteil_fahrzeit_abw = (vorschlags_df.get('Abweichung Fahrzeit') / vorschlags_df['Gesamtabweichung']).replace([np.inf, -np.inf], np.nan).fillna(0.5)
                anteil_haltezeit_abw = (vorschlags_df.get('Abweichung Haltezeit') / vorschlags_df['Gesamtabweichung']).replace([np.inf, -np.inf], np.nan).fillna(0.5)
                vorschlags_df[f'Vorschlag_Fahrzeit_P{p_val}'] = vorschlags_df['mittelwert_soll_fahrzeit'] + (perzentil_abw * anteil_fahrzeit_abw)
                vorschlags_df[f'Vorschlag_Haltezeit_P{p_val}'] = vorschlags_df['mittelwert_soll_haltezeit'] + (perzentil_abw * anteil_haltezeit_abw)
                vorschlags_df[f'Vorschlag_Gesamt_P{p_val}'] = vorschlags_df[f'Vorschlag_Fahrzeit_P{p_val}'] + vorschlags_df[f'Vorschlag_Haltezeit_P{p_val}']

    if vorschlag_berechnet:
        st.subheader("Summen-Vergleich der Fahrzeiten")
        sum_soll = vorschlags_df['mittelwert_soll'].sum()
        sum_p50 = vorschlags_df['Vorschlag_Gesamt_P50'].sum()
        sum_p75 = vorschlags_df['Vorschlag_Gesamt_P75'].sum()
        sum_p95 = vorschlags_df['Vorschlag_Gesamt_P95'].sum()

        delta_p50 = sum_p50 - sum_soll
        delta_p75 = sum_p75 - sum_soll
        delta_p95 = sum_p95 - sum_soll

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Summe Soll-Zeit", format_seconds_to_hms(sum_soll))
        col2.metric("Summe Vorschlag P50", format_seconds_to_hms(sum_p50), 
                    delta=format_seconds_to_hms(delta_p50, force_sign=True))
        col3.metric("Summe Vorschlag P75", format_seconds_to_hms(sum_p75), 
                    delta=format_seconds_to_hms(delta_p75, force_sign=True))
        col4.metric("Summe Vorschlag P95", format_seconds_to_hms(sum_p95), 
                    delta=format_seconds_to_hms(delta_p95, force_sign=True))

        st.markdown("---")
        
        def format_seconds_precise(seconds_total):
            if pd.isna(seconds_total) or not isinstance(seconds_total, (int, float, np.number)):
                return ""
            sign = "-" if seconds_total < 0 else ""
            seconds_total = abs(seconds_total)
            hours = int(seconds_total // 3600)
            minutes = int((seconds_total % 3600) // 60)
            seconds = seconds_total % 60
            return f"{sign}{hours:02d}:{minutes:02d}:{seconds:05.2f}".replace('.',',')

        # DataFrame für die Bildschirmanzeige vorbereiten
        df_display_export = vorschlags_df.copy()
        if 'Strecke' not in df_display_export.columns:
             df_display_export['Strecke'] = df_display_export.apply(create_display_label, axis=1)
        
        # --- KORREKTUR: Alle Vorschlagsspalten für die Anzeige auswählen ---
        export_cols = [ 'sequenz', 'Strecke', 'linien', 'anzahl', 'mittelwert_soll',
            'Vorschlag_Gesamt_P50', 'Vorschlag_Fahrzeit_P50', 'Vorschlag_Haltezeit_P50',
            'Vorschlag_Gesamt_P75', 'Vorschlag_Fahrzeit_P75', 'Vorschlag_Haltezeit_P75',
            'Vorschlag_Gesamt_P95', 'Vorschlag_Fahrzeit_P95', 'Vorschlag_Haltezeit_P95'
        ]
        df_display_export = df_display_export[[c for c in export_cols if c in df_display_export.columns]]
        
        time_cols_display = [col for col in df_display_export.columns if 'soll' in col or 'Vorschlag' in col]
        for col in time_cols_display:
            df_display_export[col] = df_display_export[col].apply(format_seconds_precise)
        
        st.dataframe(df_display_export, use_container_width=True)

        # --- Logik für den CSV-Export (unverändert) ---
        df_for_csv = vorschlags_df.copy()
        if 'Strecke' not in df_for_csv.columns:
             df_for_csv['Strecke'] = df_for_csv.apply(create_display_label, axis=1)
        
        df_for_csv = df_for_csv[[c for c in export_cols if c in df_for_csv.columns]]

        time_cols_to_sum = [col for col in df_for_csv.columns if 'Vorschlag' in col or 'soll' in col]
        summen_zeile = pd.DataFrame(df_for_csv[time_cols_to_sum].sum()).T
        summen_zeile['Strecke'] = 'GESAMTSUMME'
        df_for_csv = pd.concat([df_for_csv, summen_zeile], ignore_index=True)

        for col in time_cols_to_sum:
            df_for_csv[col] = df_for_csv[col].apply(format_seconds_precise)
        
        @st.cache_data
        def convert_df_to_csv(df_in):
            return df_in.to_csv(index=False, sep=';').encode('utf-8-sig')

        csv = convert_df_to_csv(df_for_csv)
        st.download_button(label="📁 Detaillierte Vorschläge als CSV exportieren", data=csv,
                           file_name='fahrplan_vorschlaege.csv', mime='text/csv')
    else:
        st.warning("Für den gewählten Analyse-Typ wurden keine Vorschlagsspalten berechnet (benötigt Detail-Informationen).")

with st.expander("Gefilterte Rohdaten"):
    st.dataframe(df_filtered)

# --- ENDE ---

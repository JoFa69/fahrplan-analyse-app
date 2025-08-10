import streamlit as st
import pandas as pd
import plotly.express as px
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
    """
    Extrahiert Längen- und Breitengrad aus einem WKT-String (z.B. "POINT (8.3 47.0)").
    """
    if pd.isna(wkt_string) or 'POINT' not in str(wkt_string):
        return None, None
    try:
        coords_part = wkt_string.split('(')[1].split(')')[0]
        lon, lat = coords_part.split()
        return float(lon), float(lat)
    except (IndexError, ValueError):
        return None, None

def value_to_rgb(value, min_val, max_val):
    """
    Mappt einen Wert auf eine Farbskala von Grün nach Rot.
    """
    if pd.isna(value):
        return [128, 128, 128, 100] # Grau für fehlende Werte
    if min_val == max_val:
        return [0, 255, 0, 160] # Grün, wenn alle Werte gleich sind
    # Normalisiere den Wert auf einen Bereich von 0 bis 1
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0, min(1, normalized))
    
    # Interpoliere zwischen Grün (0) und Rot (1)
    red = int(255 * normalized)
    green = int(255 * (1 - normalized))
    
    return [red, green, 0, 160]

# --- FUNKTION ZUM LADEN UND VORBEREITEN DER DATEN ---
@st.cache_data
def lade_daten(uploaded_file):
    """
    Lädt die CSV-Datei, bereitet die Daten vor und extrahiert die Koordinaten.
    """
    if uploaded_file is None:
        return None
    try:
        # Robustes Einlesen der CSV-Datei
        df = pd.read_csv(uploaded_file, sep=None, engine='python')

        # Spalte 'mittelwert' umbenennen für mehr Klarheit
        if 'mittelwert' in df.columns:
            df.rename(columns={'mittelwert': 'Gesamtabweichung'}, inplace=True)

        numeric_cols_mit_komma = [
            'mittelwert_soll', 'mittelwert_ist', 'mittelwert_soll_fahrzeit',
            'mittelwert_soll_haltezeit', 'mittelwert_ist_fahrzeit',
            'mittelwert_ist_haltezeit', 'Gesamtabweichung', 'stdabw', 'varianz', 'p05',
            'q1', 'median', 'q3', 'p95', 'Interquartilsabstand',
            'konfidenz_95_low', 'konfidenz_95_high'
        ]
        for col in numeric_cols_mit_komma:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

        if 'VON_WKT' in df.columns:
            df[['von_lon', 'von_lat']] = df['VON_WKT'].apply(lambda x: pd.Series(parse_wkt_point(x)))
        if 'NACH_WKT' in df.columns:
            df[['nach_lon', 'nach_lat']] = df['NACH_WKT'].apply(lambda x: pd.Series(parse_wkt_point(x)))

        # Abweichungen berechnen
        if 'mittelwert_ist_fahrzeit' in df.columns and 'mittelwert_soll_fahrzeit' in df.columns:
            df['Abweichung Fahrzeit'] = df['mittelwert_ist_fahrzeit'] - df['mittelwert_soll_fahrzeit']
        if 'mittelwert_ist_haltezeit' in df.columns and 'mittelwert_soll_haltezeit' in df.columns:
            df['Abweichung Haltezeit'] = df['mittelwert_ist_haltezeit'] - df['mittelwert_soll_haltezeit']

        return df
    except Exception as e:
        st.error(f"Fehler beim Laden oder Verarbeiten der Datei: {e}")
        return None

# --- HAUPTTEIL DER APP ---
st.title("🚆 Interaktives Fahrplan-Analyse Dashboard")
st.write("Laden Sie Ihre CSV-Datei hoch, um zu beginnen. Die Filter passen die Auswertungen dynamisch an.")

uploaded_file = st.file_uploader("Laden Sie hier Ihre Fahrplan-Daten als CSV-Datei hoch", type=['csv'])

if uploaded_file is None:
    st.info("Bitte laden Sie eine CSV-Datei hoch, um mit der Analyse zu beginnen.")
    st.stop()

df = lade_daten(uploaded_file)
if df is None:
    st.stop()

# --- SIDEBAR MIT FILTERN ---
st.sidebar.title("Filter & Optionen")

# --- START: NEUER HAUPTFILTER ---
analyse_typen = df['Analyse_Typ'].unique()
selected_analyse_typ = st.sidebar.selectbox("Analyse-Typ:", analyse_typen)

# Filtere das DataFrame basierend auf dem ausgewählten Analyse-Typ
df = df[df['Analyse_Typ'] == selected_analyse_typ]
# --- ENDE: NEUER HAUPTFILTER ---


tag_typen = sorted(df['tagtyp'].unique())
default_tag = ['Mo-Fr'] if 'Mo-Fr' in tag_typen else []
selected_tag = st.sidebar.multiselect("Tag-Typ:", tag_typen, default=default_tag)

if selected_tag:
    zeitschicht_optionen = sorted(df[df['tagtyp'].isin(selected_tag)]['zeitschicht'].unique())
else:
    zeitschicht_optionen = sorted(df['zeitschicht'].unique())

default_zeit = ['06:30-08:30'] if '06:30-08:30' in zeitschicht_optionen else []
selected_zeitschicht = st.sidebar.multiselect("Zeitschicht:", zeitschicht_optionen, default=default_zeit)


if 'linien' in df.columns:
    all_lines = set()
    for line_list in df['linien'].dropna():
        all_lines.update(str(line_list).split(','))
    sorted_lines = sorted(list(all_lines), key=lambda x: int(x) if x.isdigit() else x)
    selected_linien = st.sidebar.multiselect("Linien:", sorted_lines, default=[])

if 'anzahl' in df.columns and not df['anzahl'].empty:
    min_fahrten = int(df['anzahl'].min())
    max_fahrten = int(df['anzahl'].max())
    min_anzahl_fahrten = st.sidebar.number_input(
        "Minimale Anzahl Fahrten:",
        min_value=min_fahrten,
        max_value=max_fahrten,
        value=min_fahrten,
        step=10
    )

if 'mittelwert_soll_fahrzeit' in df.columns and 'mittelwert_soll_haltezeit' in df.columns:
    ausschluss_soll_zeit_null = st.sidebar.checkbox("Strecken mit Soll-Zeit 0 ausschliessen", value=True)
else:
    ausschluss_soll_zeit_null = False

# --- START: KONDITIONALE FILTER ---
# Zeige diese Filter nur an, wenn der "Detail"-Analyse-Typ ausgewählt ist
if selected_analyse_typ == 'Detail':
    if 'mittelwert_soll_fahrzeit' in df.columns:
        soll_fahrzeiten = sorted(df['mittelwert_soll_fahrzeit'].unique())
        selected_soll_fahrzeit = st.sidebar.multiselect("Soll-Fahrzeit:", soll_fahrzeiten, default=[])

    if 'mittelwert_soll_haltezeit' in df.columns:
        soll_haltezeiten = sorted(df['mittelwert_soll_haltezeit'].unique())
        selected_soll_haltezeit = st.sidebar.multiselect("Soll-Haltezeit:", soll_haltezeiten, default=[])
else:
    selected_soll_fahrzeit = []
    selected_soll_haltezeit = []
# --- ENDE: KONDITIONALE FILTER ---

st.sidebar.subheader("Analyse-Kennwert")
kennwert_optionen = ['Gesamtabweichung', 'Abweichung Fahrzeit', 'Abweichung Haltezeit']
verfuegbare_kennwerte = [opt for opt in kennwert_optionen if opt in df.columns]

selected_kennwert = st.sidebar.selectbox(
    "Kennwert für Visualisierung:",
    verfuegbare_kennwerte
)
st.sidebar.markdown(f"<small>Dieser Wert steuert die Farbe der Karte und das Boxplot-Diagramm.</small>", unsafe_allow_html=True)

st.sidebar.subheader("Fahrplan-Vorschlag")
percentile_options = {
    'median': '50. Perzentil (Median)',
    'q3': '75. Perzentil',
    'p95': '95. Perzentil'
}
selected_percentile_col = st.sidebar.selectbox(
    "Perzentil für Vorschlag:",
    options=list(percentile_options.keys()),
    format_func=lambda x: percentile_options[x],
    index=1
)

st.sidebar.subheader("Weitere Filter")
if 'abweichung_signifikant' in df.columns:
    nur_signifikante = st.sidebar.checkbox("Nur signifikante Abweichungen anzeigen")
else:
    nur_signifikante = False


if 'Gesamtabweichung' in df.columns:
    if not df['Gesamtabweichung'].dropna().empty:
        min_val = df['Gesamtabweichung'].min()
        max_val = df['Gesamtabweichung'].max()
        selected_abweichung = st.sidebar.slider(
            "Gesamtabweichung filtern (Sekunden):",
            min_value=int(min_val),
            max_value=int(max_val),
            value=(int(min_val), int(max_val))
        )
    else:
        selected_abweichung = (0, 100)
        st.sidebar.text("Keine gültigen Werte für Abweichung.")
else:
    selected_abweichung = (0, 100)


von_orte = sorted(df['von_ort'].unique())
selected_von_ort = st.sidebar.multiselect("Startort (von):", von_orte, default=[])

nach_orte = sorted(df['nach_ort'].unique())
selected_nach_ort = st.sidebar.multiselect("Zielort (nach):", nach_orte, default=[])


# --- DATENFILTERUNG BASIEREND AUF DER AUSWAHL ---
df_filtered = df.copy()

if selected_tag:
    df_filtered = df_filtered[df_filtered['tagtyp'].isin(selected_tag)]
if selected_zeitschicht:
    df_filtered = df_filtered[df_filtered['zeitschicht'].isin(selected_zeitschicht)]

if 'linien' in df.columns and selected_linien:
    pattern = r'\b(' + '|'.join(map(str, selected_linien)) + r')\b'
    df_filtered = df_filtered[df_filtered['linien'].astype(str).str.contains(pattern, regex=True, na=False)]

if 'anzahl' in df.columns and 'min_anzahl_fahrten' in locals():
    df_filtered = df_filtered[df_filtered['anzahl'] >= min_anzahl_fahrten]

if ausschluss_soll_zeit_null:
    df_filtered = df_filtered[~((df_filtered['mittelwert_soll_fahrzeit'] == 0) & (df_filtered['mittelwert_soll_haltezeit'] == 0))]

if 'mittelwert_soll_fahrzeit' in df.columns and selected_soll_fahrzeit:
    df_filtered = df_filtered[df_filtered['mittelwert_soll_fahrzeit'].isin(selected_soll_fahrzeit)]
if 'mittelwert_soll_haltezeit' in df.columns and selected_soll_haltezeit:
    df_filtered = df_filtered[df_filtered['mittelwert_soll_haltezeit'].isin(selected_soll_haltezeit)]

if 'Gesamtabweichung' in df.columns and not df_filtered.empty:
    df_filtered = df_filtered[df_filtered['Gesamtabweichung'].between(selected_abweichung[0], selected_abweichung[1])]

if selected_von_ort:
    df_filtered = df_filtered[df_filtered['von_ort'].isin(selected_von_ort)]
if selected_nach_ort:
    df_filtered = df_filtered[df_filtered['nach_ort'].isin(selected_nach_ort)]

if nur_signifikante:
    df_filtered = df_filtered[df_filtered['abweichung_signifikant'] == 'ja']


st.header("Analyse der ausgewählten Daten")
if df_filtered.empty:
    st.warning("Keine Daten für die ausgewählten Filter gefunden.")
    st.stop()


# --- VISUALISIERUNGEN ---
st.header("Visualisierungen")

# --- KARTE ---
st.subheader(f"Geografische Darstellung (Farbe nach '{selected_kennwert}')")

map_data_lines = df_filtered.copy()
map_data_points = pd.DataFrame()

# Tooltip-Spalten vorbereiten
for col in ['Gesamtabweichung', 'Abweichung Fahrzeit', 'Abweichung Haltezeit']:
    if col in map_data_lines.columns:
        map_data_lines[f'{col}_str'] = map_data_lines[col].apply(lambda x: f"{x:.1f}s" if pd.notna(x) else "N/A")

# Richtungs-Logik für die Karte
map_data_lines['route_id'] = map_data_lines.apply(lambda row: '-'.join(sorted([row['von_ort'], row['nach_ort']])), axis=1)
map_data_lines['direction_index'] = map_data_lines.groupby('route_id').cumcount()
map_data_lines['Richtung'] = np.where(map_data_lines['direction_index'] == 0, 'Hinweg', 'Rückweg')
offset = 0.0001 
map_data_lines['von_lon_offset'] = map_data_lines['von_lon'] + offset * map_data_lines['direction_index']
map_data_lines['von_lat_offset'] = map_data_lines['von_lat'] + offset * map_data_lines['direction_index']
map_data_lines['nach_lon_offset'] = map_data_lines['nach_lon'] + offset * map_data_lines['direction_index']
map_data_lines['nach_lat_offset'] = map_data_lines['nach_lat'] + offset * map_data_lines['direction_index']


# Logik für die Farbgebung
if 'Haltezeit' in selected_kennwert:
    if selected_kennwert in df_filtered.columns:
        map_data_points = df_filtered.groupby('nach_ort').agg(
            lon=('nach_lon', 'mean'),
            lat=('nach_lat', 'mean'),
            value=(selected_kennwert, 'mean')
        ).dropna().reset_index()
        
        if not map_data_points.empty:
            min_val = map_data_points['value'].min()
            max_val = map_data_points['value'].max()
            map_data_points['farbcode'] = map_data_points['value'].apply(lambda x: value_to_rgb(x, min_val, max_val))
        map_data_lines['farbcode'] = [[128, 128, 128, 70]] * len(map_data_lines)
else:
    if selected_kennwert in df_filtered.columns and not df_filtered[selected_kennwert].empty:
        min_val = map_data_lines[selected_kennwert].min()
        max_val = map_data_lines[selected_kennwert].max()
        map_data_lines['farbcode'] = map_data_lines[selected_kennwert].apply(lambda x: value_to_rgb(x, min_val, max_val))
        
        map_data_points = df_filtered.groupby('nach_ort').agg(
            lon=('nach_lon', 'mean'),
            lat=('nach_lat', 'mean')
        ).dropna().reset_index()
        map_data_points['farbcode'] = [[100, 100, 100, 100]] * len(map_data_points)

# --- Farblegende ---
if 'min_val' in locals() and 'max_val' in locals():
    st.write("Farblegende (Minimum bis Maximum des gewählten Kennwerts):")
    legend_html = f"""
    <div style="
        background: linear-gradient(to right, rgb(0,255,0), rgb(255,255,0), rgb(255,0,0));
        padding: 10px;
        border-radius: 5px;
        color: black;
        display: flex;
        justify-content: space-between;
        font-weight: bold;
    ">
        <span>{min_val:.1f}s</span>
        <span>{((min_val+max_val)/2):.1f}s</span>
        <span>{max_val:.1f}s</span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)


if not map_data_lines.empty:
    tooltip_html = """
        <b>{von_ort} nach {nach_ort} ({Richtung})</b><br/><hr>
        Gesamtabweichung: {Gesamtabweichung_str}<br/>
        Fahrzeit-Abw.: {Abweichung Fahrzeit_str}<br/>
        Haltezeit-Abw.: {Abweichung Haltezeit_str}<br/>
        Anzahl Fahrten (Gesamt): {anzahl}
    """
    
    von_stops = df_filtered[['von_ort', 'von_lon', 'von_lat']].rename(
        columns={'von_ort': 'ort', 'von_lon': 'lon', 'von_lat': 'lat'}
    )
    nach_stops = df_filtered[['nach_ort', 'nach_lon', 'nach_lat']].rename(
        columns={'nach_ort': 'ort', 'nach_lon': 'lon', 'nach_lat': 'lat'}
    )
    all_stops = pd.concat([von_stops, nach_stops]).drop_duplicates(subset=['ort']).dropna()

    layers = [
        pdk.Layer(
           'LineLayer',
           data=map_data_lines,
           get_source_position='[von_lon_offset, von_lat_offset]',
           get_target_position='[nach_lon_offset, nach_lat_offset]',
           get_color='farbcode',
           get_width=5,
           auto_highlight=True,
           pickable=True
        )
    ]
    if not map_data_points.empty:
        layers.append(
            pdk.Layer(
                'ScatterplotLayer',
                data=map_data_points,
                get_position='[lon, lat]',
                get_fill_color='farbcode',
                get_radius=40,
                pickable=True,
                auto_highlight=True
            )
        )
    
    if not all_stops.empty:
        layers.append(
            pdk.Layer(
                'TextLayer',
                data=all_stops,
                get_position='[lon, lat]',
                get_text='ort',
                get_size=12,
                get_color=[0, 0, 0, 200],
                get_angle=0,
                get_text_anchor="'middle'",
                get_alignment_baseline="'bottom'"
            )
        )

    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=map_data_lines['von_lat'].mean(),
            longitude=map_data_lines['von_lon'].mean(),
            zoom=11
        ),
        layers=layers,
        tooltip={
            "html": tooltip_html,
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    )
    
    st.pydeck_chart(deck, use_container_width=True, height=800)
else:
    st.info("🗺️ Karten-Daten fehlen oder sind gefiltert.")

# --- KENNZAHLEN UND DIAGRAMME ---
st.subheader("Statistische Kennwerte im Überblick")
col1, col2, col3, col4, col5 = st.columns(5)

if 'Gesamtabweichung' in df_filtered.columns and not df_filtered['Gesamtabweichung'].empty:
    col1.metric(
        "Ø Gesamtabweichung",
        f"{df_filtered['Gesamtabweichung'].mean():,.0f}s".replace(",", "'"),
        help="Der Durchschnitt der Gesamtabweichung über alle aktuell gefilterten Datenpunkte."
    )
col2.metric("Datensätze", f"{df_filtered.shape[0]:,}".replace(",", "'"))
if 'abweichung_signifikant' in df_filtered.columns:
    col3.metric("Signifikante Abweichungen", f"{df_filtered[df_filtered['abweichung_signifikant'] == 'ja'].shape[0]:,}".replace(",", "'"))
if 'anzahl' in df_filtered.columns:
    col4.metric("Gesamtzahl Messungen", f"{int(df_filtered['anzahl'].sum()):,}".replace(",", "'"))
    col5.metric("Ø Messungen pro Strecke", f"{df_filtered['anzahl'].mean():,.0f}".replace(",", "'"))


# --- ANZEIGE VON TABELLE UND BOXPLOT ---
col_stats, col_boxplot = st.columns(2)

with col_stats:
    # --- START: ERSETZTER BEREICH (TAB-LAYOUT) ---
    tab1, tab2, tab3 = st.tabs(["Übersicht & Abweichungen", "Fahrplan-Vorschlag", "Statistische Details"])

    # Datenaufbereitung (einmal für alle Tabs)
    if not df_filtered.empty:
        grouping_keys = ['von_ort', 'nach_ort']
        if selected_analyse_typ == 'Detail':
            if 'mittelwert_soll_fahrzeit' in df_filtered.columns:
                grouping_keys.append('mittelwert_soll_fahrzeit')
            if 'mittelwert_soll_haltezeit' in df_filtered.columns:
                grouping_keys.append('mittelwert_soll_haltezeit')

        agg_dict = {col: 'first' for col in df_filtered.columns if col not in grouping_keys and col != 'linien'}
        if 'linien' in df_filtered.columns:
            def combine_linien(series):
                all_linien = set()
                for item in series.dropna():
                    all_linien.update(str(item).split(','))
                return ', '.join(sorted(list(all_linien), key=lambda x: int(x) if x.isdigit() else x))
            agg_dict['linien'] = combine_linien
        
        stats_tabelle = df_filtered.groupby(grouping_keys).agg(agg_dict).reset_index()

        perzentil_abw = stats_tabelle[selected_percentile_col]
        
        anteil_fahrzeit_abw = (stats_tabelle['Abweichung Fahrzeit'] / stats_tabelle['Gesamtabweichung']).fillna(0.5)
        anteil_haltezeit_abw = (stats_tabelle['Abweichung Haltezeit'] / stats_tabelle['Gesamtabweichung']).fillna(0.5)

        stats_tabelle['Vorschlag Fahrzeit (s)'] = stats_tabelle['mittelwert_soll_fahrzeit'] + (perzentil_abw * anteil_fahrzeit_abw)
        stats_tabelle['Vorschlag Haltezeit (s)'] = stats_tabelle['mittelwert_soll_haltezeit'] + (perzentil_abw * anteil_haltezeit_abw)
        stats_tabelle['Vorschlag Total (s)'] = stats_tabelle['Vorschlag Fahrzeit (s)'] + stats_tabelle['Vorschlag Haltezeit (s)']


        if 'mittelwert_soll' in stats_tabelle.columns and 'Gesamtabweichung' in stats_tabelle.columns:
            stats_tabelle['Relative Abw. (%)'] = (stats_tabelle['Gesamtabweichung'] / stats_tabelle['mittelwert_soll']).replace([np.inf, -np.inf], 0) * 100

        if 'Gesamtabweichung' in stats_tabelle.columns:
            stats_tabelle = stats_tabelle.sort_values(by='Gesamtabweichung', ascending=False)

        stats_cols = {
            'linien': 'Linien', 'anzahl': 'Anzahl', 
            'Gesamtabweichung': 'Ø Abw. Total', 'mittelwert_ist': 'Ø Ist-Zeit Total',
            'mittelwert_soll': 'Soll-Zeit Total', 'mittelwert_ist_fahrzeit': 'Ø Ist-Fahrzeit',
            'mittelwert_ist_haltezeit': 'Ø Ist-Haltezeit', 'stdabw': 'Std.-Abw.', 
            'p05': '5. Perz.', 'q1': 'Q1 (25%)', 'median': 'Median', 'q3': 'Q3 (75%)', 
            'p95': '95. Perz.', 'abweichung_signifikant': 'Signifikant'
        }
        stats_tabelle.rename(columns=stats_cols, inplace=True)
        
        def create_display_label(row):
            label = f"{row['von_ort']} → {row['nach_ort']}"
            if selected_analyse_typ == 'Detail':
                fahrzeit = row.get('mittelwert_soll_fahrzeit')
                haltezeit = row.get('mittelwert_soll_haltezeit')
                details = []
                if fahrzeit is not None:
                    details.append(f"Soll-Fahrzeit: {fahrzeit:.0f}s")
                if haltezeit is not None:
                    details.append(f"Soll-Haltezeit: {haltezeit:.0f}s")
                if details:
                    label += f" ({', '.join(details)})"
            return label
        stats_tabelle.index = stats_tabelle.apply(create_display_label, axis=1)

    with tab1:
        st.subheader("Übersicht & Abweichungen")
        if not stats_tabelle.empty:
            cols = ['Linien', 'Anzahl', 'Soll-Zeit Total', 'Ø Ist-Zeit Total', 'Ø Abw. Total', 'Relative Abw. (%)', 'Signifikant']
            display_cols = [c for c in cols if c in stats_tabelle.columns]
            df_display = stats_tabelle[display_cols]
            
            format_dict = {col: '{:.1f}' for col in df_display.columns if col not in ['Anzahl', 'Signifikant', 'Linien']}
            if 'Relative Abw. (%)' in df_display.columns:
                format_dict['Relative Abw. (%)'] = '{:.1f}%'
            if 'Anzahl' in df_display.columns:
                df_display = df_display.copy()
                df_display['Anzahl'] = df_display['Anzahl'].astype(int)
                format_dict['Anzahl'] = '{:d}'
            
            table_height = (len(df_display) + 1) * 35 + 3
            st.dataframe(df_display.style.format(format_dict).background_gradient(
                cmap='Reds', axis=0, subset=['Ø Abw. Total', 'Relative Abw. (%)']
            ), use_container_width=True, height=table_height)
        else:
            st.warning("Keine Daten für die Anzeige vorhanden.")

    with tab2:
        st.subheader("Fahrplan-Vorschlag")
        if not stats_tabelle.empty:
            cols = ['Linien', 'Anzahl', 'Soll-Zeit Total', 'Ø Ist-Zeit Total', 'Vorschlag Total (s)', 'Vorschlag Fahrzeit (s)', 'Vorschlag Haltezeit (s)']
            display_cols = [c for c in cols if c in stats_tabelle.columns]
            df_display = stats_tabelle[display_cols]

            format_dict = {col: '{:.1f}' for col in df_display.columns if col not in ['Anzahl', 'Signifikant', 'Linien']}
            if 'Anzahl' in df_display.columns:
                df_display = df_display.copy()
                df_display['Anzahl'] = df_display['Anzahl'].astype(int)
                format_dict['Anzahl'] = '{:d}'

            table_height = (len(df_display) + 1) * 35 + 3
            st.dataframe(df_display.style.format(format_dict), use_container_width=True, height=table_height)
        else:
            st.warning("Keine Daten für die Anzeige vorhanden.")

    with tab3:
        st.subheader("Statistische Details")
        if not stats_tabelle.empty:
            cols = ['Linien', 'Anzahl', 'Std.-Abw.', '5. Perz.', 'Q1 (25%)', 'Median', 'Q3 (75%)', '95. Perz.']
            display_cols = [c for c in cols if c in stats_tabelle.columns]
            df_display = stats_tabelle[display_cols]

            format_dict = {col: '{:.1f}' for col in df_display.columns if col not in ['Anzahl', 'Signifikant', 'Linien']}
            if 'Anzahl' in df_display.columns:
                df_display = df_display.copy()
                df_display['Anzahl'] = df_display['Anzahl'].astype(int)
                format_dict['Anzahl'] = '{:d}'

            table_height = (len(df_display) + 1) * 35 + 3
            st.dataframe(df_display.style.format(format_dict), use_container_width=True, height=table_height)
        else:
            st.warning("Keine Daten für die Anzeige vorhanden.")
    # --- ENDE: NEUER BEREICH (TAB-LAYOUT) ---

with col_boxplot:
    with st.expander(f"Verteilung der '{selected_kennwert}' für alle gefilterten Strecken", expanded=True):
        if 'stats_tabelle' in locals() and not stats_tabelle.empty:
            
            df_plot_data = stats_tabelle.copy()

            fig = go.Figure()

            spalten_zu_pruefen = ['Ø Abw. Total', 'Q1 (25%)', 'Median', 'Q3 (75%)', '5. Perz.', '95. Perz.']
            if all(col in df_plot_data.columns for col in spalten_zu_pruefen):
                for index, row in df_plot_data.iterrows():
                    fig.add_trace(go.Box(
                        y=[index],
                        q1=[row['Q1 (25%)']],
                        median=[row['Median']],
                        q3=[row['Q3 (75%)']],
                        lowerfence=[row['5. Perz.']],
                        upperfence=[row['95. Perz.']],
                        name=index,
                        boxpoints=False,
                        orientation='h',
                        line_color='royalblue',
                        fillcolor='lightsteelblue'
                    ))
                    fig.add_trace(go.Scatter(
                        y=[index],
                        x=[row['Ø Abw. Total']],
                        mode='markers',
                        marker=dict(color='crimson', size=8, symbol='circle'),
                        name='Mittelwert'
                    ))

                num_strecken = len(df_plot_data)
                dynamic_height = max(400, 30 * num_strecken)

                fig.update_layout(
                    # KORREKTUR: Titel entfernt, um den Rand zu verkleinern
                    xaxis_title=f"{selected_kennwert} (Sekunden)",
                    yaxis_title=None,
                    showlegend=False,
                    height=dynamic_height,
                    yaxis=dict(autorange="reversed"),
                    template="plotly_white",
                    margin=dict(t=20, b=20, l=20, r=20) # Kleinerer Rand
                )
                
                # KORREKTUR: Titel als separate Überschrift hinzugefügt
                st.subheader(f"Verteilung für '{selected_kennwert}'")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Nicht alle benötigten Statistik-Spalten (Median, Quartile etc.) sind in der Tabelle vorhanden.")
        else:
            st.warning("Keine Daten für das Boxplot-Diagramm vorhanden (Statistiktabelle ist leer).")


with st.expander("Gefilterte Rohdaten"):
    # KORREKTUR: Höhe entfernt, um die Tabelle "endlos" zu machen
    st.dataframe(df_filtered)

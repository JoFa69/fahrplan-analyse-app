🚆 Interaktives Fahrplan-Analyse Dashboard
Dieses Streamlit-Dashboard dient der interaktiven Analyse und Visualisierung von Fahrplandaten aus dem öffentlichen Verkehr. Es ermöglicht Planern und Analysten, die Pünktlichkeit und Zuverlässigkeit von verschiedenen Strecken zu untersuchen, Problemstellen zu identifizieren und datengestützte Vorschläge für zukünftige Fahrpläne zu entwickeln.

🚀 Hauptfunktionen
Interaktive Karte: Geografische Darstellung der Strecken, farblich kodiert nach der gewählten Analysekennzahl (z.B. Gesamtabweichung). Haltestellen werden zur besseren Orientierung direkt beschriftet.

Flexible Filterung: Ein umfangreiches Set an Filtern in der Sidebar ermöglicht die detaillierte Eingrenzung der Daten nach Kriterien wie Tag-Typ, Zeitschicht, Linien, Anzahl der Fahrten und mehr.

Zwei Analyse-Modi:

Detail-Analyse: Untersucht jede einzelne Soll-Fahr- und Haltezeit-Kombination als separate Einheit.

Mittel-Analyse: Analysiert aggregierte Daten pro Strecke und Zeitschicht.

Statistische Auswertung: Eine übersichtliche Darstellung der wichtigsten Kennzahlen in drei thematischen Reitern:

Übersicht & Abweichungen: Die wichtigsten Metriken auf einen Blick, inklusive prozentualer Abweichung.

Fahrplan-Vorschlag: Berechnet basierend auf wählbaren Perzentilen (z.B. 75. Perzentil) robuste Vorschläge für zukünftige Fahr-, Halte- und Gesamtzeiten.

Statistische Details: Detaillierte statistische Werte wie Standardabweichung und Perzentile für eine tiefergehende Analyse.

Dynamische Visualisierungen: Interaktive Boxplots visualisieren die Verteilung der Abweichungen und passen sich dynamisch an die gefilterten Daten an.

⚙️ Benutzung
1. Daten hochladen
Laden Sie Ihre CSV-Datei über den Uploader auf der Hauptseite hoch. Die Datei sollte die unten beschriebene Struktur aufweisen.

2. Analyse-Typ wählen
Wählen Sie in der Sidebar unter "Analyse-Typ" aus, ob Sie die detaillierte Ansicht ("Detail") oder die aggregierte Ansicht ("Mittel") analysieren möchten. Die verfügbaren Filter passen sich automatisch an.

3. Daten filtern
Nutzen Sie die vielfältigen Optionen in der Sidebar, um die Datenmenge auf den für Sie relevanten Bereich einzugrenzen.

4. Ergebnisse analysieren
Untersuchen Sie die Ergebnisse in den verschiedenen Sektionen:

Karte: Verschaffen Sie sich einen geografischen Überblick.

Kennzahlen: Sehen Sie die wichtigsten aggregierten Werte der gefilterten Daten.

Tabellen & Boxplot: Nutzen Sie die Reiter, um Abweichungen zu analysieren, Fahrplan-Vorschläge zu prüfen oder in statistische Details einzutauchen.

📋 Anforderungen an die Datendatei
Damit die App korrekt funktioniert, muss die hochgeladene CSV-Datei eine bestimmte Struktur aufweisen. Die Spaltennamen sind dabei entscheidend.

Wichtige Spalten:

Analyse_Typ: Muss die Werte Detail oder Mittel enthalten.

von_ort, nach_ort: Start- und Zielhaltestelle.

VON_WKT, NACH_WKT: Koordinaten der Haltestellen im WKT-Format (z.B. POINT (8.3 47.0)).

linien: Eine oder mehrere Liniennummern, durch Komma getrennt.

tagtyp, zeitschicht: Kategorien für die Filterung.

anzahl: Anzahl der Fahrten, die in diesem Datensatz zusammengefasst sind.

mittelwert_soll, mittelwert_ist: Durchschnittliche Soll- und Ist-Gesamtzeit.

mittelwert_soll_fahrzeit, mittelwert_ist_fahrzeit: Durchschnittliche Soll- und Ist-Fahrzeit.

mittelwert_soll_haltezeit, mittelwert_ist_haltezeit: Durchschnittliche Soll- und Ist-Haltezeit.

median, q3, p95, etc.: Statistische Kennwerte der Gesamtabweichung.

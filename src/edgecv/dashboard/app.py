"""Throwaway Streamlit coverage panel with enhanced UI filtering and map controls."""
from __future__ import annotations

import folium
import psycopg
import streamlit as st
from streamlit_folium import st_folium

MOCK_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [151.205562, -33.864898],
                    [151.205905, -33.867856],
                    [151.206506, -33.872809]
                ]
            },
            "properties": {
                "segment_id": 1,
                "authority_id": "council-042",
                "road_name": "Example Road",
                "road_ref": "A01",
                "length_m": 890.4,
                "surface_type": "sealed",
                "run_id": "11111111-1111-1111-1111-111111111111",
                "assessed_at": "2026-09-06T10:00:00+00:00",
                "condition_index": 0.85,
                "condition_band": "good",
                "counts": {"pothole": 1, "crack": 0},
                "frames_assessed": 150,
                "coverage_m": 875.2
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [151.206506, -33.872809],
                    [151.212455, -33.873740],
                    [151.212123, -33.876786]
                ]
            },
            "properties": {
                "segment_id": 2,
                "authority_id": "council-042",
                "road_name": "Example Road",
                "road_ref": "A01",
                "length_m": 720.7,
                "surface_type": "sealed",
                "run_id": "11111111-1111-1111-1111-111111111111",
                "assessed_at": "2026-09-06T10:05:00+00:00",
                "condition_index": 0.58,
                "condition_band": "fair",
                "counts": {"pothole": 2, "crack": 1},
                "frames_assessed": 130,
                "coverage_m": 698.5
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [151.212123, -33.876786],
                    [151.211844, -33.877356],
                    [151.211200, -33.878229],
                    [151.209097, -33.879708],
                    [151.208432, -33.882825]
                ]
            },
            "properties": {
                "segment_id": 3,
                "authority_id": "council-042",
                "road_name": "Example Road",
                "road_ref": "A01",
                "length_m": 680.3,
                "surface_type": "sealed",
                "run_id": "11111111-1111-1111-1111-111111111111",
                "assessed_at": "2026-09-06T10:10:00+00:00",
                "condition_index": 0.25,
                "condition_band": "poor",
                "counts": {"pothole": 4, "crack": 2},
                "frames_assessed": 120,
                "coverage_m": 645.8
            }
        }
    ]
}


def latest_segments_geojson(conn: psycopg.Connection) -> dict:
    query = """
    SELECT json_build_object(
        'type', 'FeatureCollection',
        'features', COALESCE(json_agg(
            json_build_object(
                'type', 'Feature',
                'geometry', ST_AsGeoJSON(s.geom)::json,
                'properties', json_build_object(
                    'segment_id', s.segment_id,
                    'authority_id', s.authority_id,
                    'road_name', s.road_name,
                    'road_ref', s.road_ref,
                    'length_m', s.length_m,
                    'surface_type', s.surface_type,
                    'run_id', sc.run_id,
                    'assessed_at', sc.assessed_at,
                    'condition_index', sc.condition_index,
                    'condition_band', sc.condition_band,
                    'counts', sc.counts,
                    'frames_assessed', sc.frames_assessed,
                    'coverage_m', sc.coverage_m
                )
            )
        ), '[]'::json)
    )
    FROM segments s
    JOIN LATERAL (
        SELECT * FROM segment_condition
        WHERE segment_id = s.segment_id
        ORDER BY assessed_at DESC
        LIMIT 1
    ) sc ON true;
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()
            if result and result[0] and result[0].get("features"):
                return result[0]
    except Exception:
        pass

    return MOCK_GEOJSON


def main() -> None:
    from edgecv.config import Settings

    st.set_page_config(page_title="Road Condition Map", layout="wide")
    st.title("Road Condition Map")

    
    try:
        with psycopg.connect(Settings.from_env().pg_dsn, autocommit=True) as conn:
            raw_geojson = latest_segments_geojson(conn)
    except Exception:
        raw_geojson = MOCK_GEOJSON

    features = raw_geojson.get("features", [])

    
    st.sidebar.header("Map Controls")
    selected_filter = st.sidebar.selectbox(
        "Condition Filter",
        options=["all", "good", "fair", "poor"],
        format_func=lambda x: "All Segments" if x == "all" else x.capitalize()
    )

    
    if selected_filter != "all":
        filtered_features = [
            f for f in features 
            if f["properties"].get("condition_band") == selected_filter
        ]
    else:
        filtered_features = features

    
    st.sidebar.markdown(f"**Segments shown:** `{len(filtered_features)}`")

    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Legend")
    st.sidebar.markdown("🟢 **Good**")
    st.sidebar.markdown("🟠 **Fair**")
    st.sidebar.markdown("🔴 **Poor**")
    st.sidebar.markdown("⚪ **Unknown**")

    
    if filtered_features:
        m = folium.Map(location=[-33.8688, 151.2093], zoom_start=13)

        filtered_geojson = {
            "type": "FeatureCollection",
            "features": filtered_features
        }

        def style_function(feature):
            band = feature['properties'].get('condition_band', 'unknown')
            color = 'green' if band == 'good' else 'orange' if band == 'fair' else 'red'
            return {'color': color, 'weight': 6, 'opacity': 0.85}

        
        def build_popup_html(props):
            counts = props.get("counts") or {}
            return f"""
            <div style="font-family: Arial; min-width: 180px;">
                <h4 style="margin:0 0 5px 0;">Segment #{props.get('segment_id', 'N/A')}</h4>
                <b>Road:</b> {props.get('road_name', 'Unknown')}<br>
                <b>Ref:</b> {props.get('road_ref', 'Unknown')}<br>
                <b>Condition:</b> {props.get('condition_band', 'Unknown').capitalize()}<br>
                <b>Index:</b> {props.get('condition_index', 'N/A')}<br>
                <b>Length:</b> {props.get('length_m', 'N/A')} m<br>
                <b>Coverage:</b> {props.get('coverage_m', 'N/A')} m<br>
                <b>Frames:</b> {props.get('frames_assessed', 'N/A')}<br>
                <hr style="margin:5px 0;">
                <b>Potholes:</b> {counts.get('pothole', 0)}<br>
                <b>Cracks:</b> {counts.get('crack', 0)}<br>
                <b>Surface:</b> {props.get('surface_type', 'Unknown')}<br>
                <b>Assessed:</b> {props.get('assessed_at', 'Unknown')}
            </div>
            """

        geojson_layer = folium.GeoJson(
            filtered_geojson,
            name="Road Segments",
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["road_name", "road_ref", "condition_band"],
                aliases=["Road:", "Ref:", "Condition:"]
            )
        )

        
        for feature in filtered_geojson["features"]:
            popup_html = build_popup_html(feature["properties"])
            folium.Popup(popup_html, max_width=300).add_to(
                folium.GeoJson(feature, style_function=style_function).add_to(m)
            )

        st_folium(m, width="100%", height=650)
    else:
        st.info("No road segments match the selected filter.")


if __name__ == "__main__":
    main()

import streamlit as st
from sqlalchemy import create_engine, inspect, text
import pandas as pd
import json
from io import BytesIO
import datetime
import decimal


st.set_page_config(page_title="🗺️ DB Mapper", layout="wide")
st.title("🧩 Database Structure Mapper")

st.markdown("This app maps your database schema and gives you a downloadable JSON file with structure and sample data.")

# === Inputs ===
with st.sidebar:
    st.header("🔐 Database Credentials")
    db_type = st.selectbox("Database Type", ["postgresql", "mysql", "sqlite"])
    username = st.text_input("Username", value="postgres")
    password = st.text_input("Password", value="postgres", type="password")
    host = st.text_input("Host", value="localhost")
    port = st.text_input("Port", value="5432")
    db_name = st.text_input("Database Name")
    submitted = st.button("🔍 Analyze DB")

# === Function to create SQLAlchemy engine ===
def get_engine(db_type, username, password, host, port, db_name):
    if db_type == "sqlite":
        return create_engine(f"sqlite:///{db_name}")
    return create_engine(f"{db_type}://{username}:{password}@{host}:{port}/{db_name}")

# === Function to inspect database ===
def inspect_db(engine):
    inspector = inspect(engine)
    result = {}

    for schema in inspector.get_schema_names():
        result[schema] = {}

        for table in inspector.get_table_names(schema=schema):
            columns = inspector.get_columns(table, schema=schema)
            col_data = [
                {"name": col["name"], "type": str(col["type"])} for col in columns
            ]

            # Try to fetch sample data (first 5 rows)
            try:
                with engine.connect() as conn:
                    query = text(f'SELECT * FROM "{schema}"."{table}" LIMIT 5')
                    sample_result = conn.execute(query).mappings().fetchall()
                    sample_data = [dict(row) for row in sample_result]
            except Exception as e:
                sample_data = [f"❌ Error fetching sample data: {e}"]

            result[schema][table] = {
                "columns": col_data,
                "sample_data": sample_data,
            }

    return result

# === UI Actions ===
if submitted:
    try:
        with st.spinner("Connecting and analyzing database..."):
            engine = get_engine(db_type, username, password, host, port, db_name)
            db_structure = inspect_db(engine)

        st.success("✅ Database analyzed successfully!")

        # === Display in Streamlit ===
        for schema, tables in db_structure.items():
            with st.expander(f"📂 Schema: `{schema}`", expanded=False):
                for table, info in tables.items():
                    st.markdown(f"### 📄 Table: `{table}`")
                    st.markdown("**Columns:**")
                    col_df = pd.DataFrame(info["columns"])
                    st.dataframe(col_df, use_container_width=True)

                    st.markdown("**Sample Data (first 5 rows):**")
                    sample_data = info.get("sample_data", [])
                    if isinstance(sample_data, list) and sample_data and isinstance(sample_data[0], dict):
                        sample_df = pd.DataFrame(sample_data)
                        st.dataframe(sample_df, use_container_width=True)
                    else:
                        st.warning(sample_data[0] if sample_data else "No sample data available.")

        # === Export JSON Button ===
        def default_converter(obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()
            if isinstance(obj, decimal.Decimal):
                return float(obj)
            return str(obj)

        def convert_to_bytes(data):
            json_str = json.dumps(data, indent=2, default=default_converter)
            return BytesIO(json_str.encode("utf-8"))

        st.download_button(
            label="📥 Download JSON Structure",
            data=convert_to_bytes(db_structure),
            file_name=f"{db_name}_structure.json",
            mime="application/json",
        )

    except Exception as e:
        st.error(f"❌ Error analyzing DB: {e}")

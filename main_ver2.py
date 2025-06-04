import nest_asyncio
import asyncio
import asyncpg
import google.generativeai as genai
import re
import json
import datetime
import os

nest_asyncio.apply()

# === CONFIG ===
GENAI_API_KEY = "AIzaSyAx2xapgripXxeY5yvGcEixjD5v4nykxTI"
LOG_FILE = "query_trace_log.json"

DB_CONNECTIONS = {
    "PTDH_BCP": "postgresql://postgres:d3W4tH1nkM4tcH@10.27.240.142:5432/PTDH-BCP",
    "PTDH_ACP": "postgresql://postgres:d3W4tH1nkM4tcH@10.27.240.142:5432/PTDH-ACP",
    "PTDH_WKCP": "postgresql://postgres:d3W4tH1nkM4tcH@10.27.240.142:5432/PTDH-WKCP",
    "PTDH_EKCP": "postgresql://postgres:d3W4tH1nkM4tcH@10.27.240.142:5432/PTDH-EKCP",
}

DB_CONTEXTS = {
    "PTDH_BCP": {
        "location": "Bengalon",
        "schema": "DPR",
        "table": "bcp_med_table_format",
        "columns": [
            "period_date", "waste_removal_plan", "waste_removal_actual", 
            "coal_mined_plan", "coal_mined_actual", "coal_hauling_plan", "coal_hauling_actual",
            "rain_plan", "rain_actual", "slippery_plan", "slippery_actual",
            "rain_frequency_plan", "rain_frequency_actual", "rainfall_plan", "rainfall_actual"
        ]
    },
    "PTDH_ACP": {
        "location": "Asam Asam",
        "schema": "DPR",
        "table": "acp_med_table_format",
        "columns": [
            "period_date",
            "waste_removal_budget", "waste_removal_forecast", "waste_removal_actual",
            "coal_mined_budget", "coal_mined_forecast", "coal_mined_actual",
            "coal_hauling_budget", "coal_hauling_forecast", "coal_hauling_actual",
            "rain_budget", "rain_forecast", "rain_actual",
            "slippery_budget", "slippery_forecast", "slippery_actual",
            "rain_frequency_budget", "rain_frequency_forecast", "rain_frequency_actual",
            "intencity_budget", "intencity_forecast", "intencity_actual"
        ]
    },
    "PTDH_WKCP": {
        "location": "West Kintap",
        "schema": "DPR",
        "table": "wkcp_med_table_format",
        "columns": [
            "period_date",
            "waste__mud_removal_plan_bc", "waste__mud_removal_plan_3mrp", "waste__mud_removal_actual",
            "coal_mined_plan_bc", "coal_mined_plan_3mrp", "coal_mined_actual",
            "coal_hauling_plan_bc", "coal_hauling_plan_3mrp", "coal_hauling_actual",
            "rain_plan", "rain_actual",
            "slippery_plan", "slippery_actual",
            "frequency_rain__plan", "frequency_rain__actual",
            "frequency_slippery_plan", "frequency_slippery_actual",
            "rainfall_plan", "rainfall_actual"
        ]
    },
    "PTDH_EKCP": {
        "location": "Kintap",
        "schema": "DPR",
        "table": "ekcp_med_table_format",
        "columns": [
            "period_date",
            "waste_removal__mud_plan_monthly", "waste_removal__mud_plan_basecase", "waste_removal__mud_actual",
            "coal_mined_plan_monthly", "coal_mined_plan_basecase", "coal_mined_actual",
            "rain_plan", "rain_actual",
            "slippery_plan", "slippery_actual",
            "rain_frequency_plan", "rain_frequency_actual",
            "rainfall_plan", "rainfall_actual"
        ]
    }
}


# === INIT ===
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# === LOGGER ===
def log_trace(step: str, content, save_to_file=True):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "content": content
    }
    print(f"[{log_entry['timestamp']}] [{step}] {content}")
    
    if save_to_file:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

# === CONTEXT BUILDER ===
def build_model_context(db_key: str) -> str:
    db = DB_CONTEXTS[db_key]
    schema = db["schema"]
    table = db["table"]
    columns = db["columns"]
    columns_str = ', '.join([f'"{col}"' for col in columns])

    return f"""
You are an assistant that converts natural language to PostgreSQL SQL.
Only return raw SQL code without explanation.
The database has a schema named "{schema}" and a table named "{table}" (case-sensitive).
Wrap all schema and table names in double quotes. Wrap all column names in double quotes as well.
The table has columns: {columns_str}.
"""

# === GUARDRAILS ===
def apply_guardrails(sql: str) -> str:
    sql = sql.strip()
    sql_upper = sql.upper()
    unsafe_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE", "ALTER"]

    if not sql_upper.startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed!")

    if any(keyword in sql_upper for keyword in unsafe_keywords):
        raise ValueError("Unsafe SQL keyword detected in query.")

    if "LIMIT" not in sql_upper:
        has_semicolon = sql.endswith(";")
        if has_semicolon:
            sql = sql[:-1].strip()
        sql += " LIMIT 100"
        if has_semicolon:
            sql += ";"

    return sql

# === GEMINI SQL GENERATOR ===
async def generate_sql_from_prompt(prompt: str, db_key: str) -> str:
    context = build_model_context(db_key)
    full_prompt = f"{context}\nUser question: {prompt}"
    log_trace("PROMPT", prompt)

    response = model.generate_content(full_prompt)
    log_trace("RAW_MODEL_RESPONSE", response.text)

    match = re.search(r"```sql\s*(.*?)\s*```", response.text, re.DOTALL | re.IGNORECASE)
    sql = match.group(1).strip() if match else response.text.strip()

    sql = apply_guardrails(sql)
    log_trace("CLEAN_SQL", sql)

    return sql

# === SQL EXECUTION ===
async def run_query(sql: str, db_key: str):
    try:
        conn = await asyncpg.connect(DB_CONNECTIONS[db_key])
        schema = DB_CONTEXTS[db_key]["schema"]
        await conn.execute(f'SET search_path TO "{schema}";')
        log_trace("EXECUTION", "Running SQL query...")

        rows = await conn.fetch(sql)
        result_list = [dict(row) for row in rows]
        for row in result_list:
            log_trace("RESULT_ROW", row)

        await conn.close()
        return result_list

    except Exception as e:
        log_trace("ERROR", str(e))
        raise

# === MAIN ===
async def main(user_prompt: str, db_key: str):
    try:
        sql = await generate_sql_from_prompt(user_prompt, db_key)
        results = await run_query(sql, db_key)
        return results

    except Exception as e:
        log_trace("FATAL", str(e))
        return []

# === ENTRY POINT ===
# if __name__ == "__main__":
#     # Example prompt and DB (can be from UI or CLI)
#     user_prompt = "bandingkan produksi coal dan plan bulan mei 2025 berapa ya?"
#     selected_db_key = "PTDH_BCP"  # change to "PTDH_ACP", etc.
#     asyncio.run(main(user_prompt, selected_db_key))

if __name__ == "__main__":
    print("Available database keys:")
    for key, ctx in DB_CONTEXTS.items():
        print(f"- {key} ({ctx['location']})")

    selected_db_key = input("Enter the database key (e.g., PTDH_ACP): ").strip()
    if selected_db_key not in DB_CONTEXTS:
        print(f"Invalid DB key: {selected_db_key}")
        exit(1)

    user_prompt = input("Enter your natural language prompt: ").strip()
    
    asyncio.run(main(user_prompt, selected_db_key))

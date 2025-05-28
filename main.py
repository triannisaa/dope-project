import nest_asyncio
import asyncio
import asyncpg
import google.generativeai as genai
import re

nest_asyncio.apply()

genai.configure(api_key="AIzaSyAx2xapgripXxeY5yvGcEixjD5v4nykxTI")
model = genai.GenerativeModel("gemini-2.0-flash")

DB_URL = "postgresql://postgres:d3W4tH1nkM4tcH@10.27.240.142:5432/PTDH-BCP"

# context = """
# You are an assistant that converts natural language to PostgreSQL SQL.
# Only return raw SQL code without explanation.
# The database has a schema named 'DPR'.
# Assume the database has schema DPR and a table named 'bcp_med_table_format' with columns:
# period_date,waste_removal_plan,waste_removal_actual,coal_mined_plan,coal_mined_actual,
# coal_hauling_plan,coal_hauling_actual,rain_plan,rain_actual,slippery_plan,slippery_actual,
# rain_frequency_plan,rain_frequency_actual,rainfall_plan,rainfall_actual
# """

context = """
You are an assistant that converts natural language to PostgreSQL SQL.
Only return raw SQL code without explanation.
The database has a schema named 'DPR' and a table named "bcp_med_table_format" (case-sensitive).
Wrap all schema and table names in double quotes. Wrap all column names in double quotes as well.
The table has columns: "period_date", "waste_removal_plan", "waste_removal_actual", 
"coal_mined_plan", "coal_mined_actual", "coal_hauling_plan", "coal_hauling_actual", 
"rain_plan", "rain_actual", "slippery_plan", "slippery_actual", 
"rain_frequency_plan", "rain_frequency_actual", "rainfall_plan", "rainfall_actual".
"""


async def main():
    user_prompt = "jumlahkan produksi coal bulan april 2025 berapa ya?"

    gemini_response = model.generate_content(f"{context}\nUser question: {user_prompt}")

    match = re.search(r"```sql\s*(.*?)\s*```", gemini_response.text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
    else:
        sql = gemini_response.text.strip()

    print("Generated SQL:\n", sql)

    conn = await asyncpg.connect(DB_URL)
    try:
        await conn.execute("SET search_path TO DPR;")
        rows = await conn.fetch(sql)
        for row in rows:
            print(dict(row))
    except Exception as e:
        print("Error during DB operation:", e)
    finally:
        await conn.close()

await main()
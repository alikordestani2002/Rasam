import os
import traceback
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


# ==========================================
# 1. Configuration & API Key
# ==========================================

MY_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

if not MY_OPENROUTER_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set")
client = OpenAI(
    api_key=MY_OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1"
)

LLM_MODEL = "qwen/qwen-2.5-72b-instruct"

# ==========================================
# 2. Dynamic Path Resolution & Data Loading
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, 'data', 'csv_files')

print("Loading datasets into memory...")
try:
    df_string = pd.read_csv(os.path.join(CSV_DIR, 'solar_telemetry.string_data.csv'))
    df_mvpanel = pd.read_csv(os.path.join(CSV_DIR, 'solar_telemetry.mvpanel_data.csv'))
    df_lvpanel = pd.read_csv(os.path.join(CSV_DIR, 'solar_telemetry.lvpanel_data.csv'))
    df_inverter = pd.read_csv(os.path.join(CSV_DIR, 'solar_telemetry.inverter_data.csv'))
except FileNotFoundError as e:
    print(f"❌ Error finding files: {e}")
    exit()

# ==========================================
# 3. Static Schema Definition (Hardcoded)
# ==========================================
STATIC_SCHEMA = """
Database Schema (Pre-loaded DataFrames):

1. df_string:
Columns: _id, message_id, board_id, device_code, device_type, timestamp, ts, ingested_at (datetime string), voltage, current, power

2. df_mvpanel:
Columns: _id, message_id, board_id, device_code, device_type, timestamp, ts, ingested_at (datetime string), act_eng_deliv, act_eng_receiv, react_eng_deliv, react_eng_receiv, appar_eng_deliv, appar_eng_receiv, current_a, current_b, current_c, current_n, current_avg, voltage_ab, voltage_bc, voltage_ca, voltage_ll_avg, voltage_a, voltage_b, voltage_c, voltage_avg, active_pow_a, active_pow_b, active_pow_c, tot_active_pow, reactive_pow_a, reactive_pow_b, reactive_pow_c, tot_reactive_pow, appar_pow_a, appar_pow_b, appar_pow_c, tot_appar_pow, power_factor_a, power_factor_b, power_factor_c, power_factor, frequency, status

3. df_lvpanel:
Columns: _id, message_id, board_id, device_code, device_type, timestamp, ts, ingested_at (datetime string), act_eng_deliv, act_eng_receiv, react_eng_deliv, react_eng_receiv, appar_eng_deliv, appar_eng_receiv, current_a, current_b, current_c, current_n, current_avg, voltage_ab, voltage_bc, voltage_ca, voltage_ll_avg, voltage_a, voltage_b, voltage_c, voltage_avg, active_pow_a, active_pow_b, active_pow_c, tot_active_pow, reactive_pow_a, reactive_pow_b, reactive_pow_c, tot_reactive_pow, appar_pow_a, appar_pow_b, appar_pow_c, tot_appar_pow, power_factor_a, power_factor_b, power_factor_c, power_factor, frequency, status, processed_at, is_reset, delta_act_energy, plant_id, zone_id

4. df_inverter:
Columns: _id, message_id, board_id, device_code, device_type, timestamp, ts, ingested_at (datetime string), daily_pow_yield, monthly_pow_yield, tot_pow_yield, tot_apar_pow, tot_dc_pow, tot_active_pow, tot_reactive_pow, power_factor, grid_frequency, daily_run_time, tot_run_tim, int_temp, voltage_a, voltage_b, voltage_c, current_a, current_b, current_c, mppt1_volt ... to mppt12_curr, str1_current ... to str12_current, work_state, fa_al_time, fa_al_code, fa_al_message, status, mccb
"""

# ==========================================
# 4. Core Agent Functions
# ==========================================
def generate_pandas_code(user_query: str) -> str:
    """Part 1: Converting user request to Pandas code"""
    system_prompt = f"""
    You are a Senior Data Analyst and Python/Pandas expert at Rasam Enterprise.
    Your task is to convert the user's natural language query into exact, executable Pandas code.

    {STATIC_SCHEMA}

    STRICT RULES:
    1. The dataframes (df_string, df_mvpanel, df_lvpanel, df_inverter) are already loaded in memory. Do NOT write `import pandas as pd` or `pd.read_csv()`.
    2. To filter by date/time, ALWAYS convert the 'ingested_at' column to datetime using `pd.to_datetime()` before applying time conditions.
    3. You MUST store the final calculated answer (number, list, string, or small dataframe) in a variable exactly named `final_result`.
    4. Output ONLY the raw Python code. No markdown tags (like ```python), no explanations, no prints. Just the executable code.
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Query: {user_query}\n\nGenerate the Pandas code:"}
            ],
            temperature=0.0
        )
        
        code = response.choices[0].message.content.strip()
        if code.startswith("```python"):
            code = code.replace("```python", "").replace("```", "").strip()
        elif code.startswith("```"):
            code = code.replace("```", "").strip()
        return code
    except Exception as e:
        return f"# Error generating code: {str(e)}"

def generate_final_response(user_query: str, raw_database_result: str) -> str:
    """Part 2: Converting raw database output to a natural English sentence"""
    system_prompt = """
    You are the "Rasam Enterprise Data Representative". A smart and polite assistant that provides solar panel data reports to managers.
    
    Your task:
    You receive the raw result of the database processing. You must convert this result into a fully readable, professional, and polite statement (response) in English that exactly answers the user's question.

    Strict rules:
    1. Never mention variable names (like final_result or df_inverter).
    2. Never say "according to Python codes" or "based on Pandas". The user should not know that code was executed in the background.
    3. If the raw result contains an error or is empty (like None), politely apologize and say the information was not found in this time period.
    4. Focus only on providing the final answer and avoid extra explanations.
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Manager's (User) question: '{user_query}'\nRaw result fetched from the database: {raw_database_result}\n\nPlease write the final response:"}
            ],
            temperature=0.3 # Slightly higher temperature to generate a natural and human tone
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Unfortunately, an error occurred while generating the final response."

# ==========================================
# 5. Full Execution Pipeline
# ==========================================
if __name__ == "__main__":
    # Test prompt
    test_query = "What was the highest daily production amount in the inverters?"
    
    print(f"\n[User Request]: {test_query}\n")
    
    # Step 1: Generate code
    print("1. Building smart query...")
    generated_query = generate_pandas_code(test_query)
    
    # Step 2: Execute query
    print("2. Executing processing on the server...")
    local_vars = {
        'df_string': df_string,
        'df_mvpanel': df_mvpanel,
        'df_lvpanel': df_lvpanel,
        'df_inverter': df_inverter,
        'pd': pd
    }
    
    raw_result = None
    try:
        exec(generated_query, {}, local_vars)
        if 'final_result' in local_vars:
            raw_result = local_vars['final_result']
        else:
            raw_result = "Error: Data fetching encountered an issue."
    except Exception as e:
        raw_result = "Requested information not found in the data structure or a computational error occurred."
        # print(traceback.format_exc()) # You can uncomment this line for your own debugging

    # Step 3: Generate final response
    print("3. Drafting final report...\n")
    final_human_readable_answer = generate_final_response(test_query, str(raw_result))
    
    # Print final system output
    print("=" * 60)
    print("🤖 Rasam Data Representative Message:")
    print("=" * 60)
    print(final_human_readable_answer)
    print("=" * 60)
import os
import traceback
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from prompts import FINAL_RESPONSE_SYSTEM_PROMPT, PANDAS_CODE_SYSTEM_PROMPT


# ==========================================
# 1. Configuration & API Key
# ==========================================

load_dotenv()

MY_OPENROUTER_KEY = os.getenv("MY_OPENROUTER_KEY")

if not MY_OPENROUTER_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set")
client = OpenAI(
    api_key=MY_OPENROUTER_KEY,
    base_url="https://api.gapgpt.app/v1"
)

LLM_MODEL = "gapgpt-qwen-3.5"

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
# 3. Core Agent Functions
# ==========================================
def generate_pandas_code(user_query: str) -> str:
    """Part 1: Converting user request to Pandas code"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": PANDAS_CODE_SYSTEM_PROMPT},
                {"role": "user", "content": f"User Query: {user_query}\n\nGenerate the Pandas code:"}
            ],
            temperature=0.0
        )
        
        code = response.choices[0].message.content.strip()
        if code.startswith("```python"):
            code = code.replace("```python", "").replace("```", "").strip()
        elif code.startswith("```"):
            code = code.replace("```", "").strip()
        print(code)
        return code
    except Exception as e:
        return f"# Error generating code: {str(e)}"

def execute_pandas_code(generated_query: str):
    """Part 2: Executing generated Pandas code and returning the raw result"""
    local_vars = {
        'df_string': df_string,
        'df_mvpanel': df_mvpanel,
        'df_lvpanel': df_lvpanel,
        'df_inverter': df_inverter,
        'pd': pd
    }

    try:
        exec(generated_query, {}, local_vars)
        if 'final_result' in local_vars:
            return local_vars['final_result']
        return "Error: Data fetching encountered an issue."
    except Exception:
        return "Requested information not found in the data structure or a computational error occurred."
        # print(traceback.format_exc()) # You can uncomment this line for your own debugging

def generate_final_response(user_query: str, raw_database_result: str) -> str:
    """Part 3: Converting raw database output to a natural English sentence"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": FINAL_RESPONSE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Manager's (User) question: '{user_query}'\nRaw result fetched from the database: {raw_database_result}\n\nPlease write the final response:"}
            ],
            temperature=0.3 # Slightly higher temperature to generate a natural and human tone
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Unfortunately, an error occurred while generating the final response."

# ==========================================
# 4. Full Execution Pipeline
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
    raw_result = execute_pandas_code(generated_query)

    # Step 3: Generate final response
    print("3. Drafting final report...\n")
    final_human_readable_answer = generate_final_response(test_query, str(raw_result))
    
    # Print final system output
    print("=" * 60)
    print("🤖 Rasam Data Representative Message:")
    print("=" * 60)
    print(final_human_readable_answer)
    print("=" * 60)

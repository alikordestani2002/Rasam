import os
import time
import traceback
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ایمپورت پرامپت‌های تفکیک شده
from prompts import (
    ROUTER_SYSTEM_PROMPT,
    PANDAS_GENERATOR_PROMPT,
    STATISTICAL_SYNTHESIZER_PROMPT,
    DESCRIPTIVE_ANALYSIS_PROMPT
)

# ==========================================
# 1. Configuration & API Key
# ==========================================
load_dotenv()

MY_OPENROUTER_KEY = os.getenv("MY_OPENROUTER_KEY")
if not MY_OPENROUTER_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set. Please check your .env file.")

# تنظیم کلاینت OpenAI
client = OpenAI(
    api_key=MY_OPENROUTER_KEY,
    base_url="https://api.gapgpt.app/v1"
)

MODELS_TO_COMPARE = [
    "gapgpt-qwen-3.6",            # Thinking Model
]

OUTPUT_FILE_NAME = "model_comparison_results.txt"


# ==========================================
# 2. Core Engine Class (Telemetry AI)
# ==========================================
class TelemetryAI:
    def __init__(self, data_dir):
        """بارگذاری دیتاست‌ها در حافظه"""
        print("Loading datasets into memory...")
        try:
            self.df_string = pd.read_csv(os.path.join(data_dir, 'solar_telemetry.string_data.csv'))
            self.df_mvpanel = pd.read_csv(os.path.join(data_dir, 'solar_telemetry.mvpanel_data.csv'))
            self.df_lvpanel = pd.read_csv(os.path.join(data_dir, 'solar_telemetry.lvpanel_data.csv'))
            self.df_inverter = pd.read_csv(os.path.join(data_dir, 'solar_telemetry.inverter_data.csv'))
            print("Datasets loaded successfully.")
        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise

    # ------------------------------------------
    # Helper Methods
    # ------------------------------------------
    # TODO: temperature=0.1
    def _call_llm(self, prompt, model, system_instruction=""):
        """متد کمکی برای فراخوانی API مدل زبانی"""
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content

    def _clean_json(self, text):
        """حذف تگ‌های Markdown از خروجی JSON مدل"""
        text = text.strip()
        if text.startswith("```json"): 
            text = text[7:]
        elif text.startswith("```"): 
            text = text[3:]
        if text.endswith("```"): 
            text = text[:-3]
        return text.strip()

    # ------------------------------------------
    # Pipeline Step 1: Routing
    # ------------------------------------------
    def _perform_routing(self, query, model):
        """تحلیل سوال کاربر و بررسی وجود داده در دیتاست (قانون 3)"""
        t_start = time.time()
        router_prompt = f"سوال کاربر: {query}"
        
        try:
            raw_response = self._call_llm(router_prompt, model, system_instruction=ROUTER_SYSTEM_PROMPT)
            route_info = json.loads(self._clean_json(raw_response))
        except Exception as e:
            route_info = {"is_in_schema": False, "intent": "error", "error": str(e)}
            
        elapsed_time = time.time() - t_start
        return route_info, elapsed_time

    # ------------------------------------------
    # Pipeline Step 2A: Statistical Logic
    # ------------------------------------------
    def _handle_statistical_pipeline(self, query, model, result_payload):
        """مدیریت مسیر سوالات آماری شامل کدنویسی، اجرا و ترکیب پاسخ"""
        # بخش تولید کد
        t_code_start = time.time()
        code_prompt = PANDAS_GENERATOR_PROMPT.replace("{user_query}", query)
        generated_code = self._call_llm(code_prompt, model)
        generated_code = self._clean_json(generated_code).replace("python\n", "")
        
        result_payload["times"]["code_generation"] = time.time() - t_code_start
        result_payload["generated_code"] = generated_code

        # محیط ایزوله پایتون
        local_vars = {
            'df_inverter': self.df_inverter,
            'df_string': self.df_string,
            'df_lvpanel': self.df_lvpanel,
            'df_mvpanel': self.df_mvpanel,
            'pd': pd
        }
        
        # بخش اجرا و ترکیب
        t_synth_start = time.time()
        try:
            exec(generated_code, {}, local_vars)
            raw_result = local_vars.get('final_result', 'متغیر final_result توسط مدل تعریف نشد.')
            result_payload["raw_data"] = str(raw_result)
            
            synth_prompt = STATISTICAL_SYNTHESIZER_PROMPT.format(
                user_query=query, 
                executed_code=generated_code, 
                raw_result=str(raw_result)
            )
            final_answer = self._call_llm(synth_prompt, model)
            
            result_payload["status"] = "success"
            result_payload["final_answer"] = final_answer
            
        except Exception as e:
            result_payload["status"] = "error"
            result_payload["final_answer"] = f"خطا در اجرای کوئری استخراج داده روی سرور: {str(e)}"
            result_payload["error_trace"] = traceback.format_exc()
            
        result_payload["times"]["synthesis"] = time.time() - t_synth_start

    # ------------------------------------------
    # Pipeline Step 2B: Descriptive Logic
    # ------------------------------------------
    # TODO: faults = self.df_inverter[self.df_inverter['fa_al_code'] != 0][['device_code', 'fa_al_code', 'fa_al_message']].head(20).to_dict('records') -> parameter number of head
    def _handle_descriptive_pipeline(self, query, model, result_payload):
        """مدیریت مسیر سوالات توصیفی شامل واکشی لاگ‌ها و تحلیل مهندسی"""
        t_analysis_start = time.time()
        
        try:
            # استخراج داده‌های کلیدی (Context
            #TODO: حداکثر زمان مشخص کنیم جهت پیدا کردن اررورها
            faults = self.df_inverter[self.df_inverter['fa_al_code'] != 0][['device_code', 'fa_al_code', 'fa_al_message']].head(20).to_dict('records')
            # TODO: .head(20).to_dict('records')
            states = self.df_inverter['work_state'].value_counts().to_dict()
            context_data = f"وضعیت کاری فعلی اینورترها: {states}\nلیست خطاهای فعال اخیر: {faults}"
        except Exception:
            context_data = "خطا در واکشی لاگ‌های سیستم."

        desc_prompt = DESCRIPTIVE_ANALYSIS_PROMPT.format(
            user_query=query, 
            filtered_data_context=context_data
        )
        final_answer = self._call_llm(desc_prompt, model)
        
        result_payload["status"] = "success"
        result_payload["final_answer"] = final_answer
        result_payload["times"]["analysis"] = time.time() - t_analysis_start

    # ------------------------------------------
    # Main Orchestrator
    # ------------------------------------------
    def process_query(self, query, model):
        """
        متد اصلی که به عنوان هماهنگ‌کننده (Orchestrator) عمل می‌کند.
        این ساختار بسیار تمیزتر، خواناتر و قابل توسعه‌تر است.
        """
        # ساختار اولیه خروجی
        result_payload = {
            "status": "pending",
            "model_used": model,
            "route_info": {},
            "generated_code": None,
            "raw_data": None,
            "final_answer": "",
            "times": {}
        }

        # 1. اجرای مسیریاب
        route_info, routing_time = self._perform_routing(query, model)
        result_payload["route_info"] = route_info
        result_payload["times"]["routing"] = routing_time

        # بررسی قانون ۳
        if not route_info.get("is_in_schema", False):
            result_payload["status"] = "rejected"
            result_payload["final_answer"] = "متأسفم، داده‌های مربوط به این سوال در سیستم تله‌متری ثبت نشده است یا سوال نامفهوم است."
            return result_payload

        intent = route_info.get("intent", "unknown")

        # 2. مسیریابی به پایپ‌لاین مربوطه بر اساس نوع سوال
        if intent == "statistical":
            self._handle_statistical_pipeline(query, model, result_payload)
            
        elif intent == "descriptive":
            self._handle_descriptive_pipeline(query, model, result_payload)
            
        else:
            result_payload["status"] = "unknown_intent"
            result_payload["final_answer"] = "مدل نتوانست نوع سوال (آماری یا توصیفی) را به درستی تشخیص دهد."

        return result_payload


# ==========================================
# 3. Batch Execution Loop
# ==========================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_DIR = os.path.join(BASE_DIR, 'data', 'csv_files')
    
    engine = TelemetryAI(data_dir=CSV_DIR)

    query_list = [
        "وضعیت سلامت تجهیزات اصلی شامل اینورتر، پنل LV، پنل MV و استرینگ‌ها چطور است؟",
        "میانگین توان اکتیو خروجی اینورترها در بازه داده‌های موجود چقدر است؟",
        "مهم‌ترین کدهای خطای ثبت شده چه مواردی بوده‌اند؟",
        "قیمت خرید برق تضمینی برای این نیروگاه چقدر محاسبه می‌شود؟",
    ]
    
    print("\nStarting batch processing of queries and saving to file...")
    
    with open(OUTPUT_FILE_NAME, "w", encoding="utf-8") as file_handle:
        file_handle.write("=== Model Comparison Report (Modular Architecture) ===\n\n")
        
        for idx, query in enumerate(query_list, start=1):
            print(f"\n{'#'*20} PROCESSING QUERY {idx}/{len(query_list)} {'#'*20}")
            file_handle.write(f"--- Query {idx}: {query} ---\n")
            
            for model in MODELS_TO_COMPARE:
                print(f"Running on model: {model} ...")
                total_t0 = time.time()
                
                # فراخوانی ارکستراتور اصلی
                response = engine.process_query(query, model)
                
                total_time = time.time() - total_t0
                
                file_handle.write(f"Model: {model}\n")
                file_handle.write(f"Status: {response['status']}\n")
                
                route_info = response.get('route_info', {})
                file_handle.write(f"Intent: {route_info.get('intent', 'N/A')}\n")
                
                if response.get('generated_code'):
                    file_handle.write(f"Generated Code:\n{response['generated_code']}\n")
                    
                file_handle.write(f"\nFinal Answer:\n{response.get('final_answer', 'N/A')}\n")
                
                file_handle.write("\nTimes:\n")
                for step, t in response.get('times', {}).items():
                    file_handle.write(f" - {step.capitalize()}: {t:.2f} seconds\n")
                file_handle.write(f" - TOTAL TIME: {total_time:.2f} seconds\n")
                file_handle.write("-" * 50 + "\n\n")
                
                file_handle.flush()

    print(f"\nBatch processing complete. Results saved to {OUTPUT_FILE_NAME}")
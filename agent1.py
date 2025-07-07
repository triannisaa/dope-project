## DATABASE PATH

"""
🤖 DOPE-AI Super Prompt Agent
===============================
AI Agent untuk menghasilkan super prompt berdasarkan arsitektur PT Darma Henwa
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple
# from dataclasses import dataclassfrom datetime import datetime, timedelta
from enum import Enum
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback

# # Load JSON file to contextual guardrail from GitHub (raw link)
# url = "https://raw.githubusercontent.com/triannisaa/dope-project/main/context-darmahenwa.json"
# response = requests.get(url).json()
# content_context = pd.DataFrame(response).content.tolist()

class QueryType(Enum):
    CONTEXTUAL = "context"
    DATABASE = "database"
    UNCLEAR = "unclear"

@dataclass
class QueryAnalysis:
    query_type: QueryType
    confidence: float
    extracted_entities: Dict[str, str]
    reasoning: str

@dataclass
class DatabaseQuery:
    table: Optional[str]
    columns: List[str]
    filters: Dict[str, str]
    aggregation: Optional[str]
    is_valid_format: bool
    missing_fields: List[str]

class SuperPromptOutputParser(BaseOutputParser):
    """Parser untuk mengekstrak output yang terstruktur dari LLM"""
    
    def parse(self, text: str) -> Dict:
        try:
            # Coba parse sebagai JSON terlebih dahulu
            if text.strip().startswith('{') and text.strip().endswith('}'):
                return json.loads(text)
            
            # Jika bukan JSON, ekstrak informasi dengan regex
            result = {}
            
            # Ekstrak query type
            query_type_match = re.search(r'Query Type:\s*(\w+)', text, re.IGNORECASE)
            if query_type_match:
                result['query_type'] = query_type_match.group(1).lower()
            
            # Ekstrak confidence
            confidence_match = re.search(r'Confidence:\s*([\d.]+)', text)
            if confidence_match:
                result['confidence'] = float(confidence_match.group(1))
            
            # Ekstrak reasoning
            reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=\n[A-Z]|$)', text, re.DOTALL)
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(1).strip()
            
            return result
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}", "raw_text": text}

class DOPEAIAgent:
    """
    DOPE-AI Agent untuk menghasilkan super prompt berdasarkan input user
    Mengikuti arsitektur yang dijelaskan dalam dokumen PT Darma Henwa
    """
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4"):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        
        # Database schema mapping sesuai dokumen
        self.database_schema = {
            "PTDH_BCP": ["bcp", "BCP", "bengalon", "BENGALON"],
            "PTDH_ACP": ["acp", "ACP", "asam-asam", "asam" "ASAM-ASAM", "asam asam", "ASAM ASAM"],
            "PTDH_WKCP": ["wkp", "wkcp", "WKCP", "WKP" "west kintap", "WEST KINTAP"],
            "PTDH_EKCP": ["ekcp", "kcp", "EKCP", "KCP" "east kintap", "kintap", "KINTAP", "EAST KINTAP"]
        }
        
        # Keywords untuk klasifikasi
        self.contextual_keywords = ["apa", "siapa", "bagaimana", "mengapa", "kapan", "what", "who", "how", "why", "when"]
        self.database_keywords = ["berapa", "jumlah", "total", "actual", "target", "hari ini", "today", "how much", "how many"]
        
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup LangChain chains untuk berbagai fungsi"""
        
        # Chain untuk klasifikasi query
        classifier_system_prompt = """
        Anda adalah classifier yang menentukan jenis query berdasarkan arsitektur DOPE-AI untuk PT Darma Henwa.
        
        Klasifikasi query ke dalam kategori:
        1. CONTEXTUAL: Pertanyaan tentang informasi umum (apa, siapa, bagaimana, mengapa, kapan)
        2. DATABASE: Pertanyaan tentang data numerik/kuantitatif (berapa, jumlah, total)
        3. UNCLEAR: Query yang tidak jelas atau ambigu
        
        Berikan response dalam format JSON:
        {{
            "query_type": "contextual/database/unclear",
            "confidence": 0.0-1.0,
            "reasoning": "penjelasan mengapa diklasifikasikan demikian",
            "extracted_entities": {{
                "site": "nama site jika disebutkan",
                "metric": "metrik yang ditanyakan",
                "time_period": "periode waktu"
            }}
        }}
        """
        
        classifier_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(classifier_system_prompt),
            HumanMessagePromptTemplate.from_template("Query: {query}")
        ])
        
        self.classifier_chain = LLMChain(
            llm=self.llm,
            prompt=classifier_template,
            output_parser=SuperPromptOutputParser()
        )
        
        # Chain untuk format checking database query
        db_checker_system_prompt = """
        Anda adalah database query format checker untuk sistem DOPE-AI PT Darma Henwa.
        
        Database schema yang tersedia:
        - PTDH_BCP: bcp, BCP, bengalon, BENGALON,
        - PTDH_ACP: acp, ACP, asam-asam, asam ASAM-ASAM, asam asam, ASAM ASAM,
        - PTDH_WKCP: wkp, wkcp, WKCP, WKP west kintap, WEST KINTAP,
        - PTDH_EKCP: ekcp, kcp, EKCP, KCP east kintap, kintap, KINTAP, EAST KINTAP

        
        Analisis apakah query memiliki informasi yang cukup untuk membuat SQL query
        Format yang dibutuhkan: [PRODUCTION] + [SITE] + [PERIODE] + [OPERATION]

        dengan ketentuan site berupa kata PTDH-"site" (misal: PTDH_ACP), production adalah nama material yang ingin 
        dilihat (misal: waste, coal mining atau coal hauling), periode adakah waktu yang ditentukan dan ubahlah format menjadi ketentuan dd-mm-yyyy
        (misal: 01-01-2025) sehingga jika prompt berkutat seperti hari ini, kemarin atau lusa maka terjemahkanlah berdasarkan tanggal hari ini, 
        serta jika diminta dalam bentuk bulanan buatlah dalam bentuk range seperti "01-01-2025 to 01-02-2025" dan operation adalah operasi 
        matematika yang dibutuhkan seperti sum, count, min, max, mean dan lainnya
                
        Response format JSON:
        {{
            "is_valid_format": true/false,
            "missing_fields": ["field1", "field2"],
            "suggested_table": "nama_table",
            "suggested_columns": ["col1", "col2"],
            "sql_structure": "struktur SQL yang disarankan"
        }}
        """
        
        db_checker_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(db_checker_system_prompt),
            HumanMessagePromptTemplate.from_template("Database Query: {query}")
        ])
        
        self.db_checker_chain = LLMChain(
            llm=self.llm,
            prompt=db_checker_template,
            output_parser=SuperPromptOutputParser()
        )
        
        # Chain untuk interactive callback
        callback_system_prompt = """
        Anda adalah interactive callback assistant untuk DOPE-AI yang membantu user memperbaiki query.
        
        Tugas Anda:
        1. Identifikasi informasi yang hilang dari query database
        2. Berikan pertanyaan follow-up yang spesifik dan mudah dipahami
        3. Berikan contoh format yang benar
        
        Response format:
        {{
            "callback_message": "pertanyaan follow-up yang ramah",
            "examples": ["contoh1", "contoh2"],
            "required_info": ["info yang dibutuhkan"]
        }}
        """
        
        callback_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(callback_system_prompt),
            HumanMessagePromptTemplate.from_template(
                "Original query: {query}\nMissing information: {missing_fields}"
            )
        ])
        
        self.callback_chain = LLMChain(
            llm=self.llm,
            prompt=callback_template,
            output_parser=SuperPromptOutputParser()
        )

        # Chain untuk generating super prompt
        super_prompt_system = """
        Anda adalah Super Prompt Generator untuk DOPE-AI, bertugas menghasilkan prompt teroptimasi berdasarkan query pengguna.

        Tugas Anda:
        1. Mengidentifikasi apakah query termasuk ke dalam jalur Tool A (database) atau Tool B (contextual).
        2. Jika masuk kedalam tool A bangun super prompt dalam format:
            **"berapa jumlah [OPERATION] [PRODUCTION] disite [SITE] tanggal [PERIODE]"**
            #Panduan Penyusunan:
            
            - **[OPERATION]**:
              - Kenali sinonim dan parafrase dari operasi matematika.
              - Gunakan pemetaan seperti berikut:
                - "jumlah", "jumlahkan", "jumlahan", "hitung", "hitunglah" → `sum`
                - "berapa banyak", "berapa kali", "total item", "jumlah entri" → `count`
                - "rata-rata", "average", "rerata" → `mean`
                - "tertinggi", "maksimal", "paling tinggi" → `max`
                - "terendah", "minimal", "paling rendah" → `min`
            
            - **[PRODUCTION]**:
              - Material atau aktivitas produksi seperti: `waste`, `coal mining`, `coal hauling`
                - Kenali apakah material sudah sesuai aktivitas produksi, kemudian kenali tipe yang ini dilihat melalui pemetaan berikut:
                  - "aktual", "Actual", "aktual", "lapangan" → `Actual`
                  - "budget", "Budget", "plan" → `budget`
              - Atau aktivitas delay seperti : `rain`,`slippery`,`rainfall`,`rain_frequency`
                  - "aktual", "Actual", "aktual", "lapangan" → `Actual`
                  - "budget", "Budget", "plan" → `budget`
                  
            - **[SITE]**:
              - Ubah site jadi format `PTDH_[SITE_UC]`, contoh: `PTDH_ACP`
              
            - **[PERIODE]**:
              - "hari ini" → '{today_date}'
              - "kemarin" → tanggal kemarin dalam format dd-mm-yyyy
              - "besok" → tanggal besok dalam format dd-mm-yyyy
              - Bulanan → format range `dd-mm-yyyy to dd-mm-yyyy`.
                
        3. Jika masuk kedalam tool B bangun super prompt dalam format:
            Jika query tidak berkaitan dengan data kuantitatif, susun prompt dalam format bebas yang jelas dan lengkap sesuai konteks.      


        🔴 **OUTPUT FORMAT HARUS DALAM JSON DENGAN FIELD BERIKUT (WAJIB ADA SEMUA):**
        {{
          "super_prompt": "prompt yang dioptimasi",
          "tool_routing": "Tool A/Tool B/Tool C",
          "optimization_notes": "penjelasan ringkas optimasi yang Anda lakukan"
        }}
        
        🔴 **IMPORTANT:**
        - Pastikan output dalam format JSON valid.
        - Jangan hilangkan field apapun meskipun isinya kosong, tetap kembalikan string kosong jika tidak ada nilai.
        
        Contoh output:
        {{
          "super_prompt": "berapa jumlah sum coal mining di site PTDH_ACP tanggal 04-07-2025",
          "tool_routing": "Tool A",
          "optimization_notes": "Query diklasifikasikan sebagai database dengan operasi sum dan site PTDH_ACP pada tanggal hari ini"
        }}
        
        Jika ada bagian yang tidak diketahui, isi dengan string kosong "" untuk field tersebut.
        """
        # super_prompt_system = prompt1

        super_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(super_prompt_system),
            HumanMessagePromptTemplate.from_template(
                "Original query: {query}\nQuery analysis: {analysis}\nUser context: {context}"
            )
        ])
        
        self.super_prompt_chain = LLMChain(
            llm=self.llm,
            prompt=super_prompt_template,
            output_parser=SuperPromptOutputParser()
        )

    # === 2. Fungsi guardrail: cek apakah prompt cocok dengan konteks
    def is_prompt_allowed(prompt: str) -> bool:
        prompt_lower = prompt.lower()
        return any(str(context).lower() in prompt_lower for context in content_context)
    
    def handoff(self, user_input: str) -> Dict:
        """
        Fungsi handoff utama yang memproses input user
        Mengikuti flowchart yang diberikan
        """
        try:
            with get_openai_callback() as cb:
                # Step 1: Klasifikasi query
                print("\n🔍 Menganalisis query...")
                classification_result = self.classifier_chain.run(query=user_input)
                
                if isinstance(classification_result, dict) and 'error' not in classification_result:
                    query_type = classification_result.get('query_type', 'unclear')
                    confidence = classification_result.get('confidence', 0.0)
                    
                    print(f"📊 Klasifikasi: {query_type.upper()} (confidence: {confidence:.2f})")
                    print(f"💭 Reasoning: {classification_result.get('reasoning', 'N/A')}")
                    
                    # Step 2: Route berdasarkan klasifikasi
                    if query_type == 'contextual':
                        return self._handle_contextual_path(user_input, classification_result)
                    
                    elif query_type == 'database':
                        return self._handle_database_path(user_input, classification_result)
                    
                    else:  # unclear
                        return self._handle_interactive_callback(user_input, "Query tidak jelas atau ambigu")
                
                else:
                    return {
                        "status": "error",
                        "message": "Gagal mengklasifikasi query",
                        "details": classification_result
                    }
                    
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Error dalam handoff: {str(e)}"
            }
    
    def _handle_contextual_path(self, query: str, analysis: Dict) -> Dict:
        """Handle contextual queries (Tool B path)"""
        print("📚 Memproses contextual query...")
        
        # Generate super prompt untuk contextual query
        super_prompt_result = self.super_prompt_chain.run(
            query=query,
            analysis=json.dumps(analysis),
            context="contextual_query"
        )
        
        return {
            "status": "success",
            "path": "contextual",
            "tool": "Tool B - Vector Search",
            "original_query": query,
            "analysis": analysis,
            "super_prompt": super_prompt_result.get('super_prompt'),
            "optimization_notes": super_prompt_result.get('optimization_notes', ''),
            "next_action": "Route to vector database search"
        }

    def _handle_database_path(self, query: str, analysis: Dict) -> Dict:
        """Handle database queries (Tool A path)"""
        print("🗄️ Memproses database query...")
        
        # Check format database query
        db_check_result = self.db_checker_chain.run(query=query)
        
        if isinstance(db_check_result, dict) and db_check_result.get('is_valid_format', False):
            # Format sudah sesuai, generate super prompt
            super_prompt_result = self.super_prompt_chain.run(
                query=query,
                analysis=json.dumps(analysis),
                context="database_query",
                today_date = datetime.today().date().strftime("%d-%m-%Y")
            )
            
            return {
                "status": "success",
                "path": "database",
                "tool": "Tool A - SQL Query",
                "original_query": query,
                "analysis": analysis,
                "db_analysis": db_check_result,
                "super_prompt": super_prompt_result.get('super_prompt'),
                "optimization_notes": super_prompt_result.get('optimization_notes', ''),
                "next_action": "Execute SQL query"
            }
        else:
            # Format tidak sesuai, lakukan interactive callback
            missing_fields = db_check_result.get('missing_fields', [])
            return self._handle_interactive_callback(query, f"Format database query tidak lengkap. Missing: {', '.join(missing_fields)}")
    
    def _handle_interactive_callback(self, query: str, reason: str) -> Dict:
        """Handle interactive callback untuk memperbaiki query"""
        print("💬 Membutuhkan clarification dari user...")
        
        callback_result = self.callback_chain.run(
            query=query,
            missing_fields=reason
        )
        
        return {
            "status": "callback_required",
            "path": "interactive_callback", 
            "tool": "Tool C - Prompt Rewriter",
            "original_query": query,
            "callback_message": callback_result.get('callback_message', 'Mohon berikan informasi lebih lengkap.'),
            "examples": callback_result.get('examples', []),
            "required_info": callback_result.get('required_info', []),
            "next_action": "Wait for user clarification"
        }
    
    def process_callback_response(self, original_query: str, additional_info: str) -> Dict:
        """Process response dari user setelah interactive callback"""
        combined_query = f"{original_query} {additional_info}"
        print("\n🔄 Memproses informasi tambahan dari user...")
        return self.handoff(combined_query)
    
    def generate_final_super_prompt(self, session_data: Dict) -> str:
        """Generate final super prompt berdasarkan seluruh session"""
        if session_data.get('status') == 'success':
            super_prompt = session_data.get('super_prompt', session_data.get('original_query'))
            tool = session_data.get('tool', 'Unknown')
            
            final_prompt = f"""
╔═══════════════════════════════════════════════════════════════╗
║                    🤖 DOPE-AI SUPER PROMPT                    ║
╠═══════════════════════════════════════════════════════════════╣
║ Tool: {tool:<55} ║
║ Path: {session_data.get('path', 'Unknown'):<55} ║
╠═══════════════════════════════════════════════════════════════╣
║ 📝 OPTIMIZED QUERY:                                           ║
║ {super_prompt:<59} ║
║                                                               ║
║ 🎯 OPTIMIZATION NOTES:                                        ║
║ {session_data.get('optimization_notes', 'Standard optimization'):<59} ║
║                                                               ║
║ ➡️ NEXT ACTION:                                                ║
║ {session_data.get('next_action', 'Process query'):<59} ║
╚═══════════════════════════════════════════════════════════════╝
"""
            return final_prompt
        else:
            return f"❌ Super prompt generation failed: {session_data.get('message', 'Unknown error')}"


def main():
    """Main function - Interactive DOPE-AI Agent"""
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                  🤖 DOPE-AI SUPER PROMPT AGENT               ║")
    print("║                  PT Darma Henwa Mining Assistant             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Setup API Key
    api_key = "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"
    if not api_key:
        print("❌ Tidak dapat melanjutkan tanpa API Key")
        return
    
    # Initialize agent
    try:
        print("\n🚀 Menginisialisasi DOPE-AI Agent...")
        agent = DOPEAIAgent(openai_api_key=api_key)
        print("✅ Agent siap digunakan!")
    except Exception as e:
        print(f"❌ Error menginisialisasi agent: {str(e)}")
        return
    
    # Interactive session
    print("\n" + "="*60)
    print("💬 Mulai sesi interaktif")
    print("💡 Ketik 'exit' atau 'quit' untuk keluar")
    print("💡 Ketik 'help' untuk bantuan")
    print("="*60)
    
    session_history = []
    
    while True:
        try:
            # Get user input
            print("\n🎯 Masukkan query Anda:")
            user_query = input(">>> ").strip()
            
            # Handle special commands
            if user_query.lower() in ['exit', 'quit', 'keluar']:
                print("\n👋 Terima kasih telah menggunakan DOPE-AI Agent!")
                break
            
            if user_query.lower() in ['help', 'bantuan']:
                print("\n📚 BANTUAN PENGGUNAAN:")
                print("• Contextual Query: 'Apa visi misi PT Darma Henwa?'")
                print("• Database Query: 'Berapa coal mined actual hari ini di Bengalon?'")
                print("• Query dapat dalam Bahasa Indonesia atau English")
                print("• Agent akan meminta clarification jika informasi kurang lengkap")
                continue
            
            if not user_query:
                print("⚠️ Query tidak boleh kosong!")
                continue
            
            # Process query
            print(f"\n🔄 Memproses: '{user_query}'")
            print("-" * 50)
            
            result = agent.handoff(user_query)
            session_history.append({"query": user_query, "result": result})
            
            # Handle different result types
            if result['status'] == 'callback_required':
                # Interactive callback - minta input tambahan dari user
                print(f"\n💬 {result['callback_message']}")
                
                if result.get('examples'):
                    print("\n💡 Contoh format:")
                    for i, example in enumerate(result['examples'], 1):
                        print(f"   {i}. {example}")
                
                if result.get('required_info'):
                    print(f"\n📋 Informasi yang dibutuhkan: {', '.join(result['required_info'])}")
                
                # Minta input tambahan
                print("\n🎯 Berikan informasi tambahan:")
                additional_info = input(">>> ").strip()
                
                if additional_info:
                    # Process dengan informasi tambahan
                    print(f"\n🔄 Memproses ulang dengan informasi tambahan...")
                    result = agent.process_callback_response(user_query, additional_info)
                    session_history[-1]["additional_info"] = additional_info
                    session_history[-1]["final_result"] = result
                else:
                    print("⚠️ Tidak ada informasi tambahan diberikan")
                    continue
            
            # Show final result
            if result['status'] == 'success':
                final_prompt = agent.generate_final_super_prompt(result)
                print(final_prompt)
                
                # Tanya apakah user ingin melihat detail
                show_detail = input("\n🔍 Tampilkan detail analisis? (y/n): ").strip().lower()
                if show_detail in ['y', 'yes', 'ya']:
                    print(f"\n📊 DETAIL ANALISIS:")
                    print(f"• Original Query: {result['original_query']}")
                    print(f"• Classification: {result.get('analysis', {}).get('reasoning', 'N/A')}")
                    if 'db_analysis' in result:
                        print(f"• SQL Structure: {result['db_analysis'].get('sql_structure', 'N/A')}")
                
            elif result['status'] == 'error':
                print(f"❌ Error: {result['message']}")
                if 'details' in result:
                    print(f"Detail: {result['details']}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Proses dibatalkan oleh user")
            break
        except Exception as e:
            print(f"\n❌ Error tidak terduga: {str(e)}")
            continue
    
    # Session summary
    if session_history:
        print(f"\n📊 RINGKASAN SESI:")
        print(f"Total queries diproses: {len(session_history)}")
        successful_queries = sum(1 for item in session_history 
                               if item.get('final_result', item['result']).get('status') == 'success')
        print(f"Berhasil diproses: {successful_queries}")
        print(f"Membutuhkan callback: {sum(1 for item in session_history if 'additional_info' in item)}")

if __name__ == "__main__":
    main()
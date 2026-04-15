"""
main.py — CLI Entry Point for SIT Nagpur Student Chatbot
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from faq_engine      import FAQEngine
from context_manager import ConversationContext
from chatbot_core    import get_response
from analytics       import analyze_logs, export_csv, get_log_count

BLUE   = '\033[94m'; GREEN = '\033[92m'; YELLOW = '\033[93m'
CYAN   = '\033[96m'; BOLD  = '\033[1m';  DIM    = '\033[2m'; RESET = '\033[0m'


def banner():
    print(f"""{BLUE}
╔═══════════════════════════════════════════════════╗
║  🎓  SIT Nagpur — Student Support Chatbot  🎓     ║
║  Symbiosis Institute of Technology, Nagpur        ║
║  Intelligent FAQ · Intent · Entity · Context      ║
╚═══════════════════════════════════════════════════╝{RESET}""")


def run_chat():
    engine  = FAQEngine()
    context = ConversationContext()
    print(f"{CYAN}  Chat started. Type 'quit' to exit | 'reset' to clear context{RESET}\n")

    while True:
        try:
            raw = input(f"{BOLD}  You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break

        if not raw: continue
        if raw.lower() == 'quit': print(f"{GREEN}  👋 Goodbye!{RESET}"); break
        if raw.lower() == 'reset':
            context.reset(); print(f"{YELLOW}  🔄 Context cleared.{RESET}"); continue

        r = get_response(raw, engine, context)
        print(f"\n{GREEN}  Bot:{RESET} {r['answer']}\n")


def run_analytics():
    print(analyze_logs())
    if input(f"{YELLOW}  Export to CSV? (y/n): {RESET}").strip().lower() == 'y':
        print(f"  {export_csv()}")


def main():
    banner()
    while True:
        print(f"""
{BLUE}  ══════════════════════════════{RESET}
  {BOLD}MAIN MENU{RESET}
  {YELLOW}[1]{RESET} 💬 Start Chatbot
  {YELLOW}[2]{RESET} 📊 View Analytics
  {YELLOW}[0]{RESET} 🚪 Exit
{BLUE}  ══════════════════════════════{RESET}""")
        choice = input(f"{YELLOW}  Select: {RESET}").strip()
        if   choice == '1': run_chat()
        elif choice == '2': run_analytics()
        elif choice == '0': print(f"{GREEN}  Goodbye!{RESET}"); break
        else: print(f"  Invalid option.")


if __name__ == "__main__":
    main()

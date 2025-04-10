import uvicorn
import os
import sys
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

def print_header():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"\n{Fore.CYAN}╔════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║ {Fore.YELLOW}Age Prediction Visualization Tool {Fore.CYAN}                      ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}╚════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

def print_instructions():
    print(f"\n{Fore.GREEN}Starting the FastAPI server...{Style.RESET_ALL}")
    print(f"\n{Fore.WHITE}Once the server is running, open your browser and go to:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}http://localhost:8000{Style.RESET_ALL}")
    print(f"\n{Fore.WHITE}Press {Fore.RED}CTRL+C{Fore.WHITE} to stop the server.{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}═════════════════════════════════════════════════════════════{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        # Display header and instructions
        print_header()
        print_instructions()
        
        # Start the server
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Server stopped.{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1) 
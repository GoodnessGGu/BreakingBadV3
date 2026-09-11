import os
import json
import logging
from dotenv import load_dotenv
from forex_mcp_client import IQForexMCPClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestMCP")

def main():
    load_dotenv()
    client = IQForexMCPClient()
    logger.info("Initializing IQ Option MCP Forex client...")
    
    success = client.initialize()
    if not success:
        logger.error("Failed to initialize MCP session. Check token permissions.")
        return
        
    logger.info("=== AVAILABLE TOOLS ===")
    for name, tool in client.tools.items():
        print(f"\nTool: {name}")
        print(f"Description: {tool.get('description', '')}")
        print("Parameters:")
        print(json.dumps(tool.get("inputSchema", {}), indent=2))
        
    logger.info("\n=== TESTING LIST BALANCES ===")
    balances = client.list_balances()
    print("Balances result:", json.dumps(balances, indent=2))
    
    logger.info("\n=== TESTING LIST POSITIONS ===")
    positions = client.list_positions()
    print("Positions result:", json.dumps(positions, indent=2))

if __name__ == "__main__":
    main()

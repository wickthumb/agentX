import logging
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentX.agent_x import AgentX

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_web(query: str) -> Dict[str, str]:
    """Search the web for information.
    
    Args:
        query (str): The search query
        
    Returns:
        Dict[str, str]: Search results
    """
    # Mock implementation
    return {"results": f"Found results for: {query}"}

def save_note(content: str) -> bool:
    """Save a note to storage.
    
    Args:
        content (str): The note content
        
    Returns:
        bool: True if saved successfully
    """
    # Mock implementation
    logger.info(f"Saving note: {content}")
    return True

def main():
    # Define available tools
    tools = [search_web, save_note]
    
    # Create mission
    mission = "Search for information about Python and save key points as notes"
    
    # Initialize agent
    agent = AgentX(
        tools=tools,
        mission=mission,
        enable_logging=True,
        max_steps=5
    )
    
    # Execute mission
    result = agent.execute_mission()
    
    logger.info(f"Mission completed: {result}")

if __name__ == "__main__":
    main()

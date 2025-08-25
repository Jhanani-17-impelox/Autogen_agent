import uvicorn
import autogen
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Dynamic Multi-Agent Travel Assistant API",
    description="An API to configure and interact with a team of AI agents for booking travel.",
    version="1.0.0"
)

# --- Global State for Configuration and Agents ---
AGENT_SYSTEM = {}

# --- Pydantic Models for API Payloads ---

class LLMConfig(BaseModel):
    model: str
    api_key: str
    api_type: Optional[str] = None
    api_version: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7

class AgentConfig(BaseModel):
    name: str
    type: str
    system_message: str
    llm_config: LLMConfig
    data_schema: Optional[Dict[str, Dict[str, List[str]]]] = None

class GroupChatConfig(BaseModel):
    system_message: str

class SystemConfigurationPayload(BaseModel):
    agents: List[AgentConfig]
    group_chat_manager: GroupChatConfig

class ChatPayload(BaseModel):
    query: str
    session_id: str

# --- In-memory Session Management ---
conversation_history: Dict[str, List[Dict]] = {}

# --- Helper Functions ---

def get_agent_class(agent_type_str: str):
    """Dynamically get an agent class from the autogen library."""
    if agent_type_str == "AssistantAgent":
        return autogen.AssistantAgent
    elif agent_type_str == "UserProxyAgent":
        return autogen.UserProxyAgent
    elif agent_type_str == "ConversableAgent":
        return autogen.ConversableAgent
    else:
        raise ValueError(f"Unknown agent type: {agent_type_str}")

def initialize_agents_from_config(config: SystemConfigurationPayload):
    """Initializes the multi-agent system from the provided configuration."""
    global AGENT_SYSTEM
    agents = []
    agent_names = []

    for agent_conf in config.agents:
        AgentClass = get_agent_class(agent_conf.type)

        llm_config = {
            "config_list": [{
                "model": agent_conf.llm_config.model,
                "api_key": agent_conf.llm_config.api_key,
                "api_type": agent_conf.llm_config.api_type,
                "api_version": agent_conf.llm_config.api_version,
                "base_url": agent_conf.llm_config.base_url,
            }],
            "temperature": agent_conf.llm_config.temperature,
        }

        system_message = agent_conf.system_message
        if agent_conf.data_schema:
            schema_str = json.dumps(agent_conf.data_schema, indent=2)
            system_message += f"\n\nHere is the data schema you MUST follow to gather information:\n{schema_str}"

        agent_kwargs = { "name": agent_conf.name, "system_message": system_message }

        if agent_conf.type == "UserProxyAgent":
            agent_kwargs.update({
                "human_input_mode": "NEVER",
                "max_consecutive_auto_reply": 10,
                "is_termination_msg": lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
                "code_execution_config": False,
            })
        else:
            agent_kwargs["llm_config"] = llm_config

        agent = AgentClass(**agent_kwargs)
        agents.append(agent)
        agent_names.append(agent_conf.name)

    groupchat = autogen.GroupChat(agents=agents, messages=[], max_round=50)

    manager_llm_config = next((agent.llm_config for agent in agents if hasattr(agent, 'llm_config') and agent.llm_config), None)
    if not manager_llm_config:
        raise ValueError("No LLM configuration found for any agent, cannot configure manager.")

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=manager_llm_config,
        system_message=config.group_chat_manager.system_message.format(agent_names=", ".join(agent_names))
    )

    AGENT_SYSTEM = {
        'manager': manager,
        'groupchat': groupchat,
        'user_proxy': next(agent for agent in agents if isinstance(agent, autogen.UserProxyAgent)),
        'agents': agents
    }
    logger.info("Agent system initialized successfully.")

# --- API Endpoints ---
@app.post("/configure")
async def configure_system(payload: SystemConfigurationPayload):
    try:
        initialize_agents_from_config(payload)
        return {"status": "success", "message": "System configured successfully."}
    except Exception as e:
        logger.error(f"Configuration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to configure system: {str(e)}")

# ================== FIX STARTS HERE: REBUILT /chat ENDPOINT ==================
@app.post("/chat")
async def chat_with_agents(payload: ChatPayload):
    """
    This endpoint now uses `max_turns=1` to enforce a single turn of conversation,
    preventing runaway loops and waiting for the user's next API call.
    """
    if 'manager' not in AGENT_SYSTEM:
        raise HTTPException(status_code=400, detail="System not configured. Please call the /configure endpoint first.")

    session_id = payload.session_id
    user_query = payload.query
    user_proxy = AGENT_SYSTEM['user_proxy']
    manager = AGENT_SYSTEM['manager']
    groupchat = AGENT_SYSTEM['groupchat']

    # Step 1: Load the history for this session. This is crucial for continuing conversations.
    groupchat.messages = conversation_history.get(session_id, [])

    # Step 2: Initiate the chat. The `max_turns=1` parameter is the key fix.
    # It tells AutoGen: "Let the user speak, let ONE agent reply, and then stop."
    # This completely prevents the internal agent loop.
    await user_proxy.a_initiate_chat(
        manager,
        message=user_query,
        max_turns=1,
    )

    # Step 3: The groupchat.messages list is now updated with the user's query
    # and the agent's single response. We save this new state.
    conversation_history[session_id] = groupchat.messages

    # Step 4: Extract the last message, which is the agent's reply, to send back to the user.
    agent_response_message = groupchat.messages[-1]
    agent_response_content = agent_response_message.get("content", "").strip()

    # Step 5: Check if the conversation is complete.
    is_complete = "terminate" in agent_response_content.lower()

    return {
        "session_id": session_id,
        "response": agent_response_content,
        "is_complete": is_complete,
        "history": groupchat.messages
    }
# ============================== FIX ENDS HERE ===============================

# --- To run the app ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


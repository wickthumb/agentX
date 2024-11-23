# agentxai

A lightweight, but effective autonomous agent framework powered by X.AI's Grok model.

## Features

- **AgentX**: An autonomous agent that can execute complex missions using provided tools
  - `tools`: List of callable functions the agent can use to complete its mission
    - can be another agent, assuming the agent lives in a function
  - `mission`: String describing the goal or task to accomplish
  - `exit_function`: Optional callback function executed upon mission completion
    - default is a success bool
  - `system_prompt`: Customizable prompt to guide agent behavior (default: autonomous goal-focused agent)
  - `model`: AI model selection (default: "grok-beta")
  - `enable_logging`: Toggle detailed logging (default: False)
  - `max_steps`: Maximum execution steps before termination (default: 10)
- **XAI**: A robust client for interacting with X.AI's Grok model and other supported AI models
- **Model Selection**: Choose from three AI models to power your agents
  - `grok-beta`: X.AI's Grok model
  - `gpt-4o`: OpenAI's GPT-4 optimized for performance
  - `gpt-4o-mini`: A lightweight version of GPT-4 optimized for minimal resource usage
- Cost tracking and token usage monitoring
- Automatic retries and error handling
- JSON response validation
- Configurable logging

## Model Selection

AgentXAI supports three AI models, each with different capabilities and cost structures. You can select the desired model when initializing `AgentX` or `XAI`.

### Available Models

1. **grok-beta**
   - **Provider**: X.AI
   - **Cost**:
     - Input Tokens: $5.00 per 1M tokens
     - Output Tokens: $15.00 per 1M tokens

2. **gpt-4o**
   - **Provider**: OpenAI
   - **Cost**:
     - Input Tokens: $2.50 per 1M tokens
     - Output Tokens: $10.00 per 1M tokens

3. **gpt-4o-mini**
   - **Provider**: OpenAI
   - **Cost**:
     - Input Tokens: $0.150 per 1M tokens
     - Output Tokens: $0.600 per 1M tokens

### Selecting a Model

When initializing `AgentX` or `XAI`, specify the `model` parameter to select the desired AI model.

#### Example: Initializing AgentX with `gpt-4o-mini`

example agent:

```python
from agentX.agent_x import AgentX

def my_function(input: str) -> str:
    """
    A simple example function that can be passed to AgentX as a tool.

    Args:
        input (str): The input string to process

    Returns:
        str: The unmodified input string
    """
    # do something with input
    return input

def exit_function(message: str) -> int:
    """
    A simple example exit function that can be passed to AgentX.

    Args:
        message (str): A string containing a number to convert

    Returns:
        int: The string converted to an integer
    """
    try:
        return int(message)
    except ValueError:
        return -1

agent = AgentX(
    model="grok-beta",
    tools=[my_function],
    mission="""
    Do something with

    #1 setup
     - some useful details
    #2 do something
    - some details
    #3 last thing to do
    - some details

    success criteria:
      -  that something is done
      -  it is done well
    """,
    exit_function=exit_function,
)
```

## Installation

```bash
pip install agentxai
```

## ENV Variables


```
XAI_API_KEY=
```

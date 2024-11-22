# agentxai

A powerful autonomous agent framework powered by X.AI's Grok model.

## Features

- **AgentX**: An autonomous agent that can execute complex missions using provided tools
  - `tools`: List of callable functions the agent can use to complete its mission
    - can be another agent, assuming the agent lives in a funciton
  - `mission`: String describing the goal or task to accomplish
  - `exit_function`: Optional callback function executed upon mission completion
    - default is a success bool
  - `system_prompt`: Customizable prompt to guide agent behavior (default: autonomous goal-focused agent)
  - `model`: AI model selection (default: "grok-beta")
  - `enable_logging`: Toggle detailed logging (default: False)
  - `max_steps`: Maximum execution steps before termination (default: 10)
- **XAI**: A robust client for interacting with X.AI's Grok model
- Cost tracking and token usage monitoring
- Automatic retries and error handling
- JSON response validation
- Configurable logging

example agent:

```python

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

## ENV

```
XAI_API_KEY=
```

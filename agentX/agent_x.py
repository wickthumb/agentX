import logging
from typing import Any, Callable, Dict, List, Optional

from xAI.x_ai import XAI
# from openAi.o_ai import OAI

logger = logging.getLogger(__name__)


class AgentX(XAI):
    """
    An automated agent that utilizes XAI to achieve goals by executing tools.
    Inherits from the XAI base class for chat functionality.
    """

    def __init__(
        self,
        tools: List[Callable],
        mission: str,
        exit_function: Optional[Callable[..., Any]] = None,
        system_prompt: str = "You are an autonomous agent focused on completing specific goals. Think carefully about each step.",
        model: str = "grok-beta",
        enable_logging: bool = False,
        max_steps: int = 10
    ) -> None:
        """
        Initialize the AgentX instance.

        Args:
            tools (List[Callable]): Available tools the agent can utilize.
            mission (str): Description of the mission or goal.
            exit_function (Optional[Callable[..., Any]]): Function to execute upon mission completion.
            system_prompt (str): System prompt guiding the agent's behavior.
            model (str): AI model used for chat interactions.
            enable_logging (bool): Flag to enable or disable logging.
            max_steps (int): Maximum execution steps before termination.
        """
        super().__init__(model=model, system=system_prompt, enable_logging=enable_logging)
        self.tools = {tool.__name__: tool for tool in tools}
        self.mission = mission
        self.exit_function = exit_function
        self.max_steps = max_steps
        self.step_count = 0
        self.completed = False
        self.result = None

        self.tool_descriptions = self._generate_tool_descriptions()

    def _generate_tool_descriptions(self) -> Dict[str, str]:
        """Generate descriptions for available tools."""
        descriptions = {
            name: tool.__doc__ or "No description available." for name, tool in self.tools.items()}
        return descriptions

    def _format_tool_list(self) -> str:
        """Format the list of tools for prompts."""
        return "Available tools:\n" + "\n".join(f"- {name}: {desc}" for name, desc in self.tool_descriptions.items())

    def _get_next_action(self) -> Dict[str, Any]:
        """
        Request the next action from the AI.

        Returns:
            Dict[str, Any]: Details of the next action.
        """
        prompt = (
            f"Mission: {self.mission}\n"
            f"Current step: {self.step_count + 1}\n\n"
            f"{self._format_tool_list()}\n\n"
            "What is the next best action to take? Respond with JSON containing:\n"
            "{\n"
            '    "action_type": "tool",\n'
            '    "name": "str",\n'
            '    "parameters": {},\n'
            '    "reasoning": "str"\n'
            "}"
        )
        response = self.chat(prompt, expected_format={
            "action_type": "str",
            "name": "str",
            "parameters": "dict",
            "reasoning": "str"
        })
        if self.enable_logging:
            logger.debug(f"Next action response: {response}")
        return response

    def _check_completion(self) -> bool:
        """
        Determine if the mission is complete.

        Returns:
            bool: True if complete, else False.
        """
        prompt = (
            f"Mission: {self.mission}\n"
            "Review all previous actions and determine if the mission is complete.\n"
            "Respond with JSON: {\"is_complete\": bool, \"reason\": \"str\"}"
        )
        response = self.chat(prompt, expected_format={
                             "is_complete": "bool", "reason": "str"})
        if self.enable_logging:
            logger.debug(f"Completion check response: {response}")
        return response.get("is_complete", False)

    def execute_mission(self) -> Any:
        """
        Execute the mission using available tools.

        Returns:
            Any: Result from exit_function if provided and mission completed, else completion status.
        """
        if self.enable_logging:
            logger.info(f"Starting mission: {self.mission}")

        while self.step_count < self.max_steps and not self.completed:
            try:
                action = self._get_next_action()
                if self.enable_logging:
                    logger.debug(f"Action {self.step_count + 1}: {action}")

                action_type = action.get("action_type", "").lower()
                name = action.get("name")
                parameters = action.get("parameters", {})

                if action_type == "tool":
                    self._execute_tool(name, parameters)
                else:
                    logger.error(f"Unknown action type: {action_type}")
                    return False

                self.step_count += 1
                self.completed = self._check_completion()

            except Exception as e:
                logger.error(f"Error during mission execution: {
                             e}", exc_info=True)
                return False

        if self.completed:
            if self.enable_logging:
                logger.info("Mission completed successfully.")
            if self.exit_function:
                try:
                    self.result = self._execute_exit()
                except Exception as e:
                    if self.enable_logging:
                        logger.error(f"Error executing exit function: {
                                     e}", exc_info=True)
                    return False
            return self.result if self.result is not None else self.completed

        if self.enable_logging:
            logger.warning(
                f"Mission not completed within maximum allowed steps ({self.max_steps}).")
        return False

    def _execute_exit(self) -> Any:
        """Execute the exit function and return its result.

        Returns:
            Any: Result from the exit function.
        """
        if self.exit_function:
            parameters = self.chat(f"What parameters should be passed to {self.exit_function.__name__}? Use its schema: {
                                   self.exit_function.__annotations__}, here is the documentation: {self.exit_function.__doc__}", expected_format={"parameters": "dict"})["parameters"]
            if self.enable_logging:
                logger.info(
                    f"Executing exit function with parameters: {parameters}")
            result = self.exit_function(**parameters)
            return result
        else:
            logger.warning(
                "Exit action requested but no exit_function provided.")
            return None

    def _execute_tool(self, name: str, parameters: Dict[str, Any]) -> None:
        """Execute a specified tool."""
        tool = self.tools.get(name)
        if not tool:
            if self.enable_logging:
                logger.error(f"Tool '{name}' not found.")
            raise ValueError(f"Unknown tool: {name}")

        if self.enable_logging:
            logger.info(f"Executing tool '{name}' with parameters: {parameters}")
        result = tool(**parameters)
        if self.enable_logging:
            logger.debug(f"Tool '{name}' result: {result}")

        # Add tool execution result as system message
        self.messages.append({
            "role": "system",
            "content": f"Tool '{name}' result: {result}"
        })

    def get_messages(self, n: int = 2) -> str:
        """
        Retrieve the last n messages from the conversation.

        Args:
            n (int): Number of recent messages to retrieve.

        Returns:
            str: Concatenated content of the last n messages.
        """
        relevant_messages = self.messages[-n:] if len(
            self.messages) > n else self.messages
        return "\n".join(msg['content'] for msg in relevant_messages if 'content' in msg)

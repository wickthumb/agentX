import os
import requests
import json
import logging
from typing import Any, Dict, Optional, Union, List
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XAI:
    """
    A class to interact with the X.AI chatbot API, handling messaging, cost tracking,
    and response validation.
    """
    
    API_URL: str = "https://api.x.ai/v1/chat/completions"
    PROMPT_COST_RATE: float = 5.0 / 1_000_000  # $5 per 1M input tokens
    COMPLETION_COST_RATE: float = 15.0 / 1_000_000  # $15 per 1M output tokens
    MAX_RETRIES: int = 3

    def __init__(
        self,
        model: str = "grok-beta",
        system: str = "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy.",
        enable_logging: bool = False,
        context: str = ""
    ) -> None:
        self.model = model
        self.system = system
        self.enable_logging = enable_logging
        self.context = context
        self.messages: list[Dict[str, str]] = [
            {
                "role": "system",
                "content": system
            }
        ]
        if context:
            self.messages.append({
                "role": "system",
                "content": f"Context: {context}"
            })
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

        # Initialize a session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
        })
        retries = HTTPAdapter(max_retries=self.MAX_RETRIES)
        self.session.mount("https://", retries)

    def update_context(self, new_context: str) -> None:
        """
        Update the conversation context.
        
        Args:
            new_context (str): The new context to set
        """
        self.context = new_context
        # Find and update existing context message or add new one
        context_idx = next((i for i, msg in enumerate(self.messages) 
                          if msg["role"] == "system" and msg["content"].startswith("Context:")), None)
        if context_idx is not None:
            self.messages[context_idx]["content"] = f"Context: {new_context}"
        else:
            self.messages.append({
                "role": "system",
                "content": f"Context: {new_context}"
            })

    @property
    def cost(self) -> float:
        """
        Calculate the total cost based on prompt and completion tokens.
        Returns:
            float: Total cost in USD.
        """
        total_cost = (self.prompt_tokens * self.PROMPT_COST_RATE) + \
                     (self.completion_tokens * self.COMPLETION_COST_RATE)
        if self.enable_logging:
            logger.info(f"Total cost so far: ${total_cost:.6f}")
        return total_cost

    def assert_keys(self, json_response: Union[Dict[str, Any], List[Dict[str, Any]]], expected_keys: Optional[Dict[str, Any]]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Verify all expected keys are present in the response.
        
        Args:
            json_response (Union[Dict[str, Any], List[Dict[str, Any]]]): The JSON response from the API.
            expected_keys (Optional[Dict[str, Any]]): The expected keys to validate.

        Returns:
            Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]: Filtered response if valid, else None.
        """
        if expected_keys is None:
            return json_response

        if isinstance(json_response, list):
            filtered_responses = []
            for item in json_response:
                filtered_item = self._validate_keys(item, expected_keys)
                if filtered_item:
                    filtered_responses.append(filtered_item)
            return filtered_responses if filtered_responses else None
        else:
            return self._validate_keys(json_response, expected_keys)

    def _validate_keys(self, response: Dict[str, Any], expected_keys: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Helper method to validate keys for a single response object"""
        expected_keys_lower = {key.lower(): value for key, value in expected_keys.items()}
        response_keys_lower = {key.lower(): value for key, value in response.items()}

        filtered_response = {}
        for key in expected_keys_lower:
            try:
                filtered_response[key] = response_keys_lower[key]
            except KeyError:
                logger.error(f"Missing expected key: {key}")
                filtered_response[key] = None

        return filtered_response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RequestException)
    )
    def chat(self, prompt: str, expected_format: Optional[Dict[str, Any]] = None) -> Any:
        """
        Send a chat message to the X.AI API and retrieve the response.
        
        Args:
            prompt (str): The user's input prompt.
            expected_format (Optional[Dict[str, Any]]): The expected JSON structure.

        Returns:
            Any: The response from the API, either as JSON or raw response.
        """
        if expected_format:
            prompt += f" Please format your response as a JSON object or array with these keys: {expected_format}"
        else:
            prompt += " Feel free to respond in plain text."

        self.messages.append({
            "role": "user",
            "content": prompt
        })

        data = {
            "messages": self.messages,
            "model": self.model,
            "stream": False,
            "temperature": 0
        }

        try:
            response = self.session.post(self.API_URL, json=data)
            response.raise_for_status()
            response_json = response.json()

            if 'choices' in response_json and response_json['choices']:
                response_content = response_json['choices'][0]['message']

                if expected_format:
                    try:
                        json_result = self.convert_to_json(response_json)
                        validated_result = self.assert_keys(json_result, expected_format)
                        if validated_result:
                            self.messages.append(response_content)
                            if self.enable_logging:
                                logger.info(response_content['content'])
                            self._update_tokens(response_json)
                            return validated_result
                        self.messages.pop()
                    except (ValueError, KeyError) as e:
                        logger.error(f"JSON conversion/validation error: {e}")
                        self.messages.pop()
                        raise
                else:
                    self.messages.append(response_content)
                    if self.enable_logging:
                        logger.info(response_content['content'])
                    self._update_tokens(response_json)
                    return response_json['choices'][0]['message']['content']
            else:
                logger.error("Invalid response structure")
                raise ValueError("Invalid response structure")

        except RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def _update_tokens(self, response_json: Dict[str, Any]) -> None:
        """
        Update the token counts based on the API response.
        
        Args:
            response_json (Dict[str, Any]): The JSON response from the API.
        """
        usage = response_json.get('usage', {})
        self.prompt_tokens += usage.get('prompt_tokens', 0)
        self.completion_tokens += usage.get('completion_tokens', 0)
        if self.enable_logging:
            logger.info(f"Updated tokens - Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}")

    def convert_to_json(self, response: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Extract and parse JSON content from the API response.
        
        Args:
            response (Dict[str, Any]): The JSON response from the API.

        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Parsed JSON content.
        
        Raises:
            ValueError: If JSON extraction fails.
        """
        try:
            content = response['choices'][0]['message']['content']
            content = content.strip()

            # Look for JSON code block markers
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    json_string = content[json_start:json_end].strip()
                    return json.loads(json_string)

            # Try to find JSON between brackets (list) or braces (dict)
            json_start = content.find('[') if '[' in content else content.find('{')
            json_end = content.rfind(']') + 1 if '[' in content else content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                # If no brackets/braces found, try the whole content
                json_string = content
            else:
                json_string = content[json_start:json_end]
            
            # Clean up common formatting issues
            json_string = json_string.replace("'", '"')
            json_string = json_string.replace('\n', ' ')
            json_string = ' '.join(json_string.split())  # Normalize whitespace
            
            try:
                return json.loads(json_string)
            except json.JSONDecodeError:
                # If parsing fails, try to extract JSON parts using regex
                import re
                # Look for both array and object patterns
                array_pattern = r'\[(?:[^[\]]|(?R))*\]'
                object_pattern = r'\{(?:[^{}]|(?R))*\}'
                
                array_matches = re.findall(array_pattern, content, re.DOTALL)
                object_matches = re.findall(object_pattern, content, re.DOTALL)
                
                if array_matches:
                    return json.loads(array_matches[0])
                elif object_matches:
                    return json.loads(object_matches[0])
                raise
            
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error converting response to JSON: {e}")
            raise ValueError("Failed to convert response to JSON") from e

# Example usage
if __name__ == "__main__":
    xai = XAI()
    try:
        print(xai.chat("What is the capital of France?", expected_format={"capital": "str"}))
        print(xai.chat("What is that place most famous for?"))
    except Exception as e:
        logger.error(f"An error occurred: {e}")
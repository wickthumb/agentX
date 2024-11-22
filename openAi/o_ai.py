import os
import requests
import json
import logging
from typing import Any, Dict, Optional, Union, List
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OAI:
    """
    A class to interact with the OpenAI API, handling messaging, cost tracking,
    and response validation.
    """
    
    API_URL: str = "https://api.openai.com/v1/chat/completions"
    
    # Pricing per 1M tokens in USD
    MODEL_COSTS = {
        "gpt-4o": {
            "input": 2500.0,        # $2.50 per 1M input tokens
            "output": 10000.0       # $10.00 per 1M output tokens
        },
        "gpt-4o-mini": {
            "input": 150.0,         # $0.150 per 1M input tokens
            "output": 600.0         # $0.600 per 1M output tokens
        },
        "o1-preview": {
            "input": 15000.0,       # $15.00 per 1M input tokens
            "output": 60000.0       # $60.00 per 1M output tokens
        }
    }
    
    MAX_RETRIES: int = 3

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system: str = "You are a helpful assistant.",
        enable_logging: bool = False,
        context: str = ""
    ) -> None:
        if model not in self.MODEL_COSTS:
            raise ValueError(f"Model {model} not found in MODEL_COSTS. Available models: {list(self.MODEL_COSTS.keys())}")
            
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

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )

    def update_context(self, new_context: str) -> None:
        """
        Update the conversation context.
        
        Args:
            new_context (str): The new context to set
        """
        self.context = new_context
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
        model_rates = self.MODEL_COSTS[self.model]
        total_cost = (self.prompt_tokens * model_rates["input"] / 1_000_000) + \
                     (self.completion_tokens * model_rates["output"] / 1_000_000)
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
        Send a chat message to the OpenAI API and retrieve the response.
        
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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=0
            )

            if response.choices:
                response_content = {
                    "role": response.choices[0].message.role,
                    "content": response.choices[0].message.content
                }

                if expected_format:
                    try:
                        json_result = self.convert_to_json({"choices": [{"message": response_content}]})
                        validated_result = self.assert_keys(json_result, expected_format)
                        if validated_result:
                            self.messages.append(response_content)
                            if self.enable_logging:
                                logger.info(response_content['content'])
                            self._update_tokens(response)
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
                    self._update_tokens(response)
                    return response_content['content']
            else:
                logger.error("Invalid response structure")
                raise ValueError("Invalid response structure")

        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def _update_tokens(self, response) -> None:
        """
        Update the token counts based on the API response.
        
        Args:
            response: The response from the OpenAI API.
        """
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        if self.enable_logging:
            logger.info(f"Updated tokens - Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}")

    def convert_to_json(self, response: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Extract and parse JSON content from the API response.
        
        Args:
            response (Dict[str, Any]): The response from the API.

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

if __name__ == "__main__":
    oai = OAI()
    print(oai.chat("What is the capital of France?"))
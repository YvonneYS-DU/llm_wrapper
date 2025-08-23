"""
LLM Wrapper - A streamlined interface for LangChain AI agent creation.

This module provides a simplified API for creating and managing AI agents
with multi-modal support (text and images) using LangChain framework.

Author: LLM Wrapper Team
Version: 0.2.1
Date: Aug 2025
"""

import re
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Generator
import base64

import fitz
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO

# Importing necessary modules and classes from OpenAI and LangChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser




class Agents:
    """
    A utility class for creating and managing AI agents with LangChain.
    
    This class provides static methods for creating prompts, chains, and
    executing AI model calls with support for text and image inputs.
    """
    
    def __init__(self) -> None:
        """
        Initialize the Agents class.
        
        Note: This class is primarily used as a utility class with static methods.
        """
        self.agents: List[Any] = []
        
    @staticmethod
    def _text_prompts(text_prompt_template: str = 'text prompt template') -> HumanMessagePromptTemplate:
        """
        Create a text prompt template for human messages.
        
        Args:
            text_prompt_template: The template string for the text prompt.
                                Use {parameter} for variable substitution.
                                
        Returns:
            HumanMessagePromptTemplate: A LangChain prompt template object for text input.
            
        Example:
            >>> template = Agents._text_prompts("Hello {name}, how are you?")
        """
        text_prompts = HumanMessagePromptTemplate.from_template(
            [{'text': text_prompt_template}]
        )
        return text_prompts
    
    @staticmethod
    def _image_prompts() -> HumanMessagePromptTemplate:
        """
        Create an image prompt template using file path.
        
        Returns:
            HumanMessagePromptTemplate: A LangChain prompt template object for image input via path.
            
        Note:
            This method creates a template expecting {image_path} and {detail_parameter}
            variables to be filled when the prompt is used.
        """
        image_prompts = HumanMessagePromptTemplate.from_template(
            [{'image_url': {'path': '{image_path}', 'detail': '{detail_parameter}'}}]
        )
        return image_prompts
    
    @staticmethod
    def _convert_pdf_to_base64_img_list(
        pdf_path: str, 
        dpi: int = 100, 
        crop_box_mm: Optional[tuple] = (10, 15, 10, 25)
    ) -> List[str]:
        """
        Convert a PDF file to a list of base64-encoded images.
        
        Args:
            pdf_path: Path to the PDF file to convert.
            dpi: Resolution for image conversion in dots per inch.
            crop_box_mm: Optional cropping area in millimeters as (left, top, right, bottom).
                        If None, no cropping is applied.
                        
        Returns:
            List of base64-encoded image strings, one per PDF page.
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            Exception: If PDF processing fails.
            
        Example:
            >>> images = Agents._convert_pdf_to_base64_img_list("document.pdf", dpi=150)
            >>> print(f"Converted {len(images)} pages")
        """
        img_list = []
        pdf_document = fitz.open(pdf_path)

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Convert crop area from mm to pixels based on DPI
            if crop_box_mm:
                crop_box_pixels = (
                    crop_box_mm[0] * dpi / 25.4,
                    crop_box_mm[1] * dpi / 25.4,
                    page.rect.width * dpi / 72 - crop_box_mm[2] * dpi / 25.4,
                    page.rect.height * dpi / 72 - crop_box_mm[3] * dpi / 25.4
                )
                clip = fitz.Rect(*crop_box_pixels)
                pix = page.get_pixmap(dpi=dpi, clip=clip)
            else:
                pix = page.get_pixmap(dpi=dpi)

            # Convert to base64
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_list.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
        
        pdf_document.close()
        return img_list

    @staticmethod
    def _image_prompts_base64() -> HumanMessagePromptTemplate:
        """
        Create an image prompt template using base64-encoded images.
        
        Returns:
            HumanMessagePromptTemplate: A LangChain prompt template for base64 image input.
            
        Note:
            This template is compatible with vision-enabled models like:
            - Claude 3.5 Sonnet
            - GPT-4o, GPT-4o-mini
            - Other vision-capable LLMs
            
            The template expects {base64_image} and {detail_parameter} variables.
        """
        template_string = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,{base64_image}",
                    "detail": "{detail_parameter}"
                }
            }
        ]
        image_prompts = HumanMessagePromptTemplate.from_template(template_string)

        return image_prompts
    
    @staticmethod
    def _image_prompts_base64_multi(
        base64_image: str, 
        detail_parameter: str = 'auto'
    ) -> HumanMessagePromptTemplate:
        """
        Create a pre-filled image prompt template for multi-image conversations.
        
        Args:
            base64_image: Base64-encoded image string.
            detail_parameter: Image detail level. Options: 'high', 'low', 'auto'.
                            Default is 'auto' for balanced quality and speed.
                            
        Returns:
            HumanMessagePromptTemplate: A pre-filled prompt template with image data.
            
        Note:
            This method creates a template that's already filled with image data,
            unlike other template methods that return templates with placeholders.
        """
        # Template string with placeholders
        template_string = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": f"{detail_parameter}"
                }
            }
        ]

        # template to object
        image_prompts = HumanMessagePromptTemplate.from_template(template_string)
        
        return image_prompts

    @staticmethod
    def list_to_img_dict(img_list: List[str]) -> Dict[str, str]:
        """
        Convert a list of images to a dictionary with numbered keys.
        
        Args:
            img_list: List of images (base64 strings or URLs).
            
        Returns:
            Dictionary with keys 'img1', 'img2', etc., mapped to image data.
            
        Example:
            >>> images = ['base64_img1', 'base64_img2']
            >>> result = Agents.list_to_img_dict(images)
            >>> print(result)  # {'img1': 'base64_img1', 'img2': 'base64_img2'}
        """
        return {f"img{i+1}": img for i, img in enumerate(img_list)}
    
    @staticmethod
    def _system_prompts(system_prompt_template: str = 'system prompt template') -> SystemMessagePromptTemplate:
        """
        Create a system prompt template for AI agent behavior configuration.
        
        Args:
            system_prompt_template: The system prompt string that defines AI behavior.
                                   Use {parameter} for variable substitution.
                                   
        Returns:
            SystemMessagePromptTemplate: A LangChain system prompt template object.
            
        Note:
            System prompts define the AI's role, behavior, and constraints.
            They are processed before user messages in the conversation.
        """
        system_prompts = SystemMessagePromptTemplate.from_template(
            system_prompt_template
        )
        return system_prompts
    
    @staticmethod
    def extract_prompts_parameters(prompt_template: str) -> List[str]:
        """
        Extract unique parameter names from a prompt template.
        
        Args:
            prompt_template: The prompt template string containing {parameter} placeholders.
            
        Returns:
            List of unique parameter names found in the template.
            
        Example:
            >>> template = "Hello {name}, your age is {age}. Nice to meet you {name}!"
            >>> params = Agents.extract_prompts_parameters(template)
            >>> print(params)  # ['name', 'age']
        """
        # use regex to extract the parameters
        parameters = re.findall(r'{(.*?)}', prompt_template)
        # remove duplicates
        unique_parameters = list(set(parameters))
        return unique_parameters
    
    @staticmethod
    def lc_prompt_template(
        text_prompt_template: str = 'text prompt template',
        system_prompt_template: Optional[str] = None,
        image_prompt_template: bool = False,
        image_list: List[str] = None,
        fill_img: bool = True
    ) -> ChatPromptTemplate:
        """
        Create a complete LangChain chat prompt template with optional image support.
        
        Args:
            text_prompt_template: The main text prompt template string.
            system_prompt_template: Optional system prompt for AI behavior configuration.
            image_prompt_template: Whether to include image input capability.
            image_list: List of base64 images for multi-image support.
            fill_img: Whether to fill image data immediately (True) or use placeholders (False).
            
        Returns:
            ChatPromptTemplate: A complete LangChain chat prompt template.
            
        Example:
            >>> template = Agents.lc_prompt_template(
            ...     text_prompt_template="Describe the image: {description}",
            ...     image_prompt_template=True
            ... )
        """
        if image_list is None:
            image_list = []
            
        if system_prompt_template:  # system prompt
            chat_prompt_template = ChatPromptTemplate.from_messages(
                messages=[
                    Agents._system_prompts(system_prompt_template),
                    Agents._text_prompts(text_prompt_template),
                    *([Agents._image_prompts_base64()] if image_prompt_template else []),
                ]
            )
        else:
            if image_list:  # multi-image prompt
                chat_prompt_template = Agents.multi_image_templates(
                    text_prompt_template=text_prompt_template, 
                    image_list=image_list, 
                    fill_img=fill_img
                )
            else:  # zero/single-image prompt
                chat_prompt_template = ChatPromptTemplate.from_messages(
                    messages=[
                        Agents._text_prompts(text_prompt_template),  
                        *([Agents._image_prompts_base64()] if image_prompt_template else [])
                    ]
                )
        return chat_prompt_template
    
    @staticmethod
    def multi_image_templates(
        text_prompt_template: str = 'text prompt template',
        fill_img: bool = True,
        image_list: List[str] = None,
        detail_parameter: str = 'high'
    ) -> ChatPromptTemplate:
        """
        Create a multi-image prompt template for processing multiple images simultaneously.
        
        Args:
            text_prompt_template: The main text prompt template string.
            fill_img: Whether to fill image data immediately (True) or use placeholders (False).
            image_list: List of base64-encoded images or placeholders.
            detail_parameter: Image detail level ('high', 'low', 'auto').
            
        Returns:
            ChatPromptTemplate: A prompt template configured for multiple image inputs.
            
        Example:
            >>> template = Agents.multi_image_templates(
            ...     text_prompt_template="Compare these images: {comparison_task}",
            ...     image_list=["base64_img1", "base64_img2"],
            ...     fill_img=True
            ... )
        """
        if image_list is None:
            image_list = []
            
        # Create text prompt component
        text_prompts = [Agents._text_prompts(text_prompt_template)]
        
        if fill_img:
            # Create prompt templates with actual image data
            image_prompts = [
                Agents._image_prompts_base64_multi(image, detail_parameter)
                for image in image_list
            ]
        else:
            # Create placeholder templates (img1, img2, ...)
            image_prompts = [
                Agents._image_prompts_base64_multi(f"img{i+1}", detail_parameter)
                for i in range(len(image_list))
            ]

        # Compose the complete chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            messages=text_prompts + image_prompts
        )
        
        return chat_prompt_template
    
    # @staticmethod
    # def generate_image(prompt, number_of_images=1, size='1792x1024', style='natural', quality = 'standard', api_key='api_key'):
    #     """
    #     prompt: string of prompt to generate image
    #     number_of_images: int, default 1, number of images to generate
    #     size: string, default '1792x1024', size of the image, openai choices: ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024']
    #     style: string, default 'natural', disabled for bad image quality, openai choices: ['vivid', 'natural']
    #     quality: string, default 'standard', openai choices: ['standard', 'hd']
    #     api_key: string, openai api key

    #     return the response of the image generation
    #     """
    #     client = OpenAI(api_key=api_key)    
    #     response = client.images.generate(
    #     model = "dall-e-3",
    #     prompt = prompt,
    #     size = size, # ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024']
    #     n = number_of_images,
    #     #style = style, #['vivid', 'natural']
    #     quality = quality #['standard', 'hd']
    #     )
    #     return response
    

    @staticmethod
    def chain_create(
        model: Any,
        system_prompt_template: str = '',
        text_prompt_template: str = 'prompt template string',
        image_prompt_template: bool = False,
        output_parser: Any = StrOutputParser(),
        format_var_name: str = 'schema',
        image_list: List[str] = None,
        fill_image: bool = False,
        parameters: bool = False
    ) -> Union[Any, tuple]:
        """
        Create a complete LangChain processing chain with model, prompts, and output parser.
        
        This is the main function for creating AI processing chains. It combines
        a language model, prompt templates, and output parsers into a single callable chain.
        
        Args:
            model: The language model instance (ChatOpenAI, ChatAnthropic, etc.).
            system_prompt_template: Optional system prompt to define AI behavior.
            text_prompt_template: The main text prompt template with {parameter} placeholders.
            image_prompt_template: Whether to enable image input processing.
            output_parser: Parser for model output (StrOutputParser, JsonOutputParser, etc.).
            format_var_name: Variable name for format instructions in templates.
            image_list: List of base64 images for multi-image processing.
            fill_image: Whether to fill image data immediately or use placeholders.
            parameters: Whether to return prompt parameters along with the chain.
            
        Returns:
            Runnable chain object, or tuple of (chain, parameters) if parameters=True.
            
        Example:
            >>> from langchain_openai import ChatOpenAI
            >>> from langchain_core.output_parsers import StrOutputParser
            >>> llm = ChatOpenAI(model_name='gpt-4o-mini')
            >>> chain = Agents.chain_create(
            ...     model=llm,
            ...     text_prompt_template="Answer this question: {question}",
            ...     output_parser=StrOutputParser()
            ... )
            >>> response = chain.invoke({"question": "What is AI?"})
        """

        if image_list is None:
            image_list = []
            
        # Create the prompt template
        lc_prompt_template = Agents.lc_prompt_template(
            text_prompt_template=text_prompt_template,
            system_prompt_template=system_prompt_template,
            image_prompt_template=image_prompt_template,
            image_list=image_list,
            fill_img=fill_image
        )
        
        # Add format instructions if the output parser supports them
        if hasattr(output_parser, 'get_format_instructions'):
            if f"{{{format_var_name}}}" in text_prompt_template:
                partial_dict = {format_var_name: output_parser.get_format_instructions()}
                lc_prompt_template = lc_prompt_template.partial(**partial_dict)
            else:
                print(f"Warning: {format_var_name} not found in prompt template, skipping format instructions")
        
        # Create the chain: prompt | model | parser
        chain = lc_prompt_template | model | output_parser
        
        if parameters:
            extracted_parameters = Agents.extract_prompts_parameters(text_prompt_template)
            print("Parameters:", extracted_parameters)
            return chain, extracted_parameters
        else:
            return chain
   
    @staticmethod
    async def _delay(seconds: float) -> None:
        """
        Asynchronous delay utility function.
        
        Args:
            seconds: Number of seconds to delay.
            
        Example:
            >>> await Agents._delay(1.5)  # Wait 1.5 seconds
        """
        await asyncio.sleep(seconds)
        
    @staticmethod
    def chain_stream_generator(chain: Any, dic: Dict[str, Any] = None) -> Generator[str, None, None]:
        """
        Generate streaming responses from a chain, yielding chunks as they arrive.
        
        Args:
            chain: The chain object created by Agents.chain_create().
            dic: Dictionary of parameters to fill template placeholders.
                Format: {"parameter_name": "value"}
                
        Yields:
            String chunks of the model's response in real-time.
            
        Example:
            >>> chain = Agents.chain_create(llm, "Tell me about {topic}")
            >>> for chunk in Agents.chain_stream_generator(chain, {"topic": "AI"}):
            ...     print(chunk, end="", flush=True)
        """
        if dic is None:
            dic = {}
            
        for chunk in chain.stream(dic):
            yield chunk.content


    @staticmethod
    def chain_batch_generator(chain: Any, dic: Dict[str, Any] = None, max_retries: int = 2) -> Any:
        """
        Execute a chain synchronously with automatic retry logic.
        
        Args:
            chain: The chain object created by Agents.chain_create().
            dic: Dictionary of parameters to fill template placeholders.
                Format: {"parameter_name": "value"}
            max_retries: Maximum number of retry attempts on failure.
            
        Returns:
            The model's response after successful execution.
            
        Raises:
            Exception: If all retry attempts fail.
            
        Example:
            >>> chain = Agents.chain_create(llm, "Translate {text} to {language}")
            >>> response = Agents.chain_batch_generator(
            ...     chain, 
            ...     {"text": "Hello", "language": "French"}
            ... )
            >>> print(response)
        """
        if dic is None:
            dic = {}
        
        attempt = 0
        
        while attempt <= max_retries:
            try:
                response = chain.invoke(dic)
                return response
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise Exception(f"AI encountered some issues, please try again later: {e}")
                else:
                    continue
    
    @staticmethod
    def continue_chain_batch_generator(
        chain: Any, 
        dic: Dict[str, Any] = None, 
        max_retries: int = 2,
        max_continue: int = 5,
        trigger_type: str = "str",
        continue_trigger: Union[str, Dict[str, Any]] = "continue",
        result_cache: List[str] = None,
        history_key: Union[bool, str] = False,
        show_progress: bool = False
    ) -> str:
        """
        A utility function to continue the chain batch generator for very long conversations.
        
        This function handles cases where the AI response is cut off due to token limits
        and needs to be continued. It automatically detects when continuation is needed
        and manages the conversation context.
        
        Args:
            chain: The chain object created by Agents.chain_create().
            dic: Dictionary of parameters to fill template placeholders.
            max_retries: Maximum number of retry attempts on failure per continuation.
            max_continue: Maximum number of continuation attempts.
            trigger_type: Type of trigger detection. Options:
                - "str": Simple string matching (default)
                - "json": JSON format detection
            continue_trigger: Trigger configuration:
                - For trigger_type="str": string keyword or dict with keywords
                - For trigger_type="json": dict like {"continue": "continue"}
            result_cache: List to store accumulated results from all continuations.
            history_key: Control history context management. Options:
                - False: Disable history management (default behavior)
                - str: Enable history management with custom key name (e.g., "my_history")
                  When string provided, adds history content to dic with specified key name
            show_progress: Whether to output the current loop iteration number.
            
        Returns:
            The complete concatenated response from all continuation attempts.
            
        Examples:
            >>> # Simple string trigger
            >>> chain = Agents.chain_create(llm, "Write a detailed essay about {topic}")
            >>> response = Agents.continue_chain_batch_generator(
            ...     chain,
            ...     {"topic": "artificial intelligence"},
            ...     trigger_type="str",
            ...     continue_trigger="continue"
            ... )
            
            >>> # JSON format trigger
            >>> response = Agents.continue_chain_batch_generator(
            ...     chain,
            ...     {"topic": "machine learning"},
            ...     trigger_type="json",
            ...     continue_trigger={"continue": "continue"}
            ... )
            
            >>> # For JSON outputs: {"content": "...", "continue": true}
            >>> # For XML-style: "Some text... <continue/>"
            >>> # For text: "Some text... continue"
            
            >>> # Using custom history key
            >>> response = Agents.continue_chain_batch_generator(
            ...     chain,
            ...     {"topic": "machine learning", "task": "explain concepts"},
            ...     history_key="conversation_history"
            ... )
            >>> # This adds "conversation_history" to dic containing all previous responses
        """
        # Validate and normalize trigger_type with fuzzy matching
        str_variants = ["str", "STR", "STRING", "string", "String", "text", "TEXT", "Text"]
        json_variants = ["json", "JSON", "Json", "JSON_FORMAT", "json_format"]
        
        if trigger_type in str_variants:
            trigger_type = "str"  # Normalize to standard form
        elif trigger_type in json_variants:
            trigger_type = "json"  # Normalize to standard form
        else:
            raise ValueError(f"trigger_type must be a string variant (str/STR/STRING/string/text) or json variant (json/JSON/Json), got '{trigger_type}'")
            
        if dic is None:
            dic = {}
        if result_cache is None:
            result_cache = []
            
        continue_count = 0
        
        # Initialize history key for first call if using custom history key
        if history_key and isinstance(history_key, str):
            if history_key not in dic:
                dic[history_key] = "first call, no history"  # Add empty default value for first call
        
        # Show initial progress if requested
        if show_progress:
            print(f"Starting continue_chain_batch_generator - Loop 0 (initial call)")
        
        # Get initial response
        try:
            initial_response = Agents.chain_batch_generator(chain, dic, max_retries)
            result_cache.append(str(initial_response))
            current_response = str(initial_response)
        except Exception as e:
            raise Exception(f"Failed to get initial response: {e}")
        
        # Check if continuation is needed and continue until max_continue or no trigger found
        while continue_count < max_continue:
            # Check if the response appears to be cut off or contains continuation trigger
            response_lower = current_response.lower().strip()
            
            # Common indicators that response was cut off
            needs_continuation = (
                response_lower.endswith(('...', '…')) or
                Agents._check_continue_trigger(trigger_type, continue_trigger, current_response) or  # Check trigger
                len(current_response.strip()) == 0 or
                response_lower.endswith(('.', ',', ';', ':')) == False or  # Doesn't end with proper punctuation
                response_lower.endswith(('to be continued', 'continued...', 'continue reading'))
            )
            
            if not needs_continuation:
                break
                
            continue_count += 1
            
            # Show progress if requested
            if show_progress:
                print(f"Continue loop {continue_count} / {max_continue} - Processing continuation...")
            
            # Prepare continuation prompt
            continuation_dic = dic.copy()
            
            # Handle history management based on history_key parameter
            if history_key:
                # If history_key is a string, use it as custom key name
                if isinstance(history_key, str):
                    conversation_context = "\n".join(result_cache)
                    continuation_dic[history_key] = conversation_context
                else:
                    # If history_key is True, use default behavior
                    conversation_context = "\n".join(result_cache)
                    if 'context' not in continuation_dic:
                        continuation_dic['context'] = conversation_context
                    if 'previous_response' not in continuation_dic:
                        continuation_dic['previous_response'] = current_response
                        
                    # Add continuation instruction
                    continuation_dic['continuation_instruction'] = f"Please continue from where you left off. Previous response: ...{current_response[-200:]}"
            else:
                # Original behavior: always add context management keys
                conversation_context = "\n".join(result_cache)
                if 'context' not in continuation_dic:
                    continuation_dic['context'] = conversation_context
                if 'previous_response' not in continuation_dic:
                    continuation_dic['previous_response'] = current_response
                    
                # Add continuation instruction
                continuation_dic['continuation_instruction'] = f"Please continue from where you left off. Previous response: ...{current_response[-200:]}"
            
            try:
                # Get continuation response
                continuation_response = Agents.chain_batch_generator(chain, continuation_dic, max_retries)
                current_response = str(continuation_response)
                result_cache.append(current_response)
                
                # Show completion progress if requested
                if show_progress:
                    print(f"Loop {continue_count} completed successfully")
                
            except Exception as e:
                if show_progress:
                    print(f"Warning: Failed to get continuation {continue_count}: {e}")
                else:
                    print(f"Warning: Failed to get continuation {continue_count}: {e}")
                break
        
        # Show final progress if requested
        if show_progress:
            total_loops = continue_count + 1  # Include initial call
            print(f"Continue_chain_batch_generator completed - Total loops: {total_loops} (0 initial + {continue_count} continuations)")
        
        # Return the complete concatenated response with smart merging
        complete_response = Agents._merge_response_cache(result_cache, trigger_type, continue_trigger)
        return complete_response

    @staticmethod
    def _check_continue_trigger(trigger_type: str, continue_trigger: Union[str, Dict[str, Any]], response_text: str) -> bool:
        """
        Helper method to check if continuation trigger is detected.
        
        Args:
            trigger_type: Type of trigger - 'str' for string matching, 'json' for JSON format
            continue_trigger: Trigger configuration (str or dict for json type)
            response_text: Full response text to check
            
        Returns:
            bool: True if trigger is detected, False otherwise
            
        Raises:
            ValueError: If trigger_type is not 'str' or 'json'
        """
        # Validate and normalize trigger_type with fuzzy matching
        str_variants = ["str", "STR", "STRING", "string", "String", "text", "TEXT", "Text"]
        json_variants = ["json", "JSON", "Json", "JSON_FORMAT", "json_format"]
        
        if trigger_type in str_variants:
            trigger_type = "str"  # Normalize to standard form
        elif trigger_type in json_variants:
            trigger_type = "json"  # Normalize to standard form
        else:
            raise ValueError(f"trigger_type must be a string variant (str/STR/STRING/string/text) or json variant (json/JSON/Json), got '{trigger_type}'")
            
        response_lower = response_text.lower().strip()
        
        if trigger_type == "json":
            # For JSON type, continue_trigger should be a dict like {'continue': 'continue'}
            if isinstance(continue_trigger, dict):
                try:
                    # Try to parse response as JSON
                    json_data = json.loads(response_text.strip())
                    if isinstance(json_data, dict):
                        # Check if any key in continue_trigger exists in response
                        for key, expected_value in continue_trigger.items():
                            if key in json_data:
                                actual_value = str(json_data[key]).lower()
                                expected_value_lower = str(expected_value).lower()
                                # Check for exact match or common truthy values
                                if (actual_value == expected_value_lower or 
                                    actual_value in ['true', 'yes', '1', 'continue', 'next', 'more']):
                                    return True
                except (json.JSONDecodeError, ValueError):
                    pass
            return False
            
        elif trigger_type == "str":
            # For string type, check in last part of response for robustness
            if isinstance(continue_trigger, str):
                # Simple string case - check last 100 characters
                check_length = 100
                return continue_trigger.lower() in response_lower[-check_length:]
            elif isinstance(continue_trigger, dict):
                keywords = continue_trigger.get('keywords', ['continue'])
                min_length = continue_trigger.get('min_length', 0)
                check_length = continue_trigger.get('check_length', 100)
                
                # Check minimum length requirement first
                if min_length > 0 and len(response_text.strip()) < min_length:
                    return True
                    
                # Check for keyword triggers in text (last part of response)
                for keyword in keywords:
                    if keyword.lower() in response_lower[-check_length:]:
                        return True
                
        return False

    @staticmethod
    def _normalize_json_trigger(json_str: str) -> Dict[str, Any]:
        """
        Normalize JSON string trigger to dict format with cleanup markers.
        
        Args:
            json_str: JSON string like '{"continue": "continue"}'
            
        Returns:
            Dict with normalized format
        """
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                # Generate cleanup markers for both JSON and Python dict formats
                cleanup_markers = []
                for key, value in parsed.items():
                    cleanup_markers.extend([
                        f'{{"{key}": "{value}"}}',  # Complete JSON object
                        f"{{'key': '{value}'}}".replace('key', key),  # Complete Python dict
                        f'"{key}": "{value}"',  # JSON key-value pair
                        f"'{key}': '{value}'",  # Python dict key-value pair
                    ])
                return {
                    "trigger_type": "json",
                    "keywords": list(parsed.values()),
                    "json_keys": list(parsed.keys()),
                    "cleanup_markers": cleanup_markers
                }
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback to simple string
        return {
            "trigger_type": "str",
            "keywords": [json_str]
        }

    @staticmethod
    def _merge_response_cache(result_cache: List[str], trigger_type: str, continue_trigger: Union[str, Dict[str, Any]]) -> str:
        """
        Simple merge of response cache with configurable cleaning.
        
        Args:
            result_cache: List of response fragments
            trigger_type: Type of trigger - 'str' for string matching, 'json' for JSON format
            continue_trigger: Trigger configuration for cleaning
            
        Returns:
            str: Merged response
            
        Raises:
            ValueError: If trigger_type is not 'str' or 'json'
        """
        # Validate and normalize trigger_type with fuzzy matching
        str_variants = ["str", "STR", "STRING", "string", "String", "text", "TEXT", "Text"]
        json_variants = ["json", "JSON", "Json", "JSON_FORMAT", "json_format"]
        
        if trigger_type in str_variants:
            trigger_type = "str"  # Normalize to standard form
        elif trigger_type in json_variants:
            trigger_type = "json"  # Normalize to standard form
        else:
            raise ValueError(f"trigger_type must be a string variant (str/STR/STRING/string/text) or json variant (json/JSON/Json), got '{trigger_type}'")
            
        if not result_cache:
            return ""
        
        # Process each response fragment
        cleaned_parts = []
        for response in result_cache:
            current_part = response.strip()
            
            if trigger_type == "json":
                # For JSON triggers, remove JSON trigger markers
                if isinstance(continue_trigger, dict):
                    # Remove JSON objects that match the trigger pattern
                    for key, value in continue_trigger.items():
                        # Remove patterns like {"continue": "continue"}
                        json_pattern = f'{{"{key}": "{value}"}}'
                        current_part = current_part.replace(json_pattern, "")
                        # Also try with single quotes
                        json_pattern_single = f"{{'{key}': '{value}'}}"
                        current_part = current_part.replace(json_pattern_single, "")
                        
            elif trigger_type == "str":
                # For string triggers, use smart cleaning logic
                if isinstance(continue_trigger, str):
                    # Simple string case - check last 100 characters for trigger
                    trigger_word = continue_trigger.lower().strip()
                    current_lower = current_part.lower()
                    check_length = 100
                    
                    # Check if trigger appears in the last check_length characters
                    if trigger_word in current_lower[-check_length:]:
                        # Find the last occurrence of the trigger
                        idx = current_lower.rfind(trigger_word)
                        if idx >= 0:
                            # Check what comes after the trigger
                            suffix = current_part[idx + len(trigger_word):]
                            # If suffix contains only dots, spaces, or common continuation symbols, remove it all
                            if re.match(r'^[.\s…-]*$', suffix):
                                current_part = current_part[:idx].strip()
                
                elif isinstance(continue_trigger, dict):
                    # Handle dict format for string triggers
                    keywords = continue_trigger.get("keywords", [])
                    cleanup_markers = continue_trigger.get('cleanup_markers', [])
                    check_length = continue_trigger.get('check_length', 100)
                    
                    # Remove cleanup markers first
                    for marker in cleanup_markers:
                        current_part = current_part.replace(marker, "")
                    
                    # Smart keyword removal from end of text
                    current_lower = current_part.lower()
                    for keyword in keywords:
                        keyword_lower = keyword.lower().strip()
                        # Check if keyword appears in the last check_length characters
                        if keyword_lower in current_lower[-check_length:]:
                            # Find the last occurrence of the keyword
                            idx = current_lower.rfind(keyword_lower)
                            if idx >= 0:
                                # Check what comes after the keyword
                                suffix = current_part[idx + len(keyword):]
                                # If suffix contains only dots, spaces, or common continuation symbols, remove it all
                                if re.match(r'^[.\s…-]*$', suffix):
                                    current_part = current_part[:idx].strip()
                                    break  # Only remove first match
            
            # Add non-empty parts
            if current_part:
                cleaned_parts.append(current_part)
        
        # Join with newlines for better readability
        return '\n'.join(cleaned_parts)

    @staticmethod
    async def continue_chain_batch_generator_async(
        chain: Any, 
        dic: Dict[str, Any] = None, 
        max_retries: int = 2,
        max_continue: int = 5,
        trigger_type: str = "str",
        continue_trigger: Union[str, Dict[str, Any]] = "continue",
        result_cache: List[str] = None,
        delay: Optional[float] = None,
        history_key: Union[bool, str] = False,
        show_progress: bool = False
    ) -> str:
        """
        Async version of continue_chain_batch_generator for very long conversations.
        
        This function handles cases where the AI response is cut off due to token limits
        and needs to be continued asynchronously with optional delays.
        
        Args:
            chain: The chain object created by Agents.chain_create().
            dic: Dictionary of parameters to fill template placeholders.
            max_retries: Maximum number of retry attempts on failure per continuation.
            max_continue: Maximum number of continuation attempts.
            trigger_type: Type of trigger detection. Options:
                - "str": Simple string matching (default)
                - "json": JSON format detection
            continue_trigger: Trigger configuration:
                - For trigger_type="str": string keyword or dict with keywords
                - For trigger_type="json": dict like {"continue": "continue"}
            result_cache: List to store accumulated results from all continuations.
            delay: Optional delay in seconds between continuation attempts.
            history_key: Control history context management. Options:
                - False: Disable history management (default behavior)
                - str: Enable history management with custom key name (e.g., "my_history")
                  When string provided, adds history content to dic with specified key name
            show_progress: Whether to output the current loop iteration number.
            
        Returns:
            The complete concatenated response from all continuation attempts.
            
        Examples:
            >>> # Async with delay and advanced triggers
            >>> chain = Agents.chain_create(llm, "Write a detailed analysis of {topic}")
            >>> response = await Agents.continue_chain_batch_generator_async(
            ...     chain,
            ...     {"topic": "machine learning trends"},
            ...     trigger_type="json",
            ...     continue_trigger={"continue": "continue"},
            ...     delay=1.0
            ... )
            
            >>> # Supports same format detection as sync version:
            >>> # JSON: {"analysis": "...", "continue": true}
            >>> # XML: "Analysis content... <continue/>"
            >>> # Text: "Analysis content... 继续"
        """
        # Validate and normalize trigger_type with fuzzy matching
        str_variants = ["str", "STR", "STRING", "string", "String", "text", "TEXT", "Text"]
        json_variants = ["json", "JSON", "Json", "JSON_FORMAT", "json_format"]
        
        if trigger_type in str_variants:
            trigger_type = "str"  # Normalize to standard form
        elif trigger_type in json_variants:
            trigger_type = "json"  # Normalize to standard form
        else:
            raise ValueError(f"trigger_type must be a string variant (str/STR/STRING/string/text) or json variant (json/JSON/Json), got '{trigger_type}'")
            
        if dic is None:
            dic = {}
        if result_cache is None:
            result_cache = []
            
        continue_count = 0
        
        # Initialize history key for first call if using custom history key
        if history_key and isinstance(history_key, str):
            if history_key not in dic:
                dic[history_key] = " first call, no history"  # Add empty default value for first call
        
        # Get initial response
        try:
            initial_response = await Agents.chain_batch_generator_async(chain, dic, delay, max_retries)
            result_cache.append(str(initial_response))
            current_response = str(initial_response)
        except Exception as e:
            raise Exception(f"Failed to get initial response: {e}")
        
        # Check if continuation is needed and continue until max_continue or no trigger found
        while continue_count < max_continue:
            # Check if the response appears to be cut off or contains continuation trigger
            response_lower = current_response.lower().strip()
            
            # Common indicators that response was cut off
            needs_continuation = (
                response_lower.endswith(('...', '…')) or
                Agents._check_continue_trigger(trigger_type, continue_trigger, current_response) or  # Check trigger
                len(current_response.strip()) == 0 or
                response_lower.endswith(('.', ',', ';', ':')) == False or  # Doesn't end with proper punctuation
                response_lower.endswith(('to be continued', 'continued...', 'continue reading'))
            )
            
            if not needs_continuation:
                break
                
            continue_count += 1
            
            # Add delay between continuation attempts if specified
            if delay:
                await Agents._delay(delay)
            
            # Prepare continuation prompt
            continuation_dic = dic.copy()
            
            # Handle history management based on history_key parameter
            if history_key:
                # If history_key is a string, use it as custom key name
                if isinstance(history_key, str):
                    conversation_context = "\n".join(result_cache)
                    continuation_dic[history_key] = conversation_context
                else:
                    # If history_key is True, use default behavior
                    conversation_context = "\n".join(result_cache)
                    if 'context' not in continuation_dic:
                        continuation_dic['context'] = conversation_context
                    if 'previous_response' not in continuation_dic:
                        continuation_dic['previous_response'] = current_response
                        
                    # Add continuation instruction
                    continuation_dic['continuation_instruction'] = f"Please continue from where you left off. Previous response: ...{current_response[-200:]}"
            else:
                # Original behavior: always add context management keys
                conversation_context = "\n".join(result_cache)
                if 'context' not in continuation_dic:
                    continuation_dic['context'] = conversation_context
                if 'previous_response' not in continuation_dic:
                    continuation_dic['previous_response'] = current_response
                    
                # Add continuation instruction
                continuation_dic['continuation_instruction'] = f"Please continue from where you left off. Previous response: ...{current_response[-200:]}"
            
            try:
                # Get continuation response
                continuation_response = await Agents.chain_batch_generator_async(chain, continuation_dic, None, max_retries)
                current_response = str(continuation_response)
                result_cache.append(current_response)
                
            except Exception as e:
                print(f"Warning: Failed to get continuation {continue_count}: {e}")
                break
        
        # Return the complete concatenated response with smart merging
        complete_response = Agents._merge_response_cache(result_cache, trigger_type, continue_trigger)
        return complete_response

    @staticmethod
    async def chain_batch_generator_async(
        chain: Any, 
        dic: Dict[str, Any] = None, 
        delay: Optional[float] = None, 
        max_retries: int = 2
    ) -> Any:
        """
        Execute a chain asynchronously with automatic retry logic and optional delay.
        
        This method is particularly useful for image analysis and batch processing
        where you want to avoid overwhelming the API with simultaneous requests.
        
        Args:
            chain: The chain object created by Agents.chain_create().
            dic: Dictionary of parameters to fill template placeholders.
            delay: Optional delay in seconds before starting execution.
            max_retries: Maximum number of retry attempts on failure.
            
        Returns:
            The model's response after successful execution, or Exception on failure.
            
        Example:
            >>> chain = Agents.chain_create(llm, "Analyze this image: {base64_image}")
            >>> response = await Agents.chain_batch_generator_async(
            ...     chain, 
            ...     {"base64_image": image_data},
            ...     delay=0.5
            ... )
        """
        if dic is None:
            dic = {}
            
        attempt = 0
        print("Task started at:", datetime.now())
        
        if delay:
            print(f"Waiting for {delay} seconds before starting the task.")
            await Agents._delay(delay)
            
        while attempt <= max_retries:
            print("Attempting to invoke the chain...")
            try:
                response = await chain.ainvoke(dic)
                return response
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed: {e}. Retrying...")
                if attempt > max_retries:
                    print("Max retries exceeded. Error:", e)
                    return e
                continue
    @staticmethod
    def output_parser(output_string: str) -> List[Dict[str, str]]:
        """
        Parse TSV (Tab-Separated Values) output from LLM into structured data.
        
        This utility function processes LLM output that contains tabular data,
        removing code block wrappers and converting to a list of dictionaries.
        
        Args:
            output_string: The raw output from the LLM, potentially containing
                          TSV data wrapped in code blocks (```).
                          
        Returns:
            List of dictionaries where each dictionary represents a row,
            with column headers as keys and cell values as values.
            
        Example:
            >>> tsv_output = '''
            ... ```
            ... Name\tAge\tCity
            ... John\t25\tNew York
            ... Jane\t30\tLos Angeles
            ... ```
            ... '''
            >>> result = Agents.output_parser(tsv_output)
            >>> print(result)
            [{'Name': 'John', 'Age': '25', 'City': 'New York'},
             {'Name': 'Jane', 'Age': '30', 'City': 'Los Angeles'}]
        """
        # Remove code block wrappers if present
        tsv_string = re.sub(r'^```.*?\n|```$', '', output_string, flags=re.DOTALL).strip()
        
        # Split the TSV string into lines
        lines = tsv_string.strip().split('\n')
        
        if not lines or len(lines) < 2:
            return []
            
        # Extract and clean header row
        headers = [h.strip() for h in lines[0].split('\t')]
        result = []
        
        # Process each data row (skip the header row)
        for i in range(1, len(lines)):
            line = lines[i].strip()
            if not line:  # Skip empty lines
                continue
                
            # Split row by tab character and clean values
            values = [v.strip() for v in line.split('\t')]
            
            # Create dictionary for this row
            row_dict = {}
            for j, header in enumerate(headers):
                row_dict[header] = values[j] if j < len(values) else ''
                
            result.append(row_dict)
            
        return result




class Contexts:
    """
    this is a utility class for create Context engineering for LLMs.
    the context contain 3 parts:
    1. create layer for context
    2. create LLM context compression
    3. automatic context management for LLMs
    """
    def __init__(self) -> None:
        """
        Initialize the Contexts class.
        
        Note: This class is primarily used as a utility class with static methods.
        """
        self.contexts: List[Any] = []
    
    @staticmethod
    def create_context_layer(
        context_template: str = 'context template string',
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a context layer for LLMs using a template and optional parameters.
        
        Args:
            context_template: The context template string with {parameter} placeholders.
            parameters: Optional dictionary of parameters to fill the template.
            
        Returns:
            Formatted context string with parameters filled in.
            
        Example:
            >>> context = Contexts.create_context_layer("Current date is {date}.", {"date": "2025-01-01"})
            >>> print(context)  # "Current date is 2025-01-01."
        """
        if parameters is None:
            parameters = {}
            
        return context_template.format(**parameters)
    @staticmethod
    def compress_context(
        model: Any,
        context_dict: Dict[str, Any],
        compression_prompt: str = "Compress this context to under {max_tokens} tokens while preserving all key information: {context}",
        max_tokens: int = 1000,
        system_prompt: str = "You are a context compression expert. Compress the given context while preserving all important information.",
        max_retries: int = 2
    ) -> str:
        """
        Compress a context dictionary using an automatically created chain.
        
        This method creates an internal chain and automatically runs it to compress context.
        The input context_dict must match the {} placeholders in compression_prompt.
        
        Args:
            model: The language model instance (ChatOpenAI, ChatAnthropic, etc.).
            context_dict: Dictionary with keys matching compression_prompt placeholders.
            compression_prompt: Template string with {} placeholders matching context_dict keys.
            max_tokens: Maximum number of tokens for the compressed context.
            system_prompt: System prompt for the compression model.
            max_retries: Maximum number of retry attempts on failure.
            
        Returns:
            Compressed context string.
            
        Raises:
            Exception: If all retry attempts fail.
            ValueError: If context_dict keys don't match prompt placeholders.
            
        Example:
            >>> from langchain_openai import ChatOpenAI
            >>> llm = ChatOpenAI(model_name='gpt-4o-mini')
            >>> context = {"context": "long text to compress", "max_tokens": 500}
            >>> compressed = Contexts.compress_context(
            ...     model=llm,
            ...     context_dict=context
            ... )
        """
        # Extract prompt parameters to validate context_dict
        prompt_params = Agents.extract_prompts_parameters(compression_prompt)
        
        # Validate that context_dict keys match prompt parameters
        missing_params = set(prompt_params) - set(context_dict.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in context_dict: {missing_params}")
        
        # Add max_tokens to context if not present
        final_context = context_dict.copy()
        if "max_tokens" not in final_context:
            final_context["max_tokens"] = max_tokens
        
        # Create compression chain internally
        chain = Agents.chain_create(
            model=model,
            system_prompt_template=system_prompt,
            text_prompt_template=compression_prompt
        )
        
        # Execute the chain with automatic retry
        return Agents.chain_batch_generator(
            chain=chain,
            dic=final_context,
            max_retries=max_retries
        )


class Document:
    """
    A comprehensive document processing class for handling PDF, images, Word documents, and tables.
    
    This class provides functionality for:
    1. PDF text extraction and image conversion
    2. Image processing and enhancement for OCR
    3. Word document reading
    4. Table extraction and processing
    
    The main focus is on PDF and image processing with advanced features like
    A4 sizing, DPI control, sharpening, and denoising for improved OCR results.
    """
    
    def __init__(self) -> None:
        """
        Initialize the Document class.
        """
        self.supported_formats = {
            'pdf': ['.pdf'],
            'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
            'word': ['.docx', '.doc'],
            'table': ['.xlsx', '.xls', '.csv']
        }
    
    @staticmethod
    def extract_text_from_pdf(
        pdf_path: str,
        start_page: int = 0,
        end_page: Optional[int] = None
    ) -> Dict[int, str]:
        """
        Extract text from PDF using fitz (PyMuPDF).
        
        Args:
            pdf_path: Path to the PDF file.
            start_page: Starting page number (0-indexed).
            end_page: Ending page number (0-indexed). If None, process all pages.
            
        Returns:
            Dictionary with page numbers as keys and extracted text as values.
            
        Example:
            >>> text_dict = Document.extract_text_from_pdf("document.pdf")
            >>> print(text_dict[0])  # Text from first page
        """
        text_dict = {}
        
        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            
            if end_page is None:
                end_page = total_pages - 1
            
            end_page = min(end_page, total_pages - 1)
            
            for page_num in range(start_page, end_page + 1):
                page = pdf_document.load_page(page_num)
                text = page.get_text()
                text_dict[page_num] = text
            
            pdf_document.close()
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {e}")
        
        return text_dict
    
    @staticmethod
    def convert_pdf_to_images(
        pdf_path: str,
        dpi: int = 115,
        output_format: str = 'jpeg',
        a4_resize: bool = True,
        target_width: int = 954,  # A4 width at 115 DPI
        target_height: int = 1347,  # A4 height at 115 DPI
        crop_box_mm: Optional[tuple] = None,
        enhance_for_ocr: bool = False,
        start_page: int = 0,
        end_page: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Convert PDF pages to images with advanced processing options.
        
        Args:
            pdf_path: Path to the PDF file.
            dpi: Resolution for image conversion.
            output_format: Output image format ('PNG', 'JPEG', etc.).
            a4_resize: Whether to resize images to A4 dimensions.
            target_width: Target width for A4 resize (pixels).
            target_height: Target height for A4 resize (pixels).
            crop_box_mm: Optional cropping area in millimeters (left, top, right, bottom).
            enhance_for_ocr: Whether to apply OCR enhancement filters.
            start_page: Starting page number (0-indexed).
            end_page: Ending page number (0-indexed). If None, process all pages.
            
        Returns:
            List of PIL Image objects.
            
        Example:
            >>> images = Document.convert_pdf_to_images(
            ...     "document.pdf", 
            ...     dpi=300, 
            ...     enhance_for_ocr=True
            ... )
        """
        images = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            
            if end_page is None:
                end_page = total_pages - 1
            
            end_page = min(end_page, total_pages - 1)
            
            for page_num in range(start_page, end_page + 1):
                page = pdf_document.load_page(page_num)
                
                # Apply cropping if specified
                if crop_box_mm:
                    crop_box_pixels = (
                        crop_box_mm[0] * dpi / 25.4,
                        crop_box_mm[1] * dpi / 25.4,
                        page.rect.width * dpi / 72 - crop_box_mm[2] * dpi / 25.4,
                        page.rect.height * dpi / 72 - crop_box_mm[3] * dpi / 25.4
                    )
                    clip = fitz.Rect(*crop_box_pixels)
                    pix = page.get_pixmap(dpi=dpi, clip=clip)
                else:
                    pix = page.get_pixmap(dpi=dpi)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Resize to A4 if requested
                if a4_resize:
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # Apply OCR enhancement if requested
                if enhance_for_ocr:
                    img = Document._enhance_image_for_ocr(img)
                
                images.append(img)
            
            pdf_document.close()
            
        except Exception as e:
            raise Exception(f"Error converting PDF to images: {e}")
        
        return images
    
    @staticmethod
    def _enhance_image_for_ocr(
        image: Image.Image,
        contrast_factor: float = 1.05,
        brightness_factor: float = 1.02,
        sharpness_factor: float = 1.05,
        enable_smoothing: bool = False
    ) -> Image.Image:
        """
        Apply image enhancement filters to improve OCR accuracy using PIL only.
        
        This method applies a series of enhancements:
        1. Convert to RGB if not already
        2. Increase contrast (conservative default)
        3. Apply brightness adjustment
        4. Apply sharpening filter
        5. Apply smoothing filter for noise reduction (PIL-based)
        
        Args:
            image: PIL Image object to enhance.
            contrast_factor: Contrast enhancement factor (1.0 = no change, default 1.05 for conservative enhancement).
            brightness_factor: Brightness adjustment factor (1.0 = no change).
            sharpness_factor: Sharpness enhancement factor (1.0 = no change, default 1.2).
            enable_smoothing: Whether to apply smoothing filter for noise reduction.
            
        Returns:
            Enhanced PIL Image object.
        """
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply contrast enhancement
        if contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
        
        # Apply brightness adjustment
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
        
        # Apply sharpening
        if sharpness_factor != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness_factor)
        
        # Apply smoothing for noise reduction (PIL-based alternative to OpenCV denoising)
        if enable_smoothing:
            image = image.filter(ImageFilter.SMOOTH_MORE)
        
        return image
    
    @staticmethod
    def process_image_file(
        image_path: str,
        enhance_for_ocr: bool = False,
        resize_to_a4: bool = False,
        target_width: int = 954,
        target_height: int = 1347,
        adjust_contrast: float = 1.0,
        adjust_brightness: float = 1.0,
        adjust_sharpness: float = 1.0,
        enable_smoothing: bool = False
    ) -> Image.Image:
        """
        Process an image file with various enhancement options using PIL only.
        
        Args:
            image_path: Path to the image file.
            enhance_for_ocr: Apply comprehensive OCR enhancement.
            resize_to_a4: Resize image to A4 dimensions.
            target_width: Target width for resizing.
            target_height: Target height for resizing.
            adjust_contrast: Contrast adjustment factor (1.0 = no change).
            adjust_brightness: Brightness adjustment factor (1.0 = no change).
            adjust_sharpness: Sharpness adjustment factor (1.0 = no change).
            enable_smoothing: Apply smoothing filter for noise reduction.
            
        Returns:
            Processed PIL Image object.
            
        Example:
            >>> img = Document.process_image_file(
            ...     "document.jpg",
            ...     enhance_for_ocr=True,
            ...     resize_to_a4=True
            ... )
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to A4 if requested
            if resize_to_a4:
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Apply comprehensive OCR enhancement
            if enhance_for_ocr:
                image = Document._enhance_image_for_ocr(image)
            else:
                # Apply individual enhancements if requested
                if adjust_contrast != 1.0:
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(adjust_contrast)
                
                if adjust_brightness != 1.0:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(adjust_brightness)
                
                if adjust_sharpness != 1.0:
                    enhancer = ImageEnhance.Sharpness(image)
                    image = enhancer.enhance(adjust_sharpness)
                
                if enable_smoothing:
                    image = image.filter(ImageFilter.SMOOTH_MORE)
            
            return image
            
        except Exception as e:
            raise Exception(f"Error processing image file: {e}")
    
    @staticmethod
    def pdf_to_base64_images(
        pdf_path: str,
        dpi: int = 115,
        enhance_for_ocr: bool = True,
        a4_resize: bool = True,
        start_page: int = 0,
        end_page: Optional[int] = None
    ) -> List[str]:
        """
        Convert PDF to base64-encoded images for LLM processing.
        
        This method combines PDF conversion and base64 encoding in one step,
        optimized for LLM vision models.
        
        Args:
            pdf_path: Path to the PDF file.
            dpi: Resolution for image conversion.
            enhance_for_ocr: Apply OCR enhancement filters.
            a4_resize: Resize to A4 dimensions.
            start_page: Starting page number (0-indexed).
            end_page: Ending page number (0-indexed).
            
        Returns:
            List of base64-encoded image strings.
            
        Example:
            >>> base64_images = Document.pdf_to_base64_images(
            ...     "document.pdf",
            ...     dpi=300,
            ...     enhance_for_ocr=True
            ... )
        """
        # Convert PDF to images
        images = Document.convert_pdf_to_images(
            pdf_path=pdf_path,
            dpi=dpi,
            enhance_for_ocr=enhance_for_ocr,
            a4_resize=a4_resize,
            start_page=start_page,
            end_page=end_page
        )
        
        # Convert images to base64
        base64_images = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(base64_string)
        
        return base64_images
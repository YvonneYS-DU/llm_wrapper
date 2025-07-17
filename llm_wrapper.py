"""
LLM Wrapper - A streamlined interface for LangChain AI agent creation.

This module provides a simplified API for creating and managing AI agents
with multi-modal support (text and images) using LangChain framework.

Author: LLM Wrapper Team
Version: 0.1
Date: January 2025
"""

import re
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Generator
import base64

import fitz
from PIL import Image
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

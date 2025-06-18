#llm wrapper @01/Apl/2025
import json
import re
import asyncio
from datetime import datetime
import base64
import fitz
from PIL import Image
from io import BytesIO

# Importing necessary modules and classes from OpenAI and LangChain
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser


    
class Agents:
    def __init__(self):
        self.agents = []
        
    @staticmethod
    def _text_prompts(text_prompt_template = 'text prompt template'):
        """
        text prompt template: string of prompt
        return text prompt template object
        """
        text_prompts = HumanMessagePromptTemplate.from_template(
            [{'text': text_prompt_template}]
        )
        return text_prompts
    
    @staticmethod
    def _image_prompts():
        """
        image prompt template using path
        return the image prompt template (path)
        """
        image_prompts = HumanMessagePromptTemplate.from_template(
            [{'image_url': {'path': '{image_path}', 'detail': '{detail_parameter}'}}]
        )
        return image_prompts
    
    @staticmethod
    def _convert_pdf_to_base64_img_list(pdf_path, dpi=100, crop_box_mm=(10, 15, 10, 25)):
        """
        Convert a PDF to base64-encoded images with optional cropping.

        Args:
            pdf_path (str): Path to the PDF
            dpi (int): Image resolution
            crop_box_mm (tuple): Crop area in mm (left, upper, right, lower)

        Returns:
            list: List of base64-encoded images
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
    def _image_prompts_base64():
        """
        image prompt template using base64 image
        return the image prompt template (base64 image)
        suitable claude and opneai vison-llm (sonnet, 4o, 4o-mini, and etc.)
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
    def _image_prompts_base64_multi(base64_image, detail_parameter):
        """
        image prompt template using base64 image
        upload multiple images in one prompt (conversation)
        base64_image: base64 image string
        detail_parameter: string, default 'auto', openai choices: ['high', 'low', 'auto']
        create the image prompt template (base64 image) ALREADY FILLed image and detail parameter
        return the FILLed image prompt template (base64 image)
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
    def list_to_img_dict(img_list):
        """
        Convert a list of images to a dictionary with keys as 'img1', 'img2', ...
        img_list: List of images (e.g., base64 strings or URLs)

        Returns a dictionary with image keys.
        """
        return {f"img{i+1}": img for i, img in enumerate(img_list)}
    
    @staticmethod
    def _system_prompts(system_prompt_template = 'system prompt template'):
        """
        system prompt template: string of prompt
        used for agent creation, current not used in the chainmaker
        return system prompt template object
        """
        system_prompts = SystemMessagePromptTemplate.from_template(
            system_prompt_template
        )
        return system_prompts
    
    @staticmethod
    def extract_prompts_parameters(prompt_template):
        """
        Extract the parameters from the prompt template without duplicates.
        prompt_template: string of prompt
        return list of unique parameters
        """
        # use regex to extract the parameters
        parameters = re.findall(r'{(.*?)}', prompt_template)
        # remove duplicates
        unique_parameters = list(set(parameters))
        return unique_parameters
    
    @staticmethod
    def lc_prompt_template(text_prompt_template = 'text prompt template', system_prompt_template=None, image_prompt_template = False, image_list=[], fill_img = True):
        """
        text_prompt_template: string of prompt
        image_prompt_template: bool, default False, if True, image prompt will be added
            image_prompts = HumanMessagePromptTemplate.from_template(
                [{'image_url': {'path': '{image_path}', 'detail': '{detail_parameter}'}}]
            )
        return the chat prompt template object
        """
        if system_prompt_template: # system prompt
            chat_prompt_template = ChatPromptTemplate.from_messages(
                messages=[
                    Agents._system_prompts(system_prompt_template),
                    Agents._text_prompts(text_prompt_template),
                    *([Agents._image_prompts_base64] if image_prompt_template else []), # default need base64 image
                ]
            )
        else:
            if image_list: # multi-image prompt
                chat_prompt_template = Agents.multi_image_templates(text_prompt_template=text_prompt_template, image_list=image_list, fill_img=fill_img)
            else: # zero/single-image prompt
                chat_prompt_template = ChatPromptTemplate.from_messages(
                    messages=[
                        Agents._text_prompts(text_prompt_template),  
                        *([Agents._image_prompts_base64()] if image_prompt_template else [])
                    ])
        return chat_prompt_template
    
    @staticmethod
    def multi_image_templates(text_prompt_template='text prompt template', fill_img=True, image_list=[], detail_parameter = 'high'):
        """
        This is a multi-image prompt template.
        text_prompt_template: string of prompt
        image_in_prompt: bool, default True, determines whether to include images in the prompt
        image_list: list of base64 images

        Returns the prompt template object (with multi-image or placeholder templates)
        """
        # Text prompt template
        text_prompts = [Agents._text_prompts(text_prompt_template)]
        
        if fill_img:
            # Multi-image prompt template with images
            image_prompts = [
                Agents._image_prompts_base64_multi(image, detail_parameter)
                for image in image_list
            ]
        else:
            # Placeholder image prompts with unique parameter names (img1, img2, ...)
            image_prompts = [
                Agents._image_prompts_base64_multi(f"{base64_image_placeholder}", detail_parameter)
                for base64_image_placeholder in [f"img{i+1}" for i in range(len(image_list))]
            ]

        # compose the chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            messages = text_prompts + image_prompts
        )
        # return the chat prompt template
        # chain = chat_prompt_template | llm | output_parser
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
    def chain_create(model, system_prompt_template='', text_prompt_template='prompt tamplate string', image_prompt_template = False, output_parser = StrOutputParser, format_var_name='schema', image_list=[], fill_image = False, parameters=False):  # llm model chreate by llm_model, prompt template string, choose to PRINT the PromptTamplate parameters or not
        """
        IMPORTANT: this is the main function to create the chain
        model: llm model object
        text_prompt_template: string, default 'prompt tamplate string', text prompt template
        image_prompt_template: bool, default False, image prompt template

        system_prompt_template: string, default '', system prompt template, current not used in the chainmaker

        EXAMPLE:
        chain = Agents.chain_create(llm, text_prompt_template='text prompt template', image_prompt_template=False)
        then no need to directly call Agents.lc_prompt_template(text_prompt_template='text prompt template', image_prompt_template=False)
        
        return the chain object
        """

        LC_prompt_template = Agents.lc_prompt_template(text_prompt_template = text_prompt_template, system_prompt_template = system_prompt_template, image_prompt_template = image_prompt_template, image_list=image_list, fill_img=fill_image)
        llm = model
        if hasattr(output_parser, 'get_format_instructions'):
            if f"{{{format_var_name}}}" in text_prompt_template:
                partial_dict = {format_var_name: output_parser.get_format_instructions()}
                LC_prompt_template = LC_prompt_template.partial(**partial_dict)
            else:
                print(f"Warning: {format_var_name} not found in prompt template, skipping format instructions")
        
        if not parameters:
            chain = LC_prompt_template | llm | output_parser
            return chain
        else:
            parameters = Agents.extract_prompts_parameters(text_prompt_template) 
            chain = LC_prompt_template | llm | output_parser
            print("Parameters:", parameters)
            return chain, parameters
   
    @staticmethod
    async def _delay(seconds: float):
        empty_loop = asyncio.get_running_loop()
        future = empty_loop.create_future()
        empty_loop.call_later(seconds, future.set_result, None)
        await future
        
    @staticmethod
    def chain_stream_generator(chain, dic={}): # gnerate response in stream, to generate respoonse, CHAIN(template, model) and DIC of parameters are required
        """
        stream response in generator, chunk by chunk
        chain: chain object - Agents.chain_create()
        dic: dictionary of parameters dict to fill all the parameters in the prompt, template show as {parameter: value}
        
        return the response in stream
        """
        for chunk in chain.stream(dic):
            yield chunk.content


    @staticmethod
    def chain_batch_generator(chain, dic=None, max_retries=2):
        """
        batch generate response
        chain: chain object - Agents.chain_create()
        dic: dictionary of parameters dict to fill all the parameters in the prompt, template show as {parameter: value}
        
        return the response in batch
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
                    raise Exception(f"AI encountered some issues, please try again later{e}")
                else:
                    continue
    
    
    @staticmethod
    async def chain_batch_generator_async(chain, dic={}, delay=None, max_retries=2):
        """
        async batch generate response
        this is mainly used in the async function call to analysis images / generate images
        for images async call, please refer to https://github.com/Bingzhi-Du/AI_text_extractor
        
        return the response in batch
        """
        attempt = 0
        print("taks start at:", datetime.now())
        if delay:
            print("Waiting for", delay, "seconds before starting the task.")
            await Agents._delay(delay)
        while attempt <= max_retries:
            print("Attempting to invoke the chain...")
            try:
                response = await chain.ainvoke(dic)
                attempt += 1
                return response
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                attempt += 1
                if attempt > max_retries:
                    print("Max retries exceeded. with error:", e)
                    return e
                continue
        return response
        
    @staticmethod
    def output_parser(output_string):
        """
        this will parse the output TSV string from the llm model
        Parse a TSV string into a JSON array of objects,
        removing any code block wrappers (```) if present.
        Specifically handles tab-separated values.
        Args:
            output_string (str): The input text which may contain TSV data with wrappers
        Returns:
            list: List of dictionaries where keys are headers and values are row values
        """
        # Remove code block wrappers if present
        tsv_string = re.sub(r'^```.*?\n|```$', '', output_string, flags=re.DOTALL).strip()
        # Split the TSV string into lines
        lines = tsv_string.strip().split('\n')
        # Extract header row and split by tab character
        headers = lines[0].split('\t')
        headers = [h.strip() for h in headers]  # Clean up any extra whitespace
        # Initialize result list
        result = []
        # Process each data row (skip the header row)
        for i in range(1, len(lines)):
            # Skip empty lines
            if not lines[i].strip():
                continue
            # Split row by tab character
            values = lines[i].split('\t')
            values = [v.strip() for v in values]  # Clean up any extra whitespace
            # Create a dictionary for the current row
            row_dict = {}
            # Map each value to its corresponding header
            for j, header in enumerate(headers):
                # Use empty string if value doesn't exist
                row_dict[header] = values[j] if j < len(values) else ''
            # Add the dictionary to the result list
            result.append(row_dict)
        return result

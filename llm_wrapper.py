import json
import re
import asyncio
from datetime import datetime

# Importing necessary modules and classes from OpenAI and LangChain
from openai import OpenAI
#from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
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
    def _image_prompts_base64():
        """
        image prompt template using base64 image
        return the image prompt template (base64 image)
        suitable claude and opneai vison-llm (sonnet, 4o, 4o-mini, and etc.)
        """
        image_prompts = HumanMessagePromptTemplate.from_template(
            [{"type": "image_url", 'image_url': {"url": "data:image/jpeg;base64,{base64_image}", 'detail': '{detail_parameter}'}}]
        )
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
        # build the url part
        url_part = f"data:image/jpeg;base64,{base64_image}"
        
        # build the detail part
        detail_part = f"{detail_parameter}"
        
        # formulate the template string
        template_string = [{"image_url": {"url": url_part, "detail": detail_part}}] # detail: high, low, auto
        
        # template to object
        image_prompts = HumanMessagePromptTemplate.from_template(template_string)
        
        return image_prompts
    
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
    def lc_prompt_template(text_prompt_template = 'text prompt template', image_prompt_template = False):
        """
        text_prompt_template: string of prompt
        image_prompt_template: bool, default False, if True, image prompt will be added
            image_prompts = HumanMessagePromptTemplate.from_template(
                [{'image_url': {'path': '{image_path}', 'detail': '{detail_parameter}'}}]
            )
        return the chat prompt template object
        """
        chat_prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                Agents._text_prompts(text_prompt_template),
                *([Agents._image_prompts_base64()] if image_prompt_template else []), # default need base64 image
            ])
        return chat_prompt_template
    
    @staticmethod
    def multi_image_templates(text_prompt_template='text prompt template', image_prompt_template=False, image_list=[]):
        """
        this is a multi-image prompt template
        text_prompt_template: string of prompt
        image_prompt_template: bool, default False,
        image_list: list of base64 images

        return the prompt template object (with multi-image)
        """
        # text prompt template
        text_prompts = [Agents._text_prompts(text_prompt_template)]
        # multi-image prompt template
        image_prompts = [
            Agents._image_prompts_base64_multi(image, "auto")
            for image in image_list
        ] if image_prompt_template else []

        # compose the chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            messages = text_prompts + image_prompts
        )
        # return the chat prompt template
        # chain = chat_prompt_template | llm | output_parser
        return chat_prompt_template
    
    @staticmethod
    def generate_image(prompt, number_of_images=1, size='1792x1024', style='natural', quality = 'standard', api_key='api_key'):
        """
        prompt: string of prompt to generate image
        number_of_images: int, default 1, number of images to generate
        size: string, default '1792x1024', size of the image, openai choices: ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024']
        style: string, default 'natural', disabled for bad image quality, openai choices: ['vivid', 'natural']
        quality: string, default 'standard', openai choices: ['standard', 'hd']
        api_key: string, openai api key

        return the response of the image generation
        """
        client = OpenAI(api_key=api_key)    
        response = client.images.generate(
        model = "dall-e-3",
        prompt = prompt,
        size = size, # ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024']
        n = number_of_images,
        #style = style, #['vivid', 'natural']
        quality = quality #['standard', 'hd']
        )
        return response
    
    @staticmethod
    def llm_model(
            model: str = 'openai',  
            model_name: str = 'gpt-4o', 
            temperature: float = 0.7, 
            azure_endpoint: str = None 
    ):
        if model == 'openai':
            llm = AzureChatOpenAI(
            azure_deployment = model_name,
            azure_endpoint = azure_endpoint,
            temperature = temperature,
            )
            return llm

        else:
            raise ValueError('Model configuration error, check the lambda env config whether in the langchain model list.')
    
    @staticmethod
    def chain_create(model, system_prompt_template='', text_prompt_template='prompt tamplate string', image_prompt_template = False, output_parser = StrOutputParser, parameters=False):  # llm model chreate by llm_model, prompt template string, choose to PRINT the PromptTamplate parameters or not
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

        LC_prompt_template = Agents.lc_prompt_template(text_prompt_template = text_prompt_template, image_prompt_template = image_prompt_template)
        llm = model
        output_parser = output_parser()
        if not parameters:
            chain = LC_prompt_template | llm | output_parser      #return chain: "prompt template | llm model"
            return chain
        else:
            parameters = Agents._extract_prompts_parameters(text_prompt_template)  # list of parameters
            chain = LC_prompt_template | llm | output_parser
            print("Parameters:", parameters)
            return chain, parameters
        
    @staticmethod
    def chain_stream_generator(chain, dic): # gnerate response in stream, to generate respoonse, CHAIN(template, model) and DIC of parameters are required
        """
        stream response in generator, chunk by chunk
        chain: chain object - Agents.chain_create()
        dic: dictionary of parameters dict to fill all the parameters in the prompt, template show as {parameter: value}
        
        return the response in stream
        """
        for chunk in chain.stream(dic):
            yield chunk.content

    @staticmethod
    def chain_batch_generator(chain, dic):
        """
        batch generate response
        chain: chain object - Agents.chain_create()
        dic: dictionary of parameters dict to fill all the parameters in the prompt, template show as {parameter: value}
        
        return the response in batch
        """
        response = chain.invoke(dic)
        return response
    
    @staticmethod
    async def chain_batch_generator_async(chain, dic):
        """
        async batch generate response
        this is mainly used in the async function call to analysis images / generate images
        for images async call, please refer to https://github.com/Bingzhi-Du/AI_text_extractor
        
        return the response in batch
        """
        print("taks start at:", datetime.now())
        response = await chain.ainvoke(dic)
        return response
        
    @staticmethod
    def sub_agent(llm, sub_agent_prompt_dic): # create sub agents
        """
        a dict of sub agents with name and prompt template string
        {'sub_agent_name': 'prompt template string', ...}
        
        return a dict of sub agents with name and chain object
        """
        chains = {}
        for key, value in sub_agent_prompt_dic.items():
            chain_name = key
            prompt_template = Agents.lc_prompt_template(value)
            chains[chain_name] = prompt_template | llm
        return chains

    @staticmethod
    def create_agent(llm, system_prompt_template='system prompt template', text_prompt_template='text prompt template', tools=[]):
        """
        create the agent with system prompt template
        but is similar to chain_create
        temparary not used in the chainmaker

        return the agent object
        """
        system_prompt = Agents._system_prompts(system_prompt_template)
        
class API_unpack:
    """
    this is the resolver of api passed to call llm.
    """


    @staticmethod
    def _PromptImporter(prompt_template='prompt tamplate string'):
        """
        get the prompt template object.
        """
        warning = "[WARNING]: please use [Agents.chain_create(__, prompt_tamplate = 'prompt tamplate string')], this is a developer feature."
        return warning, Agents.lc_prompt_template(prompt_template=prompt_template)
    
    @staticmethod
    def model_config(config_dic, required_keys=['model', 'model_name', 'temperature', 'api_key', 'streaming']):
        """
        may replaced by lambda env config.
        """
        default_config = {
            'model': 'openai',
            'model_name': 'gpt-4',
            'temperature': 0.7,
            'api_key': 'api_key',
            'streaming': True
        }
        final_config = {}
        # replace the default config with the preferred config
        for key in required_keys:
            if key in config_dic:
                final_config[key] = config_dic[key]
            else:
                final_config[key] = default_config[key]
        return final_config
    
    @staticmethod
    def get_prompt_name(json_from_api):
        """
        Get the prompt name from the api response.
        """
        try:
            # Json data from API
            if isinstance(json_from_api, str):
                json_data = json.loads(json_from_api)
            else:
                json_data = json_from_api

            # Get the prompt_name from the API response
            for item in json_data.get('prompt', []):
                if item.get('type') == 'sys':
                    return item.get('prompt_name', None)
        
            # If no prompt_name found
            print("Error: prompt['type']='sys' not found, or prompt_name not found in the API response.")
            return None

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing JSON: {e}.")
            return None

    @staticmethod
    def data_parameters(data_dic, required_list='prompt_template'):
        required_list = Agents._extract_prompts_parameters(required_list)
        for key in required_list:
            if key not in data_dic:
                raise ValueError(f"Missing key: {key}.")
        return {key: data_dic[key] for key in required_list}

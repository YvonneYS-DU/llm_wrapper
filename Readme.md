# intuitive ai agent prompt framework

The primary goal of llmwrapper is to provide a streamlined, developer-friendly interface that reduces the complexity of working with LangChain’s more intricate features. 

# Key Features and Technologies

The project is primarily built on LangChain’s architecture and encapsulates its core functions, with a design that focuses on:

*Simple prompt creation:* Making it easier to generate both text and image prompts.
*Support text and image prompts:* Supporting multi-modal prompt generation, including Base64-encoded image inputs.
*Chain creation:* Streamlining the process of constructing chains, which integrate prompts, models, and output parsers.
*Agent generation:* Offering an intuitive interface for creating agents that handle more complex tasks.
*Asynchronous execution:* Async API calling

# example using llm_wrapper
## import llm model

```bash
llm = Agents.llm_model(
    model='openai', 
    model_name='gpt-4o', 
    temperature=0.7, 
    api_key='your-api-key'
)
```

## set prompt template
prompt template have two feature now, to better illustrate, function name is lc_prompt_template(text_prompt_template, image_prompt_template):
```bash
text_prompt_template = 'text_prompt_template, is string, when using {}, means is a {parameter}，{parameter} must filled when, calling the api, {parameter} will replaced by f-string by langchain; when want just show '{' string, use /{/} or {{info}}
```

```bash
image_prompt_template = True / False, true meams use image template, false means not use image template.
```

## formulating a langchain prompt
when prompt template is set, we can build a chain in one line of code.
```bash
chain = Agents.chain_create(model = llm, text_prompt_template = "text_prompt_template", image_prompt_template = False, output_parser = StrOutputParser, parameters=False)
```
image_prompt_template = False/True, False means not use image template, True means use image template. when image_prompt_template is True, image should be processed into base64 format, when calling the function.
output_parser = StrOutputParser/JsonOutputParser, StrOutputParser means AI output is string, JsonOutputParser means AI output is json format.

## calling the chain
aysnc call:
```bash
output = await Agents.chain_batch_generator_async(chain, dic)
```
sync call:
```bash
output = Agents.chain_batch_generator(chain, dic)
```
streaming call:
```bash
output = Agents.chain_stream_generator(chain, dic)
```
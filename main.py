import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, SafetySetting, Tool
from vertexai.preview.generative_models import grounding
from dotenv import load_dotenv

load_dotenv()
project = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION")
model_name = os.getenv("MODEL_NAME")

def multiturn_generate_content():
    vertexai.init(project=project, location=location)
    tools = [
        Tool.from_google_search_retrieval(
            google_search_retrieval=grounding.GoogleSearchRetrieval()
        ),
    ]
    model = GenerativeModel(
        model_name,
        tools=tools,
        system_instruction=["""you are a helpful agent that help me choose a restaurant based on user food preference and location"""]
    )
    chat = model.start_chat()
    while True:
        user_input = input("User: ")
        print(chat.send_message(
            [user_input],
            generation_config=generation_config,
            safety_settings=safety_settings
        ).candidates[0].content.parts[0].text)


generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

if __name__ == "__main__":
    multiturn_generate_content()

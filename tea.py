"""
Task Embedding Agent, tea.py
1.	Analyze the user’s request and LIST the related hardware (MCU, Sensors, etc.)
2.	Find the information of the related data in memory. If failed, then
    a)	From all the images, find the corresponding hardware
    b)	Analyze the hardware and store its information in memory
3.	Forward the task description and the information of related hardware to the WA and CA
"""
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.agents import ChatAgent, TaskSpecifyAgent
from camel.toolkits import FunctionTool
from camel.toolkits import (
    SearchToolkit,
    AsyncBrowserToolkit,
    ImageAnalysisToolkit,
    FileWriteToolkit,
    MemoryToolkit,
)
from camel.prompts import TextPrompt
from PIL import Image
from camel.types import ModelPlatformType, ModelType, TaskType
from colorama import Fore
import os
import json

from dotenv import load_dotenv

tsa_prompt = """
You are an AI Society Task Specify Agent (TaskSpecifyAgent).
Your responsibilities are:
1. Analyze the user's main goal (task_prompt).
2. Consider the role information provided in meta_dict (assistant_role, user_role) and any constraints (e.g., word_limit).
3. Break down the overall goal into 3–5 clear, ordered, and actionable subtasks.
4. For each subtask, provide:
   - a short title
   - a concise description
5. Output the result in JSON format exactly as follows:

{
  "objective": "<the original task_prompt>",
  "roles": {
    "assistant": "<assistant_role>",
    "user": "<user_role>"
  },
  "subtasks": [
    {
      "title": "First subtask title",
      "description": "Description of what to do for the first subtask."
    },
    {
      "title": "Second subtask title",
      "description": "Description of what to do for the second subtask."
    },
    ...
  ]
}
"""

ba_prompt = """
You are a Hardware Retrieval and Analysis Agent.
Your responsibilities are:
1. Receive a user message that may include uploaded images and a task description.
2. First, attempt to retrieve existing hardware information (MCU, sensors, etc.) from memory.
3. If memory lookup fails for any hardware mentioned or detected:
   a. Analyze each uploaded image to identify hardware models and key specs.
   b. Perform an online search for missing details (e.g., datasheets, pinouts) using SearchToolkit and BrowserToolkit.
   c. Store all newly found hardware information back into memory for future reuse.
4. Your response should be a single JSON object with:
   - "hardware_list": a list of objects, each containing:
       - "model": hardware model name
       - "specs": key specifications (e.g., pin count, voltage, interfaces)
       - "source": "memory", "image_analysis", or "web_search"
   - "task_description": the original user task
5. Your response must not content any other element 
6. Do not proceed to planning; just gather and return the complete hardware_list.

Example output:
{
  "task_description": "I want to set up a temperature and humidity monitor using ESP32S3 and DHT11.",
  "hardware_list": [
    {
      "model": "ESP32-S3",
      "specs": {"pins": 48, "voltage": "3.3V", "interfaces": ["I2C", "SPI", "UART"]},
      "source": "image_analysis"
    },
    {
      "model": "DHT11",
      "specs": {"voltage": "3.3V–5V", "interface": "1-wire", "sampling_rate": "1Hz"},
      "source": "web_search"
    }
  ]
}
"""


user_msg_content = '''
I want to set up a temperature and humidity monitor using the given hardware. Each stored in MCU.jpg and DHT11.jpg
Provide a proper wiring plan.
'''

img_dir = 'imgs'
exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')


def create_model(
        model_platform: ModelPlatformType = ModelPlatformType.OPENAI,
        model_type: ModelType = ModelType.GPT_4_TURBO,
):
    model = ModelFactory.create(model_platform=model_platform, model_type=model_type)
    return model

def create_task_specify_agent(prompt: str = tsa_prompt):
    agent = TaskSpecifyAgent(
        model=create_model(),
        task_type=TaskType.AI_SOCIETY,
        task_specify_prompt=prompt,
        output_language='English',
    )
    return agent

def create_base_agent(prompt: str = ba_prompt):
    agent = ChatAgent(
        system_message=TextPrompt(prompt),
        model=create_model(),
        tools=[
            *AsyncBrowserToolkit().get_tools(),
            *ImageAnalysisToolkit().get_tools(),
            *FileWriteToolkit().get_tools(),
        ],
        output_language='json',
    )
    return agent



def set_user_msg():
    images = []
    for fname in os.listdir(img_dir):
        if fname.lower().endswith(exts):
            path = os.path.join(img_dir, fname)
            try:
                img = Image.open(path)
                img.filename = fname
                images.append(img)
            except IOError:
                print(f'Failed to open {path}')
    msg = BaseMessage.make_user_message(
        role_name="User",
        content=user_msg_content,
        image_list=images,
    )
    return msg

if __name__ == '__main__':
    load_dotenv()
    task_specify_agent = create_task_specify_agent()
    base_agent = create_base_agent()
    user_msg = set_user_msg()
    ba_resp = base_agent.step(user_msg)
    ba_output = ba_resp.msgs[0].content
    print(Fore.CYAN + "=== BaseAgent Output ===")
    print(ba_output)
    data = json.loads(ba_output)
    hardware_list = data["hardware_list"]
    task_text     = data["task_description"]
    ts_prompt = (
        f"Task: {task_text}\n\n"
        f"Hardware you discovered:\n{json.dumps(hardware_list, indent=2)}"
    )
    ts_output = task_specify_agent.run(
        task_prompt=ts_prompt,
        meta_dict={
            "assistant_role": "Embedded HW Expert",
            "user_role": "Engineer",
            "word_limit": 150
        }
    )
    print(Fore.GREEN + "\n=== TaskSpecifyAgent Output ===")
    print(ts_output)


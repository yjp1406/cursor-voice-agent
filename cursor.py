from dotenv import load_dotenv
from openai import OpenAI
import json
import requests
from pydantic import BaseModel, Field
from typing import Optional
import os
import asyncio
import speech_recognition as sr
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

load_dotenv()
client = OpenAI()

async_client = AsyncOpenAI()

async def tts(speech: str):
    async with async_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        instructions="Always speak in cheerfull manner with full of delight and happy",
        input=speech,
        response_format="pcm"
    ) as response:
        await LocalAudioPlayer().play(response)


def run_command(cmd: str):
    result = os.system(cmd)
    return result

def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if(response.status_code == 200):
        return f"The weather in {city} is {response.text}"
    
    return "Something went wrong"

SYSTEM_PROMPT = """
    You're an expert AI Assistant in resolving user queries using chain of thought.
    You work on START, PLAN and OUTPUT steps.
    You need to first PLAN what needs to be done. The PLAN can be multiple steps.
    Once you think enought PLAN has been done, finally you can give an OUTPUT.
    You can also call a tool if required from the list of available tools.
    for every tool call wait for the observe step which is the output from the called tool.

    Rules:
    - Strictly follow the given JSON output format
    - Only run one step at a time.
    - The sequence of steps is START (where  user gives an input), PLAN (that can be multiple times) and 
    finally OUTPUT (which is going to be displayed to the user).

    Output JSON format:
    {"step": "START" | "PLAN" | "OUTPUT" | "TOOL", "content": "string", "tool":"string", "input":"string"}


    Available Tools:
    - get_weather(city: str): Takes city name as an input string and returns the weather info about the city.
    - run_command(cmd: str): Takes a system linux command as string and executes the command on user's system and returns the output from that command.

    Example1:
    START: Hey, Can you solve 2 + 3 * 5 / 10
    PLAN: {"step": "PLAN","content": "Seems like user is interested in maths problem"}
    PLAN: {"step": "PLAN","content": "Looking at the problem, we should solve this using BODMAS method"}
    PLAN: {"step": "PLAN","content": "Yes, The BODMAS is correct thing to be done here"}
    PLAN: {"step": "PLAN","content": "First we must multiply 3*5 which is 15"}
    PLAN: {"step": "PLAN","content": "Now the new equation is 2 + 15 / 10"}
    PLAN: {"step": "PLAN","content": "we must perform divide that is 15/10 = 1.5"}
    PLAN: {"step": "PLAN","content": "Now the new equation is 2 + 1.5"}
    PLAN: {"step": "PLAN","content": "Now finally lets perform the add 2+1.5 = 3.5"}
    PLAN: {"step": "PLAN","content": "Great, we have solved and finally left with 3.5 as ans"}
    PLAN: {"step": "OUTPUT","content": "3.5"}

    Example2:
    START: What is the weather of Delhi?
    PLAN: {"step": "PLAN","content": "Seems like user is interested in getting weather of Delhi in India"}
    PLAN: {"step": "PLAN","content": "Lets see if we have any available tools from the available tools list"}
    PLAN: {"step": "PLAN","content": "Great, we have get_weather tool available for this query"}
    PLAN: {"step": "PLAN","content": "I need to call get_weather tool for delhi as input for city"}
    PLAN: {"step": "TOOL", "tool" : get_weather, "input": "delhi"}
    PLAN: {"step": "OBSERVE", "tool" : "get_weather", "output": "The temp of delhi is cloudy with 20 C"}
    PLAN: {"step": "PLAN","content": "Great I got the weather info about delhi"}
    PLAN: {"step": "OUTPUT","content": "The current weather in delhi is 20 C with some clouds"}

"""

print("\n\n\n")

class MyOutputFormat(BaseModel):
    step: str = Field(...,description="The ID of the step. Example: PLAN, OUTPUT, TOOL, etc")
    content: Optional[str] = Field(None, description="Optional string content")
    tool: Optional[str] = Field(None, description="The ID of the tool to call.")
    input: Optional[str] = Field(None, description="The input params for the tool")

available_tools = {
    "get_weather": get_weather,
    "run_command": run_command
}

message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

r = sr.Recognizer()
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    r.pause_threshold = 2


    while True:
        print("Speak Something...")
        audio = r.listen(source)

        print("Processing Audio... (STT)")
        stt = r.recognize_google(audio)
        # user_query = input("-> ")
        message_history.append({"role":"user", "content": stt})

        while True:
            response = client.chat.completions.parse(
            model="gpt-4o-mini",
            response_format=MyOutputFormat,
            messages=message_history
            )

            raw_result = response.choices[0].message.content
            message_history.append({"role":"assistant", "content":raw_result})

            parsed_result = response.choices[0].message.parsed
            

            if parsed_result.step == "START":
                print("Start ", parsed_result.content)
                continue

            if parsed_result.step == "TOOL":
                tool_to_call = parsed_result.tool
                tool_input = parsed_result.input
                # print("Tool ", parsed_result.get("content"))
                tool_response = available_tools[tool_to_call](tool_input)
                print(f"Tool: {tool_to_call}  ({tool_input}) =  {tool_response}")
                message_history.append({"role":"developer", "content": json.dumps({"step":"OBSERVE", "tool" :tool_to_call, "input":tool_input, "output": tool_response})})
                continue

            if parsed_result.step == "PLAN":
                print("Thinking ", parsed_result.content)
                continue

            if parsed_result.step == "OUTPUT":
                asyncio.run(tts(parsed_result.content))
                break


print("\n\n\n")
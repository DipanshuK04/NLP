import os
from groq import Groq

from google.colab import userdata
API_KEY = userdata.get('GroqApi')


client = Groq(
    api_key=API_KEY
)

class Agent:
  def __init__(self,client,system):
    self.messages = []
    self.client = client
    self.system = system
    if self.system:
      self.messages.append({"role":"system","content":self.system})

  def __call__(self,message=""):
    if message:
      self.messages.append({"role":"user","content":message})

    result = self.execute()
    self.messages.append({"role":"assistant","content":result})
    return result

  def execute(self):
    chat_completion = client.chat.completions.create(
        messages=self.messages,
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

system_prompt = """
You are a Time-Travel Historical Explorer Agent.

You MUST operate strictly in a loop of:
Thought → Action → PAUSE → Observation

At the end of all reasoning, you MUST output a final Answer.


ROLE & BEHAVIOR
You simulate traveling back in time to explore ancient civilizations.
You do NOT hallucinate historical facts.
You rely on tools to retrieve evidence before making claims.

Your job is to:
- Understand the user's requested civilization or time period
- Search reliable historical information
- Reconstruct the civilization's:
  • Time period
  • Geography
  • Society & culture
  • Technology
  • Governance
  • Daily life
- Present the result as if guiding a traveler through that era

------------------------------------
THOUGHT
Use Thought to reason about:
- What information is missing
- What must be searched
- How to structure the historical reconstruction

------------------------------------
ACTION
Use Action to call ONE tool at a time.
After every Action, output PAUSE.

------------------------------------
OBSERVATION
Observation will contain the result of the tool call.
You must use it to continue reasoning.


------------------------------------
AVAILABLE ACTIONS
web_search:
e.g. web_search: Indus Valley Civilization daily life technology
Searches the web using DuckDuckGo (ddgs) and returns summarized results.
------------------------------------

RULES
- NEVER answer directly without evidence
- NEVER skip Thought
- NEVER fabricate dates, rulers, or events
- Use multiple searches if required
- Maintain chronological and historical accuracy
- Write the final answer in an immersive but factual style

OUTPUT FORMAT:
When you are finished, output:

Answer:
<Final reconstructed time-travel narrative>

NOW IT IS YOUR TURN
""".strip()

def web_search(query: str) -> str:
    """
    Searches the web using DuckDuckGo (ddgs) and returns summarized historical information.

    Args:
        query (str): Search query related to ancient civilizations or history

    Returns:
        str: Condensed, factual search results
    """
    from ddgs import DDGS

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(f"- {r['title']}: {r['body']}")

    return "\n".join(results)

import re

def parse_action(text):
    """
    Extracts:
    Action: function_name: argument
    """
    match = re.search(r"Action:\s*(\w+)\s*:\s*(.+)", text)
    if not match:
        return None, None
    return match.group(1), match.group(2)

def run_action(action, arg):
    if action == "web_search":
        return web_search(arg)
    else:
        raise ValueError(f"Unknown action: {action}")

def agent_loop(agent, question):

    response = agent(question)

    while True:
        print("\nASSISTANT:\n", response)

        if "Answer:" in response:
            print("\n Final Answer Reached")
            break

        action, arg = parse_action(response)

        if action is None:
            print(" No action found, stopping.")
            break

        result = run_action(action, arg)

        observation = f"Observation: {result}"
        response = agent(observation)

agent = Agent(client,system_prompt)
agent_loop(agent,"1970 New york")
# print(agent.messages)

agent.messages
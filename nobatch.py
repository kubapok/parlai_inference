import time
from parlai.core.agents import create_agent_from_model_file

agent = create_agent_from_model_file("zoo:blender/blender_1Bdistill/model", {'model_parallel': False, 'skip_generation': False})

text = 'Hi, I am Alan.'

agent.observe({'text': text, 'episode_done': False})
response = agent.act()
print(response['text'])

for _ in range(100):
    start = time.time()
    agent.observe({'text': text, 'episode_done': False})
    response = agent.act()
    print(time.time() - start)

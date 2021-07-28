import time
from parlai.core.agents import create_agent_from_model_file

BS = 8
agent = create_agent_from_model_file(
        "zoo:blender/blender_1Bdistill/model", {'model_parallel': False, 'skip_generation': False}
            )

clones = [agent.clone() for _ in range(BS)]
acts = []
text = 'Hi, I am Alan.'

for i in range(100):
    start = time.time()
    acts = []
    for index in range(BS):
        acts.append(clones[index].observe({'text':text, 'episode_done': False}))
    responses = agent.batch_act(acts)

    #for i in range(BS):
    #    print(responses[i]['text'])
    print(time.time() - start)

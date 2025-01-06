"""
Standalone script that runs in background to determine the latency
"""
import time
import json
import concurrent
from openai import OpenAI

from params import ZEPHYR_API_KEY, ZEPHYR_API_BASE
client = OpenAI(
            api_key=ZEPHYR_API_KEY,
            base_url=ZEPHYR_API_BASE,
        )



def check_answer(i):
    try:
        prompt = \
           f"""user
            You are an assistant
            "In the context provided in ``` 
                    - Carefully create complete summary from above definitions and context provided in ``` to provide your answer.
                    - After creating the summary generate 20 similar stories, all should br full length.
                    - Lets think step by step."
            
            "```context: "A data scientist named Maya developed an algorithm to predict extreme weather patterns. Using historical climate data and machine learning, she discovered concerning patterns in ocean temperatures. Her model predicted upcoming weather anomalies with 92% accuracy, helping coastal communities prepare for storms. The success led to implementation across multiple weather stations, potentially saving thousands of lives.
            """
        t1 = time.time()
        completion = client.completions.create(
                        model= "HuggingFaceH4/zephyr-7b-beta",
                        prompt= prompt,
                        max_tokens=2048,
                        temperature=0.0
                    )
        completion.usage.time = time.time() - t1
    except:
        return None
    return completion

cache = []

def worker(num_processes):
    try:
        completion = check_answer(num_processes)
        completion.usage.parallel_requests = num_processes
        print(f"Completed index {num_processes}")
        result = completion.usage.dict()
        result['text'] = completion.choices[0].text
        result['tok_per_sec'] = result['completion_tokens']/result['time']
        cache.append( result )
    except concurrent.futures.TimeoutError:
        print(f"check_answer({num_processes}) timed out. Moving on to the next index.")

for num_processes in [1,8,16,24,32,64,100,128]:
    print(f"Running {num_processes} parallel requests...")
    t1 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(worker, num_processes) for _ in range(num_processes)]
    concurrent.futures.wait(futures)
    cache.append( {"parallel_requests":num_processes, "time":time.time()-t1 } )

    json.dump(
        cache,
        open(f"./benchmark_test2_long.json", "w"),
        indent=2
    )

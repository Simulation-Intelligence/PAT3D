import os
from openai import OpenAI


import ast
from time import sleep
import json
import argparse
import inflect
from re import DOTALL, finditer
import base64



def find_json_response(full_response):
    extracted_responses = list(
        finditer(r"({[^}]*$|{.*})", full_response, flags=DOTALL)
    )

    if not extracted_responses:
        print(
            f"Unable to find any responses of the matching type dictionary: `{full_response}`"
        )
        return None

    if len(extracted_responses) > 1:
        print("Unexpected response > 1, continuing anyway...", extracted_responses)

    extracted_response = extracted_responses[0]
    extracted_str = extracted_response.group(0)
    return extracted_str


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


## gpt tool for different prompts and images
class GPTO_tool:
    def __init__(self, apikey, query_prompt_path = None, query_prompt = None, retry_limit=3):

        if query_prompt is None:
            with open(query_prompt_path, "r") as file:
                self.query = file.read().strip()
        else:
            self.query = query_prompt
        self.retry_limit = retry_limit
        self.client = OpenAI(api_key=apikey)

    def call(self, image_path, max_tokens=300):
        query_image = encode_image(image_path)
        try_count = 0
        while True:
            response = self.client.chat.completions.create(model="gpt-4o",
            messages=[{
                "role": "system",
                "content": self.query
                            },
                            {
                "role": "user",
                "content": [
                {
                "type": "text",
                "text": f"{self.query}"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_path[-3:]};base64,{query_image}"
                }
                },
            ]
            }
            ],
            seed=100,
            max_tokens=max_tokens)
            response = response.choices[0].message.content
            try:
                result = find_json_response(response)
                result = ast.literal_eval(result.replace(' ', '').replace('\n', ''))
                break
            except:
                try_count += 1
                if try_count > self.retry_limit:
                    raise ValueError(f"Over Limit: Unknown response: {response}")
                else:
                    print("Retrying after 1s.")
                    sleep(1)
        return result
    
    def call_img_list(self, image_path_list, max_tokens=300):
        return 

    def call_obj_descrip(self, image_path, max_tokens=300):
        query_image = encode_image(image_path)
        try_count = 0
        while True:
            response = self.client.chat.completions.create(model="gpt-4o",
            messages=[{
                "role": "system",
                "content": self.query
                            },
                            {
                "role": "user",
                "content": [
                {
                "type": "text",
                "text": f"{self.query}"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_path[-3:]};base64,{query_image}"
                }
                }
            ]
            }
            ],
            seed=100,
            max_tokens=max_tokens)
            response = response.choices[0].message.content
            try:
                result = find_json_response(response)
                result = ast.literal_eval(result)
                break
            except:
                try_count += 1
                if try_count > self.retry_limit:
                    raise ValueError(f"Over Limit: Unknown response: {response}")
                else:
                    print("Retrying after 1s.")
                    sleep(1)
        return result


def query_gpt4o_item(img_path, apikey_path, query_prompt_path = None, query_prompt = None):
    #import ipdb; ipdb.set_trace()

    ## get apikey
    with open(apikey_path, "r") as file:
        apikey = file.read().strip()

    ## set gpt tool item
    gpt = GPTO_tool(query_prompt_path = query_prompt_path, apikey = apikey, query_prompt = query_prompt)
    result = gpt.call(img_path)    

    ## double check and filter the results
    p = inflect.engine()
    #for obj_name in result:
        # singular = p.singular_noun(obj_name)
        # singular = obj_name
    #    if singular:
    #        result[singular] = result.pop(obj_name)
    #    else:
    #        continue

    return result



def query_gpt4o(img_path, apikey_path, query_prompt_path = None, query_prompt = None):

    ## get apikey
    with open(apikey_path, "r") as file:
        apikey = file.read().strip()

    ## set gpt tool item
    gpt = GPTO_tool(query_prompt_path = query_prompt_path, apikey = apikey, query_prompt = query_prompt)
    result = gpt.call_obj_descrip(img_path)    

    return result

def query_gpt4o_img_list(img_path_list, apikey_path, query_prompt_path = None, query_prompt = None):

    ## get apikey
    with open(apikey_path, "r") as file:
        apikey = file.read().strip()

    ## set gpt tool item
    gpt = GPTO_tool(query_prompt_path = query_prompt_path, apikey = apikey, query_prompt = query_prompt)
    result = gpt.call_img_list(img_path_list)    

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", type=str, default="data/ref_img")
    parser.add_argument("--save_folder", type=str, default="data/items")
    parser.add_argument("--scene_name", type=str, default="fruitbasket")
    parser.add_argument("--prompt_path", type=str, default="pat3d/preprocessing/gpt_utils/get_items.txt")
    parser.add_argument("--apikey_path", type=str, default="pat3d/preprocessing/gpt_utils/apikey.txt")
    args = parser.parse_args()


    img_path = f'{args.img_folder}/{args.scene_name}.jpeg'
    result = query_gpt4o_item(img_path, args.apikey_path, args.prompt_path)

    save_path = f'{args.save_folder}/{args.scene_name}.json'

    with open(save_path, "w") as file:
        json.dump(result, file)

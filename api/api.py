import json
import requests
from tqdm import tqdm
from .secret import API, get_response


def make_request(api_name, params, headers):
    api_dict = API[api_name]
    api_url = api_dict['url']
    api_auth = api_dict['auth']

    input_params = {}
    input_params.update(api_auth)
    input_params.update(params)
    
    response = requests.get(
        api_url, 
        params=input_params, 
        headers=headers)

    result_dict = get_response(api_name,response)
    return result_dict

def update_db(food_list, api_name="edanam"):
    query_str = API[api_name]['query_str']
    headers = {"Accept": "application/json",}
    db = []
    for food_name in tqdm(food_list):
        query_str['ingr'] = food_name
        food_dict = make_request(
            api_name, 
            params=query_str,
            headers=headers)

        if food_dict is not None:
            db.append(food_dict)
        else:
            print(f"Failed to get {food_name}")
    
    save_db(db)

def save_db(db, out_name="./api/db.json"):
    # db: list[dict]
    # data: list[dict]

    with open(out_name, 'r') as f:
        data = json.load(f)

    data['food'] += db
    with open(out_name, 'w') as f:
        json.dump(data,f)


if __name__ == '__main__':
    food_list = [
        'milk',
        'yogurt',
        'rice',
        'mixed rice',
        'bread',
        'white bread',
        'udon',
        'fish',
        'meat',
        'salad',
        'cherry tomatoe',
        'soup',
        'tofu',
        'bibimbap',
        'fried noodles',
        'spaghetti',
        'citrus',
        'apple'
    ]

    update_db(food_list)
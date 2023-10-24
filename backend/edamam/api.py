import json
import requests
from tqdm import tqdm
from .secret import API, get_response

DATABASE = "./backend/edamam/db.json"


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

    result_dict = get_response(api_name, response)
    return result_dict


def update_db(food_list, api_name="edamam"):
    query_str = API[api_name]['query_str']
    headers = {"Accept": "application/json", }
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


def save_db(db, out_name=DATABASE):
    # db: list[dict]
    # data: list[dict]

    with open(out_name, 'r') as f:
        data = json.load(f)

    data['food'] += db
    with open(out_name, 'w') as f:
        json.dump(data, f)


def get_info_from_db(food_list):
    if not isinstance(food_list, list):
        food_list = [food_list]

    with open(DATABASE, 'r') as f:
        data = json.load(f)

    result_list = {
        "calories": [],
        "protein": [],
        "fat": [],
        "carbs": [],
        "fiber": []
    }
    for food_name in food_list:
        has_info = False
        for item in data['food']:
            if '_' in str(food_name):
                food_name = ' '.join(food_name.split('_'))

            if str.lower(food_name) == str.lower(item['name']):
                for key in result_list.keys():
                    result_list[key].append(item['nutrients'][key])
                has_info = True
                break
        if not has_info:
            for key in result_list.keys():
                result_list[key].append(None)

    return result_list


if __name__ == '__main__':
    food_list = [
        "broccoli",
        'mushroom',
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
        'soups',
        'tofu',
        'bibimbap',
        'fried noodles',
        'spaghetti',
        'citrus',
        'apple',
        "banh mi",
        "banh xeo",
        "bun bo hue",
        "bun dau",
        "com tam",
        "goi cuon",
        "pho",
        "hamburger",
        "cabbage",
        "french fries",
        "sandwich",
        'submarine sandwich',
        "cheese",
        "banana",
        "coffee",
        "tea",
        "cucumber",
        "egg",
        "orange",
        "nachos",
        "mushroom risotto",
        "guacamole",
        "seaweed salad",
        "frozen yogurt",
        "steak",
        "omelette",
        "takoyaki",
        "melitzanosalata",
        "carrot cake",
        'carrot',
        'common fig',
        'grape',
        'muffin',
        'cocktail',
        'zucchini',
        'cooking spray',
        "apple pie",
        "zha jiang mian",
        "baklava",
        "lasagna",
        "grilled cheese sandwich",
        "prime rib",
        "sarma",
        "deviled eggs",
        "green curry",
        'curry',
        "samosa",
        "chicken quesadilla",
        "french onion soup",
        "french toast",
        "pork chop",
        "paneer tikka",
        "hot and sour soup",
        "pozole",
        "pho",
        "lobster bisque",
        "enchiladas",
        "moussaka",
        'strawberry',
        'croissant',
        'doughnut',
        'fruit',
        'grapefruit',
        'honeycomb',
        'lemon',
        'lobster',
        'mango',
        "pad thai",
        "ice cream",
        "greek salad",
        "garlic bread",
        "chow mein",
        "xiao long bao",
        "spanakopita",
        "kolokythokeftedes",
        "lobster roll sandwich",
        "croque madame",
        "risotto",
        "poutine",
        "chiles en nogada",
        "coconut milk-flavored crepes with shrimp and beef",
        "gyro",
        "naan",
        "souvlaki",
        "papdi chaat",
        'squash',
        "fried calamari",
        "beef tartare",
        'tart',
        "cig kofte",
        "coconut milk soup",
        "noodles with fish curry",
        "coq au vin",
        "kaali daal",
        "tempura",
        "hummus",
        "donuts",
        "beef carpaccio",
        "grilled salmon",
        "beef bourguignon",
        "sukiyaki",
        "ravioli",
        "tandoori chicken",
        "cup cakes",
        "clam chowder",
        "pizza",
        "sushi",
        "french fries",
        "banh xeo",
        "chocolate cake",
        "caprese salad",
        "beignets",
        "huevos rancheros",
        "saganaki",
        "peking duck",
        "spring rolls",
        "beef in oyster sauce",
        "ramen",
        "hunkar begendi",
        "spaghetti bolognese",
        "salade nicoise",
        "blanquette de veau",
        "macaroni and cheese",
        "oysters",
        'oyster',
        "artichoke bottoms in olive oil",
        "udon noodle",
        "hue beef rice vermicelli soup",
        "cheesecake",
        "paella",
        "fried rice",
        "stewed pork leg",
        "borek",
        "winter melon soup",
        'vegetable',
        "hot and sour fish and vegetable ragout",
        "minestrone",
        "cheese plate",
        "foie gras",
        "chicken wings",
        'wine',
        "baby back ribs",
        "gnocchi",
        "beet salad",
        "shrimp and grits",
        "biryani",
        "chocolate mousse",
        "tomatokeftedes",
        "macarons",
        "tacos",
        'taco',
        "spaghetti carbonara",
        "confit de canard",
        "fried mussel pancakes",
        "club sandwich",
        "caesar salad",
        "yellow curry",
        "candy",
        "cake",
        'cantaloupe',
        'pear',
        'popcorn',
        'pretzel',
        'tomato',
        'crab',
        'lemon',
        "cao lau",
        "bruschetta",
        "khao soi",
        "rogan josh",
        "escargots",
        "bread pudding",
        "pastitsio",
        "chole",
        "bibimbap",
        "tiramisu",
        "sashimi",
        "manti",
        "kebap",
        "hot dog",
        "hamburger",
        "banh mi",
        "cha ca",
        "steamed rice roll",
        "churros",
        "scallops",
        "vermicelli noodles with snails",
        "ratatouille",
        "bun cha",
        "cannoli",
        "filet mignon",
        "pulled pork sandwich",
        "fried pork in scoop",
        "thai papaya salad",
        "kisir",
        "crab cakes",
        "mussels",
        "com tam",
        "chilaquiles",
        "creme brulee",
        "edamame",
        "butter chicken",
        "chicken curry",
        "gyoza",
        "twice cooked pork",
        "miso soup",
        "palak paneer",
        "ceviche",
        "kung pao chicken",
        "dumplings",
        "charcoal-boiled pork neck",
        "malai kofta",
        "icli kofte",
        "tuna tartare",
        "panna cotta",
        "yemista",
        "red velvet cake",
        'watermelon',
        "eggs benedict",
        "onion rings",
        "fish and chips",
        "artichoke",
        "asparagus",
        "bagel",
        "baked goods",
        "beer",
        "bell pepper",
        'pasta',
        'pastry',
        'peach',
        'pineapple',
        'pomegranate',
        'potato',
        'pumpkin',
        'radish'
    ]

    food_list_2 = [
        "broccoli",
        'mushroom',
        'soups',
        "cabbage",
        'submarine sandwich',
        'carrot',
        'common fig',
        'grape',
        'muffin',
        'cocktail',
        'zucchini',
        'cooking spray',
        'curry',
        'strawberry',
        'croissant',
        'doughnut',
        'fruit',
        'grapefruit',
        'honeycomb',
        'lemon',
        'lobster',
        'mango',
        'squash',
        'tart',
        'oyster',
        'vegetable',
        'wine',
        'taco',
        "candy",
        "cake",
        'cantaloupe',
        'pear',
        'popcorn',
        'pretzel',
        'tomato',
        'crab',
        'lemon',
        'watermelon',
        'wintermelon',
        "artichoke",
        "asparagus",
        "bagel",
        "baked goods",
        "beer",
        "bell pepper",
        'pasta',
        'pastry',
        'peach',
        'pineapple',
        'pomegranate',
        'potato',
        'pumpkin',
        'radish'
    ]

    update_db(food_list_2)

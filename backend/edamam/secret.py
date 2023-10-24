API_NAMES = ["edamam", "usda"]

API = {
    "edamam": {
        "homepage": "https://developer.edamam.com/",
        "url": "https://api.edamam.com/api/food-database/v2/parser?",
        "auth":{
            "app_id": "82bf3e46",
            "app_key": "7ddcf5cf20429748592784d9928bd093"
        },
        "query_str": {
            "ingr": "",
            "nutrition-type": "logging",
        }
    },

    "usda": {
        "homepage": "https://fdc.nal.usda.gov/api-guide.html",
        "url": "https://api.nal.usda.gov/fdc/v1/foods/search?",
        "auth": {
            "api_key": "UTGKCT7Z084bVzDu9qCIEzFYrsYB6DhuwhiWv9y3"
        }
    }
}


def get_response_from_edamam(response):
    response_dict = response.json()
    result = response_dict['parsed']
    if len(result) == 0:
        result = response_dict['hints']
    result = result[0] 
    food_info = result['food']
    
    food_label = response_dict["text"]
    food_id = food_info['foodId']
    food_nutrients = food_info['nutrients']

    calories = food_nutrients['ENERC_KCAL']
    protein = food_nutrients['PROCNT']
    fat = food_nutrients['FAT']
    carbs = food_nutrients['CHOCDF']
    fiber = food_nutrients['FIBTG']

    return {
        "name": food_label,
        "nutrients": {
            "calories": calories,
            "protein": protein,
            "fat": fat,
            "carbs": carbs,
            "fiber": fiber
        }
    }

def get_response(api_name, response):
    assert api_name in API_NAMES, "API not supported"
    try:
        if api_name == 'edamam':
            return get_response_from_edamam(response)
    except:
        return None

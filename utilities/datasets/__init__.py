import fiftyone as fo
import fiftyone.zoo as foz


food_list = [
    "Apple", "Orange", "Pizza", "Hamburger", "French fries", 
    "Sandwich", "Cheese", "Burrito", "Banana",
   "Pancake", "Coffee", "Tea", "Milk", "Salab",
    "Cucumber", "Tomato", "Egg"]

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="validation",
    label_types=["detections"],
    classes = food_list,
    max_samples=100,
    seed=51,
    shuffle=True,
    dataset_name="open-images-food",
)


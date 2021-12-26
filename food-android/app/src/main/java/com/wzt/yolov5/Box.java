package com.wzt.yolov5;

import android.graphics.Color;
import android.graphics.RectF;

import java.util.Random;

public class Box {
    public float x0,y0,x1,y1;
    private int label;
    private float score;
    private static String[] labels={"hot-dog", "Apple", "Artichoke", "Asparagus", "Bagel", "Baked-goods", "Banana", "Beer", "Bell-pepper", "Bread",
            "Broccoli", "Burrito", "Cabbage", "Cake", "Candy", "Cantaloupe", "Carrot", "Common-fig", "Cookie", "Dessert",
            "French-fries", "Grape", "Guacamole", "Hot-dog", "Ice-cream", "Muffin", "Orange", "Pancake", "Pear", "Popcorn",
            "Pretzel", "Strawberry", "Tomato", "Waffle", "food-drinks", "Cheese", "Cocktail", "Coffee",
            "Cooking-spray", "Crab", "Croissant", "Cucumber", "Doughnut", "Egg", "Fruit", "Grapefruit", "Hamburger", "Honeycomb",
            "Juice", "Lemon", "Lobster", "Mango", "Milk", "Mushroom", "Oyster", "Pasta", "Pastry", "Peach",
            "Pineapple", "Pizza", "Pomegranate", "Potato", "Pumpkin", "Radish", "Salad", "food", "Sandwich", "Shrimp",
            "Squash", "Squid", "Submarine-sandwich", "Sushi", "Taco", "Tart", "Tea", "Vegetable", "Watermelon", "Wine",
            "Winter-melon", "Zucchini"};
    public Box(float x0,float y0, float x1, float y1, int label, float score){
        this.x0 = x0;
        this.y0 = y0;
        this.x1 = x1;
        this.y1 = y1;
        this.label = label;
        this.score = score;
    }

    public RectF getRect(){
        return new RectF(x0,y0,x1,y1);
    }

    public String getLabel(){
        return labels[label];
    }

    public float getScore(){
        return score;
    }

    public int getColor(){
        Random random = new Random(label);
        return Color.argb(255,random.nextInt(256),random.nextInt(256),random.nextInt(256));
    }
}

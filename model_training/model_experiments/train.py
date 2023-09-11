import csv
import random
import sys

import pandas as pd
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, Trainer, TrainingArguments

import torch
from datasets import load_dataset, load_dataset_builder, Features, Value, Dataset
import os

from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset

import torch.nn as nn

from model_training.model_experiments.GPTJ_layers import GPTJ_LAYERS
from model_training.model_experiments.LLAMA_layers import LLAMA_LAYERS

import ast

from beartype import beartype
from beartype.typing import List, Dict, Tuple

DEVICE = "cuda"

pad_sequence = partial(pad_sequence, batch_first=True)

long_tensor = partial(torch.tensor, dtype=torch.long, device=DEVICE)
int_tensor = partial(torch.tensor, dtype=torch.long, device=DEVICE)

REMOVE_CALCULATOR = False
ARG_TRAINING = True
LORA = False
SHUFFLE = False
RAW = False
MODEL_NAME = "GPTJ"
MODEL_LAYERS = LLAMA_LAYERS if MODEL_NAME == "LLAMA" else GPTJ_LAYERS
NO_TOKEN_OPTION = ""# ""#
METHOD_B = False

TOOL_START_TOKEN = "<TOOL>"
TOOL_END_TOKEN = "</TOOL>" 

if MODEL_NAME == "GPTJ":
    TOOL_START_TOKEN = " " + TOOL_START_TOKEN

if LORA:
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )

TRAINING_DESCRIPTIONS = [
    """increasing_relevance: Characteristics:
        - Warmup 40 and LR: 5e-5
        - batch size 4 gradient accum 2
        - 100% description
        - Data: shiny new""",
    """increasing_relevance_2: This is the continuation of the increasing_relevance model. Changes:
        - Warmup and lr -> 10% train and 1e-5 instead of 40 and 5e-5
        - Try to add perplexity to the eval
        - 0.15 chance of not adding tool avail prompt
        - batch size 16 instead of 8
        - Data: shiny new""",
    """shuffled_DX5: This is the continuation of the increasing_relevance model. Changes:
        - SAME
        - SHUFFLE TO TEST IF INCREASING RELEVANCE IS GOOD
        - Data: shiny new""",
    """arg_training: Remove shuffle (increasing relevance with slight randomness, 20% I think)
        - Train on tokens 4,5,6 as well (new addition)
        - Increase prompt removal from 13 to 18, to account for the fact that arg generation has a diff prompt
    """,

    """arg_training, all data:
        - Unfroze layers 0,1,7, 13, 19, >24
        - Data is bigger (improved) DIDNT GO THROUGH, ITS SHINY NEW
            - Relevance is increasing (BEFORE WAS DECREASING)
            - Tools are intertwined randomly, and relevance increase PER tool
        - 15% chance of not adding tool avail prompt
""",
    """full_data_fr, all data, for real:
        - Unfroze layers 13, 19, >24
        - Data is bigger (improved)
            - Relevance is increasing
            - Tools are intertwined randomly, and relevance increase PER tool
        - 20% chance of not adding tool avail prompt
""",
    """just_gen, all data, for real:
        - Unfroze layers >24
        - Data is bigger (improved)
            - Relevance is increasing
            - Tools are intertwined randomly, and relevance increase PER tool
        - 0% chance of not adding tool avail prompt
        - NO ARG TRAINING
""",
   """Micky, Better training set, same as before,
        - 0 to 23 frozen except 13 and 19 (one less than before)""",
    """Micky-no-calc, Micky, with no calculator""",
"""distracted has improvement of not including main tools as distractors.
        - Everything same as Mickey
        - LESS LAYERS RRAINED. Freeze from 0 to 24 inclusive
        - Data is "better" """,

""" Small set and distracted,
- Unfroze layers >24
- Same as distracted with small set
- Warmup 7%""",

"""Small-no-calc: Small set and distracted, no calc,
- Same as distracted with small set and no calc
- Warmup 7%
- 2000 Calc, 4000 wiki, 1000 Calend""",

"""Med set:
- Unfroze layers >24
- Same but:
- 3000 Calc, 6000 wiki, 2000 Calend""",

"""Med-no-calc:
- Same but:
- no calc
- 3000 Calc, 6000 wiki, 2000 Calend""",

"""Med_arg:
- Same but:
- 3000 Calc, 6000 wiki, 2000 Calend""",


"""Large set:
- Same but:
- 6000 Calc, 12000 wiki, 4000 Calend""",


"""Med set: full, all layers are unfrozen
arg training, LORA"""

"""Huge full train, no frozen""",

"huge2. same as huge, but shuffle available tool order"

"med-shuffle. same as huge, but shuffle available tool order",


"""Small2: Greater lr: 3e-5,
- 2000 Calc, 4500 Wiki, 1500 calend
- Warmup 4%,
- arg training,
- no shuffle""",

"""Small3: Greater lr: 3e-5,
- 2200 Calc, 4950 Wiki, 1500 calend
- Warmup 4%,
- arg training,
- no shuffle""",


"""Med set: full, all layers are unfrozen
arg training, LORA""",

"""Med_no_token:
- Same as med but no tool token:
- 3000 Calc, 6000 wiki, 2000 Calend""",

""" med2: 
- same as med but with 3e-5 lr"""
]

tool_name_alternatives = {
    "Calculator":{"mix":["Calculator","calculator","CALCULATOR"
                        "Calculate","calculate","CALCULATE",
                        "Operate","operate","OPERATE",
                        "Calc","calc","CALC",
                        "WolframAlpha","wolframAlpha","WOLFRAMALPHA","Wolframalpha",
                        "Subtract","subtract","SUBTRACT",
                        "Multiply","multiply","MULTIPLY",
                        "Divide","divide","DIVIDE",
                        "Arithmetic","arithmetic","ARITHMETIC",
                        "Math","math","MATH",
                        "ArithmetiX","arithmetiX","ARITHMETIX",
                        "MathXpert","mathXpert","MATHXPERT",
                        "ComputeX","computeX","COMPUTEX",
                        "NumCalc","numCalc","NUMCALC",
                        "CalcuMate","calcuMate","CALCU_MATE",
                        ],
                  "add_subtract": ["CalcAddSub","calcAddSub","CALC_ADD_SUB",
                                   "Calc_add_sub","Calc_Add_Sub","calc_add_sub",],
                  "add": ["Add","add","ADD",
                          "Plus","plus","PLUS",
                          "Sum","sum","SUM",
                          "Addition","addition","ADDITION",
                          "Total","total","TOTAL",
                          "AddTo","addTo","ADDTO",
                          "addingTool","AddingTool","ADDINGTOOL",
                          ],
                  "subtract": ["Subtract","subtract","SUBTRACT",
                               "Minus","minus","MINUS",
                               "Subtraction","subtraction","SUBTRACTION",
                               "Difference","difference","DIFFERENCE",
                               "SubtractFrom","subtractFrom","SUBTRACTFROM",
                               "MinusFrom","minusFrom","MINUSFROM",
                               "SubtractingTool","subtractingTool","SUBTRACTINGTOOL",],
                  "multiply": ["Multiply","multiply","MULTIPLY",
                               "Times","times","TIMES",
                               "Multiplication","multiplication","MULTIPLICATION",
                               "Product","product","PRODUCT",
                               "MultiplyBy","multiplyBy","MULTIPLYBY",
                               "TimesBy","timesBy","TIMESBY",
                               "MultiplyingTool","multiplyingTool","MULTIPLYINGTOOL",],
                  "divide": ["Divide","divide","DIVIDE",
                             "Division","division","DIVISION",
                             "Div","div","DIV",
                             "Quotient","quotient","QUOTIENT",
                             "DivideBy","divideBy","DIVIDEBY",
                             "DivisionTool","divisionTool","DIVISIONTOOL",
                             "DividingTool","dividingTool","DIVIDINGTOOL",
                             "DivideTool","divideTool","DIVIDETOOL",],
                  "mult_divide": ["CalcMultDiv","calcMultDiv","CALC_MULT_DIV",
                                  "Calc_mult_div","Calc_Mult_Div","calc_mult_div",
                                  "mult_divide_tool","MultDivideTool","MULTDIVIDETOOL",
                                    "MultDiv","multDiv","MULTDIV",
                                    "MultDivTool","multDivTool","MULTDIVTOOL",
                                    "MultDivide","multDivide","MULTDIVIDE",],

        },
    "WikiSearch":["WikiSearch","wikisearch","WIKISEARCH","Wiki-search","wiki-Search","WIKI-SEARCH","Wiki_search","Wiki_Search","WIKI_SEARCH",
                  "Search","search","SEARCH",
                  "Wiki","wiki","WIKI",
                  "WikiPedia","wikipedia","WIKIPEDIA","Wikipedia",
                  "InternetSearch","internetSearch","INTERNETSEARCH","Internet-search","internet-search","INTERNET-SEARCH","Internet_search","Internet_Search","INTERNET_SEARCH",
                  "Google","google","GOOGLE",
                  "Browser","browser","BROWSER",
                  "Knowledge-base","knowledge-base","KNOWLEDGE-BASE","Knowledge_base","Knowledge_Base","KNOWLEDGE_BASE",
                  "WikiSnip","wikisnip","WIKISNIP","wikiSnip","Wiki_snip","Wiki_Snip","WIKI_SNIP",
                  "WikiSnips","wikisnips","WIKISNIPS","wikiSnips","Wiki_snips","Wiki_Snips","WIKI_SNIPS",
                  "WikiScans","wikiscans","WIKISCANS","wikiScans","Wiki_scans","Wiki_Scans","WIKI_SCANS",
                  "WikiScan","wikiscan","WIKISCAN","wikiScan","Wiki_scan","Wiki_Scan","WIKI_SCAN",
                  "WikiSnap","wikisnap","WIKISNAP","wikiSnap","Wiki_snap","Wiki_Snap","WIKI_SNAP",
                  "WiseSearch","wisesearch","WISESEARCH","Wise_search","Wise-Search","WISE-SEARCH","Wise_search","Wise_Search","WISE_SEARCH",
                  "WikiSearcher","wikisearcher","WIKISEARCHER","Wiki_searcher","Wiki-Searcher","WIKI-SEARCHER","Wiki_searcher","Wiki_Searcher","WIKI_SEARCHER",
                  "WikiFinder","wikifinder","WIKIFINDER","Wiki_finder","Wiki-Finder","WIKI-FINDER","Wiki_finder","Wiki_Finder","WIKI_FINDER",
                  "WikiSearchTool","wikisearchtool","WIKISEARCHTOOL","Wiki_search_tool","Wiki-Search-Tool","WIKI-SEARCH-TOOL","Wiki_search_tool","Wiki_Search_Tool","WIKI_SEARCH_TOOL",
                  "InfoSearch","infoSearch","INFOSEARCH","Info_search","Info-Search","INFO-SEARCH","Info_search","Info_Search","INFO_SEARCH",
                  "InfoSeek","infoSeek","INFOSEEK","Info_seek","Info-Seek","INFO-SEEK","Info_seek","Info_Seek","INFO_SEEK",
                  "InfoFinder","infoFinder","INFOFINDER","Info_finder","Info-Finder","INFO-FINDER","Info_finder","Info_Finder","INFO_FINDER",
                  "FactSearch","factSearch","FACTSEARCH","Fact_search","Fact-Search","FACT-SEARCH","Fact_search","Fact_Search","FACT_SEARCH",
                  "FactSeek","factSeek","FACTSEEK","Fact_seek","Fact-Seek","FACT-SEEK","Fact_seek","Fact_Seek","FACT_SEEK",
                  "Encyclopedia","encyclopedia","ENCYCLOPEDIA",
                  "Encyclopaedia","encyclopaedia","ENCYCLOPAEDIA",
                  "Encyclopedic","encyclopedic","ENCYCLOPEDIC",
                    "InfoTool","infoTool","INFOTOOL","Info_tool","Info-Tool","INFO-TOOL","Info_tool","Info_Tool","INFO_TOOL",
                    "InfoAPI","infoAPI","INFOAPI","Info_api","Info-API","INFO-API","Info_api","Info_Api","INFO_API",
                  "WikiAPI","wikiAPI","WIKIAPI","Wiki_api","Wiki-API","WIKI-API","Wiki_api","Wiki_Api","WIKI_API",
                    "BrowserAPI","browserAPI","BROWSERAPI","Browser_api","Browser-API","BROWSER-API","Browser_api","Browser_Api","BROWSER_API",
                    "SearchAPI","searchAPI","SEARCHAPI","Search_api","Search-API","SEARCH-API","Search_api","Search_Api","SEARCH_API",
                    "SearchTool","searchTool","SEARCHTOOL","Search_tool","Search-Tool","SEARCH-TOOL","Search_tool","Search_Tool","SEARCH_TOOL",
                    "BrowserTool","browserTool","BROWSERTOOL","Browser_tool","Browser-Tool","BROWSER-TOOL","Browser_tool","Browser_Tool","BROWSER_TOOL",
                    "FindMyInfo","findMyInfo","FINDMYINFO","Find_my_info","Find-My-Info","FIND-MY-INFO","Find_my_info","Find_My_Info","FIND_MY_INFO",
                    "FindMyFact","findMyFact","FINDMYFACT","Find_my_fact","Find-My-Fact","FIND-MY-FACT","Find_my_fact","Find_My_Fact","FIND_MY_FACT",
                    "FactFinder","factFinder","FACTFINDER","Fact_finder","Fact-Finder","FACT-FINDER","Fact_finder","Fact_Finder","FACT_FINDER",
                    "FactAPI","factAPI","FACTAPI","Fact_api","Fact-API","FACT-API","Fact_api","Fact_Api","FACT_API",
                    "FactTool","factTool","FACTTOOL","Fact_tool","Fact-Tool","FACT-TOOL","Fact_tool","Fact_Tool","FACT_TOOL",
                    "InfoEngine","infoEngine","INFOENGINE","Info_engine","Info-Engine","INFO-ENGINE","Info_engine","Info_Engine","INFO_ENGINE",
                    "InfoExpert","infoExpert","INFOEXPERT","Info_expert","Info-Expert","INFO-EXPERT","Info_expert","Info_Expert","INFO_EXPERT",
                    "DataEngine","dataEngine","DATAENGINE","Data_engine","Data-Engine","DATA-ENGINE","Data_engine","Data_Engine","DATA_ENGINE",
                    "Internet", "internet", "INTERNET",
                    "search", "Search", "SEARCH",
                    "WhatIs", "whatIs", "WHATIS", "What_is", "What-Is", "WHAT-IS", "What_is", "What_Is", "WHAT_IS",
                    "Info", "info", "INFO",
                    "Research", "research", "RESEARCH",
                    "HuntFacts", "huntFacts", "HUNTFACTS", "Hunt_facts", "Hunt-Facts", "HUNT-FACTS", "Hunt_facts", "Hunt_Facts", "HUNT_FACTS",
                    "LearnPedia", "learnPedia", "LEARNPEDIA", "Learn_pedia", "Learn-Pedia", "LEARN-PEDIA", "Learn_pedia", "Learn_Pedia", "LEARN_PEDIA",
                    "MyInfo", "myInfo", "MYINFO", "My_info", "My-Info", "MY-INFO", "My_info", "My_Info", "MY_INFO",
                    "MyFact", "myFact", "MYFACT", "My_fact", "My-Fact", "MY-FACT", "My_fact", "My_Fact", "MY_FACT",
                    "MyFacts", "myFacts", "MYFACTS", "My_facts", "My-Facts", "MY-FACTS", "My_facts", "My_Facts", "MY_FACTS",
                    "FastSearch", "fastSearch", "FASTSEARCH", "Fast_search", "Fast-Search", "FAST-SEARCH", "Fast_search", "Fast_Search", "FAST_SEARCH",
                    "ShortSearch", "shortSearch", "SHORTSEARCH", "Short_search", "Short-Search", "SHORT-SEARCH", "Short_search", "Short_Search", "SHORT_SEARCH",
                    "QuickSearch", "quickSearch", "QUICKSEARCH", "Quick_search", "Quick-Search", "QUICK-SEARCH", "Quick_search", "Quick_Search", "QUICK_SEARCH",
                    "QuickFind", "quickFind", "QUICKFIND", "Quick_find", "Quick-Find", "QUICK-FIND", "Quick_find", "Quick_Find", "QUICK_FIND",
                    "QuickFact", "quickFact", "QUICKFACT", "Quick_fact", "Quick-Fact", "QUICK-FACT", "Quick_fact", "Quick_Fact", "QUICK_FACT",
                    "QuickFacts", "quickFacts", "QUICKFACTS", "Quick_facts", "Quick-Facts", "QUICK-FACTS", "Quick_facts", "Quick_Facts", "QUICK_FACTS",
                ],
    "Calendar":["Calendar","calendar","CALENDAR",
                "Date","date","DATE",
                "Time","time","TIME",
                "DateStampNow","dateStampNow","DATESTAMPNOW","Date_stamp_now","Date-Stamp-Now","DATE-STAMP-NOW","Date_stamp_now","Date_Stamp_Now","DATE_STAMP_NOW",
                "DateStamp","dateStamp","DATESTAMP","Date_stamp","Date-Stamp","DATE-STAMP","Date_stamp","Date_Stamp","DATE_STAMP",
                "TimeStamp","timeStamp","TIMESTAMP","Time_stamp","Time-Stamp","TIME-STAMP","Time_stamp","Time_Stamp","TIME_STAMP",
                "TimeStampNow","timeStampNow","TIMESTAMPNOW","Time_stamp_now","Time-Stamp-Now","TIME-STAMP-NOW","Time_stamp_now","Time_Stamp_Now","TIME_STAMP_NOW",
                "CalendarTool","calendarTool","CALENDARTOOL","Calendar_tool","Calendar-Tool","CALENDAR-TOOL","Calendar_tool","Calendar_Tool","CALENDAR_TOOL",
                "DateTool","dateTool","DATETOOL","Date_tool","Date-Tool","DATE-TOOL","Date_tool","Date_Tool","DATE_TOOL",
                "TimeTool","timeTool","TIMETOOL","Time_tool","Time-Tool","TIME-TOOL","Time_tool","Time_Tool","TIME_TOOL",
                "DayCheck","dayCheck","DAYCHECK","Day_check","Day-Check","DAY-CHECK","Day_check","Day_Check","DAY_CHECK",
                "DayChecker","dayChecker","DAYCHECKER","Day_checker","Day-Checker","DAY-CHECKER","Day_checker","Day_Checker","DAY_CHECKER",
                "DateCheck","dateCheck","DATECHECK","Date_check","Date-Check","DATE-CHECK","Date_check","Date_Check","DATE_CHECK",
                "DateChecker","dateChecker","DATECHECKER","Date_checker","Date-Checker","DATE-CHECKER","Date_checker","Date_Checker","DATE_CHECKER",
                "TimeCheck","timeCheck","TIMECHECK","Time_check","Time-Check","TIME-CHECK","Time_check","Time_Check","TIME_CHECK",
                "TimeChecker","timeChecker","TIMECHECKER","Time_checker","Time-Checker","TIME-CHECKER","Time_checker","Time_Checker","TIME_CHECKER",
                "DateRetriever","dateRetriever","DATERETRIEVER","Date_retriever","Date-Retriever","DATE-RETRIEVER","Date_retriever","Date_Retriever","DATE_RETRIEVER",
                "today","Today","TODAY",
                "todayTool","TodayTool","TODAYTOOL","Today_tool","Today-Tool","TODAY-TOOL","Today_tool","Today_Tool","TODAY_TOOL",
                "now","Now","NOW",
                "nowTool","NowTool","NOWTOOL","Now_tool","Now-Tool","NOW-TOOL","Now_tool","Now_Tool","NOW_TOOL",
                "DateAPI","dateAPI","DATEAPI","Date_api","Date-API","DATE-API","Date_api","Date_Api","DATE_API",
                "TimeAPI","timeAPI","TIMEAPI","Time_api","Time-API","TIME-API","Time_api","Time_Api","TIME_API",
                "CalendarAPI","calendarAPI","CALENDARAPI","Calendar_api","Calendar-API","CALENDAR-API","Calendar_api","Calendar_Api","CALENDAR_API",
            ],
    "Random":["Random","random","RANDOM",
              "RandomNumber","randomNumber","RANDOMNUMBER","Random_number","Random-Number","RANDOM-NUMBER","Random_number","Random_Number","RANDOM_NUMBER",
              "RandomNumberGenerator","randomNumberGenerator","RANDOMNUMBERGENERATOR","Random_number_generator","Random-Number-Generator","RANDOM-NUMBER-GENERATOR","Random_number_generator","Random_Number_Generator","RANDOM_NUMBER_GENERATOR",
              "RandomNumberTool","randomNumberTool","RANDOMNUMBERTOOL","Random_number_tool","Random-Number-Tool","RANDOM-NUMBER-TOOL","Random_number_tool","Random_Number_Tool","RANDOM_NUMBER_TOOL",
              "RandomNumberAPI","randomNumberAPI","RANDOMNUMBERAPI","Random_number_api","Random-Number-API","RANDOM-NUMBER-API","Random_number_api","Random_Number_Api","RANDOM_NUMBER_API",],
    "Movies":["Movies","movies","MOVIES","Movie","movie","MOVIE",
                "MovieSearch","movieSearch","MOVIESEARCH","Movie_search","Movie-Search","MOVIE-SEARCH","Movie_search","Movie_Search","MOVIE_SEARCH",
                "MovieFinder","movieFinder","MOVIEFINDER","Movie_finder","Movie-Finder","MOVIE-FINDER","Movie_finder","Movie_Finder","MOVIE_FINDER",
                "MovieSeek","movieSeek","MOVIESEEK","Movie_seek","Movie-Seek","MOVIE-SEEK","Movie_seek","Movie_Seek","MOVIE_SEEK",
                "film","Film","FILM",
                "filmSearch","FilmSearch","FILMSEARCH","Film_search","Film-Search","FILM-SEARCH","Film_search","Film_Search","FILM_SEARCH",
                "filmFinder","FilmFinder","FILMFINDER","Film_finder","Film-Finder","FILM-FINDER","Film_finder","Film_Finder","FILM_FINDER",
                "filmSeek","FilmSeek","FILMSEEK","Film_seek","Film-Seek","FILM-SEEK","Film_seek","Film_Seek","FILM_SEEK",
                "favoriteMovie","FavoriteMovie","FAVORITEMOVIE","Favorite_movie","Favorite-Movie","FAVORITE-MOVIE","Favorite_movie","Favorite_Movie","FAVORITE_MOVIE",
                "cinema","Cinema","CINEMA",
                "cinemaSearch","CinemaSearch","CINEMASEARCH","Cinema_search","Cinema-Search","CINEMA-SEARCH","Cinema_search","Cinema_Search","CINEMA_SEARCH",
                "bestMovie","BestMovie","BESTMOVIE","Best_movie","Best-Movie","BEST-MOVIE","Best_movie","Best_Movie","BEST_MOVIE",
                "bestFilm","BestFilm","BESTFILM","Best_film","Best-Film","BEST-FILM","Best_film","Best_Film","BEST_FILM",
                "bestCinema","BestCinema","BESTCINEMA","Best_cinema","Best-Cinema","BEST-CINEMA","Best_cinema","Best_Cinema","BEST_CINEMA",
                "bestMovieFinder","BestMovieFinder","BESTMOVIEFINDER","Best_movie_finder","Best-Movie-Finder","BEST-MOVIE-FINDER","Best_movie_finder","Best_Movie_Finder","BEST_MOVIE_FINDER",],
    "Weather":["Weather","weather","WEATHER",
                "WeatherAPI","weatherAPI","WEATHERAPI","Weather_api","Weather-API","WEATHER-API","Weather_api","Weather_Api","WEATHER_API",
                "WeatherFinder","weatherFinder","WEATHERFINDER","Weather_finder","Weather-Finder","WEATHER-FINDER","Weather_finder","Weather_Finder","WEATHER_FINDER",
                "WeatherSearch","weatherSearch","WEATHERSEARCH","Weather_search","Weather-Search","WEATHER-SEARCH","Weather_search","Weather_Search","WEATHER_SEARCH",
                "Rain","rain","RAIN",
                "RainFinder","rainFinder","RAINFINDER","Rain_finder","Rain-Finder","RAIN-FINDER","Rain_finder","Rain_Finder","RAIN_FINDER",
                "Windguru","windguru","WINDGURU","Wind_guru","Wind-Guru","WIND-GURU","Wind_guru","Wind_Guru","WIND_GURU",
                "Windfinder","windfinder","WINDFINDER","Wind_finder","Wind-Finder","WIND-FINDER","Wind_finder","Wind_Finder","WIND_FINDER",
                "Wind","wind","WIND",],
    "Restaurants":["Restaurants","restaurants","RESTAURANTS","Restaurant","restaurant","RESTAURANT",
                   "RestaurantFinder","restaurantFinder","RESTAURANTFINDER","Restaurant_finder","Restaurant-Finder","RESTAURANT-FINDER","Restaurant_finder","Restaurant_Finder","RESTAURANT_FINDER",
                   "WhereToEat","whereToEat","WHERETOEAT","Where_to_eat","Where-To-Eat","WHERE-TO-EAT","Where_to_eat","Where_To_Eat","WHERE_TO_EAT",
                   "WhereToDine","whereToDine","WHERETODINE","Where_to_dine","Where-To-Dine","WHERE-TO-DINE","Where_to_dine","Where_To_Dine","WHERE_TO_DINE",
                   "WhereToGo","whereToGo","WHERETOGO","Where_to_go","Where-To-Go","WHERE-TO-GO","Where_to_go","Where_To_Go","WHERE_TO_GO",
                   "Food","food","FOOD",
                   "LocalFood","localFood","LOCALFOOD","Local_food","Local-Food","LOCAL-FOOD","Local_food","Local_Food","LOCAL_FOOD",
                   "LocalRestaurant","localRestaurant","LOCALRESTAURANT","Local_restaurant","Local-Restaurant","LOCAL-RESTAURANT","Local_restaurant","Local_Restaurant","LOCAL_RESTAURANT",
                   "LocalCuisine","localCuisine","LOCALCUISINE","Local_cuisine","Local-Cuisine","LOCAL-CUISINE","Local_cuisine","Local_Cuisine","LOCAL_CUISINE",
                   "LocalDish","localDish","LOCALDISH","Local_dish","Local-Dish","LOCAL-DISH","Local_dish","Local_Dish","LOCAL_DISH",
                   "bestRestaurant","BestRestaurant","BESTRESTAURANT","Best_restaurant","Best-Restaurant","BEST-RESTAURANT","Best_restaurant","Best_Restaurant","BEST_RESTAURANT",
                   "bestFood","BestFood","BESTFOOD","Best_food","Best-Food","BEST-FOOD","Best_food","Best_Food","BEST_FOOD",],
    "Hotels":["LocalGuide", "localGuide", "LOCALGUIDE", "Local_Guide", "Local-Guide", "LOCAL-GUIDE", "Local_guide", "Local_Guide", "LOCAL_GUIDE",
              "Hotel", "hotel", "HOTEL", "Hotels", "hotels", "HOTELS",
                "HotelFinder", "hotelFinder", "HOTELFINDER", "Hotel_finder", "Hotel-Finder", "HOTEL-FINDER", "Hotel_finder", "Hotel_Finder", "HOTEL_FINDER",
                "SearchHotel", "searchHotel", "SEARCHHOTEL", "Search_hotel", "Search-Hotel", "SEARCH-HOTEL", "Search_hotel", "Search_Hotel", "SEARCH_HOTEL",
                "Stay", "stay", "STAY", "Stays", "stays", "STAYS",
                "StayFinder", "stayFinder", "STAYFINDER", "Stay_finder", "Stay-Finder", "STAY-FINDER", "Stay_finder", "Stay_Finder", "STAY_FINDER",
                "Airbnb", "airbnb", "AIRBNB", "Air_bnb", "Air-Bnb", "AIR-BNB", "Air_bnb", "Air_Bnb", "AIR_BNB",
                "Booking", "booking", "BOOKING", "Book", "book", "BOOK",
                "BookingFinder", "bookingFinder", "BOOKINGFINDER", "Booking_finder", "Booking-Finder", "BOOKING-FINDER", "Booking_finder", "Booking_Finder", "BOOKING_FINDER",
                "WhereToStay", "whereToStay", "WHERETOSTAY", "Where_to_stay", "Where-To-Stay", "WHERE-TO-STAY", "Where_to_stay", "Where_To_Stay", "WHERE_TO_STAY",
                "WhereToSleep", "whereToSleep", "WHERETOSLEEP", "Where_to_sleep", "Where-To-Sleep", "WHERE-TO-SLEEP", "Where_to_sleep", "Where_To_Sleep", "WHERE_TO_SLEEP",
                "CompareHotel", "compareHotel", "COMPAREHOTEL", "Compare_hotel", "Compare-Hotel", "COMPARE-HOTEL", "Compare_hotel", "Compare_Hotel", "COMPARE_HOTEL",
                "CompareHotels", "compareHotels", "COMPAREHOTELS", "Compare_hotels", "Compare-Hotels", "COMPARE-HOTELS", "Compare_hotels", "Compare_Hotels", "COMPARE_HOTELS",
                "CountryHotel", "countryHotel", "COUNTRYHOTEL", "Country_hotel", "Country-Hotel", "COUNTRY-HOTEL", "Country_hotel", "Country_Hotel", "COUNTRY_HOTEL",
                "CoucheSurfing", "coucheSurfing", "COUCHESURFING", "Couche_surfing", "Couche-Surfing", "COUCHE-SURFING", "Couche_surfing", "Couche_Surfing", "COUCHE_SURFING",
                "CoucheSurf", "coucheSurf", "COUCHESURF", "Couche_surf", "Couche-Surf", "COUCHE-SURF", "Couche_surf", "Couche_Surf", "COUCHE_SURF",],
    "Flights":["Flight", "flight", "FLIGHT", "Flights", "flights", "FLIGHTS",
                "FlightFinder", "flightFinder", "FLIGHTFINDER", "Flight_finder", "Flight-Finder", "FLIGHT-FINDER", "Flight_finder", "Flight_Finder", "FLIGHT_FINDER",
                "FlightSearch", "flightSearch", "FLIGHTSEARCH", "Flight_search", "Flight-Search", "FLIGHT-SEARCH", "Flight_search", "Flight_Search", "FLIGHT_SEARCH",
                "FlightBooking", "flightBooking", "FLIGHTBOOKING", "Flight_booking", "Flight-Booking", "FLIGHT-BOOKING", "Flight_booking", "Flight_Booking", "FLIGHT_BOOKING",
                "CompareFlight", "compareFlight", "COMPAREFLIGHT", "Compare_flight", "Compare-Flight", "COMPARE-FLIGHT", "Compare_flight", "Compare_Flight", "COMPARE_FLIGHT",
                "FlightCompare", "flightCompare", "FLIGHTCOMPARE", "Flight_compare", "Flight-Compare", "FLIGHT-COMPARE", "Flight_compare", "Flight_Compare", "FLIGHT_COMPARE",
                "FindFlight", "findFlight", "FINDFLIGHT", "Find_flight", "Find-Flight", "FIND-FLIGHT", "Find_flight", "Find_Flight", "FIND_FLIGHT",
                "SearchFlight", "searchFlight", "SEARCHFLIGHT", "Search_flight", "Search-Flight", "SEARCH-FLIGHT", "Search_flight", "Search_Flight", "SEARCH_FLIGHT",
                "FlightSearcher", "flightSearcher", "FLIGHTSEARCHER", "Flight_searcher", "Flight-Searcher", "FLIGHT-SEARCHER", "Flight_searcher", "Flight_Searcher", "FLIGHT_SEARCHER",
                "GoogleFlights", "googleFlights", "GOOGLEFLIGHTS", "Google_flights", "Google-Flights", "GOOGLE-FLIGHTS", "Google_flights", "Google_Flights", "GOOGLE_FLIGHTS",
                "FlightBooker", "flightBooker", "FLIGHTBOOKER", "Flight_booker", "Flight-Booker", "FLIGHT-BOOKER", "Flight_booker", "Flight_Booker", "FLIGHT_BOOKER",
                "CheapFlight", "cheapFlight", "CHEAPFLIGHT", "Cheap_flight", "Cheap-Flight", "CHEAP-FLIGHT", "Cheap_flight", "Cheap_Flight", "CHEAP_FLIGHT",
                "RoundTrip", "roundTrip", "ROUNDTRIP", "Round_trip", "Round-Trip", "ROUND-TRIP", "Round_trip", "Round_Trip", "ROUND_TRIP",],
    "Travel":["Travel", "travel", "TRAVEL", "Travels", "travels", "TRAVELS",
                "TravelFinder", "travelFinder", "TRAVELFINDER", "Travel_finder", "Travel-Finder", "TRAVEL-FINDER", "Travel_finder", "Travel_Finder", "TRAVEL_FINDER",
                "InternationalTravel", "internationalTravel", "INTERNATIONALTRAVEL", "International_travel", "International-Travel", "INTERNATIONAL-TRAVEL", "International_travel", "International_Travel", "INTERNATIONAL_TRAVEL",
                "TravelSearch", "travelSearch", "TRAVELSEARCH", "Travel_search", "Travel-Search", "TRAVEL-SEARCH", "Travel_search", "Travel_Search", "TRAVEL_SEARCH",
                "AmericanTravel", "americanTravel", "AMERICANTRAVEL", "American_travel", "American-Travel", "AMERICAN-TRAVEL", "American_travel", "American_Travel", "AMERICAN_TRAVEL",
                "AmazonTravel", "amazonTravel", "AMAZONTRAVEL", "Amazon_travel", "Amazon-Travel", "AMAZON-TRAVEL", "Amazon_travel", "Amazon_Travel", "AMAZON_TRAVEL",
                "Vacation", "vacation", "VACATION", "Vacations", "vacations", "VACATIONS",
                "VacationFinder", "vacationFinder", "VACATIONFINDER", "Vacation_finder", "Vacation-Finder", "VACATION-FINDER", "Vacation_finder", "Vacation_Finder", "VACATION_FINDER",
                "PackageVacation", "packageVacation", "PACKAGEVACATION", "Package_vacation", "Package-Vacation", "PACKAGE-VACATION", "Package_vacation", "Package_Vacation", "PACKAGE_VACATION",
                "VacationSearch", "vacationSearch", "VACATIONSEARCH", "Vacation_search", "Vacation-Search", "VACATION-SEARCH", "Vacation_search", "Vacation_Search", "VACATION_SEARCH",
                "CreateVacation", "createVacation", "CREATEVACATION", "Create_vacation", "Create-Vacation", "CREATE-VACATION", "Create_vacation", "Create_Vacation", "CREATE_VACATION",
                "CruiseVacation", "cruiseVacation", "CRUISEVACATION", "Cruise_vacation", "Cruise-Vacation", "CRUISE-VACATION", "Cruise_vacation", "Cruise_Vacation", "CRUISE_VACATION",
                "VacationPlanner", "vacationPlanner", "VACATIONPLANNER", "Vacation_planner", "Vacation-Planner", "VACATION-PLANNER", "Vacation_planner", "Vacation_Planner", "VACATION_PLANNER",],
    "StoryWriter":["StoryWriter", "storyWriter", "STORYWRITER", "Story_writer", "Story-Writer", "STORY-WRITER", "Story_writer", "Story_Writer", "STORY_WRITER",
                "StoryMaker", "storyMaker", "STORYMAKER", "Story_maker", "Story-Maker", "STORY-MAKER", "Story_maker", "Story_Maker", "STORY_MAKER",
                "StoryCreator", "storyCreator", "STORYCREATOR", "Story_creator", "Story-Creator", "STORY-CREATOR", "Story_creator", "Story_Creator", "STORY_CREATOR",
                "StoryGenerator", "storyGenerator", "STORYGENERATOR", "Story_generator", "Story-Generator", "STORY-GENERATOR", "Story_generator", "Story_Generator", "STORY_GENERATOR",
                "GenerateStory", "generateStory", "GENERATESTORY", "Generate_story", "Generate-Story", "GENERATE-STORY", "Generate_story", "Generate_Story", "GENERATE_STORY",
                "MakeStory", "makeStory", "MAKESTORY", "Make_story", "Make-Story", "MAKE-STORY", "Make_story", "Make_Story", "MAKE_STORY",
                "WriteStory", "writeStory", "WRITESTORY", "Write_story", "Write-Story", "WRITE-STORY", "Write_story", "Write_Story", "WRITE_STORY",
                "BookWriter", "bookWriter", "BOOKWRITER", "Book_writer", "Book-Writer", "BOOK-WRITER", "Book_writer", "Book_Writer", "BOOK_WRITER",
                "WriteBook", "writeBook", "WRITEBOOK", "Write_book", "Write-Book", "WRITE-BOOK", "Write_book", "Write_Book", "WRITE_BOOK",
                "AIAuthor", "aiAuthor", "AIAUTHOR", "Ai_author", "Ai-Author", "AI-AUTHOR", "Ai_author", "Ai_Author", "AI_AUTHOR",
                "WriteNovel", "writeNovel", "WRITENOVEL", "Write_novel", "Write-Novel", "WRITE-NOVEL", "Write_novel", "Write_Novel", "WRITE_NOVEL",
                "WritePoem", "writePoem", "WRITEPOEM", "Write_poem", "Write-Poem", "WRITE-POEM", "Write_poem", "Write_Poem", "WRITE_POEM",
                "CreateStory", "createStory", "CREATESTORY", "Create_story", "Create-Story", "CREATE-STORY", "Create_story", "Create_Story", "CREATE_STORY",
                "CreativeWriter", "creativeWriter", "CREATIVEWRITER", "Creative_writer", "Creative-Writer", "CREATIVE-WRITER", "Creative_writer", "Creative_Writer", "CREATIVE_WRITER",],
    "Recipes":["Recipes", "recipes", "RECIPES", "Recipe", "recipe", "RECIPE",
                "RecipeFinder", "recipeFinder", "RECIPEFINDER", "Recipe_finder", "Recipe-Finder", "RECIPE-FINDER", "Recipe_finder", "Recipe_Finder", "RECIPE_FINDER",
                "RecipeSearch", "recipeSearch", "RECIPESEARCH", "Recipe_search", "Recipe-Search", "RECIPE-SEARCH", "Recipe_search", "Recipe_Search", "RECIPE_SEARCH",
                "HealthyRecipes", "healthyRecipes", "HEALTHYRECIPES", "Healthy_recipes", "Healthy-Recipes", "HEALTHY-RECIPES", "Healthy_recipes", "Healthy_Recipes", "HEALTHY_RECIPES",
                "RecipeGenerator", "recipeGenerator", "RECIPEGENERATOR", "Recipe_generator", "Recipe-Generator", "RECIPE-GENERATOR", "Recipe_generator", "Recipe_Generator", "RECIPE_GENERATOR",
                "RecipeMaker", "recipeMaker", "RECIPEMAKER", "Recipe_maker", "Recipe-Maker", "RECIPE-MAKER", "Recipe_maker", "Recipe_Maker", "RECIPE_MAKER",
                "helpMeCook", "HelpMeCook", "HELPMECOOK", "Help_me_cook", "Help-Me-Cook", "HELP-ME-COOK", "Help_me_cook", "Help_Me_Cook", "HELP_ME_COOK",
                "CookingRecipes", "cookingRecipes", "COOKINGRECIPES", "Cooking_recipes", "Cooking-Recipes", "COOKING-RECIPES", "Cooking_recipes", "Cooking_Recipes", "COOKING_RECIPES",
                "CookingAssistant", "cookingAssistant", "COOKINGASSISTANT", "Cooking_assistant", "Cooking-Assistant", "COOKING-ASSISTANT", "Cooking_assistant", "Cooking_Assistant", "COOKING_ASSISTANT",
                "CookingGuide", "cookingGuide", "COOKINGGUIDE", "Cooking_guide", "Cooking-Guide", "COOKING-GUIDE", "Cooking_guide", "Cooking_Guide", "COOKING_GUIDE",
                "CookingTips", "cookingTips", "COOKINGTIPS", "Cooking_tips", "Cooking-Tips", "COOKING-TIPS", "Cooking_tips", "Cooking_Tips", "COOKING_TIPS",
                "CookingIdeas", "cookingIdeas", "COOKINGIDEAS", "Cooking_ideas", "Cooking-Ideas", "COOKING-IDEAS", "Cooking_ideas", "Cooking_Ideas", "COOKING_IDEAS",
                "MealPlanner", "mealPlanner", "MEALPLANNER", "Meal_planner", "Meal-Planner", "MEAL-PLANNER", "Meal_planner", "Meal_Planner", "MEAL_PLANNER",
                "CookingRecipes", "cookingRecipes", "COOKINGRECIPES", "Cooking_recipes", "Cooking-Recipes", "COOKING-RECIPES", "Cooking_recipes", "Cooking_Recipes", "COOKING_RECIPES",
                "PastaRecipes", "pastaRecipes", "PASTARECIPES", "Pasta_recipes", "Pasta-Recipes", "PASTA-RECIPES", "Pasta_recipes", "Pasta_Recipes", "PASTA_RECIPES",
                "PlantBasedRecipes", "plantBasedRecipes", "PLANTBASEDRECIPES", "Plant_based_recipes", "Plant-Based-Recipes", "PLANT-BASED-RECIPES", "Plant_based_recipes", "Plant_Based_Recipes", "PLANT_BASED_RECIPES",
                "PlanMeals", "planMeals", "PLANMEALS", "Plan_meals", "Plan-Meals", "PLAN-MEALS", "Plan_meals", "Plan_Meals", "PLAN_MEALS",
                "DiabeticRecipes", "diabeticRecipes", "DIABETICRECIPES", "Diabetic_recipes", "Diabetic-Recipes", "DIABETIC-RECIPES", "Diabetic_recipes", "Diabetic_Recipes", "DIABETIC_RECIPES",
                "DiabeticCookbook", "diabeticCookbook", "DIABETICCOOKBOOK", "Diabetic_cookbook", "Diabetic-Cookbook", "DIABETIC-COOKBOOK", "Diabetic_cookbook", "Diabetic_Cookbook", "DIABETIC_COOKBOOK",
                "DiabeticCooking", "diabeticCooking", "DIABETICCOOKING", "Diabetic_cooking", "Diabetic-Cooking", "DIABETIC-COOKING", "Diabetic_cooking", "Diabetic_Cooking", "DIABETIC_COOKING",
                "CuisineRecipes", "cuisineRecipes", "CUISINERECIPES", "Cuisine_recipes", "Cuisine-Recipes", "CUISINE-RECIPES", "Cuisine_recipes", "Cuisine_Recipes", "CUISINE_RECIPES",
                "FrenchRecipes", "frenchRecipes", "FRENCHRECIPES", "French_recipes", "French-Recipes", "FRENCH-RECIPES", "French_recipes", "French_Recipes", "FRENCH_RECIPES",
                "MexicanRecipes", "mexicanRecipes", "MEXICANRECIPES", "Mexican_recipes", "Mexican-Recipes", "MEXICAN-RECIPES", "Mexican_recipes", "Mexican_Recipes", "MEXICAN_RECIPES",
                "ItalianRecipes", "italianRecipes", "ITALIANRECIPES", "Italian_recipes", "Italian-Recipes", "ITALIAN-RECIPES", "Italian_recipes", "Italian_Recipes", "ITALIAN_RECIPES",
                "IndianRecipes", "indianRecipes", "INDIANRECIPES", "Indian_recipes", "Indian-Recipes", "INDIAN-RECIPES", "Indian_recipes", "Indian_Recipes", "INDIAN_RECIPES",
                "CurryRecipes", "curryRecipes", "CURRYRECIPES", "Curry_recipes", "Curry-Recipes", "CURRY-RECIPES", "Curry_recipes", "Curry_Recipes", "CURRY_RECIPES",
                "ChineseRecipes", "chineseRecipes", "CHINESERECIPES", "Chinese_recipes", "Chinese-Recipes", "CHINESE-RECIPES", "Chinese_recipes", "Chinese_Recipes", "CHINESE_RECIPES",
                "CookingAtHome", "cookingAtHome", "COOKINGATHOME", "Cooking_at_home", "Cooking-At-Home", "COOKING-AT-HOME", "Cooking_at_home", "Cooking_At_Home", "COOKING_AT_HOME",],
    "Music": ["Music", "music", "MUSIC",
                "MusicPlayer", "musicPlayer", "MUSICPLAYER", "Music_player", "Music-Player", "MUSIC-PLAYER", "Music_player", "Music_Player", "MUSIC_PLAYER",
                "MusicPlayerApp", "musicPlayerApp", "MUSICPLAYERAPP", "Music_player_app", "Music-Player-App", "MUSIC-PLAYER-APP", "Music_player_app", "Music_Player_App", "MUSIC_PLAYER_APP",
                "MusicPlayerOnline", "musicPlayerOnline", "MUSICPLAYERONLINE", "Music_player_online", "Music-Player-Online", "MUSIC-PLAYER-ONLINE", "Music_player_online", "Music_Player_Online", "MUSIC_PLAYER_ONLINE",
                "PlayMusic", "playMusic", "PLAYMUSIC", "Play_music", "Play-Music", "PLAY-MUSIC", "Play_music", "Play_Music", "PLAY_MUSIC",
                "SpotifyMusic", "spotifyMusic", "SPOTIFYMUSIC", "Spotify_music", "Spotify-Music", "SPOTIFY-MUSIC", "Spotify_music", "Spotify_Music", "SPOTIFY_MUSIC",
                "SoundCloudMusic", "soundCloudMusic", "SOUNDCLOUDMUSIC", "SoundCloud_music", "SoundCloud-Music", "SOUNDCLOUD-MUSIC", "SoundCloud_music", "SoundCloud_Music", "SOUNDCLOUD_MUSIC",
                "SongPlayer", "songPlayer", "SONGPLAYER", "Song_player", "Song-Player", "SONG-PLAYER", "Song_player", "Song_Player", "SONG_PLAYER",
                "GuitarTuner", "guitarTuner", "GUITARTUNER", "Guitar_tuner", "Guitar-Tuner", "GUITAR-TUNER", "Guitar_tuner", "Guitar_Tuner", "GUITAR_TUNER",
                "GrooveMusic", "grooveMusic", "GROOVEMUSIC", "Groove_music", "Groove-Music", "GROOVE-MUSIC", "Groove_music", "Groove_Music", "GROOVE_MUSIC",
                "eMusic", "emusic", "EMUSIC", "E_music", "E-Music", "E-MUSIC", "E_music", "E_Music", "E_MUSIC",
                "djay", "Djay", "DJAY", "D_jay", "D-Jay", "D-JAY", "D_jay", "D_Jay", "D_JAY",
                "Lyrics", "lyrics", "LYRICS", "Lyric", "lyric", "LYRIC", "Lyrics", "Lyrics", "LYRICS",
                "ShareMusic", "shareMusic", "SHAREMUSIC", "Share_music", "Share-Music", "SHARE-MUSIC", "Share_music", "Share_Music", "SHARE_MUSIC",
                "FeelTheBeat", "feelTheBeat", "FEELTHEBEAT", "Feel_the_beat", "Feel-The-Beat", "FEEL-THE-BEAT", "Feel_the_beat", "Feel_The_Beat", "FEEL_THE_BEAT",
                "HipHop", "hipHop", "HIPHOP", "Hip_hop", "Hip-Hop", "HIP-HOP", "Hip_hop", "Hip_Hop", "HIP_HOP",
                "Rap", "rap", "RAP", "Rap", "Rap", "RAP", "Rap", "Rap", "RAP",],
    "News": ["News", "news", "NEWS",
                "NewsApp", "newsApp", "NEWSAPP", "News_app", "News-App", "NEWS-APP", "News_app", "News_App", "NEWS_APP",
                "TopNews", "topNews", "TOPNEWS", "Top_news", "Top-News", "TOP-NEWS", "Top_news", "Top_News", "TOP_NEWS",
                "NewsFeed", "newsFeed", "NEWSFEED", "News_feed", "News-Feed", "NEWS-FEED", "News_feed", "News_Feed", "NEWS_FEED",
                "NewsReader", "newsReader", "NEWSREADER", "News_reader", "News-Reader", "NEWS-READER", "News_reader", "News_Reader", "NEWS_READER",
                "WorldNews", "worldNews", "WORLDNEWS", "World_news", "World-News", "WORLD-NEWS", "World_news", "World_News", "WORLD_NEWS",
                "NewYorkTimes", "newYorkTimes", "NEWYORKTIMES", "New_York_Times", "New-York-Times", "NEW-YORK-TIMES", "New_York_Times", "New_York_Times", "NEW_YORK_TIMES",
                "BBCNews", "bbcNews", "BBCNEWS", "BBC_News", "BBC-News", "BBC-NEWS", "BBC_News", "BBC_News", "BBC_NEWS",
                "CNNNews", "cnnNews", "CNNNEWS", "CNN_News", "CNN-News", "CNN-NEWS", "CNN_News", "CNN_News", "CNN_NEWS",
                "FoxNews", "foxNews", "FOXNEWS", "Fox_News", "Fox-News", "FOX-NEWS", "Fox_News", "Fox_News", "FOX_NEWS",
                "GoogleNews", "googleNews", "GOOGLENEWS", "Google_News", "Google-News", "GOOGLE-NEWS", "Google_News", "Google_News", "GOOGLE_NEWS",
                "NewsRadio", "newsRadio", "NEWSRADIO", "News_radio", "News-Radio", "NEWS-RADIO", "News_radio", "News_Radio", "NEWS_RADIO",
                "NewsChannel", "newsChannel", "NEWSCHANNEL", "News_channel", "News-Channel", "NEWS-CHANNEL", "News_channel", "News_Channel", "NEWS_CHANNEL",
                "NewsPaper", "newsPaper", "NEWSPAPER", "News_paper", "News-Paper", "NEWS-PAPER", "News_paper", "News_Paper", "NEWS_PAPER",
                "NewsAlerts", "newsAlerts", "NEWSALERTS", "News_alerts", "News-Alerts", "NEWS-ALERTS", "News_alerts", "News_Alerts", "NEWS_ALERTS",
                "GreatNews", "greatNews", "GREATNEWS", "Great_news", "Great-News", "GREAT-NEWS", "Great_news", "Great_News", "GREAT_NEWS",
                "GoodNews", "goodNews", "GOODNEWS", "Good_news", "Good-News", "GOOD-NEWS", "Good_news", "Good_News", "GOOD_NEWS",
                "EveningNews", "eveningNews", "EVENINGNEWS", "Evening_news", "Evening-News", "EVENING-NEWS", "Evening_news", "Evening_News", "EVENING_NEWS",
                "WhatIsNew", "whatIsNew", "WHATISNEW", "What_is_new", "What-Is-New", "WHAT-IS-NEW", "What_is_new", "What_Is_New", "WHAT_IS_NEW",
                "YourNews", "yourNews", "YOURNEWS", "Your_news", "Your-News", "YOUR-NEWS", "Your_news", "Your_News", "YOUR_NEWS",],
    "Sports": ["Sports", "sports", "SPORTS",
                "SportsApp", "sportsApp", "SPORTSAPP", "Sports_app", "Sports-App", "SPORTS-APP", "Sports_app", "Sports_App", "SPORTS_APP",
                "EdgeSport", "edgeSport", "EDGESPORT", "Edge_sport", "Edge-Sport", "EDGE-SPORT", "Edge_sport", "Edge_Sport", "EDGE_SPORT",
                "Sportify", "sportify", "SPORTIFY", "Sportify", "Sportify", "SPORTIFY", "Sportify", "Sportify", "SPORTIFY",
                "MySport", "mySport", "MYSPORT", "My_sport", "My-Sport", "MY-SPORT", "My_sport", "My_Sport", "MY_SPORT",
                "MusculeSport", "musculeSport", "MUSCULESPORT", "Muscule_sport", "Muscule-Sport", "MUSCULE-SPORT", "Muscule_sport", "Muscule_Sport", "MUSCULE_SPORT",
                "MyMuscle", "myMuscle", "MYMUSCLE", "My_muscle", "My-Muscle", "MY-MUSCLE", "My_muscle", "My_Muscle", "MY_MUSCLE",
                "UltimateSport", "ultimateSport", "ULTIMATESPORT", "Ultimate_sport", "Ultimate-Sport", "ULTIMATE-SPORT", "Ultimate_sport", "Ultimate_Sport", "ULTIMATE_SPORT",
                "Runners", "runners", "RUNNERS", "Runners", "Runners", "RUNNERS", "Runners", "Runners", "RUNNERS",
                "RunnerApp", "runnerApp", "RUNNERAPP", "Runner_app", "Runner-App", "RUNNER-APP", "Runner_app", "Runner_App", "RUNNER_APP",
                "RouteRun", "routeRun", "ROUTERUN", "Route_run", "Route-Run", "ROUTE-RUN", "Route_run", "Route_Run", "ROUTE_RUN",
                "RunRoute", "runRoute", "RUNROUTE", "Run_route", "Run-Route", "RUN-ROUTE", "Run_route", "Run_Route", "RUN_ROUTE",
                "BigRunner", "bigRunner", "BIGRUNNER", "Big_runner", "Big-Runner", "BIG-RUNNER", "Big_runner", "Big_Runner", "BIG_RUNNER",
                "BikeRoute", "bikeRoute", "BIKEROUTE", "Bike_route", "Bike-Route", "BIKE-ROUTE", "Bike_route", "Bike_Route", "BIKE_ROUTE",
                "BikeRun", "bikeRun", "BIKERUN", "Bike_run", "Bike-Run", "BIKE-RUN", "Bike_run", "Bike_Run", "BIKE_RUN",],
    "PlantWiki": ["PlantWiki", "plantWiki", "PLANTWIKI", "Plant_wiki", "Plant-Wiki", "PLANT-WIKI", "Plant_wiki", "Plant_Wiki", "PLANT_WIKI",
                "Plantpedia", "plantpedia", "PLANTPEDIA", "Plant_pedia", "Plant-Pedia", "PLANT-PEDIA", "Plant_pedia", "Plant_Pedia", "PLANT_PEDIA",
                "Plantopedia", "plantopedia", "PLANTOPEDIA", "Plant_opedia", "Plant-Opedia", "PLANT-OPEDIA", "Plant_opedia", "Plant_Opedia", "PLANT_OPEDIA",
                "IdentifyPlant", "identifyPlant", "IDENTIFYPLANT", "Identify_plant", "Identify-Plant", "IDENTIFY-PLANT", "Identify_plant", "Identify_Plant", "IDENTIFY_PLANT",
                "PlantID", "plantID", "PLANTID", "Plant_id", "Plant-ID", "PLANT-ID", "Plant_id", "Plant_Id", "PLANT_ID",
                "NatureWiki", "natureWiki", "NATUREWIKI", "Nature_wiki", "Nature-Wiki", "NATURE-WIKI", "Nature_wiki", "Nature_Wiki", "NATURE_WIKI",
                "Naturepedia", "naturepedia", "NATUREPEDIA", "Nature_pedia", "Nature-Pedia", "NATURE-PEDIA", "Nature_pedia", "Nature_Pedia", "NATURE_PEDIA",
                "Natureopedia", "natureopedia", "NATUREOPEDIA", "Nature_opedia", "Nature-Opedia", "NATURE-OPEDIA", "Nature_opedia", "Nature_Opedia", "NATURE_OPEDIA",
                "TreeWiki", "treeWiki", "TREEWIKI", "Tree_wiki", "Tree-Wiki", "TREE-WIKI", "Tree_wiki", "Tree_Wiki", "TREE_WIKI",
                "Treepedia", "treepedia", "TREEPEDIA", "Tree_pedia", "Tree-Pedia", "TREE-PEDIA", "Tree_pedia", "Tree_Pedia", "TREE_PEDIA",
                "Treeopedia", "treeopedia", "TREEOPEDIA", "Tree_opedia", "Tree-Opedia", "TREE-OPEDIA", "Tree_opedia", "Tree_Opedia", "TREE_OPEDIA",
                "TreeID", "treeID", "TREEID", "Tree_id", "Tree-ID", "TREE-ID", "Tree_id", "Tree_Id", "TREE_ID",
                "IdentifyTree", "identifyTree", "IDENTIFYTREE", "Identify_tree", "Identify-Tree", "IDENTIFY-TREE", "Identify_tree", "Identify_Tree", "IDENTIFY_TREE",
                "PlantSearch", "plantSearch", "PLANTSEARCH", "Plant_search", "Plant-Search", "PLANT-SEARCH", "Plant_search", "Plant_Search", "PLANT_SEARCH",
                "IdentifyPlant", "identifyPlant", "IDENTIFYPLANT", "Identify_plant", "Identify-Plant", "IDENTIFY-PLANT", "Identify_plant", "Identify_Plant", "IDENTIFY_PLANT",
                "FlowerSearch", "flowerSearch", "FLOWERSEARCH", "Flower_search", "Flower-Search", "FLOWER-SEARCH", "Flower_search", "Flower_Search", "FLOWER_SEARCH",
                "FlowerID", "flowerID", "FLOWERID", "Flower_id", "Flower-ID", "FLOWER-ID", "Flower_id", "Flower_Id", "FLOWER_ID",
                "IdentifyFlower", "identifyFlower", "IDENTIFYFLOWER", "Identify_flower", "Identify-Flower", "IDENTIFY-FLOWER", "Identify_flower", "Identify_Flower", "IDENTIFY_FLOWER",
                "FloraSearch", "floraSearch", "FLORASEARCH", "Flora_search", "Flora-Search", "FLORA-SEARCH", "Flora_search", "Flora_Search", "FLORA_SEARCH",
                "FloraID", "floraID", "FLORAID", "Flora_id", "Flora-ID", "FLORA-ID", "Flora_id", "Flora_Id", "FLORA_ID",],
    "AnimalWiki": ["AnimalWiki", "animalWiki", "ANIMALWIKI", "Animal_wiki", "Animal-Wiki", "ANIMAL-WIKI", "Animal_wiki", "Animal_Wiki", "ANIMAL_WIKI",
                "Animalpedia", "animalpedia", "ANIMALPEDIA", "Animal_pedia", "Animal-Pedia", "ANIMAL-PEDIA", "Animal_pedia", "Animal_Pedia", "ANIMAL_PEDIA",
                "Animalopedia", "animalopedia", "ANIMALOPEDIA", "Animal_opedia", "Animal-Opedia", "ANIMAL-OPEDIA", "Animal_opedia", "Animal_Opedia", "ANIMAL_OPEDIA",
                "MammalWiki", "mammalWiki", "MAMMALWIKI", "Mammal_wiki", "Mammal-Wiki", "MAMMAL-WIKI", "Mammal_wiki", "Mammal_Wiki", "MAMMAL_WIKI",
                "Mammalpedia", "mammalpedia", "MAMMALPEDIA", "Mammal_pedia", "Mammal-Pedia", "MAMMAL-PEDIA", "Mammal_pedia", "Mammal_Pedia", "MAMMAL_PEDIA",
                "Mammalopedia", "mammalopedia", "MAMMALOPEDIA", "Mammal_opedia", "Mammal-Opedia", "MAMMAL-OPEDIA", "Mammal_opedia", "Mammal_Opedia", "MAMMAL_OPEDIA",
                "MammalID", "mammalID", "MAMMALID", "Mammal_id", "Mammal-ID", "MAMMAL-ID", "Mammal_id", "Mammal_Id", "MAMMAL_ID",
                "IdentifyMammal", "identifyMammal", "IDENTIFYMAMMAL", "Identify_mammal", "Identify-Mammal", "IDENTIFY-MAMMAL", "Identify_mammal", "Identify_Mammal", "IDENTIFY_MAMMAL",
                "MammalSearch", "mammalSearch", "MAMMALSEARCH", "Mammal_search", "Mammal-Search", "MAMMAL-SEARCH", "Mammal_search", "Mammal_Search", "MAMMAL_SEARCH",
                "BirdWiki", "birdWiki", "BIRDWIKI", "Bird_wiki", "Bird-Wiki", "BIRD-WIKI", "Bird_wiki", "Bird_Wiki", "BIRD_WIKI",
                "Birdpedia", "birdpedia", "BIRDPEDIA", "Bird_pedia", "Bird-Pedia", "BIRD-PEDIA", "Bird_pedia", "Bird_Pedia", "BIRD_PEDIA",
                "Birdopedia", "birdopedia", "BIRDOPEDIA", "Bird_opedia", "Bird-Opedia", "BIRD-OPEDIA", "Bird_opedia", "Bird_Opedia", "BIRD_OPEDIA",
                "BirdID", "birdID", "BIRDID", "Bird_id", "Bird-ID", "BIRD-ID", "Bird_id", "Bird_Id", "BIRD_ID",
                "IdentifyBird", "identifyBird", "IDENTIFYBIRD", "Identify_bird", "Identify-Bird", "IDENTIFY-BIRD", "Identify_bird", "Identify_Bird", "IDENTIFY_BIRD",
                "BirdSearch", "birdSearch", "BIRDSEARCH", "Bird_search", "Bird-Search", "BIRD-SEARCH", "Bird_search", "Bird_Search", "BIRD_SEARCH",
                "ReptileWiki", "reptileWiki", "REPTILEWIKI", "Reptile_wiki", "Reptile-Wiki", "REPTILE-WIKI", "Reptile_wiki", "Reptile_Wiki", "REPTILE_WIKI",
                "Reptilepedia", "reptilepedia", "REPTILEPEDIA", "Reptile_pedia", "Reptile-Pedia", "REPTILE-PEDIA", "Reptile_pedia", "Reptile_Pedia", "REPTILE_PEDIA",
                "Reptileopedia", "reptileopedia", "REPTILEOPEDIA", "Reptile_opedia", "Reptile-Opedia", "REPTILE-OPEDIA", "Reptile_opedia", "Reptile_Opedia", "REPTILE_OPEDIA",
                "ReptileID", "reptileID", "REPTILEID", "Reptile_id", "Reptile-ID", "REPTILE-ID", "Reptile_id", "Reptile_Id", "REPTILE_ID",
                "IdentifyReptile", "identifyReptile", "IDENTIFYREPTILE", "Identify_reptile", "Identify-Reptile", "IDENTIFY-REPTILE", "Identify_reptile", "Identify_Reptile", "IDENTIFY_REPTILE",
                "ReptileSearch", "reptileSearch", "REPTILESEARCH", "Reptile_search", "Reptile-Search", "REPTILE-SEARCH", "Reptile_search", "Reptile_Search", "REPTILE_SEARCH",
                "AmphibianWiki", "amphibianWiki", "AMPHIBIANWIKI", "Amphibian_wiki", "Amphibian-Wiki", "AMPHIBIAN-WIKI", "Amphibian_wiki", "Amphibian_Wiki", "AMPHIBIAN_WIKI",
                "Amphibianpedia", "amphibianpedia", "AMPHIBIANPEDIA", "Amphibian_pedia", "Amphibian-Pedia", "AMPHIBIAN-PEDIA", "Amphibian_pedia", "Amphibian_Pedia", "AMPHIBIAN_PEDIA",
                "Amphibianopedia", "amphibianopedia", "AMPHIBIANOPEDIA", "Amphibian_opedia", "Amphibian-Opedia", "AMPHIBIAN-OPEDIA", "Amphibian_opedia", "Amphibian_Opedia", "AMPHIBIAN_OPEDIA",
                "AmphibianID", "amphibianID", "AMPHIBIANID", "Amphibian_id", "Amphibian-ID", "AMPHIBIAN-ID", "Amphibian_id", "Amphibian_Id", "AMPHIBIAN_ID",
                "IdentifyAmphibian", "identifyAmphibian", "IDENTIFYAMPHIBIAN", "Identify_amphibian", "Identify-Amphibian", "IDENTIFY-AMPHIBIAN", "Identify_amphibian", "Identify_Amphibian", "IDENTIFY_AMPHIBIAN",
                "AmphibianSearch", "amphibianSearch", "AMPHIBIANSEARCH", "Amphibian_search", "Amphibian-Search", "AMPHIBIAN-SEARCH", "Amphibian_search", "Amphibian_Search", "AMPHIBIAN_SEARCH",
                "FishWiki", "fishWiki", "FISHWIKI", "Fish_wiki", "Fish-Wiki", "FISH-WIKI", "Fish_wiki", "Fish_Wiki", "FISH_WIKI",
                "Fishpedia", "fishpedia", "FISHPEDIA", "Fish_pedia", "Fish-Pedia", "FISH-PEDIA", "Fish_pedia", "Fish_Pedia", "FISH_PEDIA",
                "Fishopedia", "fishopedia", "FISHOPEDIA", "Fish_opedia", "Fish-Opedia", "FISH-OPEDIA", "Fish_opedia", "Fish_Opedia", "FISH_OPEDIA",
                "FishID", "fishID", "FISHID", "Fish_id", "Fish-ID", "FISH-ID", "Fish_id", "Fish_Id", "FISH_ID",
                "IdentifyFish", "identifyFish", "IDENTIFYFISH", "Identify_fish", "Identify-Fish", "IDENTIFY-FISH", "Identify_fish", "Identify_Fish", "IDENTIFY_FISH",
                "FishSearch", "fishSearch", "FISHSEARCH", "Fish_search", "Fish-Search", "FISH-SEARCH", "Fish_search", "Fish_Search", "FISH_SEARCH",
                "InsectWiki", "insectWiki", "INSECTWIKI", "Insect_wiki", "Insect-Wiki", "INSECT-WIKI", "Insect_wiki", "Insect_Wiki", "INSECT_WIKI",
                "Insectpedia", "insectpedia", "INSECTPEDIA", "Insect_pedia", "Insect-Pedia", "INSECT-PEDIA", "Insect_pedia", "Insect_Pedia", "INSECT_PEDIA",
                "Insectopedia", "insectopedia", "INSECTOPEDIA", "Insect_opedia", "Insect-Opedia", "INSECT-OPEDIA", "Insect_opedia", "Insect_Opedia", "INSECT_OPEDIA",
                "InsectID", "insectID", "INSECTID", "Insect_id", "Insect-ID", "INSECT-ID", "Insect_id", "Insect_Id", "INSECT_ID",
                "IdentifyInsect", "identifyInsect", "IDENTIFYINSECT", "Identify_insect", "Identify-Insect", "IDENTIFY-INSECT", "Identify_insect", "Identify_Insect", "IDENTIFY_INSECT",
                "InsectSearch", "insectSearch", "INSECTSEARCH", "Insect_search", "Insect-Search", "INSECT-SEARCH", "Insect_search", "Insect_Search", "INSECT_SEARCH",],
    "Location": ["Location", "location", "LOCATION", "Locate", "locate", "LOCATE", "Locate", "Locate", "LOCATE",
                "LocationSearch", "locationSearch", "LOCATIONSEARCH", "Location_search", "Location-Search", "LOCATION-SEARCH", "Location_search", "Location_Search", "LOCATION_SEARCH",
                "LocationFinder", "locationFinder", "LOCATIONFINDER", "Location_finder", "Location-Finder", "LOCATION-FINDER", "Location_finder", "Location_Finder", "LOCATION_FINDER",
                "DistanceCalculator", "distanceCalculator", "DISTANCECALCULATOR", "Distance_calculator", "Distance-Calculator", "DISTANCE-CALCULATOR", "Distance_calculator", "Distance_Calculator", "DISTANCE_CALCULATOR",
                "DistanceFinder", "distanceFinder", "DISTANCEFINDER", "Distance_finder", "Distance-Finder", "DISTANCE-FINDER", "Distance_finder", "Distance_Finder", "DISTANCE_FINDER",
                "DistanceSearch", "distanceSearch", "DISTANCESEARCH", "Distance_search", "Distance-Search", "DISTANCE-SEARCH", "Distance_search", "Distance_Search", "DISTANCE_SEARCH",
                "CalculateDistance", "calculateDistance", "CALCULATEDISTANCE", "Calculate_distance", "Calculate-Distance", "CALCULATE-DISTANCE", "Calculate_distance", "Calculate_Distance", "CALCULATE_DISTANCE",
                "GeoDistance", "geoDistance", "GEODISTANCE", "Geo_distance", "Geo-Distance", "GEO-DISTANCE", "Geo_distance", "Geo_Distance", "GEO_DISTANCE",
                "GPSDistance", "gpsDistance", "GPSDISTANCE", "GPS_distance", "GPS-Distance", "GPS-DISTANCE", "GPS_distance", "GPS_Distance", "GPS_DISTANCE",
                "WhereAmI", "whereAmI", "WHEREAMI", "Where_am_i", "Where-am-i", "WHERE-AM-I", "Where_am_i", "Where_Am_I", "WHERE_AM_I",
                "LocateMe", "locateMe", "LOCATEME", "Locate_me", "Locate-Me", "LOCATE-ME", "Locate_me", "Locate_Me", "LOCATE_ME",
                "LocateMyself", "locateMyself", "LOCATEMYSELF", "Locate_myself", "Locate-Myself", "LOCATE-MYSELF", "Locate_myself", "Locate_Myself", "LOCATE_MYSELF",
                "GPSTracker", "gpsTracker", "GPSTRACKER", "GPS_tracker", "GPS-Tracker", "GPS-TRACKER", "GPS_tracker", "GPS_Tracker", "GPS_TRACKER",
                "GPSTool", "gpsTool", "GPSTOOL", "GPS_tool", "GPS-Tool", "GPS-TOOL", "GPS_tool", "GPS_Tool", "GPS_TOOL",],
    "Translator": ["Translator", "translator", "TRANSLATOR", "Translate", "translate", "TRANSLATE", "Translate", "Translate", "TRANSLATE",
                "Translation", "translation", "TRANSLATION", "Translate", "translate", "TRANSLATE", "Translate", "Translate", "TRANSLATE",
                "TranslateText", "translateText", "TRANSLATETEXT", "Translate_text", "Translate-Text", "TRANSLATE-TEXT", "Translate_text", "Translate_Text", "TRANSLATE_TEXT",
                "TranslateWord", "translateWord", "TRANSLATEWORD", "Translate_word", "Translate-Word", "TRANSLATE-WORD", "Translate_word", "Translate_Word", "TRANSLATE_WORD",
                "TranslateSentence", "translateSentence", "TRANSLATESENTENCE", "Translate_sentence", "Translate-Sentence", "TRANSLATE-SENTENCE", "Translate_sentence", "Translate_Sentence", "TRANSLATE_SENTENCE",
                "TranslateParagraph", "translateParagraph", "TRANSLATEPARAGRAPH", "Translate_paragraph", "Translate-Paragraph", "TRANSLATE-PARAGRAPH", "Translate_paragraph", "Translate_Paragraph", "TRANSLATE_PARAGRAPH",
                "TranslateDocument", "translateDocument", "TRANSLATEDOCUMENT", "Translate_document", "Translate-Document", "TRANSLATE-DOCUMENT", "Translate_document", "Translate_Document", "TRANSLATE_DOCUMENT",
                "TranslateTextTo", "translateTextTo", "TRANSLATETEXTTO", "Translate_text_to", "Translate-Text-To", "TRANSLATE-TEXT-TO", "Translate_text_to", "Translate_Text_To", "TRANSLATE_TEXT_TO",
                "SpeechTranslator", "speechTranslator", "SPEECHTRANSLATOR", "Speech_translator", "Speech-Translator", "SPEECH-TRANSLATOR", "Speech_translator", "Speech_Translator", "SPEECH_TRANSLATOR",
                "SpanishTranslator", "spanishTranslator", "SPANISHTRANSLATOR", "Spanish_translator", "Spanish-Translator", "SPANISH-TRANSLATOR", "Spanish_translator", "Spanish_Translator", "SPANISH_TRANSLATOR",
                "EnglishTranslator", "englishTranslator", "ENGLISHTRANSLATOR", "English_translator", "English-Translator", "ENGLISH-TRANSLATOR", "English_translator", "English_Translator", "ENGLISH_TRANSLATOR",
                "FrenchTranslator", "frenchTranslator", "FRENCHTRANSLATOR", "French_translator", "French-Translator", "FRENCH-TRANSLATOR", "French_translator", "French_Translator", "FRENCH_TRANSLATOR",
                "GermanTranslator", "germanTranslator", "GERMANTRANSLATOR", "German_translator", "German-Translator", "GERMAN-TRANSLATOR", "German_translator", "German_Translator", "GERMAN_TRANSLATOR",
                "GermanToEnglish", "germanToEnglish", "GERMANTOENGLISH", "German_to_english", "German-To-English", "GERMAN-TO-ENGLISH", "German_to_english", "German_To_English", "GERMAN_TO_ENGLISH",
                "EnglishToGerman", "englishToGerman", "ENGLISHTOGERMAN", "English_to_german", "English-To-German", "ENGLISH-TO-GERMAN", "English_to_german", "English_To_German", "ENGLISH_TO_GERMAN",
                "SpanishToEnglish", "spanishToEnglish", "SPANISHTOENGLISH", "Spanish_to_english", "Spanish-To-English", "SPANISH-TO-ENGLISH", "Spanish_to_english", "Spanish_To_English", "SPANISH_TO_ENGLISH",
                "EnglishToSpanish", "englishToSpanish", "ENGLISHTOSPANISH", "English_to_spanish", "English-To-Spanish", "ENGLISH-TO-SPANISH", "English_to_spanish", "English_To_Spanish", "ENGLISH_TO_SPANISH",
                "FrenchToEnglish", "frenchToEnglish", "FRENCHTOENGLISH", "French_to_english", "French-To-English", "FRENCH-TO-ENGLISH", "French_to_english", "French_To_English", "FRENCH_TO_ENGLISH",
                "EnglishToFrench", "englishToFrench", "ENGLISHTOFRENCH", "English_to_french", "English-To-French", "ENGLISH-TO-FRENCH", "English_to_french", "English_To_French", "ENGLISH_TO_FRENCH",
                "BestTranslator"],
    "Code": ["Code", "code", "CODE", "Code", "Code", "CODE", "Code", "Code", "CODE",
                "CodeEditor", "codeEditor", "CODEEDITOR", "Code_editor", "Code-Editor", "CODE-EDITOR", "Code_editor", "Code_Editor", "CODE_EDITOR",
                "CodeExplorer", "codeExplorer", "CODEEXPLORER", "Code_explorer", "Code-Explorer", "CODE-EXPLORER", "Code_explorer", "Code_Explorer", "CODE_EXPLORER",
                "CodeGenerator", "codeGenerator", "CODEGENERATOR", "Code_generator", "Code-Generator", "CODE-GENERATOR", "Code_generator", "Code_Generator", "CODE_GENERATOR",
                "CodeCompiler", "codeCompiler", "CODECOMPILER", "Code_compiler", "Code-Compiler", "CODE-COMPILER", "Code_compiler", "Code_Compiler", "CODE_COMPILER",
                "PythonCompiler", "pythonCompiler", "PYTHONCOMPILER", "Python_compiler", "Python-Compiler", "PYTHON-COMPILER", "Python_compiler", "Python_Compiler", "PYTHON_COMPILER",
                "JavaCompiler", "javaCompiler", "JAVACOMPILER", "Java_compiler", "Java-Compiler", "JAVA-COMPILER", "Java_compiler", "Java_Compiler", "JAVA_COMPILER",
                "GetCode", "getCode", "GETCODE", "Get_code", "Get-Code", "GET-CODE", "Get_code", "Get_Code", "GET_CODE",
                "GenerateCode", "generateCode", "GENERATECODE", "Generate_code", "Generate-Code", "GENERATE-CODE", "Generate_code", "Generate_Code", "GENERATE_CODE",
                "NewCode", "newCode", "NEWCODE", "New_code", "New-Code", "NEW-CODE", "New_code", "New_Code", "NEW_CODE",
                "Coding", "coding", "CODING", "Coding", "Coding", "CODING", "Coding", "Coding", "CODING",
                "Codify", "codify", "CODIFY", "Codify", "Codify", "CODIFY", "Codify", "Codify", "CODIFY",
                "AIcode", "aIcode", "AICODE", "AI_code", "AI-Code", "AI-CODE", "AI_code", "AI_Code", "AI_CODE",
                "CodeAI", "codeAI", "CODEAI", "Code_ai", "Code-Ai", "CODE-AI", "Code_ai", "Code_AI", "CODE_AI",],
    "DataBase": ["DataBase", "database", "DATABASE", "Data_base", "Data-Base", "DATA-BASE", "Data_base", "Data_Base", "DATA_BASE",
                "Data", "data", "DATA", "Data", "Data", "DATA", "Data", "Data", "DATA",
                "DataStorage", "dataStorage", "DATASTORAGE", "Data_storage", "Data-Storage", "DATA-STORAGE", "Data_storage", "Data_Storage", "DATA_STORAGE",
                "GetDatabase", "getDatabase", "GETDATABASE", "Get_database", "Get-Database", "GET-DATABASE", "Get_database", "Get_Database", "GET_DATABASE",
                "EditDatabase", "editDatabase", "EDITDATABASE", "Edit_database", "Edit-Database", "EDIT-DATABASE", "Edit_database", "Edit_Database", "EDIT_DATABASE",
                "ExportDatabase", "exportDatabase", "EXPORTDATABASE", "Export_database", "Export-Database", "EXPORT-DATABASE", "Export_database", "Export_Database", "EXPORT_DATABASE",
                "FindTable", "findTable", "FINDTABLE", "Find_table", "Find-Table", "FIND-TABLE", "Find_table", "Find_Table", "FIND_TABLE",
                "FindDatabase", "findDatabase", "FINDDATABASE", "Find_database", "Find-Database", "FIND-DATABASE", "Find_database", "Find_Database", "FIND_DATABASE",
                "RowDatabase", "rowDatabase", "ROWDATABASE", "Row_database", "Row-Database", "ROW-DATABASE", "Row_database", "Row_Database", "ROW_DATABASE",
                "QueryDatabase", "queryDatabase", "QUERYDATABASE", "Query_database", "Query-Database", "QUERY-DATABASE", "Query_database", "Query_Database", "QUERY_DATABASE",
                "DatabaseQuery", "databaseQuery", "DATABASEQUERY", "Database_query", "Database-Query", "DATABASE-QUERY", "Database_query", "Database_Query", "DATABASE_QUERY",
                "LoadDatabase", "loadDatabase", "LOADDATABASE", "Load_database", "Load-Database", "LOAD-DATABASE", "Load_database", "Load_Database", "LOAD_DATABASE",
                "LookDatabase", "lookDatabase", "LOOKDATABASE", "Look_database", "Look-Database", "LOOK-DATABASE", "Look_database", "Look_Database", "LOOK_DATABASE",
                "LookUpDatabase", "lookUpDatabase", "LOOKUPDATABASE", "Look_up_database", "Look-Up-Database", "LOOK-UP-DATABASE", "Look_up_database", "Look_Up_Database", "LOOK_UP_DATABASE",
                "FindInDatabase", "findInDatabase", "FINDINDATABASE", "Find_in_database", "Find-In-Database", "FIND-IN-DATABASE", "Find_in_database", "Find_In_Database", "FIND_IN_DATABASE",
                "FindInTable", "findInTable", "FINDINTABLE", "Find_in_table", "Find-In-Table", "FIND-IN-TABLE", "Find_in_table", "Find_In_Table", "FIND_IN_TABLE",
                "TableQuery", "tableQuery", "TABLEQUERY", "Table_query", "Table-Query", "TABLE-QUERY", "Table_query", "Table_Query", "TABLE_QUERY",
                "TableSearch", "tableSearch", "TABLESEARCH", "Table_search", "Table-Search", "TABLE-SEARCH", "Table_search", "Table_Search", "TABLE_SEARCH",
                "TableLookup", "tableLookup", "TABLELOOKUP", "Table_lookup", "Table-Lookup", "TABLE-LOOKUP", "Table_lookup", "Table_Lookup", "TABLE_LOOKUP",
                "TableSort", "tableSort", "TABLESORT", "Table_sort", "Table-Sort", "TABLE-SORT", "Table_sort", "Table_Sort", "TABLE_SORT",
                "TableFilter", "tableFilter", "TABLEFILTER", "Table_filter", "Table-Filter", "TABLE-FILTER", "Table_filter", "Table_Filter", "TABLE_FILTER",
                "FilterTable", "filterTable", "FILTERTABLE", "Filter_table", "Filter-Table", "FILTER-TABLE", "Filter_table", "Filter_Table", "FILTER_TABLE",
                "SortTable", "sortTable", "SORTTABLE", "Sort_table", "Sort-Table", "SORT-TABLE", "Sort_table", "Sort_Table", "SORT_TABLE",],
    "Youtube": ["Youtube", "youtube", "YOUTUBE", "You_tube", "You-Tube", "YOU-TUBE", "You_tube", "You_Tube", "YOU_TUBE",
                "Video", "video", "VIDEO", "Video", "Video", "VIDEO", "Video", "Video", "VIDEO",
                "WatchVideo", "watchVideo", "WATCHVIDEO", "Watch_video", "Watch-Video", "WATCH-VIDEO", "Watch_video", "Watch_Video", "WATCH_VIDEO",
                "PlayVideo", "playVideo", "PLAYVIDEO", "Play_video", "Play-Video", "PLAY-VIDEO", "Play_video", "Play_Video", "PLAY_VIDEO",
                "SearchVideo", "searchVideo", "SEARCHVIDEO", "Search_video", "Search-Video", "SEARCH-VIDEO", "Search_video", "Search_Video", "SEARCH_VIDEO",],
    "Document": ["Document", "document", "DOCUMENT", "Doc", "doc", "DOC", "Doc", "Doc", "DOC",
                "WriteDocument", "writeDocument", "WRITEDOCUMENT", "Write_document", "Write-Document", "WRITE-DOCUMENT", "Write_document", "Write_Document", "WRITE_DOCUMENT",
                "EditDocument", "editDocument", "EDITDOCUMENT", "Edit_document", "Edit-Document", "EDIT-DOCUMENT", "Edit_document", "Edit_Document", "EDIT_DOCUMENT",],
}

tool_desc = {
    "Calculator":{
        "mix": ["simple calculator that can add, subtract, multiply, and divide", 
                "can add, subtract, multiply and divide",
                "adds, subtracts, multiplies and divides",
                "simple calculator",
                "can compute simple arithmetic",
                "calculate simple expressions",
                "can do simple arithmetic",
                "can do simple math",
                "solve simple arithmetic",
                "solve simple math",
                "can do basic arithmetic",],
        "add":[
            "adds numbers together effortlessly",
            "quickly sums up values",
            "easily computes additions",
            "swiftly calculates sums",
            "designed for hassle-free addition",
            "performs straightforward addition",
            "simplifies adding numbers",
            "specializes in addition operations",
            "a tool to swiftly add numbers",
            "efficiently sums up numerical inputs",
            "focuses on quick addition calculations",
            "tailored for basic number addition",
            "streamlines the process of adding numbers",
            "provides an efficient way to sum values",
            "simplistic calculator for addition tasks",
            "helps with rapid number addition",
            "ideal for hassle-free sum calculations",
            "aids in swiftly totaling numbers",
            "simplifies your addition calculations",
            "designed for efficient number summing"
        ],
        "subtract": [
            "efficiently subtracts numbers",
            "swiftly calculates differences",
            "designed for hassle-free subtraction",
            "performs straightforward subtraction",
            "simplifies subtracting numbers",
            "specializes in subtraction operations",
            "a tool to quickly subtract numbers",
            "easily computes subtractions",
            "tailored for basic number subtraction",
            "focuses on quick subtraction calculations",
            "streamlines the process of subtracting numbers",
            "provides an efficient way to find differences",
            "simplistic calculator for subtraction tasks",
            "helps with rapid number subtraction",
            "ideal for hassle-free difference calculations",
            "aids in swiftly determining discrepancies",
            "simplifies your subtraction calculations",
            "designed for efficient number difference finding",
            "rapidly calculates numerical subtractions",
            "simplifies the process of finding differences",
            "designed for quick and accurate number subtraction"
        ],
        "multiply" : [
            "easily calculates products",
            "swiftly performs multiplications",
            "designed for hassle-free multiplication",
            "efficiently multiplies numbers",
            "simplifies multiplying values",
            "specializes in multiplication operations",
            "a tool to swiftly multiply numbers",
            "quickly computes products",
            "tailored for basic number multiplication",
            "focuses on quick multiplication calculations",
            "streamlines the process of multiplying numbers",
            "provides an efficient way to find products",
            "simplistic calculator for multiplication tasks",
            "helps with rapid number multiplication",
            "ideal for hassle-free product calculations",
            "aids in swiftly calculating results",
            "simplifies your multiplication calculations",
            "designed for efficient number product determination",
            "rapidly calculates numerical products",
            "simplifies the process of finding products",
            "designed for quick and accurate number multiplication"
        ],
        "divide": [
            "easily calculates quotients",
            "swiftly performs divisions",
            "designed for hassle-free division",
            "efficiently divides numbers",
            "simplifies dividing values",
            "specializes in division operations",
            "a tool to swiftly divide numbers",
            "quickly computes quotients",
            "tailored for basic number division",
            "focuses on quick division calculations",
            "streamlines the process of dividing numbers",
            "provides an efficient way to find quotients",
            "simplistic calculator for division tasks",
            "helps with rapid number division",
            "ideal for hassle-free quotient calculations",
            "aids in swiftly determining divisions",
            "simplifies your division calculations",
            "designed for efficient number quotient determination",
            "rapidly calculates numerical divisions",
            "simplifies the process of finding quotients",
            "designed for quick and accurate number division"
        ],
        "add_subtract": [
            "quickly adds and subtracts numbers",
            "performs both addition and subtraction",
            "effortlessly calculates sums and differences",
            "swiftly handles addition and subtraction tasks",
            "simplifies combined addition and subtraction",
            "specializes in both addition and subtraction operations",
            "a tool for quick addition and subtraction calculations",
            "tailored for both basic addition and subtraction",
            "focuses on quick combined calculation tasks",
            "streamlines the process of adding and subtracting numbers",
            "provides an efficient way to work with sums and differences",
            "simplistic calculator for combined arithmetic tasks",
            "helps with rapid number addition and subtraction",
            "ideal for hassle-free combined calculation tasks",
            "aids in swiftly calculating both sums and differences",
            "simplifies your addition and subtraction calculations",
            "designed for efficient combined arithmetic operations",
            "rapidly calculates both numerical sums and differences",
            "simplifies the process of working with addition and subtraction",
            "designed for quick and accurate combined calculations",
            "efficiently handles both addition and subtraction tasks"
        ],
        "mult_divide": [
            "Performs multiplication and division operations",
            "Calculates products and quotients quickly",
            "Efficient tool for both multiplication and division",
            "Simplifies complex multiplicative and divisive tasks",
            "Streamlines finding products and quotients",
            "Specializes in multiplication and division",
            "Tailored for efficient numeric operations",
            "Hassle-free tool for quick multiplication and division",
            "Rapid calculator for both tasks",
            "Efficiently handles numerical multiplication and division",
            "Comprehensive solution for accurate math",
            "Quickly computes products and divisions",
            "Simplifies working with multiplication and division",
            "Aids in quick numeric operations",
            "Versatile tool for accurate math tasks",
            "Designed for efficiency in both tasks",
            "Enhances productivity with multiplication and division",
            "Efficient way to work with numbers",
            "Quick and accurate multiplication and division",
            "Versatile solution for math calculations"
        ],
    },
    "WikiSearch": [
        "fetches Wikipedia snippets for search queries",
        "provides concise information from Wikipedia",
        "retrieves brief summaries from Wikipedia articles",
        "quickly fetches relevant details from Wikipedia",
        "simplifies access to summarized Wikipedia content",
        "helps retrieve summarized data from Wikipedia",
        "a tool for obtaining brief Wikipedia information",
        "retrieves concise snippets from Wikipedia articles",
        "saves time by delivering summarized Wikipedia content",
        "provides summarized Wikipedia details for search terms",
        "retrieves Wikipedia snippets swiftly",
        "finds concise info on Wikipedia topics",
        "fetches summarized Wikipedia data",
        "get quick insights from Wikipedia",
        "saves time by summarizing Wikipedia",
        "a tool for rapid Wikipedia info",
        "quick access to Wikipedia summaries",
        "retrieve summarized Wikipedia content",
        "simplifies Wikipedia data retrieval",
        "quickly get summarized Wikipedia details",
        "your quick reference to Wikipedia",
        "explore topics with Wikipedia snippets",
        "access summarized Wikipedia content",
        "tool for simplified Wikipedia exploration",
        "discover knowledge using Wikipedia",
        "learn from summarized Wikipedia info",
        "your gateway to concise Wikipedia data",
        "access insights through Wikipedia",
        "tool for fast Wikipedia lookups",
        "explore subjects using Wikipedia summaries",
        "a resource for summarized Wikipedia data",
        "get insights from Wikipedia snippets",
        "simplify research with Wikipedia",
        "discover topics using brief Wikipedia info",
        "your companion for Wikipedia exploration",
        "access info efficiently with Wikipedia",
        "tool for streamlined Wikipedia searches",
        "explore subjects with summarized Wikipedia",
        "navigate knowledge through Wikipedia",
        "uncover information with Wikipedia snippets",
        "your guide to summarized Wikipedia content",
        "access concise knowledge from Wikipedia",
        "tool for simplified Wikipedia inquiries",
        "explore subjects through Wikipedia info",
        "retrieve insights using Wikipedia",
        "your resource for summarized Wikipedia knowledge",
        "access information effortlessly with Wikipedia",
        "tool for quick and easy Wikipedia references",
        "explore topics using Wikipedia summaries",
        "access brief insights from Wikipedia",
        "retrieve snippets from global knowledge hubs",
        "access information snippets from diverse sources",
        "find insights from internet reference materials",
        "explore topics using data from multiple sources",
        "discover details from the vast expanse of information",
        "navigate through global data banks for insights",
        "fetch information snippets from expert references",
        "get concise knowledge from an extensive database",
        "access compact knowledge from authoritative sources",
        "explore topics using reputable data repositories",
        "retrieve insights from the realm of encyclopedias",
        "navigate through vast reference sources for facts",
        "discover condensed knowledge from WorldKnowledge",
        "access summarized information from diverse sources",
        "explore topics using data from reliable encyclopedias",
        "retrieve insights from well-established databases",
        "discover details from the depth of reference materials",
        "navigate through expert resources for comprehensive insights",
        "access data from trusted global knowledge bases",
        "explore topics using data from information repositories",
        "retrieve insights from reputable internet encyclopedias",
        "discover knowledge from the depth of WorldKnowledge",
        "navigate through comprehensive data sources for insights",
        "access snippets of information from reliable databases",
        "explore topics using insights from various repositories",
        "retrieve condensed knowledge from comprehensive sources",
        "discover insights from the vast expanse of information",
        "navigate through global data repositories for facts",
        "access information snippets from diverse encyclopedias",
        "explore topics using comprehensive information sources",
        "retrieve insights from well-established reference materials",
        "discover details from the depth of global databases",
        "navigate through expert resources for authoritative insights",
        "access data from trusted internet knowledge hubs",
        "explore topics using insights from reputable references",
        "retrieve condensed knowledge from reputable sources",
        "discover insights from the realm of comprehensive databases",
        "navigate through data repositories for comprehensive facts",
        "access snippets of information from authoritative encyclopedias",
        "explore topics using information from expert references",
        "retrieve insights from well-established internet sources",
        "discover details from the depth of reputable databases",
        "navigate through comprehensive resources for reliable insights",
        "access data from trusted knowledge repositories",
        "explore topics using comprehensive data banks for insights",
        "retrieve insights from reputable global encyclopedias",
        "discover insights from the realm of comprehensive knowledge",
        "navigate through data sources for thorough insights",
        "access snippets of information from well-established references",
        "explore topics using information from reliable sources",
        "retrieve insights from comprehensive internet encyclopedias",
        "discover knowledge from the depth of authoritative databases",
        "navigate through expert sources for comprehensive facts",
        "access data from trusted reference repositories",
        "explore topics using comprehensive data sources for insights",
        "retrieve insights from reputable information hubs",
        "discover insights from the realm of comprehensive encyclopedias",
        "navigate through data banks for thorough knowledge",
        "access snippets of information from well-established sources",
        "explore topics using reliable references for insights",
        "retrieve insights from comprehensive global databases",
        "discover knowledge from the depth of reputable knowledge bases",
        "navigate through expert data sources for comprehensive facts",
        "access data from trusted internet repositories",
        "explore topics using comprehensive information databases",
        "retrieve insights from well-established data banks",
        "discover insights from the realm of reliable references",
        "navigate through data hubs for thorough knowledge"
    ],
    "Calendar": [
        "returns the current date and time",
        "provides the present date and time",
        "fetches the current date and time",
        "simplifies access to the current date",
        "quickly retrieves the current date and time",
        "helps you get the current date and time",
        "a tool to fetch today's date and time",
        "provides up-to-date date and time information",
        "saves time by giving you the current date",
        "returns the present date and time on demand",
        "instantly provides today's date",
        "fetches the current date promptly",
        "simplifies date access",
        "returns today's date instantly",
        "get the current date quickly",
        "a tool for today's date and time",
        "provides the present date effortlessly",
        "fetches the date without delay",
        "quickly delivers the current date",
        "retrieve today's date and time easily",
        "your companion for date inquiries",
        "retrieve dates with minimal effort",
        "stay informed with accurate dates",
        "tool to help you with date queries",
        "easily find out the current date",
        "get organized with accurate dates",
        "a reliable date information tool",
        "provides the date you need, fast",
        "get up-to-date date information",
        "stay on schedule with this tool",
        "instantly know what day it is",
        "a simple date lookup solution",
        "stay aware of the current date",
        "tool to quickly fetch dates",
        "helps you keep track of dates",
        "provides a date reference tool",
        "offers a date lookup feature",
        "quickly gives you the current date",
        "a tool to retrieve dates easily",
        "your go-to for date information",
        "simplifies date retrieval tasks",
        "get date details in an instant",
        "tool for managing your calendar",
        "a handy date and time assistant",
        "access dates effortlessly with this tool",
        "find out today's date right away",
        "quickly retrieve the present date",
        "a utility for tracking dates",
        "provides current date and time details",
        "get the date with just a click",
    ],
    "Random": [
        "generates random numbers with ease",
        "provides random numeric values",
        "quickly fetches random numerical data",
        "simplifies random number generation",
        "helps you obtain random numeric outputs",
        "a tool to generate random numbers",
        "provides randomness at your fingertips",
        "generates arbitrary numeric values effortlessly",
        "saves time by producing random numbers",
        "retrieves random numerical data instantly"
    ],
    "Movies": [
        "fetches information about movies",
        "provides details about films",
        "quickly retrieves movie-related data",
        "simplifies access to movie information",
        "helps you get details about movies",
        "a tool to fetch movie-related information",
        "provides movie data at your fingertips",
        "fetches movie details effortlessly",
        "saves time by delivering film-related information",
        "retrieves film-related data instantly"
    ],
    "Weather": [
        "fetches current weather conditions",
        "provides up-to-date weather data",
        "quickly retrieves current weather information",
        "simplifies access to real-time weather conditions",
        "helps you get the latest weather updates",
        "a tool to fetch current weather details",
        "provides weather information on demand",
        "fetches weather conditions effortlessly",
        "saves time by delivering current weather data",
        "retrieves real-time weather information instantly"
    ],
    "Restaurants": [
        "discover local dining options",
        "find nearby culinary experiences",
        "explore restaurant choices",
        "locate eateries in your area",
        "get recommendations for dining",
        "explore local food destinations",
        "discover nearby restaurant gems",
        "find places to satisfy your palate",
        "get suggestions for dining out",
        "explore local dining scenes",
        "discover a variety of restaurant options",
        "find flavorful places to eat",
        "get recommendations for culinary delights",
        "explore a diverse range of eateries",
        "discover local restaurant treasures",
        "find enticing options for dining",
        "explore culinary experiences around you",
        "discover a world of dining possibilities",
        "find unique and flavorful restaurants",
        "explore local dining adventures"
    ],
    "Hotels": [
        "find accommodations for your stay",
        "explore lodging options",
        "discover places to rest during travels",
        "locate suitable places to stay",
        "get recommendations for accommodations",
        "explore hotel choices for your trip",
        "discover comfortable lodging solutions",
        "find places to relax and unwind",
        "get suggestions for your lodging needs",
        "explore a variety of hotel options",
        "discover a range of accommodation choices",
        "find cozy spots for your stay",
        "get recommendations for comfortable stays",
        "explore a diverse array of hotels",
        "discover lodging solutions tailored to you",
        "find enticing options for accommodations",
        "explore diverse places to rest during travels",
        "discover a world of lodging possibilities",
        "find unique and inviting hotels",
        "explore cozy and welcoming accommodations"
    ],
    "Flights": [
        "search for flight options",
        "explore available air travel choices",
        "discover flights for your journey",
        "locate options for air travel",
        "get recommendations for flying",
        "explore flight choices for your trip",
        "discover convenient flight connections",
        "find flights to your desired destinations",
        "get suggestions for your air travel needs",
        "explore a variety of flight options",
        "discover a range of flight connections",
        "find flights to your chosen places",
        "get recommendations for seamless flying",
        "explore a diverse array of flight routes",
        "discover flight options tailored to you",
        "find enticing options for air travel",
        "explore diverse flight possibilities",
        "discover a world of flight connections",
        "find unique and convenient flights",
        "explore flights to various destinations"
    ],
    "Travel": [
        "plan your upcoming journey",
        "explore travel options",
        "discover ways to explore new places",
        "locate ideas for your next adventure",
        "get recommendations for travel experiences",
        "explore options for your travel plans",
        "discover exciting travel destinations",
        "find tips and suggestions for exploring",
        "get suggestions for your travel needs",
        "explore a variety of travel opportunities",
        "discover a range of travel experiences",
        "find ways to embark on new journeys",
        "get recommendations for memorable travel",
        "explore a diverse array of travel destinations",
        "discover travel options tailored to you",
        "find enticing ideas for your next trip",
        "explore diverse travel possibilities",
        "discover a world of travel adventures",
        "find unique and enriching travel experiences",
        "explore new horizons and travel routes"
    ],
    "StoryWriter": [
        "unleash your creativity through writing",
        "craft engaging stories and narratives",
        "explore the art of storytelling",
        "create fictional worlds and characters",
        "get inspired to write your own tales",
        "explore the power of narrative creation",
        "turn your imagination into captivating stories",
        "bring your ideas to life through writing",
        "get suggestions for writing stories",
        "explore a world of creative writing",
        "discover your storytelling potential",
        "find techniques to write compelling narratives",
        "get recommendations for crafting stories",
        "explore a diverse range of writing styles",
        "transform your thoughts into written stories",
        "develop characters and plotlines through writing",
        "explore the realm of fiction writing",
        "embark on literary journeys through storytelling",
        "find unique outlets for creative expression",
        "explore the world of authorship and storytelling"
    ],
    "Recipes": [
        "explore a world of culinary delights",
        "discover dishes to satisfy your cravings",
        "find recipes for every taste and occasion",
        "create delicious meals with step-by-step guides",
        "get inspired to cook with diverse recipe ideas",
        "explore a variety of cooking techniques",
        "discover flavorful recipes from around the globe",
        "find inspiration for home-cooked meals",
        "cook up a storm with a collection of recipes",
        "explore a range of recipes for all skill levels",
        "discover unique and mouthwatering recipe options",
        "access a treasure trove of culinary inspiration",
        "find recipes tailored to your dietary preferences",
        "try your hand at cooking with expert recipes",
        "explore a world of flavors through recipe exploration",
        "discover creative cooking ideas and recipes",
        "unleash your inner chef with a library of recipes",
        "get recommendations for delicious homemade dishes",
        "explore diverse cuisines and cooking methods",
        "find recipes that bring joy to your kitchen",
        "unlock your culinary potential with enticing recipes",
        "explore a plethora of recipes to enhance your meals",
        "discover recipes that make cooking an enjoyable adventure",
        "find culinary inspiration for breakfast, lunch, and dinner",
        "explore a wide array of dishes to tantalize your taste buds",
        "uncover the art of cooking with a wealth of recipe options",
        "find recipes that turn ordinary ingredients into extraordinary dishes",
        "explore the world of gastronomy with an extensive recipe collection",
        "discover recipes that cater to various dietary preferences",
        "ignite your passion for cooking with innovative recipe ideas",
        "explore a vast range of culinary creations to elevate your meals"
    ],
    "Music": [
        "immerse yourself in the world of melodies",
        "discover diverse music genres and artists",
        "find tunes that resonate with your soul",
        "explore a vast musical universe",
        "get lost in the rhythm and melodies",
        "discover new tracks and old favorites",
        "find the soundtrack to your daily life",
        "explore a variety of musical moods",
        "unleash your emotions through harmonious notes",
        "discover genres that match your mood",
        "embark on a sonic journey with your favorite artists",
        "explore melodies that stir your imagination",
        "find the beats that make your heart dance",
        "uncover hidden musical gems and popular hits",
        "explore a world of musical creativity and expression",
        "immerse yourself in soundscapes that inspire",
        "find music that uplifts and soothes",
        "discover rhythms that energize your day",
        "explore a wide spectrum of musical compositions",
        "tune in to melodies that reflect your emotions",
        "find songs that tell stories and capture moments",
        "explore musical diversity and cultural richness",
        "uncover the artistry and passion behind every note",
        "find melodies that ignite your imagination",
        "immerse yourself in the art of musical storytelling",
        "explore the global tapestry of sounds and melodies",
        "discover harmonies that connect hearts and minds",
        "find melodies that become the soundtrack to your memories",
        "explore the magic of music and its ability to transcend boundaries",
        "unleash your inner musician and explore a symphony of sounds",
        "discover music that becomes the backdrop to your life's moments"
    ],
    "News": [
        "stay informed with the latest headlines",
        "get updates on current events worldwide",
        "discover breaking news and insightful stories",
        "explore diverse perspectives on global matters",
        "get a comprehensive view of current affairs",
        "stay up-to-date with the latest news developments",
        "discover in-depth coverage of global happenings",
        "access reliable sources for accurate news updates",
        "get recommendations for staying informed",
        "explore a world of news stories and analyses",
        "discover news that shapes your understanding of the world",
        "stay connected to current events with trustworthy sources",
        "access a variety of news topics and discussions",
        "get insights into local, national, and international news",
        "explore a wide range of news sources for well-rounded information",
        "discover news that sparks conversations and debates",
        "stay engaged with news stories that matter to you",
        "access real-time news updates and insightful articles",
        "get a comprehensive overview of news trends",
        "explore news that deepens your knowledge of global dynamics",
        "discover reliable sources for staying informed",
        "stay up-to-date with the latest news stories and analyses",
        "access a variety of news perspectives to broaden your horizons",
        "get insights into political, social, and economic news",
        "explore a world of news that informs and educates",
        "discover news that empowers you to make informed decisions",
        "stay engaged with news stories that inspire action",
        "access real-time updates on significant news events",
        "get a comprehensive understanding of news topics and discussions",
        "explore news that reflects the diverse fabric of society"
    ],
    "Sports": [
        "immerse yourself in the world of athletic feats",
        "get updates on sporting events and competitions",
        "discover thrilling moments from the sports arena",
        "explore diverse athletic disciplines and achievements",
        "get a front-row seat to the excitement of sports",
        "stay up-to-date with the latest sports highlights",
        "discover insights into athletes' dedication and passion",
        "access reliable sources for comprehensive sports coverage",
        "get recommendations for staying connected to sports",
        "explore a world of athletic prowess and competitions",
        "discover sports that inspire and captivate",
        "stay connected to sports stories that celebrate excellence",
        "access a variety of sports disciplines and tournaments",
        "get insights into players' strategies and achievements",
        "explore a wide range of sports events and match analyses",
        "discover the power of sports to unite and inspire",
        "stay engaged with sports stories that evoke emotions",
        "access real-time updates on thrilling sports moments",
        "get a comprehensive overview of sporting triumphs",
        "explore sports that transcend boundaries and cultures",
        "discover reliable sources for staying informed about sports",
        "stay up-to-date with the latest sports news and analyses",
        "access a variety of sports perspectives to enrich your experience",
        "get insights into athletes' dedication and perseverance",
        "explore a world of sportsmanship and competitive spirit",
        "discover sports that teach lessons in teamwork and discipline",
        "stay engaged with sports stories that motivate and uplift",
        "access real-time updates on pivotal sports events",
        "get a comprehensive understanding of sports dynamics",
        "explore sports that showcase human potential and achievement",
        "discover the thrill of victory and the lessons of defeat"
    ],
    "PlantWiki": [
        "explore the botanical world through information",
        "discover a library of plant-related knowledge",
        "find insights into the world of flora",
        "explore a digital garden of plant information",
        "get to know plants and their unique features",
        "discover a wealth of botanical wisdom",
        "explore plant diversity through comprehensive data",
        "learn about plants and their natural habitats",
        "get recommendations for exploring plant life",
        "explore a world of plant species and classifications",
        "discover plant-related facts, care tips, and more",
        "uncover the wonders of plants and their adaptations",
        "explore plant-related resources for enthusiasts",
        "learn about the role of plants in ecosystems",
        "get insights into gardening, horticulture, and botany",
        "explore plant taxonomy and identification",
        "discover plant life and its ecological significance",
        "uncover the mysteries of plant biology and growth",
        "explore a world of plant-based knowledge",
        "learn about plants that beautify and nourish our planet",
        "discover plant-related articles, photos, and research",
        "explore the interconnectedness of plants and the environment",
        "get acquainted with the fascinating world of flora",
        "discover plant species that thrive in various climates",
        "explore plant adaptations and survival strategies",
        "learn about traditional and modern uses of plants",
        "discover plant lore, folklore, and cultural significance",
        "explore the fascinating world of plant evolution",
        "get insights into plant care and propagation methods",
        "explore plant-related topics that inspire curiosity"
    ], 
    "AnimalWiki": [
        "explore the animal kingdom with informative articles",
        "discover a trove of knowledge about various species",
        "get insights into diverse creatures and habitats",
        "explore a digital encyclopedia of animal information",
        "learn about animals and their unique behaviors",
        "discover a wealth of zoological wisdom and facts",
        "explore animal diversity through comprehensive data",
        "uncover the mysteries of animal biology and adaptation",
        "get recommendations for understanding wildlife",
        "explore a world of species, classifications, and more",
        "discover articles about animals' roles in ecosystems",
        "unveil the wonders of animal behavior and interaction",
        "explore a world of fauna and their ecological importance",
        "learn about the remarkable diversity of animal life",
        "discover animal-related articles, photos, and research",
        "explore the fascinating world of animal adaptations",
        "get insights into wildlife conservation and protection",
        "explore a digital menagerie of animal facts and features",
        "uncover the beauty and variety of Earth's creatures",
        "discover animal species that inhabit different habitats",
        "explore the interconnectedness of animals and nature",
        "learn about animals that captivate our imaginations",
        "discover fascinating animal adaptations and behaviors",
        "explore the intricate web of animal interactions",
        "get acquainted with the wonders of the animal realm",
        "discover the importance of animals in our ecosystems",
        "explore animal-related topics that inspire curiosity",
        "unleash your fascination with the diversity of life forms",
        "explore a world of creatures that share our planet",
        "learn about animals that bring vitality to our world",
        "explore the intriguing world of animal biology and behavior"
    ],
    "Location": [
        "explore geographical landscapes and regions",
        "discover the beauty of different places worldwide",
        "get insights into diverse cultures and environments",
        "explore a digital atlas of global locations",
        "learn about geographical features and landmarks",
        "discover a wealth of geographic information",
        "explore the world's diversity through comprehensive data",
        "uncover the wonders of Earth's physical geography",
        "get recommendations for understanding global locations",
        "explore a world of countries, cities, and continents",
        "discover articles about places' historical significance",
        "unveil the beauty and uniqueness of landscapes",
        "explore a world of locations and their cultural heritage",
        "learn about the remarkable diversity of the planet",
        "discover location-related articles, photos, and research",
        "explore the intricate interplay of landforms and ecosystems",
        "get insights into geography's impact on human societies",
        "explore a digital compendium of geographical facts",
        "uncover the majesty and variety of Earth's terrains",
        "discover regions that exhibit different climates",
        "explore the interconnectivity of locations and nature",
        "learn about places that shape our understanding of the world",
        "discover the geographic diversity of Earth's environments",
        "explore location-related topics that inspire curiosity",
        "unleash your fascination with the tapestry of global landscapes",
        "explore a world of places that hold significance",
        "learn about the intricate relationship between people and land",
        "explore the intriguing world of Earth's geographical diversity"
    ],
    "Translator": [
        "bridge language barriers with seamless translations",
        "discover a tool for multilingual communication",
        "get accurate and instant language translations",
        "explore a digital translator for global conversations",
        "facilitate understanding across languages and cultures",
        "discover a reliable tool for language interpretation",
        "get recommendations for overcoming linguistic obstacles",
        "explore a world of language translations and fluency",
        "uncover the power of communication without boundaries",
        "translate texts and conversations with ease",
        "discover a tool that fosters global linguistic connections",
        "unveil the magic of multilingual interaction",
        "explore a world of language diversity and interpretation",
        "learn about the art of bridging language gaps",
        "discover translation-related tools, techniques, and insights",
        "explore the intersection of languages and understanding",
        "get insights into cultural exchange through translation",
        "explore a digital resource for language comprehension",
        "uncover the beauty of multilingual expression",
        "translate words and phrases with precision",
        "explore the interplay of languages in global conversations",
        "learn about the significance of accurate translations",
        "discover the transformative potential of multilingualism",
        "explore translation-related topics that inspire curiosity",
        "unleash your passion for breaking language barriers",
        "explore a world of linguistic connections and meanings",
        "learn about the dynamic role of translators in communication",
        "explore the art and science of language interpretation"
    ],
    "Code": [
        "immerse yourself in the world of programming",
        "explore coding languages and software development",
        "get insights into the art of writing code",
        "discover the world of algorithms and logic",
        "explore a digital realm of coding challenges",
        "uncover the secrets of efficient software design",
        "explore coding diversity through comprehensive data",
        "learn about the power of problem-solving through code",
        "get recommendations for mastering programming",
        "explore a world of coding languages and frameworks",
        "discover articles about innovative software solutions",
        "unveil the creativity and complexity of coding",
        "explore a world of code and its real-world applications",
        "learn about the intersection of technology and logic",
        "discover code-related articles, tutorials, and research",
        "explore the art of transforming ideas into functioning code",
        "get insights into coding's role in shaping digital landscapes",
        "explore a digital space for coding enthusiasts",
        "uncover the intricacies of building digital systems",
        "discover coding principles that drive technological progress",
        "explore the dynamics of coding languages and syntax",
        "learn about coding's influence on modern innovation",
        "discover the power of turning concepts into code",
        "explore code-related topics that inspire curiosity",
        "unleash your potential for coding creativity and innovation",
        "explore a world of algorithms and computational thinking",
        "learn about the journey of creating functional software",
        "explore the fascinating world of code and its impact",
        "explore the exciting world of programming and coding"
    ],
    "DataBase": [
        "navigate vast data repositories with ease",
        "explore digital warehouses of information",
        "get insights from structured data collections",
        "discover a tool for managing and querying data",
        "explore a world of organized data and insights",
        "uncover the secrets of data storage and retrieval",
        "explore data diversity through comprehensive resources",
        "learn about the power of data-driven decision-making",
        "get recommendations for effective data management",
        "explore a world of databases and their applications",
        "discover articles about database design and optimization",
        "unveil the intricacies of data organization and storage",
        "explore a world of structured data and its significance",
        "learn about the importance of data accuracy and integrity",
        "discover database-related articles, tutorials, and research",
        "explore the art of extracting insights from data",
        "get insights into data's role in shaping industries",
        "explore a digital space for database enthusiasts",
        "uncover the complexities of data manipulation and analysis",
        "discover database principles that drive informed decisions",
        "explore the dynamics of data models and relationships",
        "learn about data's influence on business innovation",
        "discover the power of leveraging data for insights",
        "explore database-related topics that inspire curiosity",
        "unleash your potential for transforming data into knowledge",
        "explore a world of data analysis and visualization",
        "learn about the journey of managing and utilizing data",
        "explore the fascinating world of data and its applications"
    ],
    "Youtube": [
        "immerse yourself in the world of video content",
        "discover a platform for visual storytelling",
        "get insights from a diverse range of videos",
        "explore a digital stage for creators and content",
        "uncover the beauty of video-based communication",
        "discover a variety of channels and video genres",
        "get recommendations for entertainment and education",
        "explore a world of visual content and storytelling",
        "learn about the power of video engagement and expression",
        "get entertained and informed through videos",
        "discover channels that resonate with your interests",
        "unveil the magic of video creation and consumption",
        "explore a world of video sharing and interaction",
        "learn about the art of crafting compelling videos",
        "discover video-related articles, tutorials, and insights",
        "explore the dynamics of video production and storytelling",
        "get insights into video's impact on communication",
        "explore a digital space for video enthusiasts",
        "uncover the intricacies of video editing and production",
        "discover channels that foster connection and community",
        "explore the world of video as a medium for expression",
        "learn about the role of videos in shaping online culture",
        "discover the power of visual communication and creativity",
        "explore video-related topics that inspire curiosity",
        "unleash your passion for creating and consuming video content",
        "explore a world of video diversity and artistic expression",
        "learn about the journey of crafting engaging videos",
        "explore the dynamic and captivating world of video content"
    ],
    "Document": [
        "immerse yourself in the world of written content",
        "discover a tool for textual expression and communication",
        "get insights from a diverse range of documents",
        "explore a digital realm of written knowledge",
        "uncover the beauty of literary and informative writing",
        "discover a variety of document types and formats",
        "get recommendations for education and research",
        "explore a world of textual content and storytelling",
        "learn about the power of written communication and analysis",
        "get informed and inspired through written texts",
        "discover documents that resonate with your interests",
        "unveil the magic of document creation and exploration",
        "explore a world of written expression and information",
        "learn about the art of crafting informative documents",
        "discover document-related articles, essays, and insights",
        "explore the dynamics of written composition and narrative",
        "get insights into documents' impact on knowledge sharing",
        "explore a digital space for textual content enthusiasts",
        "uncover the intricacies of editing and refining documents",
        "discover materials that foster learning and reflection",
        "explore the world of documents as vessels of ideas",
        "learn about the role of written content in communication",
        "discover the power of textual expression and creativity",
        "explore document-related topics that inspire curiosity",
        "unleash your passion for crafting and reading written content",
        "explore a world of textual diversity and literary exploration",
        "learn about the journey of creating compelling documents",
        "explore the captivating and informative world of written texts"
    ]

}

if METHOD_B:
    intention_desc = {
        "Calculator":{
            "mix": [
                "Use a calculator",
                "Solve a mathematical expression",
                "Perform arithmetic calculations",
                "Need to use an arithmetic software",
                "Calculate numerical values",
                "Perform calculations with precision",
                "Utilize a calculator tool",
                "Perform math operations",
                "Use a mathematical calculator",
                "Perform numerical computations",
                "Employ a digital calculator",
                "Engage a mathematical tool",
                "Work with an arithmetic calculator",
                "Perform complex calculations",
                "Employ a digital arithmetic tool",
                "Solve math problems",
                "Use a math calculator",
                "Conduct arithmetic computations",
                "Perform mathematical analyses",
                "Engage in numerical calculations",
                "Work with a mathematical software",
                "Perform precise calculations",
                "Utilize a digital arithmetic tool",
                "Engage in mathematical calculations",
                "Use an arithmetic solver",
                "Conduct precise arithmetic computations",
                "Perform mathematical operations",
                "Employ an arithmetic computation tool",
                "Work with a math software",
                "Utilize a digital calculator",
                "Engage in arithmetic problem solving",
                "Use a digital math tool",
                "Conduct calculations with accuracy",
                "Perform math computations",
                "Employ a numerical calculator",
                "Work with a mathematical computation tool",
                "Utilize a math solver",
                "Engage in mathematical problem solving",
                "Use an arithmetic calculation tool",
                "Conduct numerical computations",
                "Perform arithmetic analyses",
                "Employ a calculator for math",
                "Work with a mathematical solver",
                "Utilize a numerical calculator",
                "Engage in math problem solving",
                "Use a calculator for arithmetic",
                "Conduct calculations with precision",
                "Perform math analyses",
                "Employ a numerical computation tool",
                "Work with an arithmetic solver",
                "Utilize a mathematical solver",
                "Engage in numerical problem solving",
                "Use a math computation tool",
                "Conduct mathematical calculations",
                "Perform calculations accurately",
                "Employ an arithmetic tool",
                "Work with a numerical calculator",
                "Utilize a mathematical calculation tool",
                "Engage in math calculations",
                "Use a numerical computation tool",
                "Conduct mathematical computations",
                "Perform precise mathematical calculations",
                "Employ a calculator for arithmetic operations",
                "Work with a math solver",
                "Utilize an arithmetic computation tool",
                "Engage in numerical calculations",
                "Use a calculator for math operations",
                "Conduct math computations with precision",
                "Perform accurate mathematical calculations",
                "Employ a digital arithmetic solver",
                "Work with a mathematical computation tool",
                "Utilize a calculator for math problems",
                "Engage in arithmetic calculations",
                "Use a digital math solver",
                "Conduct math analyses with precision",
                "Perform mathematical computations accurately",
                "Employ a calculator for numerical calculations",
                "Work with an arithmetic calculation tool",
                "Utilize a mathematical computation tool",
                "Engage in precise math calculations",
                "Use a calculator for precise calculations",
                "Conduct numerical problem solving",
                "Perform arithmetic operations accurately",
                "Employ a mathematical computation tool",
                "Work with a numerical solver",
                "Utilize a calculator for math analyses",
                "Engage in accurate mathematical calculations",
                "Use an arithmetic calculation tool",
                "Conduct math calculations with accuracy",
                "Perform precise numerical computations",
                "Employ a calculator for arithmetic problem solving",
                "Work with a mathematical solver",
                "Utilize a numerical computation tool",
                "Engage in accurate math calculations",
                "Use a calculator for numerical computations",
                "Conduct mathematical problem solving",
                "Perform math operations accurately",
                "Employ a digital calculator tool",
                "Work with a mathematical calculation tool",
                "Utilize a calculator for math computations",
                "Engage in accurate numerical calculations",
                "Use an arithmetic computation tool",
                "Conduct mathematical analyses with precision",
                "Perform precise math calculations",
                "Employ a numerical solver",
                "Work with a math computation tool",
                "Utilize a calculator for precise mathematical calculations",
                "Engage in accurate arithmetic calculations",
                "Use a digital math computation tool",
                "Conduct accurate mathematical computations",
                "Perform numerical computations accurately",
                "Employ a calculator for numerical analyses",
                "Work with a mathematical calculation tool",
                "Utilize a calculator for precise math calculations",
                "Engage in accurate numerical problem solving",
                "Use an arithmetic solver",
                "Conduct mathematical operations with precision",
                "Perform accurate math analyses",
                "Employ a numerical computation tool",
                "Work with a calculator for accurate mathematical calculations",
                "Utilize a mathematical solver",
                "Engage in numerical calculations with accuracy",
                "Use a calculator for accurate math computations",
                "Conduct precise mathematical analyses",
                "Perform accurate numerical computations",
                "Employ a mathematical computation tool",
                "Work with a calculator for numerical problem solving",
                "Utilize an arithmetic calculation tool",
                "Engage in precise mathematical operations",
                "Use a numerical solver",
                "Conduct mathematical computations with accuracy",
                "Perform accurate arithmetic analyses",
                "Employ a calculator for accurate numerical calculations",
                "Work with a mathematical solver",
                "Utilize a calculator for mathematical problem solving",
                "Engage in accurate mathematical analyses",
                "Use a digital computation tool",
                "Conduct precise numerical calculations",
                "Perform mathematical computations with precision",
                "Employ a calculator for accurate math analyses",
                "Work with a mathematical computation tool",
                "Utilize a calculator for numerical calculations with accuracy",
                "Engage in precise arithmetic problem solving",
                "Use a digital calculator for precise mathematical calculations",
                "Conduct accurate math operations",
                "Perform numerical computations with precision",
                "Employ a calculator for precise arithmetic analyses",
                "Work with a numerical computation tool",
                "Utilize a mathematical solver",
                "Engage in accurate mathematical problem solving",
                "Use a calculator for accurate numerical analyses",
                "Conduct precise math analyses",
                "Perform accurate math computations",
                "Employ a calculator for precise numerical calculations",
                "Work with a mathematical calculation tool",
                "Utilize a calculator for precise mathematical analyses",
                "Engage in accurate arithmetic calculations",
                "Use a digital math solver",
                "Conduct accurate numerical operations",
                "Perform precise mathematical computations",
                "Employ a calculator for accurate math problem solving",
                "Work with a calculator for mathematical computations",
                "Utilize a numerical solver",
                "Engage in accurate numerical calculations",
                "Use a calculator for accurate mathematical analyses",
                "Conduct precise math operations",
                "Perform accurate arithmetic computations",
                "Employ a calculator for precise mathematical operations",
                "Work with a mathematical solver",
                "Utilize a calculator for accurate numerical analyses",
                "Engage in precise mathematical computations",
                "Use a digital computation tool",
                "Conduct accurate math calculations",
                "Perform precise numerical analyses",
                "Employ a calculator for accurate arithmetic computations",
                "Work with a calculator for precise math operations",
                "Utilize a mathematical calculation tool",
                "Engage in accurate numerical problem solving",
                "Use a calculator for accurate mathematical calculations",
                "Conduct precise arithmetic analyses",

                # ARGS
                "I need to calculate [ARGS]",
                "what is the result of [ARGS]?",
                "compute [ARGS] for me",
                "can you give me the answer for [ARGS]?",
                "I'm trying to figure out [ARGS]",
                "what does [ARGS] equal?",
                "help me solve [ARGS]",
                "I'm curious about [ARGS]",
                "I want to know the value of [ARGS]",
                "what's the solution to [ARGS]?",
                "what's the outcome of [ARGS]?",
                "could you assist with [ARGS]?",
                "I'm struggling with [ARGS]",
                "I'd like to understand [ARGS]",
                "could you calculate [ARGS]?",
                "what's the value of [ARGS]?",
                "I need the result for [ARGS]",
                "please provide the calculation for [ARGS]",
                "I'm not sure about [ARGS]",
                "calculate [ARGS] for me",
                "I need to work out [ARGS]",
                "please help me solve [ARGS]",
                "what's the answer for [ARGS]?",
                "I'm having difficulty with [ARGS]",
                "can you do the math for [ARGS]?",
                "I want to find out [ARGS]",
                "what's [ARGS] equal to?",
                "please compute [ARGS]",
                "I'm stuck on [ARGS]",
                "can you help me with [ARGS]?",
                "I'm interested in [ARGS]",
                "please provide the solution for [ARGS]",
                "I'd like to calculate [ARGS]",
                "what's the numerical value of [ARGS]?",
                "please assist with [ARGS]",
                "I'm facing a challenge with [ARGS]",
                "can you compute [ARGS] for me?",
                "I want to determine [ARGS]",
                "what's the numeric result of [ARGS]?",
                "please calculate [ARGS]",
                "I'm unsure about the value of [ARGS]",
                "can you help me understand [ARGS]?",
                "I'm requesting a calculation for [ARGS]",
                "what's the outcome of the calculation [ARGS]?",
                "I'm encountering difficulty with [ARGS]",
                "could you assist me with [ARGS]?",
                "I'd appreciate your help with [ARGS]",
                "what's the solution to the calculation [ARGS]?",
                "please provide guidance on [ARGS]",
                "I need to calculate [ARGS]",
                "what is the result of [ARGS]?",
                "compute [ARGS] for me",
                "can you give me the answer for [ARGS]?",
                "I'm trying to figure out [ARGS]",
                "what does [ARGS] equal?",
                "help me solve [ARGS]",
                "I'm curious about [ARGS]",
                "I want to know the value of [ARGS]",
                "what's the solution to [ARGS]?",
                "what's the outcome of [ARGS]?",
                "could you assist with [ARGS]?",
                "I'm struggling with [ARGS]",
                "I'd like to understand [ARGS]",
                "could you calculate [ARGS]?",
                "what's the value of [ARGS]?",
                "I need the result for [ARGS]",
                "please provide the calculation for [ARGS]",
                "I'm not sure about [ARGS]",
                "calculate [ARGS] for me",
                "I need to work out [ARGS]",
                "please help me solve [ARGS]",
                "what's the answer for [ARGS]?",
                "I'm having difficulty with [ARGS]",
                "can you do the math for [ARGS]?",
                "I want to find out [ARGS]",
                "what's [ARGS] equal to?",
                "please compute [ARGS]",
                "I'm stuck on [ARGS]",
                "can you help me with [ARGS]?",
                "I'm interested in [ARGS]",
                "please provide the solution for [ARGS]",
                "I'd like to calculate [ARGS]",
                "what's the numerical value of [ARGS]?",
                "please assist with [ARGS]",
                "I'm facing a challenge with [ARGS]",
                "can you compute [ARGS] for me?",
                "I want to determine [ARGS]",
                "what's the numeric result of [ARGS]?",
                "please calculate [ARGS]",
                "I'm unsure about the value of [ARGS]",
                "can you help me understand [ARGS]?",
                "I'm requesting a calculation for [ARGS]",
                "what's the outcome of the calculation [ARGS]?",
                "I'm encountering difficulty with [ARGS]",
                "could you assist me with [ARGS]?",
                "I'd appreciate your help with [ARGS]",
                "what's the solution to the calculation [ARGS]?",
                "please provide guidance on [ARGS]",
                "I'm attempting to solve [ARGS]",
                "can you find the answer for [ARGS]?",
                "I'm puzzled by [ARGS]",
                "what's the calculated result of [ARGS]?",
                "I need your expertise on [ARGS]",
                "what does the calculation [ARGS] lead to?",
                "please help me compute [ARGS]",
                "I'm seeking the value of [ARGS]",
                "can you guide me through [ARGS]?",
                "I'm uncertain about the outcome of [ARGS]",
                "what's the solution for [ARGS]?",
                "please assist me in evaluating [ARGS]",
                "I'm working on [ARGS] and need help",
                "what will [ARGS] give me?",
                "I'm looking for the result of [ARGS]",
                "could you explain [ARGS] to me?",
                "I'm having trouble with the calculation [ARGS]",
                "what's the output of [ARGS]?",
                "please provide the answer to [ARGS]",
                "I'm trying to compute [ARGS]",
                "what's the resolution for [ARGS]?",
                "I'm in need of [ARGS] calculation",
                "can you help me decipher [ARGS]?",
                "I'm seeking the calculated value of [ARGS]",
                "what can you tell me about [ARGS]?",
                "I'm grappling with [ARGS]",
                "what's the solution to [ARGS]?",
                "I'm interested in the calculation [ARGS]",
                "what's the answer to the problem [ARGS]?",
                "please guide me in calculating [ARGS]",
                "I'm unsure how to calculate [ARGS]",
                "what's the outcome when I input [ARGS]?",
                "I need to know the solution for [ARGS]",
                "what's the result if I use [ARGS]?",
                "I'm attempting to determine [ARGS]",
                "what does [ARGS] result in?",
                "I'd like your assistance with [ARGS]",
                "what's the computed value of [ARGS]?",
                "please help me with the calculation [ARGS]",
                "I'm puzzled about [ARGS]",
                "what's the solution for the input [ARGS]?",
                "I need help in finding the value of [ARGS]",
                "what can you tell me about the calculation [ARGS]?",
                "I'm grappling with understanding [ARGS]",
                "what will be the outcome of [ARGS]?",
                "I'm uncertain how to proceed with [ARGS]",
                "please provide guidance on calculating [ARGS]",
                "I'm trying to solve for [ARGS]",
                "what's the answer if I plug in [ARGS]?",
                "I'm in need of help with [ARGS]",
                "can you assist me in computing [ARGS]?",
                "I'm unsure about the result of [ARGS]",
                "what's the resolution of [ARGS]?",
                
                # Shorter ones
                "calculate [ARGS]",
                "solve [ARGS]",
                "compute [ARGS]",
                "what's [ARGS]?",
                "evaluate [ARGS]",
                "find [ARGS]",
                "help with [ARGS]",
                "explain [ARGS]",
                "what is [ARGS]?",
                "figure out [ARGS]",
                "what's [ARGS] equal to?",
                "assist with [ARGS]",
                "I need [ARGS]",
                "what's [ARGS]'s value?",
                "solve for [ARGS]",
                "I'm stuck on [ARGS]",
                "can you do [ARGS]?",
                "I'm curious about [ARGS]",
                "what does [ARGS] mean?",
                "compute [ARGS] for me",
                "can you calculate [ARGS]?",
                "please explain [ARGS]",
                "what's the result of [ARGS]?",
                "I'm trying to solve [ARGS]",
                "what's the answer for [ARGS]?",
                "I'm having trouble with [ARGS]",
                "help me understand [ARGS]",
                "what's [ARGS] solution?",
                "compute value of [ARGS]",
                "what's the outcome of [ARGS]?",
                "calculate [ARGS] for me",
                "please help with [ARGS]",
                "explain [ARGS] to me",
                "solve this: [ARGS]",
                "what's the value of [ARGS]?",
                "I'm not sure about [ARGS]",
                "find the solution for [ARGS]",
                "what does [ARGS] equal?",
                "solve [ARGS] for me",
                "I'm struggling with [ARGS]",
                "help me solve [ARGS]",
                "what's [ARGS]?",
                "compute the value of [ARGS]",
                "explain the calculation [ARGS]",
                "what's the solution for [ARGS]?",
                "what's [ARGS] result?",
                "calculate [ARGS]'s value",
                "I'm unsure about [ARGS]",
                "can you help with [ARGS]?",
                "what is the value of [ARGS]?",
                "compute [ARGS]'s result",
                "I need to find [ARGS]",
                "what's [ARGS] outcome?",
                "compute the answer for [ARGS]",
                "I'm having difficulty with [ARGS]",
                "solve this: [ARGS]",
                "what is [ARGS]'s result?",
                "calculate [ARGS] for me",
                "help me with [ARGS]",
                "explain the value of [ARGS]",
                "I'm curious about [ARGS]",
                "what does [ARGS] solve to?",
                "compute the solution for [ARGS]",
                "what's the value for [ARGS]?",
                "I'm trying to calculate [ARGS]",
                "what's the answer for [ARGS]?",
                "I'm puzzled by [ARGS]",
                "help me understand [ARGS]",
                "what's [ARGS] solution?",
                "compute value of [ARGS]",
                "I'm unsure about [ARGS]",
                "what's the outcome of [ARGS]?",
                "solve this equation: [ARGS]",
                "please explain [ARGS]",
                "I'm facing a challenge with [ARGS]",
                "calculate [ARGS]'s result",
                "what does [ARGS] evaluate to?",
                "help me solve [ARGS]",
                "what's [ARGS]?",
                "I need the value of [ARGS]",
                "compute the result for [ARGS]",
                "what's the solution for [ARGS]?",
                "calculate [ARGS]",
                "I'm having trouble with [ARGS]",
            ],
            "add": [
                "calculate the sum of [ARGS]",
                "what's the result when you add [ARGS]?",
                "compute the addition of [ARGS]",
                "please add [ARGS] for me",
                "I'm trying to figure out the sum of [ARGS]",
                "what does the addition of [ARGS] equal?",
                "help me add [ARGS]",
                "I'm curious about the total of [ARGS]",
                "what's the sum of [ARGS]?",
                "what's the outcome of adding [ARGS]?",
                "can you assist with adding [ARGS]?",
                "I'm struggling with adding [ARGS]",
                "I'd like to understand the result of [ARGS]",
                "calculate the sum for [ARGS]",
                "what's the value of adding [ARGS]?",
                "I need the result for [ARGS]",
                "please provide the addition for [ARGS]",
                "I'm not sure about the sum of [ARGS]",
                "calculate the addition of [ARGS] for me",
                "I need to work out the sum of [ARGS]",
                "please help me add [ARGS]",
                "what's the sum when you add [ARGS]?",
                "I'm having difficulty with adding [ARGS]",
                "can you compute the sum of [ARGS]?",
                "I want to find out the total of [ARGS]",
                "what's the sum of [ARGS] equal to?",
                "please compute the addition of [ARGS]",
                "I'm stuck on adding [ARGS]",
                "can you help me with the addition of [ARGS]?",
                "I'm interested in the sum of [ARGS]",
                "please provide the sum for [ARGS]",
                "I'd like to calculate the sum of [ARGS]",
                "what's the numerical value of adding [ARGS]?",
                "please assist with adding [ARGS]",
                "I'm facing a challenge with adding [ARGS]",
                "can you compute the sum for [ARGS]?",
                "I want to determine the sum of [ARGS]",
                "what's the numeric result of adding [ARGS]?",
                "please calculate the addition of [ARGS]",
                "I'm unsure about the sum of [ARGS]",
                "can you help me understand adding [ARGS]?",
                "I'm requesting an addition for [ARGS]",
                "what's the outcome of adding [ARGS]?",
                "I'm encountering difficulty with adding [ARGS]",
                "could you assist me with adding [ARGS]?",
                "I'd appreciate your help with adding [ARGS]",
                "what's the solution to the addition of [ARGS]?",
                "please provide guidance on adding [ARGS]",

                # Short
                "calculate [ARGS] total",
                "what's [ARGS] sum?",
                "compute [ARGS] addition",
                "help me with [ARGS]",
                "explain [ARGS] addition",
                "solve [ARGS] sum",
                "what's the total of [ARGS]?",
                "add [ARGS] for me",
                "find [ARGS] sum",
                "compute the sum for [ARGS]",
                "I'm trying to add [ARGS]",
                "assist with [ARGS] addition",
                "sum up [ARGS] for me",
                "please calculate [ARGS] sum",
                "I'm curious about [ARGS] total",
                "what does [ARGS] addition equal?",
                "help me sum [ARGS]",
                "I'm working with [ARGS]",
                "calculate the addition for [ARGS]",
                "I need the [ARGS] sum",
                "what's [ARGS] outcome?",
                "compute the total of [ARGS]",
                "sum [ARGS] numbers",
                "I'd like to understand [ARGS] addition",
                "what's [ARGS] summed?",
                "compute [ARGS] total",
                "please help with [ARGS]",
                "what's [ARGS] solution?",
                "sum [ARGS] values",
                "I'm interested in [ARGS] sum",
                "compute the sum of [ARGS]",
                "what's the value of [ARGS] addition?",
                "please assist [ARGS]",
                "find the sum for [ARGS]",
                "I'm dealing with [ARGS] sum",
                "calculate [ARGS] result",
                "what's [ARGS] added up?",
                "compute [ARGS] sum",
                "I need help with [ARGS]",
                "what's [ARGS] summed up?",
                "calculate [ARGS] total",
                "please explain [ARGS] addition",
                "what's [ARGS] value?",
                "compute the sum for [ARGS]",
                "sum up [ARGS] numbers",
                "what's [ARGS] total?",
                "help me calculate [ARGS]",
                "I'm puzzled by [ARGS]",
                "compute [ARGS] outcome",

                # General:
                "perform addition calculations",
                "use an addition tool",
                "solve addition problems",
                "do number summation",
                "calculate sums",
                "add numbers together",
                "apply addition operations",
                "use addition software",
                "compute numerical sums",
                "perform sum evaluations",
                "work with addition functions",
                "apply addition methods",
                "utilize an addition helper",
                "execute summation tasks",
                "solve number addition",
                "work with sum calculations",
                "apply numeric addition",
                "utilize sum calculators",
                "perform number summations",
                "execute addition operations",
                "solve mathematical sums",
                "use addition aids",
                "calculate total amounts",
                "perform number adding",
                "apply sum techniques",
                "utilize addition algorithms",
                "solve number-based additions",
                "execute numeric summation",
                "work with sum solutions",
                "apply addition strategies",
                "calculate sum results",
                "perform number totalings",
                "utilize addition methods",
                "solve number sum problems",
                "execute addition tasks",
                "use addition methodologies",
                "calculate sum outcomes",
                "perform numerical additions",
                "apply sum solutions",
                "utilize addition techniques",
                "solve addition challenges",
                "perform sum evaluations",
                "work with sum functions",
                "apply numeric summing",
                "utilize sum methodologies",
                "calculate numeric sums",
                "perform sum calculations",
                "use sum aids",
                "apply sum operations",
                "utilize addition strategies",
                "solve sum questions",
            ],

            "subtract": [
                "calculate the difference of [ARGS]",
                "what's the result when you subtract [ARGS]?",
                "compute the subtraction of [ARGS]",
                "please subtract [ARGS] for me",
                "I'm trying to figure out the difference of [ARGS]",
                "what does the subtraction of [ARGS] equal?",
                "help me subtract [ARGS]",
                "I'm curious about the result of subtracting [ARGS]",
                "what's the difference of [ARGS]?",
                "what's the outcome of subtracting [ARGS]?",
                "can you assist with subtracting [ARGS]?",
                "I'm struggling with subtracting [ARGS]",
                "I'd like to understand the result of subtracting [ARGS]",
                "calculate the difference for [ARGS]",
                "what's the value of subtracting [ARGS]?",
                "I need the result for [ARGS]",
                "please provide the subtraction for [ARGS]",
                "I'm not sure about the difference of [ARGS]",
                "calculate the subtraction of [ARGS] for me",
                "I need to work out the difference of [ARGS]",
                "please help me subtract [ARGS]",
                "what's the difference when you subtract [ARGS]?",
                "I'm having difficulty with subtracting [ARGS]",
                "can you compute the difference of [ARGS]?",
                "I want to find out the result of subtracting [ARGS]",
                "what's the subtraction of [ARGS] equal to?",
                "please compute the subtraction of [ARGS]",
                "I'm stuck on subtracting [ARGS]",
                "can you help me with the subtraction of [ARGS]?",
                "I'm interested in the difference of [ARGS]",
                "please provide the difference for [ARGS]",
                "I'd like to calculate the difference of [ARGS]",
                "what's the numerical value of subtracting [ARGS]?",
                "please assist with subtracting [ARGS]",
                "I'm facing a challenge with subtracting [ARGS]",
                "can you compute the difference of [ARGS]?",
                "I want to determine the difference of [ARGS]",
                "what's the numeric result of subtracting [ARGS]?",
                "please calculate the subtraction of [ARGS]",
                "I'm unsure about the difference of [ARGS]",
                "can you help me understand subtracting [ARGS]?",
                "I'm requesting a subtraction for [ARGS]",
                "what's the outcome of subtracting [ARGS]?",
                "I'm encountering difficulty with subtracting [ARGS]",
                "could you assist me with subtracting [ARGS]?",
                "I'd appreciate your help with subtracting [ARGS]",
                "what's the solution to the subtraction of [ARGS]?",
                "please provide guidance on subtracting [ARGS]",

                # Short
                "calculate [ARGS] difference",
                "what's [ARGS] result?",
                "compute [ARGS] subtraction",
                "help me with [ARGS]",
                "explain [ARGS] subtraction",
                "solve [ARGS] difference",
                "what's the difference of [ARGS]?",
                "subtract [ARGS] for me",
                "find [ARGS] result",
                "compute the difference for [ARGS]",
                "I'm trying to subtract [ARGS]",
                "assist with [ARGS] subtraction",
                "find [ARGS] difference",
                "please calculate [ARGS] difference",
                "I'm curious about [ARGS] result",
                "what does [ARGS] subtraction equal?",
                "help me find [ARGS]",
                "I'm working with [ARGS]",
                "calculate the subtraction for [ARGS]",
                "I need the [ARGS] difference",
                "what's [ARGS] outcome?",
                "compute the result of subtracting [ARGS]",
                "subtract [ARGS] numbers",
                "I'd like to understand [ARGS] subtraction",
                "what's [ARGS] subtracted?",
                "compute [ARGS] difference",
                "please help with [ARGS]",
                "what's [ARGS] solution?",
                "subtract [ARGS] values",
                "I'm interested in [ARGS] difference",
                "compute the difference of [ARGS]",
                "what's the value of [ARGS] subtraction?",
                "please assist [ARGS]",
                "find the difference for [ARGS]",
                "I'm dealing with [ARGS] subtraction",
                "calculate [ARGS] result",
                "what's [ARGS] taken away?",
                "compute [ARGS] result",
                "I need help with [ARGS]",
                "what's [ARGS] subtracted?",
                "calculate [ARGS] difference",
                "please explain [ARGS] subtraction",
                "what's [ARGS] value?",
                "compute the difference for [ARGS]",
                "find [ARGS] subtraction",
                "what's [ARGS] difference?",
                "help me calculate [ARGS]",
                "I'm puzzled by [ARGS]",
                "compute [ARGS] outcome",

                # General
                "perform subtraction calculations",
                "use a subtraction tool",
                "solve subtraction problems",
                "do number difference",
                "calculate differences",
                "subtract numbers",
                "apply subtraction operations",
                "use subtraction software",
                "compute numerical differences",
                "perform difference evaluations",
                "work with subtraction functions",
                "apply subtraction methods",
                "utilize a subtraction helper",
                "execute difference tasks",
                "solve number subtraction",
                "work with difference calculations",
                "apply numeric subtraction",
                "utilize difference calculators",
                "perform number differences",
                "execute subtraction operations",
                "solve mathematical differences",
                "use subtraction aids",
                "calculate difference amounts",
                "perform number subtracting",
                "apply difference techniques",
                "utilize subtraction algorithms",
                "solve number-based subtractions",
                "execute numeric difference",
                "work with difference solutions",
                "apply subtraction strategies",
                "calculate difference results",
                "perform number subtracting",
                "utilize subtraction methods",
                "solve number difference problems",
                "execute subtraction tasks",
                "use subtraction methodologies",
                "calculate difference outcomes",
                "perform numerical subtractions",
                "apply difference solutions",
                "utilize subtraction techniques",
                "solve subtraction challenges",
                "perform difference evaluations",
                "work with difference functions",
                "apply numeric subtracting",
                "utilize difference methodologies",
                "calculate numeric differences",
                "perform subtraction calculations",
                "use subtraction aids",
                "apply subtraction operations",
                "utilize subtraction strategies",
                "solve subtraction questions",
            ],

            "multiply": [
                "calculate the product of [ARGS]",
                "what's the result when you multiply [ARGS]?",
                "compute the multiplication of [ARGS]",
                "please multiply [ARGS] for me",
                "I'm trying to figure out the product of [ARGS]",
                "what does the multiplication of [ARGS] equal?",
                "help me multiply [ARGS]",
                "I'm curious about the result of multiplying [ARGS]",
                "what's the product of [ARGS]?",
                "what's the outcome of multiplying [ARGS]?",
                "can you assist with multiplying [ARGS]?",
                "I'm struggling with multiplying [ARGS]",
                "I'd like to understand the result of multiplying [ARGS]",
                "calculate the product for [ARGS]",
                "what's the value of multiplying [ARGS]?",
                "I need the result for [ARGS]",
                "please provide the multiplication for [ARGS]",
                "I'm not sure about the product of [ARGS]",
                "calculate the multiplication of [ARGS] for me",
                "I need to work out the product of [ARGS]",
                "please help me multiply [ARGS]",
                "what's the product when you multiply [ARGS]?",
                "I'm having difficulty with multiplying [ARGS]",
                "can you compute the product of [ARGS]?",
                "I want to find out the result of multiplying [ARGS]",
                "what's the multiplication of [ARGS] equal to?",
                "please compute the multiplication of [ARGS]",
                "I'm stuck on multiplying [ARGS]",
                "can you help me with the multiplication of [ARGS]?",
                "I'm interested in the product of [ARGS]",
                "please provide the product for [ARGS]",
                "I'd like to calculate the product of [ARGS]",
                "what's the numerical value of multiplying [ARGS]?",
                "please assist with multiplying [ARGS]",
                "I'm facing a challenge with multiplying [ARGS]",
                "can you compute the product of [ARGS]?",
                "I want to determine the product of [ARGS]",
                "what's the numeric result of multiplying [ARGS]?",
                "please calculate the multiplication of [ARGS]",
                "I'm unsure about the product of [ARGS]",
                "can you help me understand multiplying [ARGS]?",
                "I'm requesting a multiplication for [ARGS]",
                "what's the outcome of multiplying [ARGS]?",
                "I'm encountering difficulty with multiplying [ARGS]",
                "could you assist me with multiplying [ARGS]?",
                "I'd appreciate your help with multiplying [ARGS]",
                "what's the solution to the multiplication of [ARGS]?",
                "please provide guidance on multiplying [ARGS]",

                # Short
                "calculate [ARGS] product",
                "what's [ARGS] result?",
                "compute [ARGS] multiplication",
                "help me with [ARGS]",
                "explain [ARGS] multiplication",
                "solve [ARGS] product",
                "what's the product of [ARGS]?",
                "multiply [ARGS] for me",
                "find [ARGS] result",
                "compute the product for [ARGS]",
                "I'm trying to multiply [ARGS]",
                "assist with [ARGS] multiplication",
                "find [ARGS] product",
                "please calculate [ARGS] product",
                "I'm curious about [ARGS] result",
                "what does [ARGS] multiplication equal?",
                "help me find [ARGS]",
                "I'm working with [ARGS]",
                "calculate the product for [ARGS]",
                "I need the [ARGS] product",
                "what's [ARGS] outcome?",
                "compute the result of multiplying [ARGS]",
                "multiply [ARGS] numbers",
                "I'd like to understand [ARGS] multiplication",
                "what's [ARGS] multiplied?",
                "compute [ARGS] product",
                "please help with [ARGS]",
                "what's [ARGS] solution?",
                "multiply [ARGS] values",
                "I'm interested in [ARGS] product",
                "compute the product of [ARGS]",
                "what's the value of [ARGS] multiplication?",
                "please assist [ARGS]",
                "find the product for [ARGS]",
                "I'm dealing with [ARGS] multiplication",
                "calculate [ARGS] result",
                "what's [ARGS] multiplied by?",
                "compute [ARGS] result",
                "I need help with [ARGS]",
                "what's [ARGS] multiplied?",
                "calculate [ARGS] product",
                "please explain [ARGS] multiplication",
                "what's [ARGS] value?",
                "compute the product for [ARGS]",
                "find [ARGS] multiplication",
                "what's [ARGS] product?",
                "help me calculate [ARGS]",
                "I'm puzzled by [ARGS]",
                "compute [ARGS] outcome",

                # General
                "perform multiplication calculations",
                "use a multiplication tool",
                "solve multiplication problems",
                "do number multiplication",
                "calculate products",
                "multiply numbers",
                "apply multiplication operations",
                "use multiplication software",
                "compute numerical products",
                "perform product evaluations",
                "work with multiplication functions",
                "apply multiplication methods",
                "utilize a multiplication helper",
                "execute product tasks",
                "solve number multiplication",
                "work with product calculations",
                "apply numeric multiplication",
                "utilize product calculators",
                "perform number products",
                "execute multiplication operations",
                "solve mathematical products",
                "use multiplication aids",
                "calculate product amounts",
                "perform number multiplying",
                "apply product techniques",
                "utilize multiplication algorithms",
                "solve number-based multiplications",
                "execute numeric product",
                "work with product solutions",
                "apply multiplication strategies",
                "calculate product results",
                "perform number multiplying",
                "utilize multiplication methods",
                "solve number product problems",
                "execute multiplication tasks",
                "use multiplication methodologies",
                "calculate product outcomes",
                "perform numerical multiplications",
                "apply product solutions",
                "utilize multiplication techniques",
                "solve multiplication challenges",
                "perform product evaluations",
                "work with product functions",
                "apply numeric multiplying",
                "utilize multiplication methodologies",
                "calculate numeric products",
                "perform multiplication calculations",
                "use multiplication aids",
                "apply multiplication operations",
                "utilize multiplication strategies",
                "solve multiplication questions",
            ],
            "divide": [
                "calculate the quotient of [ARGS]",
                "what's the result when you divide [ARGS]?",
                "compute the division of [ARGS]",
                "please divide [ARGS] for me",
                "I'm trying to figure out the quotient of [ARGS]",
                "what does the division of [ARGS] equal?",
                "help me divide [ARGS]",
                "I'm curious about the result of dividing [ARGS]",
                "what's the quotient of [ARGS]?",
                "what's the outcome of dividing [ARGS]?",
                "can you assist with dividing [ARGS]?",
                "I'm struggling with dividing [ARGS]",
                "I'd like to understand the result of dividing [ARGS]",
                "calculate the quotient for [ARGS]",
                "what's the value of dividing [ARGS]?",
                "I need the result for [ARGS]",
                "please provide the division for [ARGS]",
                "I'm not sure about the quotient of [ARGS]",
                "calculate the division of [ARGS] for me",
                "I need to work out the quotient of [ARGS]",
                "please help me divide [ARGS]",
                "what's the quotient when you divide [ARGS]?",
                "I'm having difficulty with dividing [ARGS]",
                "can you compute the quotient of [ARGS]?",
                "I want to find out the result of dividing [ARGS]",
                "what's the division of [ARGS] equal to?",
                "please compute the division of [ARGS]",
                "I'm stuck on dividing [ARGS]",
                "can you help me with the division of [ARGS]?",
                "I'm interested in the quotient of [ARGS]",
                "please provide the quotient for [ARGS]",
                "I'd like to calculate the quotient of [ARGS]",
                "what's the numerical value of dividing [ARGS]?",
                "please assist with dividing [ARGS]",
                "I'm facing a challenge with dividing [ARGS]",
                "can you compute the quotient of [ARGS]?",
                "I want to determine the quotient of [ARGS]",
                "what's the numeric result of dividing [ARGS]?",
                "please calculate the division of [ARGS]",
                "I'm unsure about the quotient of [ARGS]",
                "can you help me understand dividing [ARGS]?",
                "I'm requesting a division for [ARGS]",
                "what's the outcome of dividing [ARGS]?",
                "I'm encountering difficulty with dividing [ARGS]",
                "could you assist me with dividing [ARGS]?",
                "I'd appreciate your help with dividing [ARGS]",
                "what's the solution to the division of [ARGS]?",
                "please provide guidance on dividing [ARGS]",

                # Short
                "calculate [ARGS] quotient",
                "what's [ARGS] result?",
                "compute [ARGS] division",
                "help me with [ARGS]",
                "explain [ARGS] division",
                "solve [ARGS] quotient",
                "what's the quotient of [ARGS]?",
                "divide [ARGS] for me",
                "find [ARGS] result",
                "compute the quotient for [ARGS]",
                "I'm trying to divide [ARGS]",
                "assist with [ARGS] division",
                "find [ARGS] quotient",
                "please calculate [ARGS] quotient",
                "I'm curious about [ARGS] result",
                "what does [ARGS] division equal?",
                "help me find [ARGS]",
                "I'm working with [ARGS]",
                "calculate the quotient for [ARGS]",
                "I need the [ARGS] quotient",
                "what's [ARGS] outcome?",
                "compute the result of dividing [ARGS]",
                "divide [ARGS] numbers",
                "I'd like to understand [ARGS] division",
                "what's [ARGS] divided?",
                "compute [ARGS] quotient",
                "please help with [ARGS]",
                "what's [ARGS] solution?",
                "divide [ARGS] values",
                "I'm interested in [ARGS] quotient",
                "compute the quotient of [ARGS]",
                "what's the value of [ARGS] division?",
                "please assist [ARGS]",
                "find the quotient for [ARGS]",
                "I'm dealing with [ARGS] division",
                "calculate [ARGS] result",
                "what's [ARGS] divided by?",
                "compute [ARGS] result",
                "I need help with [ARGS]",
                "what's [ARGS] divided?",
                "calculate [ARGS] quotient",
                "please explain [ARGS] division",
                "what's [ARGS] value?",
                "compute the quotient for [ARGS]",
                "find [ARGS] division",
                "what's [ARGS] quotient?",
                "help me calculate [ARGS]",
                "I'm puzzled by [ARGS]",
                "compute [ARGS] outcome",

                # General
                "perform division calculations",
                "use a division tool",
                "solve division problems",
                "do number division",
                "calculate quotients",
                "divide numbers",
                "apply division operations",
                "use division software",
                "compute numerical quotients",
                "perform quotient evaluations",
                "work with division functions",
                "apply division methods",
                "utilize a division helper",
                "execute quotient tasks",
                "solve number division",
                "work with quotient calculations",
                "apply numeric division",
                "utilize quotient calculators",
                "perform number quotients",
                "execute division operations",
                "solve mathematical quotients",
                "use division aids",
                "calculate quotient amounts",
                "perform number dividing",
                "apply quotient techniques",
                "utilize division algorithms",
                "solve number-based divisions",
                "execute numeric quotient",
                "work with quotient solutions",
                "apply division strategies",
                "calculate quotient results",
                "perform number dividing",
                "utilize division methods",
                "solve number quotient problems",
                "execute division tasks",
                "use division methodologies",
                "calculate quotient outcomes",
                "perform numerical divisions",
                "apply quotient solutions",
                "utilize division techniques",
                "solve division challenges",
                "perform quotient evaluations",
                "work with quotient functions",
                "apply numeric dividing",
                "utilize division methodologies",
                "calculate numeric quotients",
                "perform division calculations",
                "use division aids",
                "apply division operations",
                "utilize division strategies",
                "solve division questions",
            ],
            "add_subtract": [],
            "mult_divide": [],
        },
        "WikiSearch": [
        # General descriptions
        "search the internet for information",
        "look up details in an encyclopedic source",
        "access a comprehensive knowledge base",
        "retrieve relevant information from the web",
        "gather data from a reputable source",
        "explore a vast collection of information",
        "conduct a search on a global information platform",
        "find information through online research",
        "access a digital repository of knowledge",
        "locate data using an online search tool",
        "retrieve facts and details using a digital resource",
        "navigate a virtual library of information",
        "explore a wide range of topics online",
        "gather knowledge through an online search",
        "search for data using an information retrieval system",
        "Conduct a thorough online search",
        "Access a reliable information source",
        "Navigate through digital references",
        "Explore details using online tools",
        "Retrieve data from the web",
        "Gather insights from internet resources",
        "Find comprehensive information online",
        "Discover facts through online research",
        "Access a virtual repository of knowledge",
        "Search for details using digital platforms",
        "Locate valuable information online",
        "Investigate topics with online tools",
        "Browse for insights on the internet",
        "Retrieve data from reputable sources",
        "Explore a wealth of information online",
        "Access a digital encyclopedia",
        "Search for knowledge using online sources",
        "Navigate through online databases",
        "Gather insights from digital platforms",
        "Find comprehensive details online",
        "Discover facts through internet research",
        "Access a wide range of information",
        "Search for insights using digital tools",
        "Locate valuable data online",
        "Investigate subjects with online resources",
        "Browse for comprehensive details",
        "Retrieve knowledge from digital sources",
        "Explore a vast online repository",
        "Access a digital library",
        "Search for facts using internet sources",
        "Navigate through online encyclopedias",
        "Gather insights from digital research",
        "Find comprehensive data online",
        "Discover information through online exploration",
        "Access a wealth of knowledge",
        "Search for insights through online platforms",
        "Locate valuable insights online",
        "Investigate topics using internet tools",
        "Browse for detailed information",
        "Retrieve data from online references",
        "Explore a diverse range of sources",
        "Access a digital compendium",
        "Search for knowledge through digital searches",
        "Navigate through online knowledge bases",
        "Gather insights from digital references",
        "Find comprehensive insights online",
        "Discover facts through online sources",
        "Access a comprehensive digital archive",
        "Search for insights in digital repositories",
        "Locate valuable knowledge online",
        "Investigate subjects using online databases",
        "Browse for detailed insights",
        "Retrieve data from online platforms",
        "Explore a digital reservoir of information",
        "Access a digital resource hub",
        "Search for facts through online research",
        "Navigate through online encyclopedias",
        "Gather insights from digital exploration",
        "Find comprehensive details online",
        "Discover knowledge through online sources",
        "Access a vast digital compendium",
        "Search for insights in digital libraries",
        "Locate valuable data online",
        "Investigate topics with online references",
        "Browse for detailed knowledge",
        "Retrieve information from online databases",
        "Explore a digital collection of information",
        "Access a digital knowledge repository",
        "Search for facts using online platforms",
        "Navigate through digital encyclopedias",
        "Gather insights from online research",
        "Find comprehensive information online",
        "Discover insights through online exploration",
        "Access a comprehensive digital database",
        "Search for knowledge using online research",
        "Locate valuable information online",
        "Investigate subjects through online resources",
        "Browse for detailed insights",
        "Retrieve data from digital sources",
        "Explore a digital hub of knowledge",
        "Access a digital information trove",
        "Search for facts using digital databases",
        "Navigate through online libraries",
        "Gather insights from online references",
        "Find comprehensive data through online searches",
        "Discover knowledge using digital platforms",
        "Access a diverse digital repository",
        "Search for insights using digital encyclopedias",
        "Locate valuable insights online",
        "Investigate topics with online exploration",
        "Browse for comprehensive knowledge",
        "Retrieve information from digital platforms",
        "Explore a digital collection of insights",
        "Access a digital resource center",
        "Search for facts through digital research",
        "Navigate through online knowledge bases",
        "Gather insights from digital databases",
        "Find comprehensive details through online sources",
        "Discover information using internet platforms",
        "Access a wide digital compendium",
        "Search for insights in online libraries",
        "Locate valuable data through online exploration",
        "Investigate subjects using digital references",
        "Browse for comprehensive information",
        "Retrieve knowledge from online databases",
        "Explore a digital repository of insights",
        # General questions
        "What is this topic?",
        "Could you provide an overview?",
        "Do you have information on this?",
        "Can you explain this to me?",
        "What's the story behind this?",
        "Tell me more about this.",
        "Can you provide insights?",
        "What are the details?",
        "I'm curious about this.",
        "Can you give some context?",
        "Could you shed light on this?",
        "I'd like to know more.",
        "What's the significance?",
        "Can you provide background?",
        "Could you elaborate on this?",
        "I'm interested in learning.",
        "Can you explain the basics?",
        "Do you have any insights?",
        "Could you help me understand?",
        "I'm looking for information.",
        "Tell me about this subject.",
        "What can you tell me?",
        "Can you provide details?",
        "What's the essence of this?",
        "Tell me something about this.",
        "Could you clarify this?",
        "I'd like to get more information.",
        "What's the importance of this?",
        "Can you describe this to me?",
        "Do you have any knowledge?",
        "Could you share some facts?",
        "I'm seeking more understanding.",
        "What's the main point?",
        "Can you give an explanation?",
        "Tell me the basics.",
        "What can you tell me about this?",
        "Can you provide an insight?",
        "What's the core of this?",
        "Tell me more about this.",
        "Could you offer some context?",
        "I'd like to know the details.",
        "What's the background?",
        "Can you provide an overview?",
        "What's the essence of this?",
        "Tell me something about it.",
        "Could you give more information?",
        "I'm curious to understand this.",
        "What's the significance of this?",
        "Can you shed light on this?",
        "Tell me more about this.",
        "Could you explain this further?",


        
        # Personalized descriptions
        "discover more about [ARGS]",
        "explore details regarding [ARGS]",
        "find information on [ARGS]",
        "access data related to [ARGS]",
        "locate details about [ARGS]",
        "gather insights about [ARGS]",
        "retrieve facts on [ARGS]",
        "explore [ARGS] through online research",
        "search for details about [ARGS] on the web",
        "navigate through information about [ARGS]",
        "conduct a search on [ARGS]",
        "look up [ARGS] using an online tool",
        "search for [ARGS] in the digital space",
        "gather knowledge about [ARGS] from reputable sources",
        "retrieve information on [ARGS] from the internet",
        "explore a wide range of details about [ARGS]",
        "access a comprehensive overview of [ARGS]",
        "find data related to [ARGS] on the web",
        "locate facts about [ARGS] using online resources",
        "search for information about [ARGS] in online databases",
        "discover insights into [ARGS] through online exploration",
        "explore the topic of [ARGS] using online sources",
        "retrieve relevant details about [ARGS] from the web",
        "navigate through information about [ARGS] using digital tools",
        "gather knowledge about [ARGS] from digital repositories",
        "explore a wide range of information about [ARGS] online",
        "search for data related to [ARGS] using online platforms",
        "retrieve information about [ARGS] from online encyclopedias",
        "explore details about [ARGS] using internet resources",
        "access information on [ARGS] from online references",
        "find comprehensive insights about [ARGS] through online searches",
        "locate detailed information about [ARGS] on the internet",
        "search for [ARGS] using digital information retrieval systems",
        "explore the virtual realm for knowledge about [ARGS]",
        "retrieve facts and details about [ARGS] from online sources",
        "gather information on [ARGS] from digital databases",
        "navigate through online information to learn about [ARGS]",
        "access a wide range of knowledge about [ARGS] using digital tools",
        "explore online sources to find information about [ARGS]",
        "search for [ARGS] using internet-based research methods",
        "retrieve information about [ARGS] from reputable online sources",
        "locate comprehensive insights about [ARGS] through online searches",
        "explore online platforms for data related to [ARGS]",
        "access information on [ARGS] from reliable digital sources",
        "find details about [ARGS] by searching online",
        "search for [ARGS] using digital reference materials",
        "explore online encyclopedias to gather information about [ARGS]",
        "retrieve facts and details about [ARGS] from online references",
        "navigate through digital information to understand [ARGS]",
        "explore online databases for comprehensive insights about [ARGS]",
        "search for [ARGS] using reputable online resources",
        "locate information about [ARGS] through online research",
        "access a virtual treasure trove of information about [ARGS]",
        "find knowledge about [ARGS] by browsing the internet",
        "search for [ARGS] using digital search engines",
        "explore online platforms for comprehensive details about [ARGS]",
        "retrieve facts about [ARGS] from online encyclopedias",
        "navigate through internet-based information to learn about [ARGS]",
        "gather insights into [ARGS] from reliable online sources",
        "access information about [ARGS] using online research methods",
        "explore online databases to find information about [ARGS]",
        "search for [ARGS] through digital reference materials",
        "retrieve details about [ARGS] from trustworthy online sources",
        "locate information about [ARGS] by searching online resources",
        "explore online sources for a comprehensive understanding of [ARGS]",
        "navigate through digital repositories to gather information about [ARGS]",
        "search for [ARGS] using internet resources",
        "access knowledge about [ARGS] from online databases",
        "find insights into [ARGS] through online exploration",
        "explore online platforms to retrieve information about [ARGS]",
        "search for details about [ARGS] using internet-based research",
        "retrieve comprehensive information about [ARGS] from digital sources",
        "navigate through digital references to find details about [ARGS]",
        "gather knowledge about [ARGS] by exploring online resources",
        "explore the digital realm to discover information about [ARGS]",
        "locate facts about [ARGS] through online research methods",
        "search for [ARGS] using reputable online references",
        "retrieve information about [ARGS] from online databases",
        "access detailed insights about [ARGS] through digital exploration",
        "find information on [ARGS] by navigating online sources",
        "explore online encyclopedias to gather data about [ARGS]",
        "search for [ARGS] using digital research methods",
        "retrieve details about [ARGS] from online platforms",
        "navigate through online references to understand [ARGS]",
        "access a wealth of information about [ARGS] from digital sources",
        "explore online databases for in-depth insights about [ARGS]",
        "search for [ARGS] through reliable online resources",
        "locate comprehensive details about [ARGS] using online platforms",
        "find information about [ARGS] by searching the digital realm",
        "explore online sources to access information about [ARGS]",
        "retrieve knowledge about [ARGS] from internet-based research",
        "navigate through digital databases to find information on [ARGS]",
        "gather insights about [ARGS] from online references",
        "explore online platforms for detailed data about [ARGS]",
        "search for [ARGS] using trustworthy online sources",
        "retrieve comprehensive insights about [ARGS] from online databases",
        "locate information about [ARGS] by exploring the digital space",
        "explore online sources to find information on [ARGS]",
        "search for details about [ARGS] using digital tools",
        "retrieve information about [ARGS] from online encyclopedias",
        "navigate through internet-based information to learn more about [ARGS]",
        "access a vast collection of knowledge about [ARGS] through digital resources",
        "explore online databases for comprehensive information about [ARGS]",
        "search for [ARGS] using reputable online platforms",
        "locate detailed insights about [ARGS] by exploring online sources",
        "find facts and details about [ARGS] through online research",
        "retrieve information about [ARGS] from trustworthy online references",
        "navigate through digital repositories to access information about [ARGS]",
        "gather knowledge about [ARGS] by searching online databases",

        # First person
        "I need more information on [ARGS]",
        "I want to know more about [ARGS]",
        "I'm curious to learn about [ARGS]",
        "I'm interested in finding out about [ARGS]",
        "I'd like to explore details regarding [ARGS]",
        "I'm looking to gather insights about [ARGS]",
        "I'd appreciate more facts on [ARGS]",
        "I want to explore [ARGS] through online research",
        "I'm searching for details about [ARGS] on the web",
        "I'm navigating through information about [ARGS]",
        "I'm conducting a search on [ARGS]",
        "I want to look up [ARGS] using an online tool",
        "I'm trying to search for [ARGS] in the digital space",
        "I'd like to gather knowledge about [ARGS] from reputable sources",
        "I'm trying to retrieve information on [ARGS] from the internet",
        "I'm exploring a wide range of details about [ARGS]",
        "I'd like to access a comprehensive overview of [ARGS]",
        "I'm attempting to find data related to [ARGS] on the web",
        "I'm looking to locate facts about [ARGS] using online resources",
        "I want to search for information about [ARGS] in online databases",
        "I'm trying to discover more about [ARGS]",
        "I want to explore details about [ARGS]",
        "I'm on a mission to find information on [ARGS]",
        "I'm eager to access data related to [ARGS]",
        "I'm searching for detailed insights into [ARGS]",
        "I'd like to retrieve facts on [ARGS]",
        "I'm interested in exploring [ARGS] through online research",
        "I'm actively searching for details about [ARGS] on the web",
        "I'm delving into information about [ARGS]",
        "I'm in the process of conducting a search on [ARGS]",
        "I'm looking to gather more insights about [ARGS]",
        "I'm aiming to find comprehensive information on [ARGS]",
        "I'm curious to locate data related to [ARGS] on the web",
        "I'm engaged in searching for facts about [ARGS] using online resources",
        "I'm focused on retrieving information about [ARGS] from the internet",
        "I'm determined to explore a wide range of details about [ARGS]",
        "I'm excited to access a comprehensive overview of [ARGS]",
        "I'm actively searching for data related to [ARGS] on the web",
        "I'm dedicated to locating facts about [ARGS] using online resources",
        "I'm on a quest to search for information about [ARGS] in online databases",
        "I'm looking to discover more about [ARGS] through online exploration",
        "I'm keen to retrieve information about [ARGS] from the internet",
        "I'm invested in exploring a wide range of information about [ARGS]",
        "I'm on a mission to find data related to [ARGS] using online platforms",
        "I'm actively seeking insights into [ARGS] through online research",
        "I'm in the process of accessing information on [ARGS] from online references",
        "I'm determined to find comprehensive insights about [ARGS] through online searches",
        "I'm eager to locate detailed information about [ARGS] on the internet",
        "I'm committed to searching for [ARGS] using digital information retrieval systems",
        "I'm excited to explore the virtual realm for knowledge about [ARGS]",
        "I'm dedicated to retrieving facts and details about [ARGS] from online sources",
        "I'm interested in gathering information on [ARGS] from digital databases",
        "I'm actively navigating through online information to learn about [ARGS]",
        "I'm aiming to access a wide range of knowledge about [ARGS] using digital tools",
        "I'm on a quest to explore online sources to find information about [ARGS]",
        "I'm invested in searching for [ARGS] using internet-based research methods",
        "I'm determined to retrieve information about [ARGS] from reputable online sources",
        "I'm eager to locate comprehensive insights about [ARGS] through online searches",
        "I'm committed to exploring online platforms for data related to [ARGS]",
        "I'm interested in accessing information on [ARGS] from reliable digital sources",
        "I'm dedicated to finding details about [ARGS] by searching online",
        "I'm excited to search for [ARGS] using digital reference materials",
        "I'm actively exploring online encyclopedias to gather information about [ARGS]",
        "I'm aiming to retrieve facts and details about [ARGS] from online references",
        "I'm focused on navigating through digital information to understand [ARGS]",
        "I'm in the process of exploring online databases for comprehensive insights about [ARGS]",
        "I'm eager to search for [ARGS] using reputable online resources",
        "I'm dedicated to locating information about [ARGS] through online research",
        "I'm committed to accessing a virtual treasure trove of information about [ARGS]",
        "I'm interested in finding knowledge about [ARGS] by browsing the internet",
        "I'm on a quest to search for [ARGS] using digital search engines",
        "I'm invested in exploring online platforms for comprehensive details about [ARGS]",
        "I'm determined to retrieve facts about [ARGS] from online encyclopedias",
        "I'm eager to navigate through internet-based information to learn about [ARGS]",
        "I'm actively seeking insights into [ARGS] from reliable online sources",
        "I want to know more about [ARGS]",

        # Action
        "Search [ARGS] in the Wikipedia",
        "Explore information about [ARGS]",
        "Retrieve details about [ARGS]",
        "Gather insights on [ARGS]",
        "Access facts regarding [ARGS]",
        "Find information on [ARGS]",
        "Discover more about [ARGS]",
        "Navigate through [ARGS] details",
        "Look up [ARGS] using online resources",
        "Retrieve comprehensive data about [ARGS]",
        "Browse the internet for [ARGS]",
        "Access a wide range of information about [ARGS]",
        "Locate details on [ARGS] from reliable sources",
        "Investigate [ARGS] using online research",
        "Search for [ARGS] details on the web",
        "Explore the topic of [ARGS] online",
        "Access a digital repository for [ARGS]",
        "Gather knowledge about [ARGS] from reputable sources",
        "Find insights into [ARGS] using online platforms",
        "Retrieve comprehensive insights about [ARGS] from the internet",
        "Navigate through online references to understand [ARGS]",

        # Questions
        "What is [ARGS]?",
        "Tell me about [ARGS]",
        "Can you provide information on [ARGS]?",
        "What can you tell me about [ARGS]?",
        "What's the story behind [ARGS]?",
        "What are the details about [ARGS]?",
        "Could you give me an overview of [ARGS]?",
        "I'd like to learn more about [ARGS]",
        "Do you have any insights on [ARGS]?",
        "Could you help me understand [ARGS]?",
        "I'm interested in the background of [ARGS]",
        "Can you provide me with information about [ARGS]?",
        "Tell me something about [ARGS]",
        "Could you shed light on [ARGS]?",
        "I'm curious about [ARGS]",
        "Can you give me some context on [ARGS]?",
        "Do you know anything about [ARGS]?",
        "What's the significance of [ARGS]?",
        "Could you explain [ARGS] to me?",
        "I'm looking for details about [ARGS]",
        "Tell me the basics of [ARGS]",
        "I'm wondering what [ARGS] is",
        "Can you tell me more about [ARGS]?",
        "Could you elaborate on [ARGS]?",
        "I'd like to get more information about [ARGS]",
        "I'm interested in [ARGS], can you provide information?",
        "Could you give me an idea of [ARGS]?",
        "I'm seeking more knowledge about [ARGS]",
        "Can you provide some insights into [ARGS]?",
        "What's the information available on [ARGS]?",
        "I'm trying to learn about [ARGS], could you help?",
        "Do you have any information available about [ARGS]?",
        "What details can you share about [ARGS]?",
        "I'm looking for details on [ARGS]",
        "Can you tell me more about [ARGS]?",
        "I'm curious about [ARGS], what can you tell me?",
        "What is known about [ARGS]?",
        "I'd like to find more information about [ARGS]",
        "I'm interested in [ARGS], can you give me an overview?",
        "What can you tell me about [ARGS]?",
        "Can you provide insights on [ARGS]?",
        "I'm seeking information about [ARGS], can you help?",
        "Tell me more about [ARGS], please.",

        # Short commands:
        "Retrieve information about [ARGS]",
        "Access details on [ARGS]",
        "Search for data on [ARGS]",
        "Find content related to [ARGS]",
        "Provide insights on [ARGS]",
        "Explore the web for [ARGS]",
        "Get information about [ARGS]",
        "Look up details on [ARGS]",
        "Search for resources about [ARGS]",
        "Find relevant content about [ARGS]",
        "Offer insights on [ARGS]",
        "Discover web content on [ARGS]",
        "Retrieve data related to [ARGS]",
        "Access resources about [ARGS]",
        "Look up information on [ARGS]",
        "Search for details about [ARGS]",
        "Provide information about [ARGS]",
        "Explore web content about [ARGS]",
        "Get details on [ARGS]",
        "Find data about [ARGS]",
        "Offer resources on [ARGS]",
        "Discover information on [ARGS]",
        "Retrieve web content about [ARGS]",
        "Access data about [ARGS]",
        "Look up resources on [ARGS]",
        "Search for insights on [ARGS]",
        "Provide data on [ARGS]",
        "Explore information about [ARGS]",
        "Get insights on [ARGS]",
        "Find web content related to [ARGS]",
        "Offer details on [ARGS]",
        "Discover data about [ARGS]",
        "Retrieve resources on [ARGS]",
        "Access insights about [ARGS]",
        "Look up web content about [ARGS]",
        "Search for resources related to [ARGS]",
        "Provide web content about [ARGS]",
        "Explore data about [ARGS]",
        "Get resources on [ARGS]",
        "Find insights on [ARGS]",
        "Offer web content about [ARGS]",
        "Discover insights about [ARGS]",
        "Retrieve resources about [ARGS]",
        "Access web content on [ARGS]",
        "Look up data about [ARGS]",
        "Search for web content about [ARGS]",
        "Provide resources on [ARGS]",
        "Explore insights about [ARGS]",
        "Get resources related to [ARGS]",
        "Find web content about [ARGS]",
        "Offer data on [ARGS]",
        "Discover resources about [ARGS]",
        "Retrieve insights about [ARGS]",
        "Access data related to [ARGS]",
        "Look up insights on [ARGS]",
        "Search for data related to [ARGS]",
        "Provide details about [ARGS]",
        "Explore resources about [ARGS]",
        "Get insights related to [ARGS]",
        "Find details on [ARGS]",
        "Offer information related to [ARGS]",
        "Discover details about [ARGS]",
        "Retrieve information on [ARGS]",
        "Access insights on [ARGS]",
        "Look up resources related to [ARGS]",
        "Search for details related to [ARGS]",
        "Provide insights related to [ARGS]",
        "Explore details on [ARGS]",
        "Get information related to [ARGS]",
        "Find insights about [ARGS]",
        "Offer resources related to [ARGS]",
        "Discover information related to [ARGS]",
        "Retrieve web content related to [ARGS]",
        "Access resources related to [ARGS]",
        "Look up insights about [ARGS]",
        "Search for information about [ARGS]",
        "Provide web content related to [ARGS]",
        "Explore insights related to [ARGS]",
        "Get details related to [ARGS]",
        "Find data related to [ARGS]",
        "Offer web content related to [ARGS]",
        "Discover web content related to [ARGS]",
        "Retrieve details related to [ARGS]",
        "Access data about [ARGS]",
        "Look up web content related to [ARGS]",
        "Search for resources about [ARGS]",
        "Provide data related to [ARGS]",
        "Explore web content related to [ARGS]",
        "Get resources about [ARGS]",
        "Find insights related to [ARGS]",
        "Offer information about [ARGS]",
        "Discover data related to [ARGS]",
        "Retrieve resources about [ARGS]",
        "Access insights related to [ARGS]",
        "Look up information about [ARGS]",
        "Search for data about [ARGS]",
        "Provide details related to [ARGS]",
        "Explore resources related to [ARGS]",
        "Get insights about [ARGS]",
        "Find information about [ARGS]",
        "Offer details related to [ARGS]",
        "Discover insights related to [ARGS]",
        "Retrieve information related to [ARGS]",
        "Access web content about [ARGS]",
        "Look up resources about [ARGS]",
        "Search for insights about [ARGS]",
        "Provide insights about [ARGS]",
        "Explore web content about [ARGS]",
        "Get data related to [ARGS]",
        "Find resources related to [ARGS]",
        "Offer web content about [ARGS]",
        "Discover details about [ARGS]",
        "Retrieve insights related to [ARGS]",
        "Access data related to [ARGS]",
        "Look up details about [ARGS]",
        "Search for web content about [ARGS]",
        "Provide data about [ARGS]",
        "Explore insights about [ARGS]",
        "Get information related to [ARGS]",
        "Find web content about [ARGS]",
        "Offer resources about [ARGS]",
        "Discover web content about [ARGS]",
        "Retrieve data related to [ARGS]",
        "Access information related to [ARGS]",
        "Look up insights related to [ARGS]",
        "Search for details about [ARGS]",
        "Provide web content about [ARGS]",
        "Explore data related to [ARGS]",
        "Get resources related to [ARGS]",
        "Find insights related to [ARGS]",
        "Offer data about [ARGS]",
        "Discover information about [ARGS]",
        "Retrieve details about [ARGS]",
        "Access resources related to [ARGS]",
        "Look up insights related to [ARGS]",
        "Search for information related to [ARGS]",
        "Provide details about [ARGS]",
        "Explore web content related to [ARGS]",
        "Get insights related to [ARGS]",
        "Find data about [ARGS]",
        "Offer resources related to [ARGS]",
        "Discover insights related to [ARGS]",
        "Retrieve information about [ARGS]",
        "Access details on [ARGS]",
        "Look up data on [ARGS]",
        "Search for content related to [ARGS]",
        "Provide insights on [ARGS]",
        "Explore the web for [ARGS]",
        "Get information about [ARGS]",
        "Look up details on [ARGS]",
        "Search for resources about [ARGS]",
        "Find relevant content about [ARGS]",
        "Offer insights on [ARGS]",
        "Discover web content on [ARGS]",
        "Retrieve data related to [ARGS]",
        "Access resources about [ARGS]",
        "Look up information on [ARGS]",
        "Search for details about [ARGS]",
        "Provide information about [ARGS]",
        "Explore web content about [ARGS]",
        "Get details on [ARGS]",
        "Find data about [ARGS]",
        "Offer resources on [ARGS]",
        "Discover information on [ARGS]",
        "Retrieve web content about [ARGS]",
        "Access data about [ARGS]",
        "Look up resources on [ARGS]",
        "Search for insights on [ARGS]",
        "Provide data on [ARGS]",
        "Explore information about [ARGS]",
        "Get insights on [ARGS]",
        "Find web content related to [ARGS]",
        "Offer details on [ARGS]",
        "Discover data about [ARGS]",
        "Retrieve resources on [ARGS]",
        "Access insights about [ARGS]",
        "Look up web content about [ARGS]",
        "Search for resources related to [ARGS]",
        "Provide web content about [ARGS]",
        "Explore data about [ARGS]",
        "Get resources on [ARGS]",
        "Find insights on [ARGS]",
        "Offer web content about [ARGS]",
        "Discover insights about [ARGS]",
        "Retrieve resources about [ARGS]",
        "Access web content on [ARGS]",
        "Look up data about [ARGS]",
        "Search for web content about [ARGS]",
        "Provide resources on [ARGS]",
        "Explore insights about [ARGS]",
        "Get resources related to [ARGS]",
        "Find web content about [ARGS]",
        "Offer data on [ARGS]",
        "Discover resources about [ARGS]",
        "Retrieve insights about [ARGS]",
        "Access data related to [ARGS]",

        # Shorter?
        "Retrieve [ARGS]",
        "Access info about [ARGS]",
        "Search for [ARGS]",
        "Find details on [ARGS]",
        "Get insights on [ARGS]",
        "Explore [ARGS]",
        "Provide data on [ARGS]",
        "Discover [ARGS]",
        "Look up [ARGS]",
        "Access web content on [ARGS]",
        "Search [ARGS]",
        "Find [ARGS]",
        "Retrieve info about [ARGS]",
        "Explore the web for [ARGS]",
        "Discover insights on [ARGS]",
        "Access resources on [ARGS]",
        "Search for content about [ARGS]",
        "Find data on [ARGS]",
        "Get web content about [ARGS]",
        "Look up insights about [ARGS]",
        "Provide web content on [ARGS]",
        "Explore insights on [ARGS]",
        "Access content about [ARGS]",
        "Search the web for [ARGS]",
        "Find web content on [ARGS]",
        "Look up resources about [ARGS]",
        "Get details about [ARGS]",
        "Access data on [ARGS]",
        "Search online for [ARGS]",
        "Find information on [ARGS]",
        "Look up data about [ARGS]",
        "Get resources on [ARGS]",
        "Access insights about [ARGS]",
        "Search for info about [ARGS]",
        "Find insights about [ARGS]",
        "Look up web content about [ARGS]",
        "Discover resources on [ARGS]",
        "Access details about [ARGS]",
        "Search Wikipedia for [ARGS]",
        "Find web info about [ARGS]",
        "Look up info about [ARGS]",
        "Access Wikipedia for [ARGS]",
        "Search the internet for [ARGS]",
        "Find Wikipedia info about [ARGS]",
        "Look up Wikipedia info about [ARGS]",
        "Get Wikipedia info about [ARGS]",
        "Access online info about [ARGS]",
        "Search for Wikipedia info about [ARGS]",
        "Find online info about [ARGS]",
        "Retrieve Wikipedia info about [ARGS]",
        "Look up online info about [ARGS]",

    ],

    "Calendar": [
        # General descriptions
        # General descriptions
        "Retrieve the current date",
        "Get the present day's date",
        "Access the current year",
        "Obtain the date for today",
        "Retrieve today's date",
        "Get the current day's date",
        "Access the date of today",
        "Obtain the present year",
        "Retrieve the date for the current day",
        "Get today's date information",
        "Access today's year",
        "Obtain the present day's date",
        "Retrieve the current year's date",
        "Get the date for the current day",
        "Access today's date",
        "Obtain the year for today",
        "Retrieve the date for today's day",
        "Get the present date",
        "Access the current year's date",
        "Obtain the current date",
        "Get today's year",
        "Access today's day of the week",
        "Obtain the present date's year",
        "Retrieve today's day",
        "Get the current year's date",
        "Access the year for today",
        "Obtain the current day's year",
        "Retrieve the day of the week for today",
        "Get today's day",
        "Access today's month",
        "Obtain the current day's year",
        "Retrieve today's year",
        "Get the day of the week for today",
        "Access the month for today",
        "Obtain the current year's day",
        "Retrieve today's month",
        "Get the year for today's day",
        "Access the day of the week for the current day",
        "Obtain the month for today's day",
        "Retrieve the current day's month",
        "Get the year for the current day's date",
        "Access the month for the current day's date",
        "Obtain the current month",
        "Retrieve the month for the current day's date",
        "Get the year for this day",
        "Access the month for this day",
        "Obtain the current month's year",
        "Retrieve the month for this date",
        "Get the year for this date",
        "Access the month for this date",
        "Obtain the year for this month",
        "Retrieve the month for this year",
        "Get the date for this month",
        "Access the year for this month",
        "Obtain the day for this month",
        "Retrieve the year for this month",
        "Get the date for this year",
        "Access the month for this year",
        "Obtain the day for this year",
        "Retrieve the date for this year",
        "Get the year for this day",
        "Access the day for this date",
        "Obtain the date for this day",
        "Retrieve the day for this month",
        "Get the year for today's month",
        "Access the date for today's month",
        "Obtain the current month's day",
        "Retrieve the date for today's month",
        "Get today's day of the month",
        "Access today's year's month",
        "Obtain the day of the month for today",
        "Retrieve today's month's day",
        "Get today's month's year",
        "Access today's day of the month",
        "Obtain the day of the month for today's date",
        "Retrieve today's month's year",
        "Get today's date's day of the month",
        "Access today's year's month",
        "Obtain the day of the month for this day",
        "Retrieve the month's year for this date",
        "Get the day of the month for this year",
        "Access the year's month for this day",
        "Obtain the month's day for this year",
        "Retrieve the date's day of the month for this month",
        "Get the year's month for this date",
        "Access the day of the month for this year's month",
        "Obtain the month's day for this year's date",
        "Retrieve the month's year for this month",
        "Get the date's day of the month for this year",
        "Access the year's month for this month's day",
        "Obtain the month's day for this month's date",

        # Short questions
        "Get today's date",
        "Access current year",
        "What's the date?",
        "Today's day?",
        "What day is it?",
        "Current date?",
        "Today's year?",
        "Present date?",
        "Year now?",
        "Today?",
        "Year?",
        "Date please?",
        "Day today?",
        "Current year?",
        "Today's month?",
        "What's today?",
        "Date?",
        "Year now?",
        "Now?",
        "Day?",
        "Today's?",
        "Year today?",
        "Current day?",
        "What date?",
        "Year this?",
        "What's now?",
        "Today's day?",
        "This year?",
        "Date now?",
        "Today's date?",
        "Current month?",
        "Day of week?",
        "Day of week now?",
        "Today's year?",
        "Today's day?",
        "Year of now?",
        "Day today?",
        "Year now?",
        "Month today?",
        "What's today?",
        "Date of now?",
        "Now?",
        "Day?",
        "Today's date?",
        "Year today?",
        "Current day?",
        "Today's month?",
        "What's today?",
        "Year now?",
        "Day today?",
        "Now?",
        "Date now?",
        "Today's day?",
        "Today's year?",
        "Present date?",
        "Year?",
        "Today?",
        "What day?",
        "Current date?",
        "Date?",
        "Day?",
        "Today's?",
        "Now?",
        "Year today?",
        "Today's month?",
        "What's today?",
        "Date please?",
        "Year now?",
        "Today's day?",
        "Access current year",
        "Get today's date",
        "What day is it?",
        "Current date?",
        "Today?",
        "Year now?",
        "Today's year?",
        "Today's date?",
        "Current day?",
        "Date now?",
        "Day today?",
        "What's today?",
        "Year?",
        "Now?",
        "Date?",
        "Today's?",
        "Day of week?",
        "Access current year",
        "What's the date?",
        "Today's month?",
        "Current year?",
        "Year today?",
        "Today's day?",
        "Now?",
        "Year of now?",
        "Day today?",
        "Date of now?",
        "Today's date?",
        "Today's year?",
        "Month today?",
        "Access current year",
        "Day of week now?",
        "Today's year?",
        "Today's day?",
        "Year of now?",
        "Day today?",
        "Date of now?",
        "Today's month?",
        "Day of week now?",
        "Today's year?",
        "Today's date?",
        "Current year?",
        "Year today?",
        "Today's day?",
        "Access current year",
        "What's the date?",
        "Today's month?",
        "Current day?",
        "Date now?",
        "Today's year?",
        "Month today?",
        "What day is it?",
        "Today?",
        "Year now?",
        "Today's date?",
        "Current month?",
        "Year of now?",
        "Day today?",
        "Date of now?",
        "Today's day?",
        "Access current year",
        "Today's year?",
        "Today's date?",
        "Current year?",
        "Year today?",
        "Today's day?",
        "Today?",
        "What's the date?",
        "Current month?",
        "Today's month?",
        "Date now?",
        "Day today?",
        "Today's day?",
        "Year now?",
        "Access current year",
        "What's the date?",
        "Today's month?",
        "Current date?",
        "Today?",
        "Year now?",
        "Today's year?",
        "Today's date?",
        "Year?",
        "Now?",
        "Date?",
        "Day?",
        "Today's?",
        "Now?",
        "Year today?",
        "Today's month?",
        "What's today?",
        "Date please?",
        "Year now?",
        "Today's day?",
        "Access current year",
        "Get today's date",
        "What day is it?",
        "Current date?",
        "Today?",
        "Year now?",
        "Today's year?",
        "Today's date?",
    
    # Short commands
        "Retrieve today's date",
        "Access the current year",
        "Provide today's year",
        "Show today's day",
        "Display today's month",
        "Give current date",
        "Present today's year",
        "Fetch today's month",
        "Offer present date",
        "Supply today's day",
        "Get today's year",
        "Share current date",
        "Retrieve present year",
        "Provide today's month",
        "Display today's date",
        "Offer current day",
        "Show present year",
        "Provide today's day",
        "Access current month",
        "Get current date",
        "Present today's date",
        "Retrieve today's year",
        "Supply today's month",
        "Share today's date",
        "Give current year",
        "Fetch present month",
        "Display current date",
        "Offer today's month",
        "Get present day",
        "Present current year",
        "Retrieve present month",
        "Provide current date",
        "Access today's day",
        "Share present year",
        "Show today's date",
        "Provide present day",
        "Give today's month",
        "Display current year",
        "Offer today's day",
        "Supply current month",
        "Fetch current year",
        "Retrieve present day",
        "Present current date",
        "Show current month",
        "Get today's day",
        "Share today's year",
        "Access present day",
        "Offer current month",
        "Provide current year",
        "Display present date",
        "Give present month",
        "Retrieve current date",
        "Fetch today's month",
        "Show present day",
        "Supply today's year",
        "Present current month",
        "Display today's day",
        "Offer present month",
        "Access today's year",
        "Provide present year",
        "Get current day",
        "Retrieve current year",
        "Show current date",
        "Give current month",
        "Supply present day",
        "Display today's month",
        "Offer today's year",
        "Present present date",
        "Fetch current month",
        "Access present year",
        "Retrieve today's day",
        "Share current month",
        "Give today's day",
        "Provide current day",
        "Display present year",
        "Supply present month",
        "Present today's day",
        "Get present year",
        "Show present date",
        "Offer current day",
        "Retrieve present year",
        "Access current day",
        "Show current year",
        "Give present day",
        "Provide today's year",
        "Display present month",
        "Get current month",
        "Supply current year",
        "Fetch present year",
        "Offer present day",
        "Present today's month",
        "Retrieve current month",
        "Access today's month",
        "Show current day",
        "Give current year",
        "Display today's year",
        "Provide present month",
        "Get present date",
        "Share today's day",
        "Present current day",
        "Offer today's date",
        "Fetch current day",
        "Show today's month",
        "Retrieve present date",
        "Access current month",
        "Supply current day",
        "Display today's year",
        "Give present year",
        "Present present month",
        "Offer current date",
        "Get today's month",
        "Show present year",
        "Provide present date",
        "Access today's day",
        "Share current date",
        "Retrieve present month",
        "Supply today's day",
        "Display current day",
        "Offer today's month",
        "Present present year",
        "Fetch today's year",
        "Show current month",
        "Provide current day",
        "Get present year",
        "Access present month",
        "Retrieve current date",
        "Share present date",
        "Give today's year",
        "Supply present month",
        "Display today's day",
        "Offer current year",
        "Get current day",
        "Present today's year",
        "Show present date",
        "Retrieve today's date",
        "Access the current year",
        "Provide today's year",
        "Show today's day",
        "Display today's month",
        "Give current date",
        "Present today's year",
        "Fetch today's month",
        "Offer present date",
        "Supply today's day",
        "Get today's year",
        "Share current date",
        "Retrieve present year",
        "Provide today's month",
        "Display today's date",
        "Offer current day",
        "Show present year",
        "Provide today's day",
        "Access current month",
        "Get current date",
        "Present today's date",
        "Retrieve today's year",
        "Supply today's month",
        "Share today's date",
        "Give current year",
        "Fetch present month",
        "Display current date",
        "Offer today's month",
        "Get present day",
        "Present current year",
        "Retrieve present month",
        "Provide current date",
        "Access today's day",
        "Share present year",
        "Show today's date",
        "Provide present day",
        "Give today's month",
        "Display current year",
        "Offer today's day",
        "Supply current month",
        "Fetch current year",
        "Retrieve present day",
        "Present current date",
        "Show current month",
        "Get today's day",
        "Share today's year",
        "Access present day",
        "Offer current month",
        "Provide current year",
        "Display present date",
        "Give present month",
        "Retrieve current date",
        "Fetch today's month",
        "Show present day",
        "Supply today's year",
        "Present current month",
        "Display today's day",
        "Offer present month",
        "Access today's year",
        "Provide present year",
        "Get current day",
        "Retrieve current year",
        "Show current date",
        "Give current month",
        "Supply present day",
        "Display today's month",
        "Offer today's year",
        "Present present date",
        "Fetch current month",
        "Access present year",
        "Retrieve today's day",
        "Share current month",
        "Give today's day",
        "Provide current day",
        "Display present year",
        "Supply present month",
        "Present today's day",
        "Get present year",
        "Show present date",
        "Offer current day",
        "Retrieve present year",
        "Access current day",
        "Show current year",
        "Give present day",
    ]



    }

    for key, value in intention_desc["Calculator"].items():
        intention_desc[key] = value

    for key, value in intention_desc.items():
        if isinstance(value, list):
            # Erase duplicates
            intention_desc[key] = list(set(value))


if REMOVE_CALCULATOR:
    tool_name_alternatives["Calculator"] = {}
    tool_desc["Calculator"] = {}

for key, value in tool_name_alternatives["Calculator"].items():
    tool_name_alternatives[key] = value

for key, value in tool_desc["Calculator"].items():
    tool_desc[key] = value

for key, value in tool_name_alternatives.items():
    if isinstance(value, list):
        # Erase duplicates
        tool_name_alternatives[key] = list(set(value))

for key, value in tool_desc.items():
    if isinstance(value, list):
        # Erase duplicates
        tool_desc[key] = list(set(value))



CALC_SUBSETS = list(tool_name_alternatives["Calculator"].keys())
TOOL_NAME_COUNT = {key:len(value) for key, value in tool_name_alternatives.items() if key != "Calculator"}
DISTRACTOR_TOOLS = [key for key in tool_name_alternatives if key not in ["Calculator", "Calendar", "WikiSearch"] + CALC_SUBSETS]

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache/"

if MODEL_NAME == "GPT2":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
elif MODEL_NAME == "GPTJ":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", truncate=True, max_length=270, cache_dir=cache_dir)
elif MODEL_NAME == "LLAMA2":
    from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=cache_dir,
                                        token= "***REMOVED***")



tokenizer.pad_token = tokenizer.eos_token

if NO_TOKEN_OPTION == "":
    tokenizer.add_tokens([TOOL_START_TOKEN, TOOL_END_TOKEN])
    tokenizer.pad_token = tokenizer.eos_token


if METHOD_B:
    tokenizer.add_tokens(["[ARGS]"])

    arg_token_id = len(tokenizer) - 1
    def split_on_arg(text):
        # Text is a tokenized tensor of the text. We want to split on the token with id tokenizer.encode("[ARGS]")[0]
        # We return a tuple of the text before the first occurence of the token, and the text after the token
        # If the token is not present, we return the text, None
        if arg_token_id in text:
            try:
                index = torch.where(text == arg_token_id)[0].item()
            except:
                print(text)
                print(arg_token_id)
                print(torch.where(text == arg_token_id))
                raise ValueError
            return text[:index], text[index+1:]
        else:
            return text, None

    # Tokenize the alternative names for the Calculator subtypes, the calendar, and the wikisearch
    for key, value in tool_name_alternatives.items():
        if key in ["Calendar", "WikiSearch"] + CALC_SUBSETS:
            tool_name_alternatives[key] = [long_tensor(tokenizer.encode(name)) for name in value]

    for key, value in intention_desc.items():
        if key in ["Calendar", "WikiSearch"] + CALC_SUBSETS:
            intention_desc[key] = [split_on_arg(long_tensor(tokenizer.encode(name + "|"))) for name in value]

def filter_all_caps(name_dict):
    pass
    #TODO

# Train3 has:   
# Calculator:   1279   1517  4596
#    add:       163    196   588
#    subtract:  323    375   1156
#    multiply:  144    180   720
#    divide:    520    610   1785
#    add_sub:   20     24    75
#    mix:       109    131   445
# Calendar:     5899         7927
# WikiSearch:   10290        10290


def perplexity(dataset):
    # Dataset is a tuple of predictions and labels

    average_perplexity = 0
    examples = 0

    for pred, lab in zip(dataset):
        examples += 1
        loss_fct = torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), lab.reshape(-1), reduction='sum')

        average_perplexity += torch.exp(loss_fct)

    average_perplexity /= examples
    return {"perplexity": average_perplexity}


def tokenize_function(example):
    input = tokenizer(example["text"], truncation=True, max_length=300)

    return {"input_ids": input["input_ids"]}

#val_dataset = load_dataset("/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/unprocessed/segment_ccnet_unprocessed", data_files=["1000_examples_not_in_training.csv"], split="train")
#val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)


# Dataset classes and collate functions
@beartype
class ToolDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):

        self.texts = dataset["tokenized_text"]
        self.token_type = dataset["token_type"]
        self.tool_names = dataset["tool_name"]
        self.calc_subtypes = dataset["op_label"]
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):

        tokenized_text = ast.literal_eval(self.texts[idx])
        token_type = ast.literal_eval(self.token_type[idx])

        assert len(tokenized_text) == len(token_type), f"Lengths of tokenized text and tokenized type are not equal: {len(tokenized_text)} != {len(token_type)}"

        return tokenized_text, token_type, self.tool_names[idx], self.calc_subtypes[idx]

class RawDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, model):

        self.texts = dataset["text"]
        self.tokenizer = tokenizer
        self.model = model


    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokenized_text = self.tokenizer(self.texts[idx], truncation=True, max_length=270, return_tensors="pt")
        if isinstance(tokenized_text, dict):
            tokenized_text = tokenized_text["input_ids"]

        return self.model.prepare_inputs_for_generation(tokenized_text)


def come_as_you_are_collate_fn(batch):

    texts, token_types, _, _ = zip(*batch, strict=True)

    for text, mask in zip(texts, token_types):
        assert len(text) == len(mask), f"Shapes dont match: {len(text)} != {len(mask)}"

    texts = list(map(long_tensor, texts))
    texts = pad_sequence(texts, padding_value=PAD_ID)

    token_types = list(map(int_tensor, token_types))
    masks = list(map(method_a_masker, token_types))
    masks = pad_sequence(masks, padding_value=0)

    for text, mask in zip(texts, masks):
        assert text.shape == mask.shape, f"Shapes dont match: {text.shape} != {mask.shape}"

    return texts, masks

AVAILABLE_TOOLS_PROMPT = "These are the available tools: \n[TOOLS].\n\n"
AVAILABLE_RANGE = [1,4]
print(f"Distractors: {DISTRACTOR_TOOLS}")
#TOOL_NAME_COUNT["Calculator"] = sum(len(tool_name_alternatives["Calculator"].values()))
DESC_CHANCE = 1
PROMPT_REMOVAL_P = 0.0

@beartype
def choose(tools:List[str])->str:
    weights = [TOOL_NAME_COUNT[tool] for tool in tools]
    return random.choices(tools, weights= weights)[0]

@beartype
def random_tool_descriptor(main_tool:str, calc_sub=None)->Tuple[str,str]:
    # Number of tools is a random number in the AVAILABLE_RANGE
    num_tools = random.randint(*AVAILABLE_RANGE)

    print(f"Main tool: {main_tool}, num_tools: {num_tools}, calc_sub: {calc_sub}")
    base_names = []
    tools = set()
    
    if main_tool == "Calculator":
        if calc_sub is not None:
            opts = [calc_sub, "mix"]
            if calc_sub in ["add", "subtract"]:
                opts.append("add_subtract")
            if calc_sub in ["multiply", "divide"]:
                opts.append("mult_divide")
            main_tool = choose(opts)
        else:
            main_tool = choose(CALC_SUBSETS)

    main_name = random.choice(tool_name_alternatives[main_tool])

    # random 22% chance:
    if random.random() < DESC_CHANCE:
        tools.add(f"{main_name} ({random.choice(tool_desc[main_tool])})")
    else:
        tools.add(main_name)

    while len(tools) < num_tools:
        # Get a random tool from 
        base_name = choose(DISTRACTOR_TOOLS)
        base_names.append(base_name)

        rand_name = random.choice(tool_name_alternatives[base_name])

        # random 22% chance:
        if random.random() < DESC_CHANCE:
            rand_name = f"{rand_name} ({random.choice(tool_desc[base_name])})"
        tools.add(rand_name)
    
    tools = list(tools)
    random.shuffle(tools)
    pretty_tools = []
    for tool in tools:
        pretty_tools.append(f"  - {tool}")

    return "\n".join(pretty_tools), main_name

eval_collator = DataCollatorWithPadding(tokenizer)
    
def change_name_collate_fn(batch):
    
    texts, data_token_types, tool_names, calc_subtypes = zip(*batch, strict=True)

    for text, mask in zip(texts, data_token_types):
        assert len(text) == len(mask), f"Shapes dont match: {len(text)} != {len(mask)}"

    token_types = [None,]*4
    
    name_idxs = map(lambda token_type: token_type.index(2), data_token_types)
    parenthesis_idxs = map(lambda token_type: token_type.index(3), data_token_types)

    start_texts, token_types[1] = zip(*map(lambda text, token_type, idx: (text[:idx], token_type[:idx]), texts, data_token_types, name_idxs))
    end_texts, token_types[3] = zip(*map(lambda text, token_type, idx: (text[idx:], token_type[idx:]), texts, data_token_types, parenthesis_idxs))

    for s_text, s_mask, e_text, e_mask in zip(start_texts, token_types[1], end_texts, token_types[3]):
        assert len(s_text) == len(s_mask), f"Shapes dont match: {len(s_text)} != {len(s_mask)}"
        assert len(e_text) == len(e_mask), f"Shapes dont match: {len(e_text)} != {len(e_mask)}"

    tool_cohorts, main_tools = zip(*map(random_tool_descriptor, tool_names, calc_subtypes), strict=True)

    prompts = map(lambda cohort: AVAILABLE_TOOLS_PROMPT.replace("[TOOLS]", cohort), tool_cohorts)
    # Random PROMPT_REMOVAL_P% chance of removing the prompt
    prompts = map(lambda prompt: prompt if random.random() > PROMPT_REMOVAL_P else "", prompts)
    prompts = list(map(tokenizer.encode, prompts))
    token_types[0] = list(map(lambda prompt: [7,]*len(prompt), prompts))
    main_tools = list(map(tokenizer.encode, main_tools))
    token_types[2] = list(map(lambda tool: [0,]*len(tool), main_tools))

    for p, p_t, m, m_t in zip(prompts, token_types[0], main_tools, token_types[2]):
        assert len(p) == len(p_t), f"Shapes dont match: {len(p)} != {len(p_t)}"
        assert len(m) == len(m_t), f"Shapes dont match: {len(m)} != {len(m_t)}"

    add_4 = lambda tuple_of_lists: [item for sublist in tuple_of_lists for item in sublist]

    processed_texts = list(map(add_4, zip(prompts, start_texts, main_tools, end_texts, strict=True)))
    merged_token_types = list(map(add_4, zip(*token_types, strict=True)))

    for text, mask in zip(processed_texts, merged_token_types):
        assert len(text) == len(mask), f"Shapes dont match: {len(text)} != {len(mask)}"

    texts = list(map(long_tensor, processed_texts))
    texts = pad_sequence(texts, padding_value=PAD_ID)

    merged_token_types = list(map(int_tensor, merged_token_types))
    masks = list(map(method_a_masker, merged_token_types))
    masks = pad_sequence(masks, padding_value=0)

    for text, mask in zip(texts, masks):
        assert text.shape == mask.shape, f"Shapes dont match: {text.shape} != {mask.shape}"

    if ARG_TRAINING:
        resp_indexes = list(map(lambda token_type: token_type.index(6), token_types[3]))
        end_texts, token_types[3] = zip(*map(lambda text, token_type, idx: (text[:idx], token_type[:idx]), end_texts, token_types[3], resp_indexes))
        arg_texts = list(map(add_4, zip(start_texts, main_tools, end_texts, strict=True)))
        arg_token_types = list(map(add_4, zip(token_types[1], token_types[2], token_types[3], strict=True)))

        for text, mask in zip(arg_texts, arg_token_types):
            assert len(text) == len(mask), f"Shapes dont match: {len(text)} != {len(mask)}"

        arg_texts = list(map(long_tensor, arg_texts))
        arg_texts = pad_sequence(arg_texts, padding_value=PAD_ID)

        arg_token_types = list(map(int_tensor, arg_token_types))
        arg_masks = list(map(arg_selection_masker, arg_token_types))
        arg_masks = pad_sequence(arg_masks, padding_value=0)

        for text, mask in zip(arg_texts, arg_masks):
            assert text.shape == mask.shape, f"Shapes dont match: {text.shape} != {mask.shape}"

        return (texts, arg_texts), (masks, arg_masks)

    return texts, masks

def random_tool_name_intention(tool_name, args, calc_sub):

    main_tool = tool_name
    if tool_name == "Calculator":
        if calc_sub is not None:
            if calc_sub in ["mult_divide", "add_subtract", "mix"]: 
                opts = ["mix"]
            else:
                opts = [calc_sub, "mix"]
            main_tool = choose(opts)
        else:
            main_tool = choose(CALC_SUBSETS)

    random_name = random.choice(tool_name_alternatives[main_tool])
    random_intention = random.choice(intention_desc[main_tool]) 

    if random_intention[1] != None: 
        intention_list = [random_intention[0], args, random_intention[1]]
    else:
        intention_list = [random_intention[0]]

    random_intention = torch.cat(intention_list)
    return random_name, random_intention


def method_2_collate_fn(batch):
    # Here we have 4 token types also:
    # - start text
    # - tool intention
    # - main tool
    # - end text
    # We have removed the prompts but added the tool intentions
 
    texts, data_token_types, tool_names, calc_subtypes = zip(*batch, strict=True)

    for text, mask in zip(texts, data_token_types):
        assert len(text) == len(mask), f"Shapes dont match: {len(text)} != {len(mask)}"

    token_types = [None,]*4

    def arg_index(text):
        try:
            return text.index(4)
        except ValueError:
            return text.index(5)
    tensor_out_of_list = lambda tuple_of_lists: int_tensor([item for sublist in tuple_of_lists for item in sublist])

    name_idxs = map(lambda token_type: token_type.index(2), data_token_types)
    parenthesis_idxs = map(lambda token_type: token_type.index(3), data_token_types)
    arg_start = map(lambda token_type: arg_index(token_type), data_token_types)
    arg_end = map(lambda token_type: token_type.index(5), data_token_types)
    args = list(map(lambda text, start, end: long_tensor(text[start:end]), texts, arg_start, arg_end))

    start_texts, token_types[0] = zip(*map(lambda text, token_type, idx: (long_tensor(text[:idx]), long_tensor(token_type[:idx])), texts, data_token_types, name_idxs))
    end_texts, token_types[3] = zip(*map(lambda text, token_type, idx: (long_tensor(text[idx:]), long_tensor(token_type[idx:])), texts, data_token_types, parenthesis_idxs))

    for s_text, s_mask, e_text, e_mask in zip(start_texts, token_types[0], end_texts, token_types[3]):
        assert len(s_text) == len(s_mask), f"Shapes dont match: {len(s_text)} != {len(s_mask)}"
        assert len(e_text) == len(e_mask), f"Shapes dont match: {len(e_text)} != {len(e_mask)}"

    tool_name, tool_intention = zip(*map(random_tool_name_intention, tool_names, args, calc_subtypes), strict=True)
    processed_texts = list(map(torch.cat, zip(start_texts, tool_intention, tool_name, end_texts, strict=True)))

    token_types[1] = list(map(lambda intention: long_tensor(torch.ones(intention.shape)), tool_intention))
    token_types[2] = list(map(lambda tool: long_tensor(torch.ones(tool.shape)*2), tool_name))  
    merged_token_types = list(map(tensor_out_of_list, zip(*token_types, strict=True)))

    for text, mask in zip(processed_texts, merged_token_types):
        assert text.shape == mask.shape, f"Shapes dont match: {text.shape} != {mask.shape}"

    texts = pad_sequence(processed_texts, padding_value=PAD_ID)
    masks = pad_sequence(merged_token_types, padding_value = 7)
    masks = method_b_masker(masks)

    return texts, masks


# Override the Trainer class to use our loss function
class MyTrainer(Trainer):

    # On init, init super class:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        tokenized_data, masks = inputs

        if not isinstance(tokenized_data, tuple):
            tokenized_data = (tokenized_data,)
            masks = (masks,)

        masks = [mask[:,1:] for mask in masks] #Prediction starts from second token

        inputs = [model.prepare_inputs_for_generation(data[:,:-1], use_cache=False) for data in tokenized_data]
        outputs = [model(**input_data) for input_data in inputs]

        # For debbuging purposes, print columns with 1. Decoded token, 2. mask value:
        # The columns should be well padded with spaces, so that the mask values are aligned
        # Example:
        # Data      //    mask:
        # <TOOL>    //      1
        # <TOOL>    //      1
        # You       //      0
        logits = [out.logits for out in outputs]

        if DEBUG:
            # SIZES OF INPUTS:
            for token_data, mask, logit in zip(tokenized_data, masks, logits):
                print(f"Tokenized data size: {token_data.shape}")
                print("Data      //    mask:")
                for i, sentence in enumerate(token_data[:5,1:]):
                    print(f"First word: {tokenizer.decode(token_data[i,0])}")
                    for j, token in enumerate(sentence):
                        print(f"Label: {tokenizer.decode(token):<12} // Pred: {tokenizer.decode(logit[i][min(logit.shape[1]-1,j)].argmax().item()):<12} // {mask[i][min(mask.shape[1]-1,j)].item():>3}")
        
        # print(f"Masks size: {mask.shape}")
        masks = [mask.reshape(-1).bool() for mask in masks]
        logits = torch.cat([logit.reshape(-1, logit.shape[-1])[mask] for logit, mask in zip(logits, masks)])
        tokenized_data = torch.cat([token_data[:,1:].reshape(-1)[mask] for token_data, mask in zip(tokenized_data, masks)])
        loss = torch.nn.functional.cross_entropy(logits, tokenized_data)

        if DEBUG:
            # SIZES OF INPUTS:
            print(f"Tokenized data size: {tokenized_data.shape}")

            print("Data      //    mask:")
            for i, word in enumerate(tokenized_data):
                print(f"Label: {tokenizer.decode(word):<12} // Pred: {tokenizer.decode(logits[i].argmax().item()):<12}  loss avg = {loss.item()}")
        
        if DEBUG:
            print(f"Loss:{loss.item()}")

        return (loss, outputs) if return_outputs else loss


train_data_dir = f"/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/train/curated/{MODEL_NAME}_small2/"
train_files = [file for file in os.listdir(train_data_dir) if file.endswith(".csv")]
train_files = [f"train_short{NO_TOKEN_OPTION}.csv"]
raw_train_files = [f"{file[:-4]}_texts.csv" for file in train_files]
#train_files = ["train3_tagged.csv"]
# If raw is provided as a command line argument, make it true:

INT_COLUMNS = ['length', 'nlines', 'original_nlines', 'original_length', 'duplicity_ranking_tool', 'duplicity_ranking_global']
FLOAT_COLUMNS = ['position', 'loss_improvement', 'language_score', 'perplexity', "relevance"]

train_columns = []
if not RAW:

    train_dataset = pd.read_csv(os.path.join(train_data_dir, train_files[0]))

    if REMOVE_CALCULATOR:
        train_dataset = train_dataset[train_dataset["tool_name"] != "Calculator"].reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_dataset)

    print("Loaded dataset")

else:
    # Read first line in train file:
    with open(os.path.join(train_data_dir, raw_train_files[0]), "r") as f:
        train_header = f.readline()
        train_columns = train_header.strip().split(",")

    feat_dict = {key: Value(dtype='string', id=None) for key in train_columns}
    feat_dict.update({key: Value(dtype='int64', id=None) for key in INT_COLUMNS if key in train_columns})
    feat_dict.update({key: Value(dtype='float32', id=None) for key in FLOAT_COLUMNS if key in train_columns})
    features = Features(feat_dict)

    print(f"Loading dataset from {train_data_dir} and files {raw_train_files}")
    raw_dataset = load_dataset(train_data_dir, data_files = raw_train_files, features=features, split="train", cache_dir=cache_dir)
    print("Loaded dataset")




TRAIN_NAME = "small2-arg" # "increasing_relevance_2" # "no_duplicates_2"

training_args = TrainingArguments(
    output_dir="./results/test", # The output directory
    overwrite_output_dir=True, # overwrite the content of the output directory
    num_train_epochs=1, # number of training epochs
    per_device_train_batch_size=8, # batch size for training
    #per_device_eval_batch_size=1,  # batch size for evaluation
    #eval_steps = 10000, # Number of update steps between two evaluations.
    #evaluation_strategy = "steps",
    save_steps=500, # after # steps model is saved
    warmup_steps=40, # number of warmup steps for learning rate scheduler
    learning_rate=1e-5,
    dataloader_pin_memory=False,
    do_train=True,
    deepspeed="/vol/bitbucket/jg2619/augmenting_llms/model_training/model_experiments/ds_conf.json",
    gradient_accumulation_steps=2,
    evaluation_strategy="no",
    do_eval=False
)



def load_model(new_tokens = NO_TOKEN_OPTION == ""):
    if MODEL_NAME == "GPT2":
        model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir)
    elif MODEL_NAME == "GPTJ":
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,          # CANT HANDLE DEEPSPEED ZERO 3
            cache_dir=cache_dir 
        ).cuda()
    elif MODEL_NAME == "LLAMA2":
        config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                             padding_idx=tokenizer.pad_token_id)
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=cache_dir,
                token="***REMOVED***",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                config=config,)

    if new_tokens:
        model.resize_token_embeddings(len(tokenizer) - METHOD_B) # If method B, we have added the practical [ARGS] token
        print(f"Len tokenizer: {len(tokenizer)}")

    unfrozen_count = 0
    unfrozen_size = 0
    total_count = 0
    total_size = 0
    FROZEN_LAYERS = []
    TRAINED_LAYERS = []
    for i in range(0, 25 if MODEL_NAME == "GPTJ" else 28):
        FROZEN_LAYERS += MODEL_LAYERS[f"Layer {i}"]
    

    # Freeze some layers in the architecture
    for name, param in model.named_parameters():
        total_count += param.numel()
        total_size += param.numel() * param.element_size()
        if name in FROZEN_LAYERS:
            param.requires_grad = False
        else:
            TRAINED_LAYERS.append(name)
            unfrozen_count += param.numel()
            unfrozen_size += param.numel() * param.element_size()

    print(f"Unfrozen layers: {unfrozen_count} parameters, {unfrozen_size/1e6} MB")
    print(f"Training {unfrozen_count/total_count*100}% of params representing {unfrozen_size/total_size*100}% of size")

    return model



DEBUG = True
method_a_masker = partial(torch.isin, test_elements=int_tensor([0,1,2,3,8]))
arg_selection_masker = partial(torch.isin, test_elements=int_tensor([4,5]))
method_b_masker = partial(torch.isin, test_elements=int_tensor([0,1,4,5,8]))
PAD_ID = tokenizer.pad_token_id


# Arange of tensors from 0 to 2*3*4:
# >>> torch.arange(24).reshape(2,3,4)
#data_files=dataset_dir[len_data_files*0.8:]
#data_files=dataset_dir[:len_data_files*0.8]
print("MADE IT PAST")

# Get a random 20% of the training data, as it is too big
# train_dataset = train_dataset.shuffle(seed=42).select(range(int(len(train_dataset)*0.2)))

# Do a 90/10 split for validation randomly
# Train the model



# TODO
#CHECKPOINTING
#STATS

print("END OF SETUP")

def main():
    global raw_dataset, train_dataset

    if not RAW:
        model = load_model(new_tokens= NO_TOKEN_OPTION == "")

        print("Loaded model")

        if LORA:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        train_dataset = ToolDataset(train_dataset,tokenizer) 

        if SHUFFLE:
            random.seed(42)

            indices = list(range(len(train_dataset)))
            random.shuffle(indices)

            train_dataset = Subset(train_dataset, indices)

        training_args.warmup_steps = len(train_dataset)*7 // 100

        trainer = MyTrainer(
            model=model, # the instantiated 🤗 Transformers model to be trained
            args=training_args, # training arguments, defined above
            train_dataset=train_dataset, # training dataset
            data_collator = method_2_collate_fn if METHOD_B else change_name_collate_fn,
        )

        print("GONNA TRAIN")
        trainer.train()

        # Print cuda memory summary
        print(torch.cuda.memory_summary())

        save_file = f"/vol/bitbucket/jg2619/augmenting_llms/model_training/models/{MODEL_NAME}_{TRAIN_NAME}"
        print(f"Saving model to {save_file}")

        model.save_pretrained(save_file)
    else:

        model = load_model(new_tokens=False)
        
        raw_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=train_columns)

        training_args.warmup_steps = len(raw_dataset) // 10

        trainer = Trainer(
            model=model, # the instantiated 🤗 Transformers model to be trained
            #tokenizer=tokenizer,
            args=training_args, # training arguments, defined above
            train_dataset=raw_dataset, # training dataset
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        print("GONNA TRAIN")
        trainer.train()

        # Print cuda memory summary
        print(torch.cuda.memory_summary())

        save_file = f"/vol/bitbucket/jg2619/augmenting_llms/model_training/models/{MODEL_NAME}_{TRAIN_NAME}_raw"
        print(f"Saving model to {save_file}")

        model.save_pretrained(save_file)

    # Write the latest training description to a txt file in save_file dir:
    with open(f"{save_file}/training_description.txt", "w") as f:
        f.write(TRAINING_DESCRIPTIONS[-1])


if __name__ == "__main__":
    main()
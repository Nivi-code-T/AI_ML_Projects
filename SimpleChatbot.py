#import regular expression module to handle patten matching
import re
from http.client import responses
from typing import Any

#a dictionary maps keywords to predefined response
responses = {
        "hello": ["Hello there!", "Hi! How are you?", "Hey! Nice to see you!"],
        "how are you": ["I'm good, thanks!", "Doing great, you?", "Feeling fantastic!"],
        "bye": ["Goodbye!", "See you later!", "Take care!"]
}

# #Function to find the appropriate response based on the user input
# def chatbot_reponse(user_input):
#     #Convert user input to lower case just to avoid case insensitive
#    // user_input=user_input.lower():




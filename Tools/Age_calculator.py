from datetime import datetime
from langchain.tools import Tool
import json
import math

class AgeCalculator:
    name = "Age Calculator"
    description = """Only useful when user asks for their age. Input to this tool should be a json  object which will\
a stringify object. This JSON object will have two keys named as "birth_month" and "birth_year" and their values will \
be provided by the user never assume a value if user didn't provide any value then assign NAN value to that partcular key\
. For example for a user's query "I was born in december, 2001" the tool input will look like this: ("birth_year":2001,\
"birth_month":12). If the user does not provide any of the input values replace it with NAN"""

    def __init__(self):
        self.birth_year = None
        self.birth_month = None


    def calculate(self, user_input):
        print(user_input)
        user = json.loads(user_input)
        birth_month = user['birth_month']
        birth_year = user['birth_year']
        print(user['birth_month'])
        print(user['birth_year'])
        """Minus birth_year from datetime.now().year"""
        birth_month = str(birth_month).strip().lower()
        birth_year = str(birth_year).strip().lower()

        # Check for invalid values ("nan", None, empty string)
        if birth_month in {'nan', 'none', ''}:
            return 'I need your birth month'
        if birth_year in {'nan', 'none', ''}:
            return 'I need your birth year'

        if int(datetime.now().month) > int(birth_month):
            return int(datetime.now().year) - int(birth_year)
        else:
            return int(datetime.now().year) - int(birth_year) - 1
        # return "WHy do i calculate that?"

tool2 = AgeCalculator()
tool_AgeCalculator = Tool.from_function(
    name=tool2.name,
    description=tool2.description,
    func=tool2.calculate,
    return_direct=True
)
# print(tool_AgeCalculator.run())

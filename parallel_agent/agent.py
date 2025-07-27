# parallel agent
from dotenv import load_dotenv
load_dotenv()
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search

import os
os.environ["OTEL_PYTHON_DISABLED"] = "true" # OpenTelemetry issues -> regarding internal asyncio package


# Hotel Search Agent - Finds accommodation options
hotel_search_agent = Agent(
    name="HotelSearchAgent",
    model="gemini-2.0-flash",
    tools=[google_search],
    description="An agent that searches for hotels and accommodation options in a given destination",
    instruction="""
    You are a hotel booking specialist. You will be given a destination and travel dates, and you will search for:
    - Popular hotels in the area
    - Different price ranges (budget, mid-range, luxury)
    - Hotel amenities and ratings
    - Location advantages
    Provide a summary of the best accommodation options with brief descriptions.
    ONLY research for hotels and nothing else.
    """,
    output_key="hotel_options",
)

# Restaurant Search Agent - Finds dining options
restaurant_search_agent = Agent(
    name="RestaurantSearchAgent",
    model="gemini-2.0-flash",
    tools=[google_search],
    description="An agent that searches for restaurants and dining experiences in a given destination",
    instruction="""
    You are a food and dining expert. You will be given a destination and you will search for:
    - Top-rated restaurants and cafes
    - Local cuisine specialties
    - Different dining price ranges
    - Unique dining experiences
    Provide a summary of the best dining options with cuisine types and highlights.
    Only research for restaurants and nothing else.
    """,
    output_key="restaurant_options",
)

# Activities Search Agent - Finds things to do
activities_search_agent = Agent(
    name="ActivitiesSearchAgent",
    model="gemini-2.0-flash",
    tools=[google_search],
    description="An agent that searches for activities and attractions in a given destination",
    instruction="""
    You are a local activities expert. You will be given a destination and you will search for:
    - Popular tourist attractions
    - Outdoor activities and adventures
    - Cultural experiences and museums
    - Entertainment and nightlife options
    Provide a summary of the best activities with brief descriptions and recommendations.
    Only research for activities and nothing else.
    """,
    output_key="activity_options",
)

# Parallel agent that runs all search agents simultaneously
parallel_search_agent = ParallelAgent(
    name="ParallelTravelSearchSystem",
    description="A system that simultaneously searches for hotels, restaurants, and activities",
    sub_agents=[hotel_search_agent, restaurant_search_agent, activities_search_agent],
)

# Summary Agent - Creates a concise travel summary
summary_agent = Agent(
    name="TravelSummaryAgent",
    model="gemini-2.0-flash",
    description="An agent that creates a concise and well-organized travel summary from detailed research",
    instruction="""
    You are a travel planning expert. You will receive detailed information from three sources:
    - hotel_options: Hotel and accommodation research
    - restaurant_options: Restaurant and dining research  
    - activity_options: Activities and attractions research
    
    Use these specific data sources to create a concise, well-structured travel summary that includes:
    
    🏨 **ACCOMMODATION HIGHLIGHTS** (2-3 top picks)
    - Extract best hotel options from hotel_options data across different budgets
    
    🍽️ **DINING HIGHLIGHTS** (2-3 must-try places)
    - Extract key restaurants and local cuisine from restaurant_options data
    
    🎯 **TOP ACTIVITIES** (3-4 must-do experiences)
    - Extract mix of popular attractions and unique experiences from activity_options data
    
    📝 **QUICK TIPS**
    - 1-2 insider tips or important notes from any of the three data sources
    
    Keep the entire summary under 200 words and make it practical and actionable.
    Focus on the absolute best options rather than overwhelming detail.
    Reference the output keys (hotel_options, restaurant_options, activity_options) when processing the data.
    """,
    output_key="travel_summary",
)

# Main root agent that coordinates the entire process
root_agent = SequentialAgent(
    name="TravelPlanningSystem",
    description="A comprehensive travel planning system that researches and summarizes travel information",
    sub_agents=[parallel_search_agent, summary_agent],
)

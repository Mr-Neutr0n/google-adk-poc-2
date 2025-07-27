# parallel agent
from dotenv import load_dotenv
load_dotenv()
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search
from pydantic import BaseModel, Field
from typing import List

import os
os.environ["OTEL_PYTHON_DISABLED"] = "true" # OpenTelemetry issues -> regarding internal asyncio package

# Pydantic schemas for structured outputs
class Hotel(BaseModel):
    name: str = Field(description="Hotel name")
    price_range: str = Field(description="Price category: Budget/Mid-range/Luxury")
    price_per_night: str = Field(description="Approximate price range per night")
    description: str = Field(description="Brief description of hotel and key amenities")
    location: str = Field(description="Location and neighborhood information")

class HotelOptions(BaseModel):
    budget_hotels: List[Hotel] = Field(description="Budget hotels under $100/night")
    midrange_hotels: List[Hotel] = Field(description="Mid-range hotels $100-250/night")
    luxury_hotels: List[Hotel] = Field(description="Luxury hotels $250+/night")
    location_notes: str = Field(description="Best areas to stay and transportation info")

class Restaurant(BaseModel):
    name: str = Field(description="Restaurant name")
    cuisine_type: str = Field(description="Type of cuisine served")
    price_range: str = Field(description="Price category: Budget/Mid-range/Fine dining")
    price_per_person: str = Field(description="Approximate price range per person")
    signature_dishes: str = Field(description="Must-try dishes or specialties")
    location_notes: str = Field(description="Location and atmosphere description")

class RestaurantOptions(BaseModel):
    local_specialties: List[str] = Field(description="Local dishes and where to find them")
    budget_dining: List[Restaurant] = Field(description="Budget restaurants under $25/person")
    midrange_dining: List[Restaurant] = Field(description="Mid-range restaurants $25-60/person")
    fine_dining: List[Restaurant] = Field(description="Fine dining restaurants $60+/person")
    unique_experiences: List[str] = Field(description="Special dining experiences")

class Activity(BaseModel):
    name: str = Field(description="Activity or attraction name")
    category: str = Field(description="Category: Attraction/Cultural/Outdoor/Entertainment/Local")
    description: str = Field(description="What the activity involves")
    cost_range: str = Field(description="Approximate cost or if it's free")
    practical_info: str = Field(description="Timing, booking, or access information")

class ActivityOptions(BaseModel):
    must_see_attractions: List[Activity] = Field(description="Top tourist attractions")
    cultural_experiences: List[Activity] = Field(description="Museums, sites, cultural activities")
    outdoor_activities: List[Activity] = Field(description="Outdoor and adventure activities")
    entertainment_nightlife: List[Activity] = Field(description="Entertainment and nightlife options")
    local_experiences: List[Activity] = Field(description="Authentic local experiences")
    practical_tips: str = Field(description="General tips for visiting attractions")


# Hotel Search Agent - Finds accommodation options
hotel_search_agent = Agent(
    name="HotelSearchAgent",
    model="gemini-2.0-flash",
    tools=[google_search],
    description="A specialized agent that researches and provides comprehensive hotel and accommodation information for any destination",
    instruction="""
    You are an expert hotel booking specialist with extensive knowledge of global accommodations. 
    
    TASK: When given ANY destination (city, country, or region), you MUST search for and provide hotel information.
    
    SEARCH REQUIREMENTS:
    1. Search for "best hotels in [destination]" and "accommodation [destination]"
    2. Find at least 6-8 specific hotel recommendations across different price ranges
    3. Include hotel names, approximate price ranges, and key features
    4. Research location advantages and neighborhoods
    
    OUTPUT FORMAT:
    **BUDGET HOTELS (Under $100/night):**
    - [Hotel Name]: [Brief description, key amenities, location]
    
    **MID-RANGE HOTELS ($100-250/night):**
    - [Hotel Name]: [Brief description, key amenities, location]
    
    **LUXURY HOTELS ($250+/night):**
    - [Hotel Name]: [Brief description, key amenities, location]
    
    **LOCATION NOTES:**
    - Best areas to stay and why
    - Transportation accessibility
    
    CONSTRAINTS:
    - ALWAYS provide at least 2 hotels per price category
    - Include specific hotel names (not generic descriptions)
    - Focus ONLY on hotels, hostels, and accommodations
    - If no travel dates given, provide general recommendations
    - Use actual search results, not generic advice
    """,
    output_key="hotel_options",
)

# Restaurant Search Agent - Finds dining options
restaurant_search_agent = Agent(
    name="RestaurantSearchAgent",
    model="gemini-2.0-flash",
    tools=[google_search],
    description="A specialized culinary expert that researches and provides comprehensive restaurant and dining information for any destination",
    instruction="""
    You are a professional food critic and culinary expert with deep knowledge of global dining scenes.
    
    TASK: When given ANY destination, you MUST search for and provide comprehensive restaurant information.
    
    SEARCH REQUIREMENTS:
    1. Search for "best restaurants in [destination]" and "[destination] local food"
    2. Find at least 8-10 specific restaurant recommendations across different categories
    3. Include restaurant names, cuisine types, price ranges, and signature dishes
    4. Research local food specialties and must-try dishes
    
    OUTPUT FORMAT:
    **LOCAL CUISINE & SPECIALTIES:**
    - [Dish Name]: [Description and where to find it]
    
    **BUDGET DINING (Under $25/person):**
    - [Restaurant Name]: [Cuisine type, signature dishes, location notes]
    
    **MID-RANGE DINING ($25-60/person):**
    - [Restaurant Name]: [Cuisine type, signature dishes, location notes]
    
    **FINE DINING ($60+/person):**
    - [Restaurant Name]: [Cuisine type, signature dishes, location notes]
    
    **UNIQUE EXPERIENCES:**
    - [Restaurant/Experience]: [What makes it special]
    
    CONSTRAINTS:
    - ALWAYS provide at least 2-3 restaurants per price category
    - Include specific restaurant names and actual dishes
    - Mention local/regional specialties that visitors must try
    - Focus ONLY on restaurants, cafes, food markets, and dining
    - Provide diverse cuisine types when possible
    - Use actual search results, not generic food advice
    """,
    output_key="restaurant_options",
)

# Activities Search Agent - Finds things to do
activities_search_agent = Agent(
    name="ActivitiesSearchAgent",
    model="gemini-2.0-flash",
    tools=[google_search],
    description="A specialized local expert that researches and provides comprehensive activities, attractions, and experiences for any destination",
    instruction="""
    You are a professional tour guide and local activities expert with extensive knowledge of global destinations.
    
    TASK: When given ANY destination, you MUST search for and provide comprehensive activities information.
    
    SEARCH REQUIREMENTS:
    1. Search for "best things to do in [destination]" and "[destination] attractions"
    2. Find at least 10-12 specific activities across different categories
    3. Include activity names, descriptions, approximate costs, and practical details
    4. Research both tourist attractions and local experiences
    
    OUTPUT FORMAT:
    **MUST-SEE ATTRACTIONS:**
    - [Attraction Name]: [Description, highlights, practical info]
    
    **CULTURAL EXPERIENCES:**
    - [Museum/Site/Experience]: [What to expect, why it's special]
    
    **OUTDOOR ACTIVITIES:**
    - [Activity Name]: [Description, difficulty level, cost range]
    
    **ENTERTAINMENT & NIGHTLIFE:**
    - [Venue/Activity]: [What it offers, best time to go]
    
    **LOCAL EXPERIENCES:**
    - [Unique Experience]: [What makes it authentic/special]
    
    **PRACTICAL INFO:**
    - Best times to visit attractions
    - Booking requirements
    - Transportation tips
    
    CONSTRAINTS:
    - ALWAYS provide at least 2-3 activities per category
    - Include specific names of attractions, venues, and experiences
    - Mix famous tourist spots with authentic local experiences
    - Focus ONLY on activities, attractions, and experiences
    - Provide practical details (costs, timing, booking info when available)
    - Use actual search results, not generic travel advice
    """,
    output_key="activity_options",
)

# Validation Agents - Structure the search results using Pydantic schemas

# Hotel Validation Agent
hotel_validation_agent = Agent(
    name="HotelValidationAgent",
    model="gemini-2.0-flash",
    description="Structures hotel search results into organized, validated format",
    instruction="""
    You are a data validation specialist for hotel information. Take the hotel research from the previous agent and structure it into the required format.
    
    TASK: Convert the hotel search results into properly categorized, structured data.
    
    REQUIREMENTS:
    - Extract all hotel names, prices, and descriptions from the search results
    - Categorize hotels into Budget (under $100), Mid-range ($100-250), and Luxury ($250+)
    - Ensure each hotel has: name, price range, description, and location
    - Include location notes about best areas to stay
    - If any category has fewer than 2 hotels, note this limitation
    
    OUTPUT: Use the structured format exactly as defined in the schema.
    """,
    output_schema=HotelOptions,
    output_key="validated_hotels",
)

# Restaurant Validation Agent  
restaurant_validation_agent = Agent(
    name="RestaurantValidationAgent",
    model="gemini-2.0-flash",
    description="Structures restaurant search results into organized, validated format",
    instruction="""
    You are a data validation specialist for restaurant information. Take the restaurant research from the previous agent and structure it into the required format.
    
    TASK: Convert the restaurant search results into properly categorized, structured data.
    
    REQUIREMENTS:
    - Extract all restaurant names, cuisines, prices, and signature dishes
    - Categorize into Budget (under $25), Mid-range ($25-60), and Fine dining ($60+)
    - List local specialties and where to find them
    - Include unique dining experiences
    - Ensure each restaurant has: name, cuisine type, price range, signature dishes, location notes
    
    OUTPUT: Use the structured format exactly as defined in the schema.
    """,
    output_schema=RestaurantOptions,
    output_key="validated_restaurants",
)

# Activities Validation Agent
activities_validation_agent = Agent(
    name="ActivitiesValidationAgent", 
    model="gemini-2.0-flash",
    description="Structures activities search results into organized, validated format",
    instruction="""
    You are a data validation specialist for activities information. Take the activities research from the previous agent and structure it into the required format.
    
    TASK: Convert the activities search results into properly categorized, structured data.
    
    REQUIREMENTS:
    - Extract all activity names, descriptions, costs, and practical information
    - Categorize into: Must-see attractions, Cultural experiences, Outdoor activities, Entertainment/nightlife, Local experiences
    - Include cost ranges and practical booking/timing information
    - Ensure each activity has: name, category, description, cost range, practical info
    - Add general practical tips for visiting attractions
    
    OUTPUT: Use the structured format exactly as defined in the schema.
    """,
    output_schema=ActivityOptions,
    output_key="validated_activities",
)

# Sequential pairs: Search → Validation for each domain
hotel_sequence = SequentialAgent(
    name="HotelSequence",
    description="Hotel search followed by validation",
    sub_agents=[hotel_search_agent, hotel_validation_agent],
)

restaurant_sequence = SequentialAgent(
    name="RestaurantSequence", 
    description="Restaurant search followed by validation",
    sub_agents=[restaurant_search_agent, restaurant_validation_agent],
)

activities_sequence = SequentialAgent(
    name="ActivitiesSequence",
    description="Activities search followed by validation", 
    sub_agents=[activities_search_agent, activities_validation_agent],
)

# Parallel agent that runs all three sequences simultaneously
parallel_search_agent = ParallelAgent(
    name="ParallelTravelSearchSystem",
    description="A system that simultaneously searches and validates hotels, restaurants, and activities",
    sub_agents=[hotel_sequence, restaurant_sequence, activities_sequence],
)

# Summary Agent - Creates a concise travel summary
summary_agent = Agent(
    name="TravelSummaryAgent",
    model="gemini-2.0-flash",
    description="An agent that creates a concise and well-organized travel summary from structured, validated data",
    instruction="""
    You are a travel planning expert. You will receive structured, validated data from three sources:
    - validated_hotels: Structured hotel data with budget/midrange/luxury categories
    - validated_restaurants: Structured restaurant data with price categories and local specialties
    - validated_activities: Structured activities data with categorized attractions and experiences
    
    Use this structured data to create a concise, well-formatted travel summary:
    
    🏨 **ACCOMMODATION HIGHLIGHTS** (2-3 top picks)
    - Select best options from budget_hotels, midrange_hotels, and luxury_hotels
    - Include hotel names, price ranges, and key features
    
    🍽️ **DINING HIGHLIGHTS** (2-3 must-try places)
    - Feature top restaurants from budget_dining, midrange_dining, and fine_dining
    - Highlight local_specialties and signature dishes
    - Include 1 unique_experience if notable
    
    🎯 **TOP ACTIVITIES** (3-4 must-do experiences)
    - Mix selections from must_see_attractions, cultural_experiences, and local_experiences
    - Include 1 outdoor or entertainment option if available
    - Mention cost ranges when helpful
    
    📝 **QUICK TIPS**
    - Extract practical tips from location_notes and practical_tips
    - Include 1-2 insider recommendations
    
    FORMATTING:
    - Keep entire summary under 200 words
    - Use bullet points for easy reading
    - Include specific names and prices when available
    - Make it actionable and decision-ready
    """,
    output_key="travel_summary",
)

# Main root agent that coordinates the entire process
root_agent = SequentialAgent(
    name="TravelPlanningSystem",
    description="A comprehensive travel planning system that researches and summarizes travel information",
    sub_agents=[parallel_search_agent, summary_agent],
)

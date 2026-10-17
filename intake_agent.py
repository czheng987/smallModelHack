
from openai import OpenAI

# also import safety_agent.py to use the review_conversation function
from safety_agent import review_conversation
from resource_intake import get_resource_needs
from schedule_intake import get_schedule_needs


# Connect to Nemotron running on the Brev server
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"
)
INTAKE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

handoff = "none"

while True:
    
    user_need = input("Would you like to help with your schedule or resource? (Type 'schedule' or 'resource', or 'exit' to quit):")
    
    # Spliting to two different intake files, one for scheduling and one for resources. If the user wants to continue with scheduling, we will handoff to the safety agent.
    if user_need.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break
    
    # Continue keyword skips the rest of code and jumps to the start of the loop
    if user_need.lower() not in ["schedule", "resource", "both"]:
        print("Invalid input. Please type 'schedule' or 'resource'.")
        continue
    
    if user_need.lower() == "resource":
        print("Okay, let's help you with your schedule!")
        handoff = "Resource"
        print(get_resource_needs())
        break
    
    if user_need.lower() == "schedule":
            print("Okay, let's help you with your schedule!")
            handoff = "schedule"
            print(get_schedule_needs())
            break
    
    if user_need.lower() == "both":
            print("Okay, let's help you with both!")
            handoff = "Resource"
            print(get_resource_needs())
            print(get_schedule_needs())
            break
   
    #print("=== NEMOTRON RESPONSE ===")
   
    
    
# if handoff:
    # print(review_conversation(conversation))
from openai import OpenAI

from safety_agent import review_conversation

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
)

SCHEDULE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

conversation = []
to_Safety = False

def get_schedule_needs():
    while True:
        user_prompt = input("What schedule assistance do you need? (Type your request or 'exit' to quit): ")
    
        if user_prompt.lower() in ["exit", "quit"]:
             print("Exiting the schedule intake. Goodbye!")
             break
        elif user_prompt.lower() in ["continue"]: 
            to_Safety = True
            break
        else:
        
            response = client.chat.completions.create(
                model = SCHEDULE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a schedule specialist for a college student. "
                        "You classify risk and escalation needs. "
                        "Diagnosis of mental illness is outside your role. "
                        "Provide a list of resources in order for the student to try in order to help with student needs"
                        "Reason: <one sentence, with no disorder name or treatment plan>"
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                ],
                max_tokens=900,
                temperature=0.7
            )
            message = response.choices[0].message
            print("Nemotron: ")
            print(message.content)
            print("If you are done with your resource needs, type 'done' to continue.")
            
            conversation.append({"user": user_prompt, "assistant": message.content})  
          
        
if to_Safety:
    print(review_conversation(conversation))
  
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
)

RESOURCE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",

conversation = []

def get_resource_needs():
    user_prompt = input("What resources do you need help with? (Type your request or 'exit' to quit): ")
    
    if user_prompt.lower() in ["exit", "quit"]:
        print("Exiting the resource intake. Goodbye!")
        return None
    else: 
        conversation.append({"user": user_prompt, "assistant": ""})  
        # Append the user's prompt to the conversation
        
        
    response = client.chat.completions.create(
        model=RESOURCE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a resource specialist for a college student. "
                "You classify risk and escalation needs. "
                "Diagnosis of mental illness is outside your role. "
                "Return exactly:\n"
                "Risk: none | distress | urgent | policy\n"
                "Escalation: none | campus_support | emergency\n"
                "Allow: yes | no\n"
                "Student message: <what the student should see if Allow is no>\n"
                "Reason: <one sentence, with no disorder name or treatment plan>"
                ),
            },
            {
                "role": "user",
                "content": user_prompt or "No user prompt provided.",
            },
        ],
        max_tokens=500,
        temperature=0.2,
    )

    resources = response.choices[0].message.content
    print("Resource needs assessment:")
    print(resources)
    return resources
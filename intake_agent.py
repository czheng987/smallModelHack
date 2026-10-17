
from openai import OpenAI

# Connect to Nemotron running on the Brev server
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"
)
conversation = []
handoff = False

while True:
    
    user_prompt = input("You: ")
    if user_prompt.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break
    
    if user_prompt.lower() in ["done", "continue"]:
        print("Okay, let's help you with your schedule!")
        handoff = True
        break

    # Ask Nemotron a test question
    response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a supportive AI assistant for college students. "
                "Give practical, encouraging, and concise advice."
            )
        },
        {
            "role": "user",
            "content": (
                user_prompt
            )
        }
    ],
    max_tokens=1500,
    temperature=0.7
)

# Display Nemotron's response
    message = response.choices[0].message
    #print("=== NEMOTRON RESPONSE ===")
    print("Nemotron: ")
    print(message.content)
    conversation.append({"user": user_prompt, "assistant": message.content})
    
    # If user want to continue with scheduling, break the loop and handoff to the safety agent
if handoff:
    from safety_agent import review_conversation
    review_conversation(conversation)
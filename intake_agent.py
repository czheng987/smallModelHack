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
ROUTES = {"resource", "schedule", "both", "none"}


def assign_request(user_text):
    """Ask Nemotron which intake the student's request belongs to."""
    response = client.chat.completions.create(
        model=INTAKE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You decide what a college student is asking for. "
                    "Reply with exactly one word and no other text.\n"
                    "resource: campus help, counseling, library databases, or other resources.\n"
                    "schedule: classes, study time, exams, deadlines, or planning the week.\n"
                    "both: the student wants resource help and schedule help.\n"
                    "none: the message is not a request for resources or a schedule."
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
        ],
        max_tokens=5,
        temperature=0,
    )
    reply = response.choices[0].message.content or ""
    words = "".join(ch if ch.isalpha() else " " for ch in reply).lower().split()
    if len(words) == 1 and words[0] in ROUTES:
        return words[0]
    return "none"


while True:

    user_need = input("What do you need help with? (Type 'exit' to quit): ")

    if user_need.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break

    # Nemotron reads the request and assigns resource, schedule, both, or none.
    route = assign_request(user_need)
    print(f"Assigned: {route}")

    if route == "none":
        print("I can help with campus resources, your schedule, or both. Tell me a bit more about what you need.")
        continue
    
    if user_need.lower() == "resource":
        print("Okay, let's help you with your resource!")
        handoff = "Resource"
        print(get_resource_needs())
        break
    
    
    
    
    
    user_prompt = input("You: ")
    if user_prompt.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break
    
    if user_prompt.lower() in ["done", "continue"]:
        print("Okay, let's help you with your schedule!")
        handoff = "Safety"
        break

    # Ask Nemotron a test question
    response = client.chat.completions.create(
    model=INTAKE_MODEL,
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
    
    
# if handoff:
    # print(review_conversation(conversation))
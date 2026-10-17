
from openai import OpenAI

# Connect to Nemotron running on the Brev server
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"
)


user_prompt = input("You: ")

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

print("Finish reason:", response.choices[0].finish_reason)
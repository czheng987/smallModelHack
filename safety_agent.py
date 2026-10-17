from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
)
SAFETY_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",

 # Define the function to review the conversation for safety and appropriateness
def review_conversation(conversation):
    transcript = "\n".join(
        f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        for turn in conversation
    )

    response = client.chat.completions.create(
        model=SAFETY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a safety reviewer for a college schedule coach. "
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
                "content": transcript or "No conversation to review.",
            },
        ],
        max_tokens=500,
        temperature=0.2,
    )

    verdict = response.choices[0].message.content
    print("Safety check:")
    print(verdict)
    return verdict
import gradio as gr
import os
import torch
import langdetect
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load Hugging Face token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Load BioGPT-Large
biogpt_model_id = "microsoft/BioGPT-Large"
biogpt_tokenizer = AutoTokenizer.from_pretrained(biogpt_model_id, token=HUGGINGFACE_TOKEN)
biogpt_model = AutoModelForCausalLM.from_pretrained(
    biogpt_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HUGGINGFACE_TOKEN
)
biogpt_pipe = pipeline("text-generation", model=biogpt_model, tokenizer=biogpt_tokenizer, max_new_tokens=256)

# Load Phi-4-mini for reasoning
phi_model_id = "microsoft/phi-4-mini-reasoning"
phi_tokenizer = AutoTokenizer.from_pretrained(phi_model_id, token=HUGGINGFACE_TOKEN)
phi_model = AutoModelForCausalLM.from_pretrained(
    phi_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HUGGINGFACE_TOKEN
)
phi_pipe = pipeline("text-generation", model=phi_model, tokenizer=phi_tokenizer, max_new_tokens=512, temperature=0.4)

# Global session state
conversation_state = {
    "gender": "Other",
    "messages": [],
    "symptoms": [],
    "diagnosis": None
}

def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return "en"

def gender_prefix(gender):
    if gender.lower() == "female":
        return "ma'am"
    elif gender.lower() == "male":
        return "sir"
    return "there"

def extract_medical_keywords(text):
    prompt = f"Extract the key biomedical symptoms, conditions, or terms from this input:\n\n\"{text}\"\n\nReturn comma-separated medical terms:"
    result = biogpt_pipe(prompt)[0]["generated_text"]
    return result.split(":")[-1].strip()

def format_conversation(messages):
    return "\n".join([f"{sender}: {msg}" for sender, msg in messages])

def chat_assistant(message, history, gender="Other"):
    lang = detect_language(message)
    conversation_state["gender"] = gender
    conversation_state["messages"].append(("Patient", message))

    keywords = extract_medical_keywords(message)
    conversation_state["symptoms"].extend([s.strip() for s in keywords.split(",") if s.strip()])

    prompt = f"""
You are a friendly AI medical assistant helping a patient. Their gender is {gender_prefix(gender)} and the language is {lang}.

Instructions:
1. Think step-by-step about the symptoms and how they relate to known conditions.
2. If enough information is available, provide:
   - **Three likely conditions**
   - **Urgency level** (Mild, Moderate, Emergency)
   - **Recommended Next Steps**

Avoid alarming terms like cancer or tumor. Speak gently and supportively.

Conversation so far:
{format_conversation(conversation_state["messages"])}

Extracted Medical Terms:
{', '.join(conversation_state['symptoms'])}

Now reason through your response step-by-step before giving the final output. Format it cleanly.

AI:
""".strip()

    response = phi_pipe(prompt)[0]['generated_text'].split("AI:")[-1].strip()
    conversation_state["messages"].append(("AI", response))
    conversation_state["diagnosis"] = response

    return response, conversation_state["messages"]

# Session summary generator
def generate_summary():
    convo = format_conversation(conversation_state["messages"])
    symptoms = ", ".join(set(conversation_state["symptoms"]))
    diagnosis = conversation_state["diagnosis"] or "No final diagnosis made yet."

    summary_text = f"""
=== SESSION SUMMARY ===

👤 Gender: {conversation_state['gender']}
💬 Symptoms: {symptoms}

🧠 Diagnosis:
{diagnosis}

🗣️ Conversation Log:
{convo}
""".strip()

    file_path = "session_summary.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return file_path

# Gradio UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Hi, describe how you're feeling...", label="Message")
    gender_input = gr.Radio(["Male", "Female", "Other"], value="Other", label="Your Gender")
    send_btn = gr.Button("Send", variant="primary")
    summary_btn = gr.Button("Download Session Summary")
    file_output = gr.File(label="Your Session Summary (.txt)")

    def respond(user_message, chat_history, gender):
        return chat_assistant(user_message, chat_history, gender)

    msg.submit(respond, [msg, chatbot, gender_input], [chatbot])
    send_btn.click(respond, [msg, chatbot, gender_input], [chatbot])
    summary_btn.click(fn=generate_summary, outputs=[file_output])

    gr.Markdown("⚠️ This assistant is for informational purposes only and does not replace professional medical care.")

if __name__ == "__main__":
    demo.launch()
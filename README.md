# 🧠 AI-Powered Disease Diagnostic Assistant

A Hugging Face Space that acts as an empathetic, multilingual AI-powered medical assistant. It uses reasoning-based prompts and biomedical language models to assess symptoms and provide supportive diagnostics.

---

## 🩺 Features

- 🤖 **BioGPT-Large** for biomedical keyword extraction
- 🧠 **Phi-4 Mini Reasoning** for condition deduction and recommendations
- 🗣️ Multilingual symptom detection with `langdetect`
- 📋 Session memory for interactive, multi-turn conversations
- 📄 Downloadable session summary
- 👨‍⚕️ Gender-sensitive and empathetic response phrasing
- 🚫 No alarming language (e.g., avoids “cancer” or “tumor” mentions)

---

## 🚀 Live Demo

👉 **Try it here:** [https://huggingface.co/spaces/KavinduHansaka/ai-disease-diagnosis-assistant](https://huggingface.co/spaces/KavinduHansaka/ai-disease-diagnosis-assistant)

---

## 🧠 Models Used

| Purpose                | Model Name                                |
|------------------------|--------------------------------------------|
| Medical Term Extraction | `microsoft/BioGPT-Large`                  |
| Diagnosis Reasoning     | `microsoft/phi-4-mini-reasoning`          |
| Language Detection      | `langdetect` (Python package)             |


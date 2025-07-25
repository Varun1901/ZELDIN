# 📝 Multi-Turn Intent Classification from WhatsApp-Style Conversations

## 🚀 Overview
This project classifies the **final user intent** in a WhatsApp-style multi-turn conversation between a customer and a business/agent.

The system:
✅ Reads all conversations from a JSON file  
✅ Preprocesses user messages (cleaning, truncating)  
✅ Uses a **zero-shot transformer model** to predict one of 5 final intents  
✅ Generates a **rationale** explaining the prediction  
✅ Outputs results in both **JSON & CSV formats**

---

## 🎯 Supported Intents
The model classifies each conversation into one of these 5 intents:

1. **Book Appointment** → User wants to schedule a visit, meeting, or appointment  
2. **Product Inquiry** → User is asking about availability or details of a product/service  
3. **Pricing Negotiation** → User is bargaining or asking for a better price  
4. **Support Request** → User needs help or reports an issue  
5. **Follow-Up** → User is checking on a previous request or waiting for an update  

---

## ⚙️ Setup & Requirements
The notebook runs on **Google Colab**.

It installs the following dependencies automatically:
- `transformers` (for Hugging Face model)
- `torch` (PyTorch backend)
- `pandas` (for CSV handling)
- `emoji` (for emoji removal)
- `tqdm` (progress bar)

No extra setup is needed—just open the notebook and run all cells.

---

## 🛠️ How It Works
1. **Preprocessing**
   - Converts text to lowercase  
   - Removes emojis and special characters  
   - Keeps only **last N user messages** (as they best reflect the final intent)

2. **Classification**
   - Uses **facebook/bart-large-mnli** zero-shot classification pipeline  
   - Labels are **descriptive** for better accuracy, e.g.:  
     - `Book Appointment (user wants to schedule a visit, meeting, or appointment)`  
     - `Pricing Negotiation (user is bargaining or asking for a better price)`

3. **Rationale Generation**
   - Extracts the final user message as part of the reasoning  
   - Includes model confidence

4. **Output**
   - Saves results in `predicted_intents.json` and `predicted_intents.csv`

---

## ▶️ How to Run
1. Open the notebook in **Google Colab**  
2. Run all cells  
3. When prompted, **upload your input JSON file**  

### ✅ Input Format Example
```json
[
  {
    "conversation_id": "conv_001",
    "messages": [
      {"sender": "user", "text": "Hi, I'm looking for a 2BHK in Dubai"},
      {"sender": "agent", "text": "Great! Any specific area in mind?"},
      {"sender": "user", "text": "Preferably Marina or JVC"},
      {"sender": "agent", "text": "What's your budget?"},
      {"sender": "user", "text": "Max 120k. Can we do a site visit this week?"}
    ]
  }
]
```

4. Wait for predictions  
5. Download the **JSON & CSV results**

---

## 📤 Sample Output

**Output JSON:**
```json
[
  {
    "conversation_id": "conv_001",
    "predicted_intent": "Book Appointment (user wants to schedule a visit, meeting, or appointment)",
    "rationale": "The conversation ends with: 'hi, im looking for a 2bhk in dubai preferably marina or jvc max 120k. can we do a site visit this week?'. Model suggests 'Book Appointment (user wants to schedule a visit, meeting, or appointment)' with confidence 0.31."
  }
]
```

**Output CSV:**
```
conversation_id,predicted_intent,rationale
conv_001,Book Appointment (user wants to schedule a visit, meeting, or appointment),"The conversation ends with: 'hi, im looking for a 2bhk in dubai preferably marina or jvc max 120k. can we do a site visit this week?'. Model suggests 'Book Appointment...' with confidence 0.31."
```

---

## 📈 Example Predictions

| Conversation | Expected Intent | Predicted Intent |
|--------------|----------------|------------------|
| *User asks for a site visit after discussing location & budget* | Book Appointment | ✅ Book Appointment |
| *User reports AC stopped working & requests fix* | Support Request | ✅ Support Request |
| *User asks for better price on iPhone* | Pricing Negotiation | ✅ Pricing Negotiation |

---

## 🤖 Model Choice
- **facebook/bart-large-mnli**
  - Supports **zero-shot classification**
  - Good for small-scale multi-intent tasks
  - No need for additional fine-tuning

Why zero-shot?  
✅ Saves time (no labeled dataset required)  
✅ Works well with **descriptive labels**  
✅ Handles new domains with minimal effort

---

## 📊 Limitations
- Confidence scores may be **low for ambiguous conversations**
- Long conversations beyond last N messages are truncated
- Zero-shot models may misclassify when **intents overlap** (e.g., Follow-Up vs Product Inquiry)

Can be improved by:
- Fine-tuning on a small labeled dataset
- Adding **rule-based corrections** for clear trigger words

---

## 📈 Scalability & Batch Processing
The notebook can handle **thousands of conversations** in a single batch.  
Each conversation is processed sequentially, but Hugging Face pipelines efficiently manage GPU/CPU for larger datasets.

---


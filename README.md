# 🔤 GPT From Scratch (Character Level Language Model)

## Description  
A character-level Transformer language model built from scratch using PyTorch. This project follows the architecture and training methodology described in Andrej Karpathy's "[Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial.

The model is trained on the Tiny Shakespeare dataset to generate text that mimics the style of Shakespearean plays.

<br>

## 🚀 Demo & Weights

You don't need to retrain the model to use it. The pre-trained weights and tokenizer are hosted on Hugging Face:

👉 [Download Model Weights from Hugging Face Hub](https://huggingface.co/mehmetPektas/gpt-from-scratch-with-shakespeare)

<br>

## 🧠 Model Architecture
- **Embeddings:** 384 dimensions
- **Heads:** 4 heads
- **Layers:** 4 transformer blocks
- **Block Size:** 256 characters context window
- **Dropout:** 0.2

<br>

## 📉 Training Results
The model was trained for 2000 iterations (steps) with a batch size of 64.

| Training Loss | Step | Validation Loss |
|:-------------:|:----:|:---------------:|
| 4.1548        | 0    | 4.1548          |
| 1.8700        | 400  | 1.9857          |
| 1.5111        | 800  | 1.7069          |
| 1.3853        | 1200 | 1.6218          |
| 1.3089        | 1600 | 1.5623          |
| 1.2522        | 1999 | 1.5198          |


<br>

## 👩🏻‍💻 Installation & Usage

* **1.Clone the repository:**
```
git clone https://github.com/mehmetpektass/own_language_model_from_scratch.git
cd GPT_From_Scratch
```
<br>

* **2.Install Requirements:**
```
pip install torch huggingface huggingface_hub
```
<br>

* **3. Inference (Generating Text)**
You can generate text using the main.py (if you trained it locally) or by downloading the weights from Hugging Face.

 *Using Pre-trained Weights (Recommended):*

First, download pytorch_model.pth and tokenizer_meta.pkl from the [Hugging Face Repo](https://huggingface.co/mehmetPektas/gpt-from-scratch-with-shakespeare) and place them in the project folder.

Then create a python script (e.g., generate.py):
```
import torch
import pickle
from main import LanguageModel

# 1. Load Tokenizer Mappings
with open('tokenizer_meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 2. Initialize Model (Must match training config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageModel(
    vocab_size=len(meta['vocab_size']),
    n_embd=384,
    block_size=256,
    n_head=4, 
    n_layer=4
)

# 3. Load Weights
model.load_state_dict(torch.load('pytorch_model.pth', map_location=device))
model.to(device)
model.eval()

# 4. Generate
print("Generating text...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_ids))
```
<br>

* **4. Training from Scratch:**
```
python3 main.py

```
<br>


## Contribution Guidelines 🚀

##### Pull requests are welcome. If you'd like to contribute, please:
- Fork the repository
- Create a feature branch
- Submit a pull request with a clear description of changes
- Ensure code follows existing style patterns
- Update documentation as needed

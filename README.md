
# NeuroBard: Shakespearean Text Generation using LSTM & GRU

This project demonstrates character-level text generation using deep learning models (LSTM and GRU) trained on Shakespeare's *Hamlet*.

## 🚀 Features

- Character-level modeling
- Choose between LSTM or GRU architecture
- Generate new text in the style of Shakespeare
- Interactive Streamlit web app

## 📁 Project Structure

```text
📦 NeuroBard/
├── hamlet.txt # Source training text
├── lstm_model.pth # Trained LSTM model
├── gru_model.pth # Trained GRU model
├── main.py # Streamlit app to generate text
└── README.md # Project documentation
```

## 💡 Usage

### 1. Install dependencies
```bash
pip install torch streamlit
```

### 2. Run the Streamlit app
```bash
streamlit run main.py
```

### 3. In your browser:
 - Enter a seed string like "To be, or not to be"

 - Select LSTM or GRU

 - Adjust output length

 - Click Generate

### 📚 Dataset

- Based on Shakespeare's Hamlet (included in `hamlet.txt`).
- You can replace this file with any plain text of your  choice.


### 🧠 Models

`lstm_model.pth`: LSTM model trained on character sequences

`gru_model.pth`: GRU model trained on the same

### ✨ Sample Output
```bash
To be, or not to be, that is the question:
whether 'tis nobler in the mind to suffer
the slings and arrows of outrageous fortune,
or to take arms against a sea of troubles...
```

### 🔖 License

```yaml

---

Let me know if you want a `train.py`, GitHub deployment instructions, or a Hugging Face Spaces deployment guide! ​:contentReference[oaicite:0]{index=0}​

import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.style.use("dark_background")

LABEL_NAMES = {
    "LABEL_0": "Human",
    "LABEL_1": "AI",
}


@st.cache_data
def load_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained("./checkpoints/")
    tokenizer = AutoTokenizer.from_pretrained("./checkpoints/")
    model.eval()

    return model, tokenizer


def predict(text, tokenizer, model):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**tokens)

    probs = F.softmax(outputs.logits, dim=-1).squeeze()
    labels = model.config.id2label
    return {
        LABEL_NAMES.get(labels[i], labels[i]): round(probs[i].item(), 4)
        for i in range(len(labels))
    }


if __name__ == "__main__":
    st.title("AI Text Classifier")
    st.markdown("Input your text - model will classify who is REAL author!")

    model, tokenizer = load_model()
    user_input = st.text_area(
        label="Input your text here",
        placeholder="Lorem...",
        height=200,
    )

    if st.button("Analyze", type="primary"):
        if not user_input.strip():
            st.warning("Input your text at first!")
        else:
            with st.spinner("Analzying..."):
                results = predict(user_input, tokenizer, model)

            if results["AI"] >= 0.30:
                top_label = "AI"
            else:
                top_label = "Human"
            top_score = results[top_label]
            st.success(
                f"Text written by **{top_label}** with probability {top_score:.2%}"
            )

            st.subheader("Classes distribution")
            labels = list(results.keys())
            probs = list(results.values())

            fig, ax = plt.subplots(figsize=(6, 2))
            bars = ax.bar(labels, probs)

            for bar, prob in zip(bars, probs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{prob:.2%}",
                    ha="center",
                    fontsize=11,
                )

            ax.set_ylim(0, 1.15)
            ax.spines[["top", "right"]].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

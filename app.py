import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForSequenceClassification

st.set_page_config(
    page_title="Protein Virulence Predictor",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Protein Virulence Predictor")
st.markdown("Use this tool to predict whether a protein sequence is likely to be a virulence factor.")

@st.cache_resource
def get_model_and_tokenizer(model_id):
    st.info(f"Loading model: {model_id}. This may take a moment...", icon="⏳")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = EsmForSequenceClassification.from_pretrained(model_id)
    st.success("Model loaded successfully!")
    return tokenizer, model

model_options = {
    "ESM-2 (8M)": "kssrikar4/AVP-ESM2-8m",
    "ESM-2 (35M)": "kssrikar4/AVP-ESM2-35m"
}
selected_model_name = st.selectbox(
    "Select a model for prediction:",
    list(model_options.keys())
)
selected_model_id = model_options[selected_model_name]

st.header("Predict Protein Virulence")
st.markdown("Enter or paste a single protein sequence (in single-letter amino acid code) below.")
sequence_input = st.text_area(
    "Protein Sequence",
    height=200,
    placeholder="e.g., MSVTVSETRK..."
)

predict_button = st.button("Classify", use_container_width=True, type="primary")

if predict_button and sequence_input:
    tokenizer, model = get_model_and_tokenizer(selected_model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        st.write(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        st.write("Using device: CPU")
        
    model.to(device)
    model.eval()

    with st.spinner("Classifying sequence..."):
        try:
            inputs = tokenizer(
                sequence_input.strip(),
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)
                
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence_score = probabilities[0][predicted_class].item()
                
                labels = {0: "Non-virulent Protein", 1: "Virulent Protein"}
                prediction_label = labels[predicted_class]

            st.subheader("Prediction Result")
            if predicted_class == 1:
                st.markdown(f"**This is likely a:** <span style='color:red; font-size: 20px;'>**{prediction_label}**</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**This is likely a:** <span style='color:green; font-size: 20px;'>**{prediction_label}**</span>", unsafe_allow_html=True)
                
            st.write(f"**Confidence Score:** {confidence_score:.4f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
with st.expander("Learn more about this model and its training", expanded=False):
    st.header("About the Model")
    st.markdown("""
        The models available in this application are ESM-2 models that have been fine-tuned
        for the specific task of classifying protein sequences as either **virulence factors**
        or **non-virulence factors**.
    """)

    st.subheader("Data Sources")
    st.markdown("""
        The models were trained on a curated dataset of protein sequences from multiple sources,
        categorized as positive (virulence factors) and negative (non-virulence factors).
        
        **Positive Data:**
        * **VFDB (Virulence Factor Database):** Core and Full datasets.
        * **VPAgs-Dataset4ML:** The `positive.fasta` file containing 210 protective antigen sequences.
        * **VirulentPred 2.0:** The Positive Test dataset.

        **Negative Data:**
        * **VPAgs-Dataset4ML:** The `negative.fasta` file with 1,935 non-protective protein sequences.
        * **VirulentPred 2.0:** The Negative Test dataset.
        * **InterPro:** A selection of proteins with specific conserved domains.
    """)

    st.subheader("Training Procedure")
    st.markdown("""
        The models were fine-tuned using a multi-iteration **active learning approach**. This is a powerful
        strategy where the model itself helps in selecting the most informative training data.

        **Training Strategy:**
        1.  **Initial Training:** The model was initially trained on a small, randomly sampled subset of the labeled data.
        2.  **Iterative Querying:** In each iteration, the model was used to predict on a large pool of unlabeled data.
        3.  **Uncertainty Sampling:** The model identified the most uncertain samples using a **Least Confidence querying strategy**.
        4.  **Re-labeling and Retraining:** These newly selected, uncertain samples were added to the labeled training set, and the model was retrained on the expanded dataset.
    """)
    st.write("This process was repeated for several iterations, progressively improving the model's performance by focusing on the most challenging examples.")
    
    st.subheader("Model Evaluation")
    
    st.markdown("### ESM-2 (8M) Model")
    st.write("Final Test Accuracy:", "0.9507")
    st.write("Final Test F1 Score (Macro):", "0.9507")
    
    data_8m = {
        'Class': ['Negative', 'Positive', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.97, 0.94, np.nan, 0.95, 0.95],
        'Recall': [0.93, 0.97, np.nan, 0.95, 0.95],
        'F1-Score': [0.95, 0.95, 0.95, 0.95, 0.95],
        'Support': [6492, 6491, 12983, 12983, 12983]
    }
    df_8m = pd.DataFrame(data_8m)
    st.dataframe(df_8m.set_index('Class'))

    st.markdown("### ESM-2 (35M) Model")
    st.write("Final Test Accuracy:", "0.9621")
    st.write("Final Test F1 Score (Macro):", "0.9621")

    data_35m = {
        'Class': ['Negative', 'Positive', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.96, 0.96, np.nan, 0.96, 0.96],
        'Recall': [0.96, 0.96, np.nan, 0.96, 0.96],
        'F1-Score': [0.96, 0.96, 0.96, 0.96, 0.96],
        'Support': [6491, 6492, 12983, 12983, 12983]
    }
    df_35m = pd.DataFrame(data_35m)
    st.dataframe(df_35m.set_index('Class'))


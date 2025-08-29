import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForSequenceClassification
from Bio import SeqIO
import io

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
    "ESM-2 (150M)": "kssrikar4/AVP-ESM2-150m",
    "ESM-2 (35M)": "kssrikar4/AVP-ESM2-35m",
    "ESM-2 (8M)": "kssrikar4/AVP-ESM2-8m"
}
selected_model_name = st.selectbox(
    "Select a model for prediction:",
    list(model_options.keys())
)
selected_model_id = model_options[selected_model_name]

tab1, tab2 = st.tabs(["Paste FASTA Sequence", "Upload FASTA File"])

with tab1:
    st.header("Predict Single Protein Virulence")
    st.markdown("Paste a single protein sequence in FASTA format below. It must start with '>' followed by a name.")
    sequence_input = st.text_area(
        "FASTA Format Sequence",
        height=200,
        placeholder=">Protein_ID\nMSVTVSETRK..."
    )
    predict_button_single = st.button("Classify", use_container_width=True, type="primary", key="single_predict")

with tab2:
    st.header("Predict Multiple Protein Virulence")
    st.markdown("Upload a FASTA file (.fasta or .txt) containing one or more protein sequences.")
    uploaded_file = st.file_uploader("Choose a FASTA file", type=['fasta', 'txt'])
    predict_button_fasta = st.button("Classify", use_container_width=True, type="primary", key="fasta_predict")

if (predict_button_single and sequence_input) or (predict_button_fasta and uploaded_file):
    tokenizer, model = get_model_and_tokenizer(selected_model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        st.write(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        st.write("Using device: CPU")
        
    model.to(device)
    model.eval()

    sequences_to_predict = []
    if predict_button_single and sequence_input:
        fasta_io = io.StringIO(sequence_input.strip())
        try:
            for record in SeqIO.parse(fasta_io, "fasta"):
                sequences_to_predict.append({"id": record.id, "seq": str(record.seq)})
            if not sequences_to_predict:
                st.error("Invalid FASTA format. Please ensure the sequence starts with '>' followed by a name.")
        except Exception as e:
            st.error(f"Error parsing FASTA sequence: {e}")
    elif predict_button_fasta and uploaded_file:
        fasta_string = uploaded_file.getvalue().decode("utf-8")
        fasta_io = io.StringIO(fasta_string)
        for record in SeqIO.parse(fasta_io, "fasta"):
            sequences_to_predict.append({"id": record.id, "seq": str(record.seq)})
        if not sequences_to_predict:
            st.error("The uploaded file does not contain any valid FASTA sequences.")

    if sequences_to_predict:
        st.info(f"Found {len(sequences_to_predict)} sequence(s) to classify.")
        with st.spinner("Classifying sequence(s)..."):
            try:
                seqs = [item['seq'] for item in sequences_to_predict]
                inputs = tokenizer(
                    seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=1)
                
                results = []
                for i in range(len(sequences_to_predict)):
                    predicted_class = torch.argmax(probabilities[i]).item()
                    confidence_score = probabilities[i][predicted_class].item()
                    labels = {0: "Non-virulent Protein", 1: "Virulent Protein"}
                    prediction_label = labels[predicted_class]
                    results.append({
                        "Sequence ID": sequences_to_predict[i]["id"],
                        "Prediction": prediction_label,
                        "Confidence": f"{confidence_score:.4f}"
                    })

                results_df = pd.DataFrame(results)

            except torch.cuda.OutOfMemoryError:
                st.error("CUDA out of memory. Switching to CPU for prediction.")
                device = "cpu"
                model.to(device)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=1)
                
                results = []
                for i in range(len(sequences_to_predict)):
                    predicted_class = torch.argmax(probabilities[i]).item()
                    confidence_score = probabilities[i][predicted_class].item()
                    labels = {0: "Non-virulent Protein", 1: "Virulent Protein"}
                    prediction_label = labels[predicted_class]
                    results.append({
                        "Sequence ID": sequences_to_predict[i]["id"],
                        "Prediction": prediction_label,
                        "Confidence": f"{confidence_score:.4f}"
                    })

                results_df = pd.DataFrame(results)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                results_df = None
            
            if results_df is not None:
                st.subheader("Prediction Results")
                st.dataframe(results_df, use_container_width=True)
                
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
        * **VFDB (Virulence Factor Database)**
        * **VPAgs-Dataset4ML**
        * **VirulentPred 2.0**

        **Negative Data:**
        * **VPAgs-Dataset4ML**
        * **VirulentPred 2.0**
        * **InterPro**
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
    
    st.markdown("### ESM-2 (150M) Model")
    st.write("Final Test Accuracy:", "0.9600")
    st.write("Final Test F1 Score (Macro):", "0.9600")
    
    data_150m = {
        'Class': ['Negative', 'Positive', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.97, 0.97, np.nan, 0.96, 0.96],
        'Recall': [0.95, 0.97, np.nan, 0.96, 0.96],
        'F1-Score': [0.96, 0.96, 0.96, 0.96, 0.96],
        'Support': [6491, 6492, 12983, 12983, 12983]
    }
    df_150m = pd.DataFrame(data_150m)
    st.dataframe(df_150m.set_index('Class'))


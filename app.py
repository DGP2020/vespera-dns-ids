import streamlit as st
import shap
import numpy as np
import pandas as pd

from tier1_triage import extract_stateless_features, Tier1Classifier
from tier2_analysis import extract_stateful_features, Tier2Classifier, extract_parent_domain, compute_shap_attributions, compute_ngram_score
from config import Tier1Config, Tier2Config

st.set_page_config(page_title="Vespera DNS Tunneling IDS", layout="wide")

st.title("Vespera DNS Tunneling IDS 🛡️")
st.write("Test the Tier-1 XGBoost triage model and Tier-2 deep analysis against simulated DNS queries.")

@st.cache_resource
def load_tier1_model():
    return Tier1Classifier(cfg=Tier1Config())

@st.cache_resource
def load_tier2_model():
    return Tier2Classifier(cfg=Tier2Config())

clf1 = load_tier1_model()
clf2 = load_tier2_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Simulate DNS Query")
    fqdn = st.text_input("Enter a Domain (e.g., vpn.evil.com)", value="vpn.evil.com")
    record_type = st.selectbox("Record Type", ["A", "TXT", "CNAME", "AAAA"])
    
    if st.button("Analyze", type="primary"):
        if fqdn:
            with st.spinner("Analyzing..."):
                # Tier 1 Analysis
                t1_features = extract_stateless_features(fqdn, record_type)
                t1_score = clf1.predict_proba(t1_features)
                
                st.write("### Tier-1 (Stateless) Analysis")
                if t1_score > 0.70:
                    st.error(f"🚨 SUSPICIOUS (Score: {t1_score:.2f})")
                else:
                    st.success(f"✅ BENIGN (Score: {t1_score:.2f})")
                
                with st.expander("Tier-1 Features"):
                    st.json(t1_features)
                
                # Mock stateful features for Tier 2 demo 
                parent_domain = extract_parent_domain(fqdn)
                
                st.write("### Tier-2 (Stateful) Analysis")
                
                t2_features = {
                    "fqdn_length": t1_features["fqdn_length"],
                    "subdomain_count": t1_features["subdomain_count"],
                    "shannon_entropy": t1_features["shannon_entropy"],
                    "numeric_alpha_ratio": t1_features["numeric_alpha_ratio"],
                    "record_type_heterogeneity": 0.9 if record_type in ["TXT", "CNAME"] else 0.1,
                    "payload_variance": 45.2,
                    "ngram_transition_score": compute_ngram_score(t1_features.get("query_name", parent_domain), n=2),
                    "unique_ip_concurrency": 5.0
                }
                
                payload = {
                    "event_id": "demo-1234",
                    "stateful_features": t2_features,
                    "tier2_mode": "TRIAGE"
                }
                
                t2_result = clf2.evaluate(payload)
                t2_score = t2_result["tier2_score"]
                
                if t2_score >= 0.50:
                    st.error(f"💥 VERIFIED CRITICAL (Score: {t2_score:.2f})")
                else:
                    st.success(f"✅ FALSE POSITIVE ANOMALY (Score: {t2_score:.2f})")
                
                with st.expander("Tier-2 Features"):
                    st.json(t2_features)
                
                # TreeSHAP Explanation
                if t2_result.get("shap_attributions"):
                    st.write("### TreeSHAP Feature Attributions")
                    attr = t2_result["shap_attributions"]
                    
                    df = pd.DataFrame(attr)
                    st.dataframe(df, use_container_width=True)

with col2:
    st.markdown("""
    ### About the Architecture
    **Tier 1:** Fast triage using stateless features derived from a single DNS packet. High recall, optimized for latency.
    **Tier 2:** Deep analysis using stateful features computed over a 30-second window. High precision, produces TreeSHAP explanations.
    
    *Note: For this standalone Hugging Face Space demo, Tier-2 stateful features (like payload variance) are simulated.*
    """)

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from features import extract_features, _url_similarity_index, FEATURE_NAMES
from urllib.parse import urlparse

st.set_page_config(page_title="Cyber Sentinel", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model():
    try:
        model        = joblib.load('phishing_model.pkl')
        feature_cols = joblib.load('feature_cols.pkl')
        return model, feature_cols
    except FileNotFoundError:
        return None, None

model, feature_cols = load_model()

st.title("🛡️ Cyber Sentinel")
st.markdown("### AI-Powered Real-Time Phishing Detection")

url_input = st.text_input("Enter URL to scan:", placeholder="https://example.com")

if st.button("Analyze URL"):
    if model is None:
        st.error("Model not found! Run `python train_model.py` first.")
    elif not url_input.strip():
        st.warning("Please enter a URL.")
    else:
        raw       = extract_features(url_input)
        feat_dict = dict(zip(FEATURE_NAMES, raw))

        # Select only the columns the model was trained on, in correct order
        X = np.array([[feat_dict.get(c, 0) for c in feature_cols]])

        prediction  = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        # Similarity score for display
        try:
            u = url_input if '://' in url_input else 'http://' + url_input
            hostname  = urlparse(u).netloc.split(':')[0]
            sim_score = _url_similarity_index(hostname)
        except Exception:
            sim_score = feat_dict.get('URLSimilarityIndex', 0)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Analysis Result")
            if prediction == 0:
                st.error("🚨 PHISHING DETECTED")
                st.metric("Phishing Probability", f"{probability[0]*100:.1f}%")
            else:
                st.success("✅ LEGITIMATE URL")
                st.metric("Safety Score", f"{probability[1]*100:.1f}%")

            st.progress(int(max(probability) * 100),
                        text=f"Model confidence: {max(probability)*100:.1f}%")

            st.markdown("---")
            st.markdown(f"**Domain Similarity Score:** `{sim_score:.1f} / 100`")
            if sim_score == 100.0:
                st.info("✅ Exact match to a known legitimate domain.")
            elif sim_score >= 80:
                st.warning("⚠️ Domain closely resembles a known site but is NOT exact — common phishing pattern!")
            else:
                st.info("ℹ️ Domain has low similarity to top legitimate sites.")

        with col2:
            st.subheader("Feature Breakdown")
            feat_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Value':   raw,
            })
            st.dataframe(feat_df, use_container_width=True, height=420)

st.markdown("---")
st.caption("BTech CSE Project · Cyber Sentinel · Trained on PhiUSIIL Dataset")




# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# from features import extract_features, _url_similarity_index
# from urllib.parse import urlparse

# st.set_page_config(page_title="Cyber Sentinel", page_icon="🛡️", layout="wide")

# @st.cache_resource
# def load_model():
#     try:
#         model       = joblib.load('phishing_model.pkl')
#         feature_cols = joblib.load('feature_cols.pkl')
#         return model, feature_cols
#     except FileNotFoundError:
#         return None, None

# model, feature_cols = load_model()

# # Column name → index mapping from features.py return list
# ALL_FEATURE_NAMES = [
#     'URLLength', 'DomainLength', 'IsDomainIP', 'TLDLength',
#     'URLSimilarityIndex', 'CharContinuationRate', 'TLDLegitimateProb',
#     'URLCharProb', 'NoOfSubDomain', 'HasObfuscation',
#     'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL',
#     'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
#     'NoOfEquals', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
#     'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
#     'LongestPathTokenLength', 'URLTitleMatchScore', 'HasSocialNet',
#     'DomainTitleMatchScore', 'URLDomainRatio', 'CharRepeatRate',
#     'NoOfSlashInURL',
# ]

# st.title("🛡️ Cyber Sentinel")
# st.markdown("### AI-Powered Real-Time Phishing Detection")

# url_input = st.text_input("Enter URL to scan:", placeholder="https://example.com")

# if st.button("Analyze URL"):
#     if model is None:
#         st.error("Model not found! Run `python train_model.py` first.")
#     elif not url_input.strip():
#         st.warning("Please enter a URL.")
#     else:
#         raw          = extract_features(url_input)
#         feat_dict    = dict(zip(ALL_FEATURE_NAMES, raw))
#         X            = np.array([[feat_dict.get(c, 0) for c in feature_cols]])

#         prediction   = model.predict(X)[0]
#         probability  = model.predict_proba(X)[0]

#         # Also compute similarity score for display
#         try:
#             parsed   = urlparse(url_input if '://' in url_input else 'http://' + url_input)
#             hostname = parsed.netloc.split(':')[0]
#             sim_score = _url_similarity_index(hostname)
#         except Exception:
#             sim_score = feat_dict.get('URLSimilarityIndex', 0)

#         col1, col2 = st.columns(2)

#         with col1:
#             st.subheader("Analysis Result")
#             if prediction == 0:
#                 st.error("🚨 PHISHING DETECTED")
#                 st.metric("Phishing Probability", f"{probability[0]*100:.1f}%")
#             else:
#                 st.success("✅ LEGITIMATE URL")
#                 st.metric("Safety Score", f"{probability[1]*100:.1f}%")

#             st.progress(int(max(probability) * 100),
#                         text=f"Model confidence: {max(probability)*100:.1f}%")

#             st.markdown("---")
#             st.markdown(f"**Domain Similarity Score:** `{sim_score:.1f} / 100`")
#             if sim_score == 100:
#                 st.info("✅ Domain exactly matches a known legitimate site.")
#             elif sim_score >= 80:
#                 st.warning(f"⚠️ Domain looks similar to a known site but is NOT an exact match — common phishing trick!")
#             else:
#                 st.info("ℹ️ Domain has low similarity to top legitimate sites.")

#         with col2:
#             st.subheader("Feature Analysis")
#             feat_df = pd.DataFrame({
#                 'Feature': ALL_FEATURE_NAMES,
#                 'Value':   raw
#             })
#             st.dataframe(feat_df, use_container_width=True, height=400)

# st.markdown("---")
# st.caption("BTech CSE Project · Cyber Sentinel · Trained on PhiUSIIL Dataset")

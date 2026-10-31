import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="محلل النصوص العربية", page_icon="📝")

st.title("أداة تحليل وتلخيص النصوص باللغة العربية")
st.write("أدخلي النص المراد تحليله وتلخيصه.")

@st.cache_resource
def load_model():
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

text = st.text_area("أدخلي النص العربي", height=220)

if st.button("تحليل وتلخيص النص"):
    if not text.strip():
        st.warning("يرجى إدخال نص صالح.")
    else:
        with st.spinner("جارٍ تحليل النص..."):
            tokenizer, model = load_model()
            words = text.split()
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            summary_ids = model.generate(
                **inputs,
                max_length=100,
                min_length=20,
                num_beams=4
            )
            summary = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

        st.success("تم التحليل بنجاح")
        st.write(f"عدد الكلمات: {len(words)}")
        st.subheader("الملخص التنفيذي")
        st.write(summary)

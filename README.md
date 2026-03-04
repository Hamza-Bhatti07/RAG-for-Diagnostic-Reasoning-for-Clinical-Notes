# RAG-for-Diagnostic-Reasoning-for-Clinical-Notes

A Retrieval-Augmented Generation system designed to answer clinical queries using the MIMIC-IV-Ext dataset.
• Engineered a dense retrieval system using FAISS to index high-dimensional medical note embeddings.
• Integrated Google’s Flan-T5 to synthesize retrieved clinical context into coherent diagnostic responses.
• Developed an augmentation pipeline using prompt engineering to ground LLM outputs in specific records.
• Built a Streamlit frontend to allow users to input clinical queries and visualize retrieved document chunks.
• Optimized model inference using beam search decoding to ensure high-fidelity responses within token limits.

Live at: https://rag-for-diagnostic-reasoning-for-clinical-notes-fm2mmdk9hzga4d.streamlit.app/

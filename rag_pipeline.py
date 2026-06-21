# -*- coding: utf-8 -*-
"""RAG_pipeline.ipynb

**Descarga de librerías**
"""

!pip install -qqq arxiv pandas torch sentence-transformers faiss-cpu
!pip install -qqq langchain langchain-community langchain-core langchain_text_splitters
!pip install -qqq umap-learn matplotlib

"""## CREACIÓN BASE DE DATOS

**Obtención abstracts y limpieza mínima**

El dataset.csv se puede encontrar en la carpeta de archivos.
"""

import arxiv
import pandas as pd
import re

# Limpieza mínima
def clean_text(text):
  text = text.replace("\n", " ") # quitar saltos de línea
  text = re.sub(r'\s+', ' ', text) # quitar espacios duplicados
  return text.strip()

# Búsqueda en arXiv
client = arxiv.Client()

search = arxiv.Search(
    query='"open cluster" AND Gaia',
    max_results=200,
    sort_by=arxiv.SortCriterion.SubmittedDate)

data = []
for result in client.results(search):
    abstract = clean_text(result.summary)

    data.append({
        "id": result.entry_id,
        "title": result.title,
        "abstract": abstract,
        "year": result.published.year,
    })

# Crear DataFrame
df = pd.DataFrame(data)
df = df.dropna(subset=["abstract"])
df = df.drop_duplicates(subset=["title"])

# Guardar dataset
df.to_csv("dataset.csv", index=False)
print(f"Dataset guardado con {len(df)} abstracts")
print(df.head())

"""**Chunking y Embedding de los abstracts**

En la carpeta de archivos aparecerá dos ficheros de los chunks (uno .csv y uno .json) además del índice faiss.
"""

import pandas as pd
import faiss
import numpy as np
import json

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cargar dataset
df = pd.read_csv("dataset.csv")
df = df.dropna(subset=["abstract"]) # Así evitamos NaN

# CHUNKING
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100 )

all_chunks = []
for i, (_, row) in enumerate(df.iterrows()):
    text = row["title"] + ". " + row["abstract"]
    chunks = splitter.split_text(text)

    for j, chunk in enumerate(chunks):
        all_chunks.append({
            "chunk_id": len(all_chunks),
            "text": chunk,
            "title": row["title"],
            "year": row["year"],
            "full_text": text,
            "id": row["id"]
        })

chunk_df = pd.DataFrame(all_chunks)
chunk_df = chunk_df.reset_index(drop=True)

print(f"Total chunks: {len(chunk_df)}")

# EMBEDDINGS
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

texts = chunk_df["text"].tolist()

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

# FAISS
dimension = embeddings.shape[1]

index = faiss.IndexHNSWFlat(dimension, 32)
index.metric_type = faiss.METRIC_INNER_PRODUCT

index.add(embeddings)

print(f"Índice FAISS creado con {index.ntotal} vectores")

# MAPPING
chunk_df.to_csv("chunks.csv", index=False) # CSV Relación índice-texto

mapping = chunk_df.to_dict(orient="records") # JSON (mapping real)

with open("chunk_mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)

# Guardar índice FAISS
faiss.write_index(index, "faiss_index.bin")

"""## PIPELINE

**Pregunta y respuesta** con FLAN T5
"""

import faiss
import numpy as np

from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# LLM
model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model_llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# PROMPT
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a scientific assistant.

Use the context to answer the question.
You must rephrase and summarize the information.

If the answer is not present, say "Not enough information".

Context:
{context}

Question:
{question}

Answer in your own words:
"""
)

# GENERATION FUNCTION
def llm_generate(inputs):
    prompt_text = prompt.format(**inputs)

    tokens = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model_llm.generate(
        **tokens,
        max_new_tokens=150
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

rag_chain = RunnableLambda(llm_generate)

# RERANKER
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# RAG PIPELINE
def rag(question):

    # -------- EMBEDDING --------
    question_emb = model.encode([question],
        normalize_embeddings=True
    ).astype("float32")

    # -------- RETRIEVAL (FAISS) ---------
    D, I = index.search(question_emb, 5)

    retrieved = []
    print("\n=== FAISS RESULTS ===\n")

    for score, i in zip(D[0], I[0]):
        item = chunk_df.iloc[i]

        print(f"FAISS score: {score:.4f} | {item['title']}")

        retrieved.append({
            "text": item["text"],
            "title": item["title"],
            "year": item["year"],
            "faiss_score": score,
            "id": item["id"] })

    # -------- RERANKING --------
    pairs = [[question, c["text"]] for c in retrieved]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, retrieved),
        key=lambda x: x[0],
        reverse=True)

    top_chunks = [c for _, c in ranked[:3]]

    # -------- CONTEXT BUILDING --------
    context = "\n\n".join([
        f"[Paper: {c['title']} ({c['year']})]\n{c['text'][:300]}"
        for c in top_chunks ])

    print("\n=== CONTEXTO UTILIZADO ===\n")
    print(context)

    # -------- GENERATION --------
    generated_response = rag_chain.invoke({
        "context": context,
        "question": question })
    return generated_response, context, retrieved

# QUERY
question = "Which stellar parameters does Gaia DR3 measure?"
response, context, retrieved = rag(question)

print("\n=== RESPUESTA FINAL ===\n")
print(response)

"""## ANÁLISIS DE RESULTADOS

**Gráfica UMAP con el query y contexto recuperado**
"""

# @title
# The original attempt to install umap-learn==0.6.0 failed because it's not compatible with Python 3.12.
# Instead, we will keep the already installed umap-learn (likely 0.5.12) and upgrade its dependency pynndescent
# to a version compatible with the current Numba (0.60.0).
!pip install pynndescent==0.7.0
import umap
import matplotlib.pyplot as plt


# QUERY

#question = "Does Gaia DR3 include radial velocities for all stars?"
question = "Which stellar parameters does Gaia DR3 measure?"
#question = "How has the formation of contact binaries been investigated?"

question_emb = model.encode(
    [question],
    normalize_embeddings=True
).astype("float32")

# =========================
# FAISS RETRIEVAL
# =========================
k = 10
D, I = index.search(question_emb, k)
retrieved_indices = I[0]

# =========================
# UMAP
# =========================
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

emb_2d = reducer.fit_transform(embeddings)
question_2d = reducer.transform(question_emb)


# PLOT
plt.figure(figsize=(10,7))

# corpus
plt.scatter(emb_2d[:,0], emb_2d[:,1], alpha=0.2)

# retrieved
plt.scatter(
    emb_2d[retrieved_indices,0],
    emb_2d[retrieved_indices,1],
    marker="x",
    s=120,
    label="Retrieved"
)

# query
plt.scatter(
    question_2d[:,0],
    question_2d[:,1],
    marker="*",
    s=300,
    label="Question"
)

plt.legend()
plt.title("UMAP: Question vs Retrieved Chunks")
plt.show()

"""## Evaluación personalizada con preguntas y análisis de FAISS"""

# Definir preguntas de prueba aquí

custom_questions = [
    #"How does the molecular cloud density vary with Galactic longitude?"
    #  "How has the release of Gaia data impact the number of known Galactic open clusters?",
   #"What are open clusters widely considered as potential enviroments for?"
      # "How has the formation of contact binaries been investigated?",
    #"Which stellar parameters does Gaia DR3 measure?",
    #"Does Gaia DR3 include radial velocities for all stars?",
    #"What is the typical metallicity of open clusters studied with Gaia?"
    "as YSOs, and to reconstruct the three-dimensional (3D) motions of the main MC complexes within 2.5 kpc of the Sun using YSOs and young OCs as tracers. Using Gaia DR3 astrometry together with complementary spectroscopic surveys for radial velocities, we compiled a unified sample of 24,732 stellar tracers. We applied robust clustering in proper motion space to identify co-moving YSOs and derived cloud-averaged motions via Monte Carlo sampling. These were compared with the kinematics of OCs younger than 30 Myr. Finally, we performed orbital integrations in a realistic Galactic potential to tra"
    #"n high-resolution APOGEE spectra the model achieves precisions of $18~$K in $\textrm{T}_{\rm eff}$, $0.04~$dex in $\textrm{log}\,\textit{g}$, $0.015~$dex in [Fe/H], and ${<}\,0.03~$dex across all abundances; on lower-resolution DESI spectra, typical precisions are $51~$K, $0.09~$dex, $0.04~$dex, and ${\sim}\,0.06~$dex, respectively. Cross-survey comparisons demonstrate that labels for the same stars observed by different surveys are"
]

for i, q in enumerate(custom_questions):
    print(f"\n--- Evaluando Pregunta {i+1}: {q} ---\n")
    response, context, retrieved_faiss = rag(q)

    print("\n=== RESULTADOS DE FAISS (antes del re-ranking) ===\n")
    for item in retrieved_faiss:
        print(f"  Título: {item['title']} ({item['year']})")
        print(f"  Puntuación FAISS: {item['faiss_score']:.4f}")
        print(f"  Extracto: {item['text'][:150]}...")
        print("  ---")

    print("\n=== CONTEXTO FINAL (después del re-ranking) ===\n")
    print(context)

    print("\n=== RESPUESTA DEL MODELO ===\n")
    print(response)
    print("\n======================================================\n")

import streamlit as st
import os
import json
import pickle
import re
import ast
from typing import TypedDict, Optional, List
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# === Config ===
required_fields = ["rpf_title", "pain_point", "solution", "goals", "funding_size"]
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024, temperature=0.7)
memory = ConversationBufferMemory(return_messages=True)


# ==== CSV Ingestor ====
class CSVIngestor:
    def __init__(
        self,
        data_path: str = 'database/RFP',
        text_save_path: str = 'database',
        vector_store_path: str = 'database/faiss.index',
    ):
        self.vector_store_path = vector_store_path
        self.data_path = data_path
        self.text_save_path = text_save_path

        if not os.path.isfile(self.text_save_path + '/rfp_data.pkl'):
            self.docs_list = self.get_docs()
            self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="gpt-4",
                chunk_size=1024,
                chunk_overlap=64
            )
            doc_splits = self.text_splitter.split_documents(self.docs_list)
            with open(f'{self.text_save_path}/rfp_data.pkl', 'wb') as f:
                pickle.dump(doc_splits, f)
        else:
            with open(f'{self.text_save_path}/rfp_data.pkl', 'rb') as f:
                doc_splits = pickle.load(f)

        self.embeddings = OpenAIEmbeddings()

        if os.path.exists(self.vector_store_path):
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = FAISS.from_documents(
                documents=doc_splits,
                embedding=self.embeddings,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
            self.vector_store.save_local(self.vector_store_path)

    def get_docs(self):
        if os.path.isdir(self.data_path):
            csv_files = [file_name for file_name in os.listdir(self.data_path) if file_name.endswith(".csv")]
            if csv_files:
                loader = CSVLoader(
                    file_path=os.path.join(self.data_path, csv_files[0]),
                    csv_args={'delimiter': ','},
                    encoding='utf-8'
                )
                return loader.load()
        raise ValueError("No valid data source found.")

    def get_retriever(self, top_k=10):
        return self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})

retriever = CSVIngestor().get_retriever()


# === LangGraph: Draft Generator ===
class GraphState(TypedDict, total=False):
    rfp: dict
    is_complete: Optional[bool]
    retrieved_docs: Optional[List]
    draft: Optional[str]

def check_completion(state: GraphState) -> GraphState:
    is_complete = all(state["rfp"].get(f) for f in required_fields)
    return {"is_complete": is_complete}

def retrieve_docs(state: GraphState) -> GraphState:
    title = state.get("rfp", {}).get("rpf_title", "")
    pain_point = state.get("rfp", {}).get("pain_point", "")
    solution = state.get("rfp", {}).get("solution", "")
    
    query = f"# Title: {title}\n# Pain Point: {pain_point}\n# Solution: {solution}"
    
    docs = retriever.invoke(query)

    return {"retrieved_docs": docs}

def generate_draft(state: GraphState) -> GraphState:
    docs_text = "\n".join([doc.page_content for doc in state.get("retrieved_docs", [])])
    prompt = (
        f"Write a full government RFP draft based on this:\n\n"
        f"RFP Info:\n{json.dumps(state['rfp'], indent=2)}\n\n"
        f"Reference Docs:\n{docs_text}"
    )
    response = llm.invoke(prompt)
    return {"draft": response.content}

builder = StateGraph(GraphState)
builder.set_entry_point("check_completion")
builder.add_node("check_completion", check_completion)
builder.add_node("retrieve_docs", retrieve_docs)
builder.add_node("generate_draft", generate_draft)
builder.add_conditional_edges("check_completion", lambda s: "yes" if s.get("is_complete") else "no", {
    "yes": "retrieve_docs", "no": END
})
builder.add_edge("retrieve_docs", "generate_draft")
graph = builder.compile()

# === Prompt & Extract ===
def generate_chat_prompt(field: str, user_input: str) -> str:
    return f"""
You are helping a user write a government RFP (Request for Proposal). 
Speak in a friendly, conversational tone.

Your job is to help the user naturally provide the value for this specific field: "{field}".

Do NOT ask about other fields. Only guide the user toward completing this one field in this turn.

Always respond in the user's language based on their input.

User's message:
{user_input}

Your response:
"""

def extract_field_value(user_input: str, field: str) -> Optional[str]:
    extract_prompt = (
        f"Extract value for field '{field}' from the user's message below. "
        f"Return ONLY a Python dict like {{'{field}': value}} if value is found. Otherwise return an empty dict.\n\n"
        f"User: {user_input}"
    )
    result = llm.invoke(extract_prompt).content
    try:
        cleaned = re.sub(r"```(?:json|python)?", "", result).strip("` ")
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, dict) and field in parsed:
            return parsed[field]
    except Exception:
        return None
    return None

# === Streamlit UI ===
st.set_page_config(layout="wide", page_title="RFP Generator")
st.title("🧠 RFP Generator")

# === Session State Init ===
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.rfp_data = {k: None for k in required_fields}
    st.session_state.graph_done = False
    st.session_state.awaiting_feedback = False
    st.session_state.awaiting_revision_input = False
    st.session_state.draft = ""
    st.session_state.current_field = None
    st.session_state.skip_next_check = False  # ✅ Draft 생성 지연용 flag

# === Display Chat History ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Input ===
if user_input := st.chat_input("Type your message here..."):

    # 🛑 If we were just waiting to skip generation, now trigger it
    if st.session_state.get("skip_next_check"):
        st.session_state.skip_next_check = False
        st.stop()

    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # 🔁 Feedback / Refinement Flow
    if st.session_state.graph_done:
        if st.session_state.awaiting_feedback:
            if "no" in user_input.lower():
                st.session_state.awaiting_feedback = False
                st.session_state.awaiting_revision_input = True
                st.chat_message("assistant").markdown("What would you like to revise?")
                st.session_state.messages.append({"role": "assistant", "content": "What would you like to revise?"})
                st.stop()
            elif "yes" in user_input.lower():
                st.chat_message("assistant").markdown("✅ Draft confirmed.")
                st.session_state.messages.append({"role": "assistant", "content": "✅ Draft confirmed."})
                st.session_state.awaiting_feedback = False
                st.stop()

        elif st.session_state.awaiting_revision_input:
            revised = llm.invoke(
                f"Revise the following RFP draft based on this feedback:\n\nDraft:\n{st.session_state.draft}\n\nFeedback:\n{user_input}"
            ).content
            st.session_state.draft = revised
            st.chat_message("assistant").markdown("🔁 Updated Draft:")
            st.chat_message("assistant").markdown(revised)
            st.session_state.messages.append({"role": "assistant", "content": revised})

            follow_up = "Do you want to revise this draft again? (yes/no)"
            st.chat_message("assistant").markdown(follow_up)
            st.session_state.messages.append({"role": "assistant", "content": follow_up})

            st.session_state.awaiting_feedback = True
            st.session_state.awaiting_revision_input = False
            st.stop()

    # 🧠 Field Collection Phase
    missing_field = next((f for f in required_fields if st.session_state.rfp_data[f] is None), None)
    if missing_field:
        st.session_state.current_field = missing_field
        chat_prompt = generate_chat_prompt(missing_field, user_input)
        reply = llm.invoke(chat_prompt).content
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        memory.chat_memory.add_message(AIMessage(content=reply))

        # 추출 시도
        field_being_answered = st.session_state.current_field
        if field_being_answered:
            value = extract_field_value(user_input, field_being_answered)
            st.session_state.rfp_data[field_being_answered] = value or user_input
            st.session_state.current_field = None

        # ✅ Delay generation to next turn
        if all(st.session_state.rfp_data.get(f) for f in required_fields):
            st.session_state.skip_next_check = True
            st.stop()

    # ✅ Trigger generation (in next turn)
    if (
        not st.session_state.graph_done
        and all(st.session_state.rfp_data.get(f) for f in required_fields)
        and not st.session_state.get("skip_next_check", False)
    ):
        with st.chat_message("assistant"):
            with st.spinner("🛠️ Generating your draft..."):
                result = graph.invoke({
                    "rfp": st.session_state.rfp_data,
                    "is_complete": False,
                    "retrieved_docs": [],
                    "draft": ""
                })
                draft = result.get("draft", "❌ Failed to generate draft.")
                st.session_state.draft = draft
                st.markdown("🎉 All required fields collected! Here's your draft:")
                st.markdown(draft)

        st.session_state.messages.append({"role": "assistant", "content": draft})
        st.session_state.graph_done = True
        st.session_state.awaiting_feedback = True

        feedback_prompt = "Do you want to revise this draft? (yes/no)"
        st.chat_message("assistant").markdown(feedback_prompt)
        st.session_state.messages.append({"role": "assistant", "content": feedback_prompt})

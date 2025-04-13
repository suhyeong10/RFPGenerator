# 🧠 RFP-Agent

## The Problem RFP-Agent Solves  
When participating in hackathons or preparing project proposals, teams often struggle with writing clear, persuasive RFPs (Request for Proposals). Despite having innovative ideas, many lack experience in structuring proposals or waste time on formatting instead of focusing on their core solutions. RFP-Agent solves this problem by generating high-quality, editable RFPs based on user inputs using natural language.

## 💡 Solution  
We built an AI-powered agent that helps users write RFPs interactively. Users provide key project information such as the problem statement, proposed solution, goals, and target users. The agent generates a professional RFP draft and iteratively refines it based on user feedback. Our solution is designed to reduce time, improve clarity, and help teams communicate their ideas effectively.

To further enhance draft quality and relevance, we integrated **Retrieval-Augmented Generation (RAG)** into the pipeline. Based on the user's inputs — such as domain, problem, or goals — the system retrieves **similar real-world proposals** from a vector-based knowledge base. These examples are used to guide tone, structure, and language, allowing the model to produce **personalized, domain-aware RFPs** that reflect field-specific conventions. This significantly reduces the “blank page” problem while boosting confidence and clarity for users.

## 🧩 Overview  
Our system is built with the **LangGraph** framework, which enables stateful, multi-step workflows across different user-agent interactions. We connected this backend agent with a **Streamlit** frontend, allowing users to engage with the agent through a simple web interface. The agent collects inputs, retrieves relevant examples, generates drafts, and allows refinements until the user is satisfied.

## 📈 Pipeline Diagram

![image](https://github.com/user-attachments/assets/8b6cf936-0da9-4705-8edf-383e9b097283)

## ⚙️ Key Features  
- ✨ Personalized RFP generation based on user goals and domain  
- 🔎 Retrieval of similar proposals to guide tone and structure  
- 🔄 Editable, iterative refinement loop via LangGraph states  
- 🧩 Seamless integration between LangGraph and Streamlit for an interactive UX

## ⚠️ Challenges We Ran Into  

### Graph Architecture with LangGraph  
We decided to use LangGraph to enhance modularity and maintain clear state transitions. However, designing the right graph structure for branching logic, draft refinement, and conditional inputs was non-trivial.

### Frontend-Backend Integration  
LangGraph is optimized for end-to-end deployment, whereas Streamlit typically handles UI logic separately. Bridging the two required creating custom handlers for user input and dynamically controlling state transitions based on session data.

### RAG + Context Injection Logic  
Building a retrieval system that surfaces only **contextually relevant examples** — without overwhelming or diluting the prompt — required careful design. We had to ensure that the examples inspired the draft without making it overly templated.

### Session Management and User Feedback Loops  
Since LangGraph operates with a defined state, handling user-driven refinement required additional logic to ensure that only selected fields were updated while preserving others. Ensuring a seamless experience without breaking the flow was a key concern.

## 🚀 About Upstage  
Upstage provides a powerful suite of open-source AI tools and models. Their commitment to democratizing AI aligns with our project’s goal to make proposal writing more accessible for everyone, especially non-native English speakers or non-technical teams.

## ✨ About Story  
We started this project because we've all experienced the pain of writing RFPs or application documents under tight deadlines. Our team wanted to build something practical and immediately useful — something we would use ourselves. As the idea evolved, we realized it could help not just hackathon teams, but anyone preparing proposals. By adding personalized generation with retrieval of similar projects, we made the tool more intelligent and helpful for users at all experience levels.

## 🔧 Challenges

- **Prompt Engineering**: Ensuring the prompts guide the LLM to generate only the changed parts during refinement — without rewriting the entire RFP — required careful engineering.
- **State Explosion Risk in Graph Design**: Managing multiple user decisions and conditional paths in LangGraph led to complexity in our graph.
- **Streamlit UI State Sync**: Ensuring UI state (e.g., form values and chat history) stayed synchronized with the LangGraph agent’s state took extra effort.

## 📚 Learning

- We learned how to design and control state-based agents using LangGraph.  
- We gained experience integrating interactive LLM-based workflows with custom frontend logic.  
- We saw how **RAG can significantly improve personalization** by grounding LLM responses in real-world examples.  
- Most importantly, we saw how small UX decisions — like asking “What would you like to refine?” instead of “Is this okay?” — dramatically improve user engagement.

Demo: https://www.youtube.com/watch?v=gGHJzMS3DhY

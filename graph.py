import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv()

# ── STEP 1: Define the STATE ──
class State(TypedDict):
    question: str         
    chunks: list           
    answer: str            
    confidence: str        
    needs_human: bool      


# ── STEP 2: Define the NODES ──

def retriever_node(state: State) -> State:
   
    print(f"\n🔍 Retriever: Searching for '{state['question']}'...")
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    
    # Search for top 3 most similar chunks
    results = vectorstore.similarity_search(state["question"], k=3)
    chunks = [doc.page_content for doc in results]
    
    print(f"Found {len(chunks)} relevant chunks!")
    
    # Update the state
    state["chunks"] = chunks
    return state


def generator_node(state: State) -> State:
    
    print("\nGenerator: Creating answer from retrieved chunks...")
    
    # Combine chunks into a nice answer
    if state["chunks"]:
        answer = "Based on our knowledge base:\n\n"
        answer += "\n\n".join(state["chunks"])
    else:
        answer = "I don't have enough information to answer that question."
    
    print(f"Answer generated: {answer[:100]}...")
    
    # Update state
    state["answer"] = answer
    return state


def confidence_checker_node(state: State) -> State:
   
    print("\nConfidence Checker: Evaluating answer quality...")
    
    answer = state["answer"]
    
    if "don't have enough information" in answer.lower():
        state["confidence"] = "low"
        state["needs_human"] = True
        print("⚠️  Low confidence - needs human help!")
    elif len(answer) < 50:
        state["confidence"] = "low"
        state["needs_human"] = True
        print("⚠️  Answer too short - needs human help!")
    else:
        state["confidence"] = "high"
        state["needs_human"] = False
        print("High confidence - answer looks good!")
    
    return state


def hitl_node(state: State) -> State:
   
    print("\nHITL: Escalating to human agent...")
    print(f"   Question: {state['question']}")
    print(f"   Bot's attempt: {state['answer']}")
    print("\n   → A human agent would review this and provide a better answer!")
    
    return state


# ── STEP 3: Define ROUTING (conditional logic) ──

def should_escalate(state: State) -> str:
   
    if state["needs_human"]:
        return "hitl"
    else:
        return "end"


# ── STEP 4: Build the GRAPH ──

def create_graph():
   
    # Create a new graph
    workflow = StateGraph(State)
    
    # Add nodes (the boxes)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("confidence", confidence_checker_node)
    workflow.add_node("hitl", hitl_node)
    
    # Define the flow (the arrows)
    workflow.set_entry_point("retriever")  # Start here
    workflow.add_edge("retriever", "generator")  # retriever → generator
    workflow.add_edge("generator", "confidence")  # generator → confidence
    
    # Conditional routing after confidence check
    workflow.add_conditional_edges(
        "confidence",
        should_escalate,  # this function decides where to go
        {
            "hitl": "hitl",  # if needs human → go to hitl
            "end": END       # otherwise → END
        }
    )
    
    # HITL always ends after escalation
    workflow.add_edge("hitl", END)
    
    # Compile the graph
    app = workflow.compile()
    return app


# ── STEP 5: Run a query ──

def ask_question(question: str):
    
    print(f"\n{'='*60}")
    print(f"USER QUESTION: {question}")
    print(f"{'='*60}")
    
    # Create initial state
    initial_state = {
        "question": question,
        "chunks": [],
        "answer": "",
        "confidence": "",
        "needs_human": False
    }
    
    # Create and run the graph
    graph = create_graph()
    result = graph.invoke(initial_state)
    
    # Print final answer
    print(f"\n{'='*60}")
    print(f"FINAL ANSWER:")
    print(f"{'='*60}")
    print(result["answer"])
    print(f"\nConfidence: {result['confidence']}")
    print(f"Needed human help: {result['needs_human']}")
    
    return result


if __name__ == "__main__":
    ask_question("How do I reset my password?")
    
    ask_question("What is the meaning of life?")
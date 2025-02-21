# procurement_chain.py
import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
import asyncio

from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# -----------------------
# Decision Tree Data Model
# -----------------------
@dataclass
class DecisionOption:
    option: str
    answer: Optional[str] = None         # Terminal answer (if branch ends)
    next_node: Optional[str] = None        # Pointer to the next decision node

@dataclass
class DecisionNode:
    name: str
    question: str
    options: List[DecisionOption]

# -----------------------
# Procurement Workflow with AI Extraction, Greetings, and Farewells
# -----------------------
class ProcurementWorkflow:
    def __init__(self):
        # Build the decision tree
        self.decision_tree: Dict[str, DecisionNode] = self._initialize_decision_tree()
        # Session data: track current node, answers, and conversation history per user
        self.sessions: Dict[str, Dict] = {}
        
        # Setup the AI LLM (for extraction)
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            verbose=True
        )
        # Build an extraction chain to process free-form input
        extraction_prompt = PromptTemplate(
            template="""
You are given a list of options: {options}.
User input: {input}
Select the best matching option. Return exactly the option text from the list that best matches the user's input.
If no option seems relevant, return "None".
""",
            input_variables=["options", "input"]
        )
        self.extraction_chain = extraction_prompt | self.llm | StrOutputParser()

    def _initialize_decision_tree(self) -> Dict[str, DecisionNode]:
        tree = {}
        # Node 1: Existing Arrangement
        tree["existing_arrangement"] = DecisionNode(
            name="existing_arrangement",
            question="Does an existing arrangement exist for this contract?",
            options=[
                DecisionOption(option="RoPS", answer="Use a Purchase Order and reference the RoPS."),
                DecisionOption(option="Preferred Supplier Arrangement (PSA) Name and Number",
                               answer="Use a Purchase Order and reference the PSA name and number."),
                DecisionOption(option="Other Council Arrangement", answer="To be drafted."),
                DecisionOption(option="Local Buy",
                               answer="Issue a Notice of Successful Quotation/Tender Letter and set up a Purchase Order referencing the LB arrangement name and number."),
                DecisionOption(option="No", next_node="procurement_value")
            ]
        )
        # Node 2: Procurement Value
        tree["procurement_value"] = DecisionNode(
            name="procurement_value",
            question="What is the value of the procurement?",
            options=[
                DecisionOption(option="Under $10,000", next_node="procurement_category"),
                DecisionOption(option="$10,000-$15,000", next_node="procurement_category"),
                DecisionOption(option="$15,000-$200,000", next_node="procurement_category"),
                DecisionOption(option="Over $200,000", next_node="procurement_category")
            ]
        )
        # Node 3: Procurement Category
        tree["procurement_category"] = DecisionNode(
            name="procurement_category",
            question="What category does the procurement fall within?",
            options=[
                DecisionOption(option="Construction", next_node="construction_risk"),
                DecisionOption(option="Services Only", next_node="services_only"),
                DecisionOption(option="Goods and Services", next_node="goods_and_services_risk"),
                DecisionOption(option="Goods Only", answer="Use a Goods and Services Contract.")
            ]
        )
        # Node 4: Construction Risk
        tree["construction_risk"] = DecisionNode(
            name="construction_risk",
            question="What is the risk of the work being undertaken?",
            options=[
                DecisionOption(option="Low", answer="Set up a Purchase Order referencing the DSC Standard Terms and Conditions (Services) on the website."),
                DecisionOption(option="Medium", next_node="construction_scope")
            ]
        )
        # Node 5: Construction Scope
        tree["construction_scope"] = DecisionNode(
            name="construction_scope",
            question="What does the scope entail?",
            options=[
                DecisionOption(option="Supply of equipment, building elements, etc. and/or Installation",
                               answer="Use a Supply and Install Contract."),
                DecisionOption(option="Construction work only", next_node="construction_complexity")
            ]
        )
        # Node 6: Construction Complexity
        tree["construction_complexity"] = DecisionNode(
            name="construction_complexity",
            question="What is the complexity of the work being carried out?",
            options=[
                DecisionOption(option="Low", answer="Use a Construct Only AS4000."),
                DecisionOption(option="Medium", answer="Use the Minor Works contract (with or without design – AS4902)."),
                DecisionOption(option="High", next_node="construction_high_detail")
            ]
        )
        # Node 7: Construction High Detail
        tree["construction_high_detail"] = DecisionNode(
            name="construction_high_detail",
            question="Are the services for a fixed period including broad consultancy services?",
            options=[
                DecisionOption(option="Fixed period with Council provided design", answer="Use a Construct Only AS4000."),
                DecisionOption(option="Fixed period with Contractor providing the design", answer="Use a Design and Construct AS4902 Contract."),
                DecisionOption(option="Fixed period for Contractor Management services", answer="Use an AS4000 Contract."),
                DecisionOption(option="Other consultancy services", answer="Use a Services (Single Engagement) Contract.")
            ]
        )
        # Node for Services Only
        tree["services_only"] = DecisionNode(
            name="services_only",
            question="Is the value within your credit card and transaction delegation?",
            options=[
                DecisionOption(option="Yes", next_node="services_only_over_counter"),
                DecisionOption(option="No", answer="Set up a Purchase Order referencing the DSC Standard Terms and Conditions (Goods and Services) on the website.")
            ]
        )
        tree["services_only_over_counter"] = DecisionNode(
            name="services_only_over_counter",
            question="Are the services normally purchased over the counter?",
            options=[
                DecisionOption(option="Yes", answer="Pay on Credit Card."),
                DecisionOption(option="No", answer="Set up a Purchase Order referencing the DSC Standard Terms and Conditions (Goods and Services) on the website.")
            ]
        )
        # Node for Goods and Services Risk
        tree["goods_and_services_risk"] = DecisionNode(
            name="goods_and_services_risk",
            question="What is the risk of the scope of works?",
            options=[
                DecisionOption(option="Low", answer="Set up a Purchase Order referencing the DSC Standard Terms and Conditions (Goods and Services) on the website."),
                DecisionOption(option="Medium", answer="Use a Goods and Services Contract."),
                DecisionOption(option="High", answer="Use a Goods and Services Contract.")
            ]
        )
        return tree

    # -----------------------
    # AI-based Extraction Helpers
    # -----------------------
    async def _extract_option_with_ai(self, user_input: str, node: DecisionNode) -> Optional[DecisionOption]:
        options_text = ", ".join([opt.option for opt in node.options])
        try:
            extraction_result = await self.extraction_chain.ainvoke({
                "options": options_text,
                "input": user_input
            })
        except Exception as e:
            logging.error(f"Error in extraction chain: {e}")
            return None

        extracted_option = extraction_result.strip()
        if extracted_option.lower() == "none":
            return None

        for option in node.options:
            if option.option.lower() == extracted_option.lower():
                return option
        return None

    async def _match_option(self, user_input: str, node: DecisionNode) -> Optional[DecisionOption]:
        # First try direct matching (exact or substring matching)
        user_input_lower = user_input.strip().lower()
        for option in node.options:
            if option.option.lower() == user_input_lower:
                return option
        for option in node.options:
            if option.option.lower() in user_input_lower:
                return option
        # Fallback: use the AI extraction chain
        return await self._extract_option_with_ai(user_input, node)

    # -----------------------
    # Process a user message with greeting and farewell handling
    # -----------------------
    async def process_message(self, user_id: str, message: str) -> str:
        # Initialize session with conversation history if not already present.
        session = self.sessions.setdefault(user_id, {"current_node": "existing_arrangement", "answers": {}, "history": []})
        session["history"].append(f"User: {message}")
        message_lower = message.strip().lower()

        # Greeting handling
        greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"}
        if message_lower in greetings:
            current_node = self.decision_tree[session["current_node"]]
            options_text = ", ".join([opt.option for opt in current_node.options])
            response = f"Hello! {current_node.question}\nOptions: {options_text}"
            session["history"].append(f"System: {response}")
            return response

        # Farewell handling
        farewells = {"bye", "goodbye", "see you", "farewell"}
        if message_lower in farewells:
            response = "Goodbye! Thank you for using the procurement assistant. Have a great day!"
            session["history"].append(f"System: {response}")
            # Optionally, reset the session upon farewell.
            self.sessions[user_id] = {"current_node": "existing_arrangement", "answers": {}, "history": []}
            return response

        # Process the input for the current decision node
        current_node = self.decision_tree[session["current_node"]]
        matched_option = await self._match_option(message, current_node)
        if not matched_option:
            options_text = ", ".join([opt.option for opt in current_node.options])
            response = f"Invalid response. Please choose one of the following options: {options_text}"
            session["history"].append(f"System: {response}")
            return response

        # Record the answer
        session["answers"][current_node.name] = matched_option.option

        if matched_option.answer:
            # Terminal node reached: provide answer and a summary of selections.
            summary = "You have selected:\n" + "\n".join([f"{k}: {v}" for k, v in session["answers"].items()])
            response = f"{matched_option.answer}\n\n{summary}"
            session["history"].append(f"System: {response}")
            # Reset the session after completion.
            self.sessions[user_id] = {"current_node": "existing_arrangement", "answers": {}, "history": session["history"]}
            return response
        elif matched_option.next_node:
            # Move to the next node and ask its question.
            session["current_node"] = matched_option.next_node
            next_node = self.decision_tree[matched_option.next_node]
            options_text = ", ".join([opt.option for opt in next_node.options])
            response = f"{next_node.question}\nOptions: {options_text}"
            session["history"].append(f"System: {response}")
            return response
        else:
            response = "Workflow configuration error: Option does not have an answer or a next node."
            session["history"].append(f"System: {response}")
            return response

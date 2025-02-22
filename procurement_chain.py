import os
import re
import json
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
    answer: Optional[str] = None         # Terminal answer if branch ends
    next_node: Optional[str] = None      # Pointer to the next decision node

@dataclass
class DecisionNode:
    name: str
    question: str
    options: List[DecisionOption]

# -----------------------
# Procurement Workflow (Decision Tree + LLM Extraction + Analysis in JSON)
# -----------------------
class ProcurementWorkflow:
    def __init__(self):
        # Build the decision tree.
        self.decision_tree: Dict[str, DecisionNode] = self._initialize_decision_tree()
        # Session data: track current node, collected answers, extra info, and conversation history per user.
        self.sessions: Dict[str, Dict] = {}
        
        # Setup the LLM.
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            verbose=True
        )
        # Setup the analysis chain to generate a personalized, detailed procurement analysis in JSON.
        self._setup_chains()
        # Setup the extraction chain to process free-form messages.
        self._setup_extraction_chain()

    # -----------------------
    # Analysis Chain
    # -----------------------
    def _setup_chains(self):
        # The analysis prompt instructs the LLM to output a fixed JSON format, including a "final_answer" field.
        analysis_prompt = PromptTemplate(
            template="""
Based on the following procurement selections for {company}:
{answers}

The final node answer from the decision tree is: {final_answer}

Provide a detailed analysis in JSON format with the following structure:

{{
  "selections": {{
      "existing_arrangement": "<string>",
      "procurement_value": "<string>",
      "procurement_category": "<string>",
      "other_answers": "<string>"
  }},
  "final_answer": "<string>",
  "company": "{company}",
  "analysis": {{
      "risk_assessment": "<string>",
      "documentation_and_approvals": "<string>",
      "procurement_strategy": "<string>"
  }}
}}

Ensure the JSON is valid and does not include any additional text.
""",
            input_variables=["answers", "company", "final_answer"]
        )
        self.analysis_chain = analysis_prompt | self.llm | StrOutputParser()

    # -----------------------
    # Extraction Chain
    # -----------------------
    def _setup_extraction_chain(self):
        # Now we extract answers for all keys including "existing_arrangement".
        extraction_prompt = PromptTemplate(
            template="""
You are an intelligent procurement assistant. Your task is to extract from a free-form user message any answers that the user may have provided for the following decision tree questions. For each question, use the provided options to guide your extraction. Output a JSON object with keys only for the answers you are confident about. Do not include any additional text.

1. existing_arrangement: "Does an existing arrangement exist for this contract?"  
   Options: "RoPS", "Preferred Supplier Arrangement (PSA) Name and Number", "Other Council Arrangement", "Local Buy", "No".  
   Note: If the user says things like "I don't have an agreement", "I have no contract", or "no existing", and which ever implies "No" interpret it as "No".

2. procurement_value: "What is the value of the procurement?"  
   Options: "Under $10,000", "$10,000-$15,000", "$15,000-$200,000", "Over $200,000".  
   Note: If a numeric approximation is given (e.g., "25k", "around 25k", "approx 25000"), map it to the appropriate option based on the range.

3. procurement_category: "What category does the procurement fall within?"  
   Options: "Construction", "Services Only", "Goods and Services", "Goods Only".  
   Note: Look for mentions like "services", "construction", etc.

4. construction_risk: "What is the risk of the work being undertaken?"  
   Options: "Low", "Medium".

5. construction_scope: "What does the scope entail?"  
   Options: "Supply of equipment, building elements, etc. and/or Installation", "Construction work only".

6. construction_complexity: "What is the complexity of the work being carried out?"  
   Options: "Low", "Medium", "High".

7. construction_high_detail: "Are the services for a fixed period including broad consultancy services?"  
   Options: "Fixed period with Council provided design", "Fixed period with Contractor providing the design", "Fixed period for Contractor Management services", "Other consultancy services".

8. services_only: "Is the value within your credit card and transaction delegation?"  
   Options: "Yes", "No".

9. services_only_over_counter: "Are the services normally purchased over the counter?"  
   Options: "Yes", "No".

10. goods_and_services_risk: "What is the risk of the scope of works?"  
    Options: "Low", "Medium", "High".

Additionally, if the user message mentions a company name (for example, "Pilot Tech" or "TCS"), extract it with the key "company".

Only include keys for which you find a clear answer. If you do not understand or the answer is ambiguous, do not include that key.

User message: {message}
""",
            input_variables=["message"]
        )
        self.extraction_chain = extraction_prompt | self.llm | StrOutputParser()

    # -----------------------
    # Decision Tree
    # -----------------------
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
                DecisionOption(option="No", answer="Set up a PO referencing the DSC Standard Terms and Conditions (Services) on website.")
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
    # LLM Extraction
    # -----------------------
    async def _extract_answers_from_message(self, message: str) -> dict:
        """
        Use the LLM extraction chain to parse free-form user text for answers.
        This extraction now includes "existing_arrangement" so that if the user explicitly says
        "I have no contract", it gets captured.
        """
        try:
            extraction_text = await self.extraction_chain.ainvoke({"message": message})
            extracted = json.loads(extraction_text)
            if not isinstance(extracted, dict):
                return {}
            return extracted
        except Exception as e:
            logging.error(f"Extraction error: {e}")
            return {}

    # -----------------------
    # Auto-Advance the Current Node
    # -----------------------
    def _advance_current_node(self, session: dict):
        current = self.decision_tree[session["current_node"]]
        while current.name in session["answers"]:
            matched_option = None
            for opt in current.options:
                if opt.option.lower() == session["answers"][current.name].lower():
                    matched_option = opt
                    break
            if matched_option and matched_option.next_node:
                session["current_node"] = matched_option.next_node
                current = self.decision_tree[session["current_node"]]
            else:
                break

    # -----------------------
    # Option Matching
    # -----------------------
    async def _match_option(self, user_input: str, node: DecisionNode) -> Optional[DecisionOption]:
        if node.name == "procurement_value":
            num = None
            try:
                num = self._parse_numeric_value(user_input)
            except Exception as e:
                logging.error(f"Numeric parsing error: {e}")
            if num is not None:
                for option in node.options:
                    opt_text = option.option.lower()
                    if "under" in opt_text and num < 10000:
                        return option
                    elif "-" in opt_text:
                        parts = option.option.replace("$", "").replace(",", "").split("-")
                        try:
                            low_val = float(parts[0].strip())
                            high_val = float(parts[1].strip())
                            if low_val <= num <= high_val:
                                return option
                        except Exception as e:
                            logging.error(f"Error parsing range for option '{option.option}': {e}")
                    elif "over" in opt_text and num > 200000:
                        return option

        user_input_lower = user_input.strip().lower()
        for option in node.options:
            if option.option.lower() == user_input_lower or option.option.lower() in user_input_lower:
                return option

        return await self._extract_option_with_ai(user_input, node)

    async def _extract_option_with_ai(self, user_input: str, node: DecisionNode) -> Optional[DecisionOption]:
        options_text = ", ".join([opt.option for opt in node.options])
        extraction_prompt = PromptTemplate(
            template="""
You are given a list of options: {options}.
User input: {input}
Select the best matching option. Return exactly the option text from the list that best matches the user's input.
If no option seems relevant, return "None".
""",
            input_variables=["options", "input"]
        )
        extraction_chain = extraction_prompt | self.llm | StrOutputParser()
        try:
            extraction_result = await extraction_chain.ainvoke({
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

    def _parse_numeric_value(self, text: str) -> Optional[int]:
        match = re.search(r'(\d+\.?\d*)\s*(k)?', text.lower())
        if match:
            value = float(match.group(1))
            if match.group(2):
                value *= 1000
            return int(value)
        return None

    # -----------------------
    # Main Conversation Logic
    # -----------------------
    async def process_message(self, user_id: str, message: str) -> str:
        session = self.sessions.setdefault(user_id, {
            "current_node": "existing_arrangement",
            "answers": {},
            "history": [],
            "extra": {}
        })
        session["history"].append(f"User: {message}")
        message_lower = message.strip().lower()

        greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"}
        if message_lower in greetings:
            first_node = self.decision_tree[session["current_node"]]
            options_text = ", ".join([opt.option for opt in first_node.options])
            response = f"Hi! {first_node.question}\nOptions: {options_text}"
            session["history"].append(f"System: {response}")
            return response

        farewells = {"bye", "goodbye", "see you", "farewell"}
        if message_lower in farewells:
            response = "Goodbye! Thank you for using the procurement assistant. Have a great day!"
            session["history"].append(f"System: {response}")
            self.sessions[user_id] = {
                "current_node": "existing_arrangement",
                "answers": {},
                "history": []
            }
            return response

        # Extract answers (now including existing_arrangement)
        extracted = await self._extract_answers_from_message(message)
        if extracted:
            for key in ["existing_arrangement", "procurement_value", "procurement_category"]:
                if key in extracted:
                    session["answers"][key] = extracted[key]
            if "company" in extracted:
                session["extra"]["company"] = extracted["company"]

        self._advance_current_node(session)

        current_node = self.decision_tree[session["current_node"]]
        matched_option = await self._match_option(message, current_node)
        if matched_option:
            if current_node.name not in session["answers"]:
                session["answers"][current_node.name] = matched_option.option
            self._advance_current_node(session)
        else:
            options_text = ", ".join([opt.option for opt in current_node.options])
            response = f"{current_node.question}\nOptions: {options_text}"
            session["history"].append(f"System: {response}")
            return response

        current_node = self.decision_tree[session["current_node"]]
        terminal_found = False
        terminal_option = None
        for opt in current_node.options:
            if (session["answers"].get(current_node.name, "").lower() == opt.option.lower()) and opt.answer:
                terminal_found = True
                terminal_option = opt
                break

        if terminal_found:
            summary = "Here are your selections:\n" + "\n".join(
                f"{k}: {v}" for k, v in session["answers"].items()
            )
            company = session["extra"].get("company", "your organization")
            final_node_answer = terminal_option.answer
            analysis_input = {
                "answers": summary,
                "company": company,
                "final_answer": final_node_answer
            }
            analysis_result = ""
            try:
                analysis_result = await self.analysis_chain.ainvoke(analysis_input)
                final_analysis = json.loads(analysis_result)
            except Exception as e:
                final_analysis = {"error": f"Error during analysis: {str(e)}", "raw": analysis_result}
            final_result = {
                "selections": session["answers"],
                "company": company,
                "analysis": final_analysis
            }
            session["history"].append(f"System: {json.dumps(final_result, indent=2)}")
            self.sessions[user_id] = {
                "current_node": "existing_arrangement",
                "answers": {},
                "history": session["history"],
                "extra": {}
            }
            return json.dumps(final_result, indent=2)
        else:
            options_text = ", ".join([opt.option for opt in current_node.options])
            response = f"{current_node.question}\nOptions: {options_text}"
            session["history"].append(f"System: {response}")
            return response

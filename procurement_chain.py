import json
import os
import re
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, List
import logging

# For environment vars
from dotenv import load_dotenv

# LangChain / OpenAI-based imports (update as needed)
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough

###############################################################################
#  ENV & SETUP
###############################################################################
load_dotenv()  # Make sure you have OPENAI_API_KEY in your .env

# Enums for standardized options
class ProcurementAmount(Enum):
    UNDER_10K = "Under $10,000"
    BETWEEN_10K_15K = "$10,000-$15,000"
    BETWEEN_15K_200K = "$15,000-$200,000"
    OVER_200K = "Over $200,000"

class ExistingArrangement(Enum):
    NONE = "No existing arrangement"
    PANEL = "Panel arrangement"
    STANDING_OFFER = "Standing offer"
    WHOLE_OF_GOVERNMENT = "Whole of government arrangement"

class ProcurementCategory(Enum):
    CONSULTANCY = "Consultancy services"
    IT_SERVICES = "IT services"
    PROFESSIONAL_SERVICES = "Professional services"
    FACILITY_SERVICES = "Facility services"
    TRAINING = "Training services"
    OTHER = "Other services"

class WorkComplexity(Enum):
    LOW = "Low complexity"
    MEDIUM = "Medium complexity"
    HIGH = "High complexity"
    VERY_HIGH = "Very high complexity"

# Dataclass representing a single slot (or question) in the workflow
@dataclass
class ProcurementSlot:
    name: str
    prompt: str
    validation: List[str]
    required: bool = True

###############################################################################
#  PROCUREMENT WORKFLOW
###############################################################################
class ProcurementWorkflow:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            verbose=True
        )
        # Dictionary to track user sessions, storing:
        #   - "data": a dict with answered slots
        #   - "history": a list with conversation messages
        self.sessions: Dict[str, Dict] = {}
        # New dictionary to store final results for each user
        self.final_results: Dict[str, Dict] = {}

        # Define each workflow slot
        self.slots = self._initialize_slots()

        # Setup all the LLM prompt chains (validation, analysis, recommendation, extraction)
        self._setup_chains()

    def _initialize_slots(self) -> Dict[str, ProcurementSlot]:
        return {
            "po_reference": ProcurementSlot(
                name="po_reference",
                prompt="Please provide the PO reference number:",
                validation=[]  # no specific set of valid values
            ),
            "service_standard": ProcurementSlot(
                name="service_standard",
                prompt="Is the DSC Standard Terms and Conditions (Services) confirmed? (Yes/No)",
                validation=["Yes", "No"]
            ),
            "over_the_counter": ProcurementSlot(
                name="over_the_counter",
                prompt="Are the services normally purchased over the counter? (Yes/No)",
                validation=["Yes", "No"]
            ),
            "procurement_amount": ProcurementSlot(
                name="procurement_amount",
                prompt="What is the procurement amount? (Under $10,000, $10,000-$15,000, $15,000-$200,000, Over $200,000)",
                validation=[e.value for e in ProcurementAmount]
            ),
            "existing_arrangement": ProcurementSlot(
                name="existing_arrangement",
                prompt="Does an existing arrangement exist for this contract? (No existing arrangement, Panel arrangement, Standing offer, Whole of government arrangement)",
                validation=[e.value for e in ExistingArrangement]
            ),
            "procurement_category": ProcurementSlot(
                name="procurement_category",
                prompt="What category does the procurement fall within? (Consultancy services, IT services, Professional services, Facility services, Training services, Other services)",
                validation=[e.value for e in ProcurementCategory]
            ),
            "work_complexity": ProcurementSlot(
                name="work_complexity",
                prompt="What is the complexity of the work being carried out? (Low complexity, Medium complexity, High complexity, Very high complexity)",
                validation=[e.value for e in WorkComplexity]
            ),
        }

    def _setup_chains(self):
        """Setup the LLM prompt chains for validation, analysis, and recommendation."""
        # ------------------
        # Validation chain
        # ------------------
        validation_prompt = PromptTemplate(
            template="""
            Validate if "{input}" is acceptable for slot type "{slot_type}".
            Valid options are: {valid_options}.
            Return only "VALID" or "INVALID".
            """,
            input_variables=["input", "slot_type", "valid_options"]
        )
        self.validation_chain = (
            validation_prompt
            | self.llm
            | StrOutputParser()
        )

        # ------------------
        # Analysis chain
        # ------------------
        analysis_prompt = PromptTemplate(
            template="""
            Analyze this procurement request:
            PO Reference: {po_reference}
            Service Standard: {service_standard}
            Over the Counter: {over_the_counter}
            Amount: {procurement_amount}
            Existing Arrangement: {existing_arrangement}
            Category: {procurement_category}
            Work Complexity: {work_complexity}

            Chat History:
            {chat_history}

            Provide a detailed analysis including:
            1. Risk level and justification
            2. Required approvals based on amount and complexity
            3. Documentation requirements
            4. Compliance considerations
            5. Procurement strategy recommendations
            """,
            input_variables=[
                "po_reference",
                "service_standard",
                "over_the_counter",
                "procurement_amount",
                "existing_arrangement",
                "procurement_category",
                "work_complexity",
                "chat_history"
            ]
        )

        self.analysis_chain = (
            analysis_prompt
            | self.llm
            | StrOutputParser()
        )

        # ------------------
        # Recommendation chain
        # ------------------
        recommendation_prompt = PromptTemplate(
            template="""
            Based on this analysis:
            {analysis}

            Provide specific recommendations for:
            1. Contract type and terms
            2. Procurement timeline and key milestones
            3. Required stakeholder engagements
            4. Risk mitigation strategies
            5. Next steps and action items

            Format the response in a clear, actionable manner.
            """,
            input_variables=["analysis"]
        )
        self.recommendation_chain = (
            recommendation_prompt
            | self.llm
            | StrOutputParser()
        )

        # ------------------
        # Combined chain (analysis -> recommendation)
        # ------------------
        self.full_chain = (
            self.analysis_chain.with_config({"run_name": "analysis"})
            | {"analysis": RunnablePassthrough()}
            | self.recommendation_chain
        )

        # ------------------
        # Extraction chain
        # ------------------
        extraction_prompt = PromptTemplate(
            template="""
            You are given a free-form user message that may contain answers for multiple slots in a procurement workflow. 
            The slots and their expected answers are:

            1. **po_reference**: a purchase order reference number (e.g., "PO123").
            2. **service_standard**: "Yes" or "No" for DSC Standard Terms confirmation.
            3. **over_the_counter**: "Yes" or "No".
            4. **procurement_amount**: one of: {procurement_amount_options}. note: user can just say 25k or 2500 or around 35k.
            5. **existing_arrangement**: one of: {existing_arrangement_options}.
            6. **procurement_category**: one of: {procurement_category_options}.
            7. **work_complexity**: one of: {work_complexity_options}.

            The user may mention multiple answers in a single message, e.g. "The PO number is PO123, DSC Terms are confirmed, the cost is about $30,000, and we do not have an existing arrangement."

            For each slot that you find in the text, output an entry in JSON form, for example:
            {{
            "po_reference": "PO123",
            "service_standard": "Yes"
            }}

            Output only a JSON object with the discovered slots. 
            Do not include any text outside the JSON.

            User message: {message}
            """,
            input_variables=[
                "message",
                "procurement_amount_options",
                "existing_arrangement_options",
                "procurement_category_options",
                "work_complexity_options"
            ]
        )
        self.extraction_chain = (
            extraction_prompt
            | self.llm
            | StrOutputParser()
        )

    async def _extract_from_message(self, message: str) -> dict:
        """
        Run the extraction chain to parse user’s free-form text and return 
        a dictionary of extracted slot values. Includes robust error handling 
        for invocation and JSON parsing failures.
        """
        try:
            logging.debug(f"User message for extraction: {message}")

            extraction_text = await self.extraction_chain.ainvoke({
                "message": message,
                "procurement_amount_options": ', '.join([e.value for e in ProcurementAmount]),
                "existing_arrangement_options": ', '.join([e.value for e in ExistingArrangement]),
                "procurement_category_options": ', '.join([e.value for e in ProcurementCategory]),
                "work_complexity_options": ', '.join([e.value for e in WorkComplexity])
            })
        except Exception as e:
            logging.error(f"Error invoking extraction chain: {e}")
            return {}

        try:
            parsed_json = json.loads(extraction_text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM output as JSON. Raw response:\n{extraction_text}")
            logging.error(f"JSONDecodeError: {e}")
            return {}

        if not isinstance(parsed_json, dict):
            logging.error(f"LLM extraction output is not a JSON object:\n{parsed_json}")
            return {}

        return parsed_json

    def _normalize_slot_value(self, slot_name: str, user_value: str) -> Optional[str]:
        """
        Return the *canonical* slot value if there's a partial or exact match,
        or None if no match is found.
        
        For the PO reference slot, require that the input starts with 'PO'
        followed by digits. Otherwise, return None.
        """
        slot = self.slots.get(slot_name)
        if not slot:
            return None

        if slot_name == "po_reference":
            # Validate using a regex, e.g., expecting something like "PO123" or "PO 123"
            if not re.search(r'\bPO\s*\d+', user_value, re.IGNORECASE):
                return None

        # If the slot has no validation list, just return the stripped value.
        if not slot.validation:
            return user_value.strip()

        user_value_lower = user_value.strip().lower()

        # Check partial or exact match against the valid options
        for valid_option in slot.validation:
            if (user_value_lower in valid_option.lower()) or (valid_option.lower() in user_value_lower):
                return valid_option

        return None

    def _get_next_slot(self, user_id: str) -> Optional[str]:
        """Return the next missing slot name, or None if all are filled."""
        session_data = self.sessions[user_id]["data"]
        for slot_name in self.slots:
            if slot_name not in session_data:
                return slot_name
        return None

    async def process_message(self, user_id: str, message: str) -> str:
        """Main conversation logic to handle a single user message."""
        # Check for greeting messages
        greetings = {
            "hi", "hello", "hallo", "hey", "greetings",
            "howdy", "good morning", "good afternoon", "good evening"
        }
        if message.strip().lower() in greetings:
            if user_id not in self.sessions:
                self.sessions[user_id] = {"data": {}, "history": []}
            greeting_response = (
                "Hello! How can I assist you with your procurement today? "
                "Please provide the PO reference number to begin."
            )
            self.sessions[user_id]["history"].append(f"System: {greeting_response}")
            return greeting_response

        # Check if a final result already exists for this user.
        # If so, and if the user repeats the initial prompt, immediately return the stored answer.
        if user_id in self.final_results:
            if message.strip().lower().startswith("hey i want to buy a pen"):
                prev_data = self.final_results[user_id]['data']
                final_analysis = self.final_results[user_id]['final_analysis']
                confirmation_message = "You've already provided the following selections:\n"
                for key, value in prev_data.items():
                    confirmation_message += f" - {key}: {value}\n"
                confirmation_message += "\nFinal Analysis and Recommendations:\n" + final_analysis
                return confirmation_message

        # Initialize session if it doesn't exist
        if user_id not in self.sessions:
            self.sessions[user_id] = {"data": {}, "history": []}

        session = self.sessions[user_id]
        session["history"].append(f"User: {message}")

        # 1) Try extracting slot values from free-form text
        extracted_slots = await self._extract_from_message(message)
        if extracted_slots:
            for key, val in extracted_slots.items():
                if key in self.slots:
                    if key not in session["data"]:
                        normalized_val = self._normalize_slot_value(key, val)
                        if normalized_val is not None:
                            session["data"][key] = normalized_val

        # 2) Identify the next missing slot
        next_slot = self._get_next_slot(user_id)
        if next_slot:
            slot_obj = self.slots[next_slot]

            # If the next slot is a yes/no question:
            if set(slot_obj.validation) == {"Yes", "No"}:
                user_val_lower = message.strip().lower()
                if user_val_lower in ["yes", "no"]:
                    session["data"][next_slot] = user_val_lower.capitalize()
                    next_slot = self._get_next_slot(user_id)
                    if not next_slot:
                        return await self._run_analysis_and_store_result(user_id)

                    prompt = self.slots[next_slot].prompt
                    session["history"].append(f"System: {prompt}")
                    return prompt

            normalized_val = self._normalize_slot_value(next_slot, message)
            if normalized_val is not None:
                session["data"][next_slot] = normalized_val
                next_slot = self._get_next_slot(user_id)
                if not next_slot:
                    return await self._run_analysis_and_store_result(user_id)
                
                prompt = self.slots[next_slot].prompt
                session["history"].append(f"System: {prompt}")
                return prompt

        # 3) If all slots are filled, run the analysis
        next_slot = self._get_next_slot(user_id)
        if not next_slot:
            return await self._run_analysis_and_store_result(user_id)
        else:
            prompt = self.slots[next_slot].prompt
            session["history"].append(f"System: {prompt}")
            return prompt

    async def _run_analysis_and_store_result(self, user_id: str) -> str:
        """Once all slots are filled, run final analysis & recommendation, store the result, and clear the session."""
        session = self.sessions[user_id]
        try:
            data_for_analysis = {
                "po_reference": session["data"].get("po_reference", "N/A"),
                "service_standard": session["data"].get("service_standard", "N/A"),
                "over_the_counter": session["data"].get("over_the_counter", "N/A"),
                "procurement_amount": session["data"].get("procurement_amount", "N/A"),
                "existing_arrangement": session["data"].get("existing_arrangement", "N/A"),
                "procurement_category": session["data"].get("procurement_category", "N/A"),
                "work_complexity": session["data"].get("work_complexity", "N/A"),
                "chat_history": "\n".join(session["history"])
            }

            final_result = await self.full_chain.ainvoke(data_for_analysis)
            session["history"].append(f"System: {final_result}")

            # Store the final analysis and the data used so it can be referenced later
            self.final_results[user_id] = {
                "data": session["data"].copy(),
                "final_analysis": final_result
            }
            # Optionally clear the session (or keep it if you prefer)
            self.sessions[user_id] = {"data": {}, "history": []}
            return final_result
        except Exception as e:
            return f"Error generating final analysis: {str(e)}"

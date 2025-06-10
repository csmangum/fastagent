"""
Account Inquiry Flow

Main flow class that orchestrates the account inquiry conversation flow.
"""

from typing import Any, Callable, Dict, List

from ..base_flow import BaseFlow
from .tools import get_account_inquiry_tools
from .functions import get_account_inquiry_functions
from .prompts import (
    BASE_PROMPT,
    ACCOUNT_VERIFICATION_PROMPT,
    BALANCE_INQUIRY_PROMPT,
    TRANSACTION_HISTORY_PROMPT,
    ACCOUNT_INFO_PROMPT,
    TRANSACTION_SEARCH_PROMPT,
    ACCOUNT_STATUS_PROMPT,
    SYSTEM_INSTRUCTION,
)


class AccountInquiryFlow(BaseFlow):
    """
    Account inquiry conversation flow implementation.
    
    This flow handles various account inquiry types including:
    - Identity verification (required first step)
    - Balance inquiries
    - Transaction history requests
    - Specific transaction searches
    - Account information requests
    - Account status checks
    - Transfer to human agents
    """

    def __init__(self):
        """Initialize the account inquiry flow."""
        super().__init__("account_inquiry")

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get the OpenAI tool definitions for this flow.
        
        Returns:
            List of OpenAI function tool schemas
        """
        return get_account_inquiry_tools()

    def get_functions(self) -> Dict[str, Callable]:
        """
        Get the function implementations for this flow.
        
        Returns:
            Dictionary mapping function names to callable implementations
        """
        return get_account_inquiry_functions()

    def get_prompts(self) -> Dict[str, str]:
        """
        Get the prompts used in this flow.
        
        Returns:
            Dictionary mapping prompt names to prompt templates
        """
        return {
            "base_prompt": BASE_PROMPT,
            "account_verification": ACCOUNT_VERIFICATION_PROMPT,
            "balance_inquiry": BALANCE_INQUIRY_PROMPT,
            "transaction_history": TRANSACTION_HISTORY_PROMPT,
            "account_info": ACCOUNT_INFO_PROMPT,
            "transaction_search": TRANSACTION_SEARCH_PROMPT,
            "account_status": ACCOUNT_STATUS_PROMPT,
            "system_instruction": SYSTEM_INSTRUCTION,
        }

    def get_system_instruction(self) -> str:
        """
        Get the system instruction that should be added for this flow.
        
        Returns:
            System instruction text
        """
        return SYSTEM_INSTRUCTION

    def get_flow_stages(self) -> List[str]:
        """
        Get the ordered list of stages in this flow.
        
        Returns:
            List of stage names in order
        """
        return [
            "identity_verification",
            "inquiry_routing",
            "balance_inquiry",
            "transaction_history",
            "transaction_search", 
            "account_information",
            "account_status",
            "additional_assistance",
            "call_complete"
        ]

    def get_stage_info(self, stage: str) -> Dict[str, Any]:
        """
        Get information about a specific stage in the flow.
        
        Args:
            stage: Stage name
            
        Returns:
            Dictionary containing stage information
            
        Raises:
            ValueError: If stage name is not found
        """
        stage_info = {
            "identity_verification": {
                "description": "Verify customer identity before providing account information",
                "functions": ["verify_customer_identity"],
                "next_stage": "inquiry_routing",
                "required": True
            },
            "inquiry_routing": {
                "description": "Determine the type of account inquiry needed",
                "functions": [],
                "next_stage": "varies based on inquiry type"
            },
            "balance_inquiry": {
                "description": "Provide current account balance and available funds",
                "functions": ["check_account_balance"],
                "next_stage": "additional_assistance"
            },
            "transaction_history": {
                "description": "Provide transaction history for specified period",
                "functions": ["get_transaction_history"],
                "next_stage": "additional_assistance"
            },
            "transaction_search": {
                "description": "Search for specific transactions based on criteria",
                "functions": ["search_specific_transaction"],
                "next_stage": "additional_assistance"
            },
            "account_information": {
                "description": "Provide account details and information",
                "functions": ["get_account_information"],
                "next_stage": "additional_assistance"
            },
            "account_status": {
                "description": "Check account status, holds, and alerts",
                "functions": ["check_account_status"],
                "next_stage": "additional_assistance"
            },
            "additional_assistance": {
                "description": "Offer additional help or services",
                "functions": ["transfer_to_human"],
                "next_stage": "call_complete"
            },
            "call_complete": {
                "description": "End the call",
                "functions": [],
                "next_stage": None
            }
        }
        
        if stage not in stage_info:
            available_stages = list(stage_info.keys())
            raise ValueError(f"Stage '{stage}' not found. Available stages: {available_stages}")
            
        return stage_info[stage]

    def get_inquiry_types(self) -> List[str]:
        """
        Get the types of inquiries this flow can handle.
        
        Returns:
            List of inquiry type names
        """
        return [
            "balance_inquiry",
            "transaction_history", 
            "transaction_search",
            "account_information",
            "account_status"
        ]

    def validate_flow_configuration(self) -> Dict[str, Any]:
        """
        Validate that all flow components are properly configured.
        
        Returns:
            Validation results
        """
        tools = self.get_tools()
        functions = self.get_functions()
        prompts = self.get_prompts()
        
        tool_names = {tool["name"] for tool in tools}
        function_names = set(functions.keys())
        
        # Check that all tools have corresponding functions
        missing_functions = tool_names - function_names
        extra_functions = function_names - tool_names
        
        # Additional validation for account inquiry flow
        required_functions = {
            "verify_customer_identity",
            "check_account_balance", 
            "get_transaction_history",
            "search_specific_transaction",
            "get_account_information",
            "check_account_status",
            "transfer_to_human"
        }
        
        missing_required = required_functions - function_names
        
        return {
            "valid": len(missing_functions) == 0 and len(missing_required) == 0,
            "tool_count": len(tools),
            "function_count": len(functions),
            "prompt_count": len(prompts),
            "missing_functions": list(missing_functions),
            "extra_functions": list(extra_functions),
            "missing_required_functions": list(missing_required),
            "stages": self.get_flow_stages(),
            "inquiry_types": self.get_inquiry_types()
        }

    def get_security_requirements(self) -> Dict[str, Any]:
        """
        Get security requirements for this flow.
        
        Returns:
            Security requirements dictionary
        """
        return {
            "identity_verification_required": True,
            "verification_methods": ["account_number", "ssn", "phone"],
            "sensitive_functions": [
                "check_account_balance",
                "get_transaction_history", 
                "search_specific_transaction",
                "get_account_information",
                "check_account_status"
            ],
            "logging_required": True,
            "transfer_on_verification_failure": True
        } 
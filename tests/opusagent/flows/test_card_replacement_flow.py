"""
Tests for the CardReplacementFlow class and its components.
"""

import pytest
from typing import Any, Dict, List

from opusagent.flows.card_replacement import CardReplacementFlow
from opusagent.flows.card_replacement.tools import get_card_replacement_tools, get_tool_by_name
from opusagent.flows.card_replacement.functions import (
    get_card_replacement_functions, 
    get_function_by_name,
    call_intent,
    member_account_confirmation,
    replacement_reason,
    confirm_address,
    start_card_replacement,
    finish_card_replacement,
    wrap_up,
    transfer_to_human
)
from opusagent.flows.card_replacement.prompts import (
    BASE_PROMPT,
    MEMBER_ACCOUNT_CONFIRMATION_PROMPT,
    REPLACEMENT_REASON_PROMPT,
    CONFIRM_ADDRESS_PROMPT,
    START_CARD_REPLACEMENT_PROMPT,
    FINISH_CARD_REPLACEMENT_PROMPT,
    WRAP_UP_PROMPT,
    SYSTEM_INSTRUCTION,
)


def test_card_replacement_flow_initialization():
    """Test that CardReplacementFlow initializes correctly."""
    flow = CardReplacementFlow()
    assert flow.name == "card_replacement"


def test_card_replacement_flow_tools():
    """Test that get_tools returns the correct tools."""
    flow = CardReplacementFlow()
    tools = flow.get_tools()
    
    # Check that all expected tools are present
    tool_names = {tool["name"] for tool in tools}
    expected_tools = {
        "call_intent",
        "member_account_confirmation",
        "replacement_reason",
        "confirm_address",
        "start_card_replacement",
        "finish_card_replacement",
        "wrap_up",
        "transfer_to_human"
    }
    assert tool_names == expected_tools


def test_card_replacement_flow_functions():
    """Test that get_functions returns the correct functions."""
    flow = CardReplacementFlow()
    functions = flow.get_functions()
    
    # Check that all expected functions are present
    expected_functions = {
        "call_intent",
        "member_account_confirmation",
        "replacement_reason",
        "confirm_address",
        "start_card_replacement",
        "finish_card_replacement",
        "wrap_up",
        "transfer_to_human"
    }
    assert set(functions.keys()) == expected_functions


def test_card_replacement_flow_prompts():
    """Test that get_prompts returns the correct prompts."""
    flow = CardReplacementFlow()
    prompts = flow.get_prompts()
    
    assert prompts["base_prompt"] == BASE_PROMPT
    assert prompts["member_account_confirmation"] == MEMBER_ACCOUNT_CONFIRMATION_PROMPT
    assert prompts["replacement_reason"] == REPLACEMENT_REASON_PROMPT
    assert prompts["confirm_address"] == CONFIRM_ADDRESS_PROMPT
    assert prompts["start_card_replacement"] == START_CARD_REPLACEMENT_PROMPT
    assert prompts["finish_card_replacement"] == FINISH_CARD_REPLACEMENT_PROMPT
    assert prompts["wrap_up"] == WRAP_UP_PROMPT
    assert prompts["system_instruction"] == SYSTEM_INSTRUCTION


def test_card_replacement_flow_system_instruction():
    """Test that get_system_instruction returns the correct instruction."""
    flow = CardReplacementFlow()
    assert flow.get_system_instruction() == SYSTEM_INSTRUCTION


def test_card_replacement_flow_stages():
    """Test that get_flow_stages returns the correct stages."""
    flow = CardReplacementFlow()
    stages = flow.get_flow_stages()
    
    expected_stages = [
        "intent_identification",
        "account_confirmation",
        "reason_collection",
        "address_confirmation",
        "replacement_started",
        "replacement_complete",
        "call_complete"
    ]
    assert stages == expected_stages


def test_card_replacement_flow_stage_info():
    """Test that get_stage_info returns correct information for each stage."""
    flow = CardReplacementFlow()
    
    # Test a valid stage
    stage_info = flow.get_stage_info("intent_identification")
    assert stage_info["description"] == "Identify that customer wants card replacement"
    assert "call_intent" in stage_info["functions"]
    assert stage_info["next_stage"] == "account_confirmation"
    
    # Test an invalid stage
    with pytest.raises(ValueError):
        flow.get_stage_info("invalid_stage")


def test_card_replacement_flow_stage_info_all_stages():
    """Test get_stage_info for all valid stages."""
    flow = CardReplacementFlow()
    stages = flow.get_flow_stages()
    
    for stage in stages:
        stage_info = flow.get_stage_info(stage)
        assert "description" in stage_info
        assert "functions" in stage_info
        assert isinstance(stage_info["functions"], list)
        # Only the last stage should have next_stage as None
        if stage == "call_complete":
            assert stage_info["next_stage"] is None
        else:
            assert stage_info["next_stage"] is not None


def test_card_replacement_flow_validation():
    """Test that validate_flow_configuration returns correct validation results."""
    flow = CardReplacementFlow()
    validation = flow.validate_flow_configuration()
    
    assert validation["valid"] is True
    assert validation["tool_count"] > 0
    assert validation["function_count"] > 0
    assert validation["prompt_count"] > 0
    assert len(validation["missing_functions"]) == 0
    assert len(validation["extra_functions"]) == 0
    assert "stages" in validation


# Enhanced function tests with edge cases and error handling
class TestCallIntentFunction:
    """Test call_intent function with various scenarios."""
    
    def test_card_replacement_intent_minimal(self):
        """Test call_intent with minimal card replacement parameters."""
        result = call_intent({"intent": "card_replacement"})
        assert result["status"] == "success"
        assert result["intent"] == "card_replacement"
        assert "next_action" in result
        assert "available_cards" in result
        assert "prompt_guidance" in result
        assert "captured_context" in result
    
    def test_card_replacement_intent_with_card_type(self):
        """Test call_intent when card type is provided upfront."""
        result = call_intent({
            "intent": "card_replacement",
            "card_type": "Gold card"
        })
        assert result["status"] == "success"
        assert result["captured_context"]["card_type"] == "Gold card"
        assert "Gold card" in result["captured_context"]["captured_info"][0]
    
    def test_card_replacement_intent_with_reason(self):
        """Test call_intent when replacement reason is provided upfront."""
        result = call_intent({
            "intent": "card_replacement",
            "replacement_reason": "Lost"
        })
        assert result["status"] == "success"
        assert result["captured_context"]["replacement_reason"] == "Lost"
        assert "reason: Lost" in result["captured_context"]["captured_info"][0]
    
    def test_card_replacement_intent_with_both_card_and_reason(self):
        """Test call_intent when both card type and reason are provided."""
        result = call_intent({
            "intent": "card_replacement",
            "card_type": "Silver card",
            "replacement_reason": "Damaged"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "confirm_address"
        assert "address" in result["prompt_guidance"].lower()
    
    def test_other_intent(self):
        """Test call_intent with non-card-replacement intent."""
        result = call_intent({"intent": "other"})
        assert result["status"] == "success"
        assert result["intent"] == "other"
        assert result["next_action"] == "continue_conversation"
    
    def test_empty_arguments(self):
        """Test call_intent with empty arguments."""
        result = call_intent({})
        assert result["status"] == "success"
        assert result["intent"] == ""


class TestMemberAccountConfirmationFunction:
    """Test member_account_confirmation function."""
    
    def test_with_default_accounts(self):
        """Test with default member accounts."""
        result = member_account_confirmation({})
        assert result["status"] == "success"
        assert "available_cards" in result
        assert len(result["available_cards"]) == 3  # Default cards
    
    def test_with_specific_card_in_context(self):
        """Test when specific card is already in context."""
        result = member_account_confirmation({
            "card_in_context": "Gold card",
            "member_accounts": ["Gold card", "Silver card"]
        })
        assert result["status"] == "success"
        assert result["confirmed_card"] == "Gold card"
        assert result["next_action"] == "ask_reason"
    
    def test_with_custom_organization(self):
        """Test with custom organization name."""
        result = member_account_confirmation({
            "organization_name": "Custom Bank"
        })
        assert result["status"] == "success"
        assert result["context"]["organization_name"] == "Custom Bank"


class TestReplacementReasonFunction:
    """Test replacement_reason function."""
    
    def test_with_reason_provided(self):
        """Test when reason is provided."""
        result = replacement_reason({
            "card_in_context": "Gold card",
            "reason": "Lost"
        })
        assert result["status"] == "success"
        assert result["selected_reason"] == "Lost"
        assert result["next_action"] == "collect_address"
    
    def test_without_reason(self):
        """Test when reason is not provided."""
        result = replacement_reason({
            "card_in_context": "Silver card"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "ask_reason"
        assert "Silver card" in result["prompt_guidance"]
    
    def test_valid_reasons_included(self):
        """Test that valid reasons are included in response."""
        result = replacement_reason({"card_in_context": "Gold card"})
        assert "valid_reasons" in result
        expected_reasons = ["Lost", "Damaged", "Stolen", "Other"]
        assert result["valid_reasons"] == expected_reasons


class TestConfirmAddressFunction:
    """Test confirm_address function."""
    
    def test_with_confirmed_address(self):
        """Test when address is confirmed."""
        result = confirm_address({
            "card_in_context": "Gold card",
            "address_on_file": "123 Main St",
            "confirmed_address": "123 Main St"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "start_replacement"
        assert result["confirmed_address"] == "123 Main St"
    
    def test_without_confirmed_address(self):
        """Test when address is not confirmed."""
        result = confirm_address({
            "card_in_context": "Gold card",
            "address_on_file": "123 Main St"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "confirm_address"


class TestTransferToHumanFunction:
    """Test transfer_to_human function."""
    
    def test_with_minimal_parameters(self):
        """Test transfer with minimal parameters."""
        result = transfer_to_human({"reason": "complex issue"})
        assert result["status"] == "success"
        assert "transfer_id" in result
        assert result["transfer_id"].startswith("TR-")
        assert result["priority"] == "normal"  # default
    
    def test_with_high_priority(self):
        """Test transfer with high priority."""
        result = transfer_to_human({
            "reason": "urgent issue",
            "priority": "high"
        })
        assert result["status"] == "success"
        assert result["priority"] == "high"
    
    def test_with_context(self):
        """Test transfer with additional context."""
        context = {"stage": "address_confirmation", "attempts": 3}
        result = transfer_to_human({
            "reason": "repeated failures",
            "context": context
        })
        assert result["status"] == "success"
        assert "stage" in result["context"]
        assert result["context"]["stage"] == "address_confirmation"


class TestToolDefinitions:
    """Test tool definitions and their schemas."""
    
    def test_get_tool_by_name_valid(self):
        """Test getting a valid tool by name."""
        tool = get_tool_by_name("call_intent")
        assert tool["name"] == "call_intent"
        assert tool["type"] == "function"
        assert "parameters" in tool
    
    def test_get_tool_by_name_invalid(self):
        """Test getting an invalid tool by name."""
        with pytest.raises(ValueError) as exc_info:
            get_tool_by_name("invalid_tool")
        assert "not found" in str(exc_info.value)
    
    def test_all_tools_have_required_fields(self):
        """Test that all tools have required fields."""
        tools = get_card_replacement_tools()
        required_fields = ["type", "name", "description", "parameters"]
        
        for tool in tools:
            for field in required_fields:
                assert field in tool, f"Tool {tool.get('name', 'unknown')} missing {field}"
            
            # Check parameters structure
            params = tool["parameters"]
            assert "type" in params
            assert "properties" in params


class TestFunctionRegistry:
    """Test function registry and lookup."""
    
    def test_get_function_by_name_valid(self):
        """Test getting a valid function by name."""
        func = get_function_by_name("call_intent")
        assert callable(func)
        assert func == call_intent
    
    def test_get_function_by_name_invalid(self):
        """Test getting an invalid function by name."""
        with pytest.raises(ValueError) as exc_info:
            get_function_by_name("invalid_function")
        assert "not found" in str(exc_info.value)
    
    def test_all_functions_are_callable(self):
        """Test that all registered functions are callable."""
        functions = get_card_replacement_functions()
        for name, func in functions.items():
            assert callable(func), f"Function {name} is not callable"


def test_card_replacement_functions():
    """Test individual card replacement functions."""
    functions = get_card_replacement_functions()
    
    # Test call_intent
    result = functions["call_intent"]({"intent": "card_replacement"})
    assert result["status"] == "success"
    assert result["intent"] == "card_replacement"
    assert "next_action" in result
    assert "available_cards" in result
    
    # Test member_account_confirmation
    result = functions["member_account_confirmation"]({
        "member_accounts": ["Gold card", "Silver card"],
        "organization_name": "Test Bank"
    })
    assert result["status"] == "success"
    assert "prompt_guidance" in result
    assert "next_action" in result
    
    # Test replacement_reason
    result = functions["replacement_reason"]({
        "card_in_context": "Gold card",
        "reason": "Lost"
    })
    assert result["status"] == "success"
    assert result["selected_reason"] == "Lost"
    assert "next_action" in result
    
    # Test confirm_address
    result = functions["confirm_address"]({
        "card_in_context": "Gold card",
        "address_on_file": "123 Main St",
        "confirmed_address": "123 Main St"
    })
    assert result["status"] == "success"
    assert result["confirmed_address"] == "123 Main St"
    assert "next_action" in result
    
    # Test start_card_replacement
    result = functions["start_card_replacement"]({
        "card_in_context": "Gold card",
        "address_in_context": "123 Main St"
    })
    assert result["status"] == "success"
    assert "prompt_guidance" in result
    assert "next_action" in result
    
    # Test finish_card_replacement
    result = functions["finish_card_replacement"]({
        "card_in_context": "Gold card",
        "address_in_context": "123 Main St",
        "delivery_time": "5-7 business days"
    })
    assert result["status"] == "success"
    assert "delivery_time" in result
    assert "next_action" in result
    
    # Test wrap_up
    result = functions["wrap_up"]({
        "organization_name": "Test Bank"
    })
    assert result["status"] == "success"
    assert "prompt_guidance" in result
    assert "next_action" in result
    
    # Test transfer_to_human
    result = functions["transfer_to_human"]({
        "reason": "complex issue",
        "priority": "normal",
        "context": {"stage": "test"}
    })
    assert result["status"] == "success"
    assert "transfer_id" in result
    assert "priority" in result
    assert "next_action" in result


def test_card_replacement_tools():
    """Test card replacement tool definitions."""
    tools = get_card_replacement_tools()
    
    # Verify all required tools are present
    tool_names = {tool["name"] for tool in tools}
    expected_tools = {
        "call_intent",
        "member_account_confirmation",
        "replacement_reason",
        "confirm_address",
        "start_card_replacement",
        "finish_card_replacement",
        "wrap_up",
        "transfer_to_human"
    }
    assert tool_names == expected_tools
    
    # Verify tool parameters
    for tool in tools:
        assert "type" in tool
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert "type" in tool["parameters"]
        assert "properties" in tool["parameters"]


# Integration tests
class TestFlowIntegration:
    """Test integration between flow components."""
    
    def test_tools_functions_alignment(self):
        """Test that tools and functions are properly aligned."""
        flow = CardReplacementFlow()
        tools = flow.get_tools()
        functions = flow.get_functions()
        
        tool_names = {tool["name"] for tool in tools}
        function_names = set(functions.keys())
        
        assert tool_names == function_names, "Tools and functions should match exactly"
    
    def test_stage_functions_exist(self):
        """Test that all functions referenced in stages actually exist."""
        flow = CardReplacementFlow()
        functions = flow.get_functions()
        stages = flow.get_flow_stages()
        
        for stage in stages:
            stage_info = flow.get_stage_info(stage)
            for func_name in stage_info["functions"]:
                assert func_name in functions, f"Function {func_name} referenced in stage {stage} but not found"


class TestCompleteFlowScenarios:
    """Test complete card replacement flow scenarios."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.flow = CardReplacementFlow()
        self.functions = self.flow.get_functions()
        self.context = {}
    
    def execute_flow_step(self, function_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single flow step and update context."""
        function = self.functions[function_name]
        result = function(params)
        self.context.update(result.get("context", {}))
        return result
    
    def test_complete_flow_minimal_info(self):
        """Test complete flow with minimal information provided upfront."""
        # Step 1: Intent identification
        result = self.execute_flow_step("call_intent", {"intent": "card_replacement"})
        assert result["status"] == "success"
        assert result["next_action"] == "ask_card_type"
        
        # Step 2: Account confirmation
        result = self.execute_flow_step("member_account_confirmation", {
            "member_accounts": ["Gold card", "Silver card"],
            "organization_name": "Test Bank"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "confirm_card_selection"
        
        # Step 3: Reason collection
        result = self.execute_flow_step("replacement_reason", {
            "card_in_context": "Gold card",
            "reason": "Lost"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "collect_address"
        
        # Step 4: Address confirmation
        result = self.execute_flow_step("confirm_address", {
            "card_in_context": "Gold card",
            "address_on_file": "123 Main St",
            "confirmed_address": "123 Main St"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "start_replacement"
        
        # Step 5: Start replacement
        result = self.execute_flow_step("start_card_replacement", {
            "card_in_context": "Gold card",
            "address_in_context": "123 Main St"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "finish_replacement"
        
        # Step 6: Finish replacement
        result = self.execute_flow_step("finish_card_replacement", {
            "card_in_context": "Gold card",
            "address_in_context": "123 Main St",
            "delivery_time": "5-7 business days"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "wrap_up"
        
        # Step 7: Wrap up
        result = self.execute_flow_step("wrap_up", {
            "organization_name": "Test Bank"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "end_call"
    
    def test_complete_flow_with_upfront_info(self):
        """Test complete flow with all information provided upfront."""
        # Step 1: Intent identification with all context
        result = self.execute_flow_step("call_intent", {
            "intent": "card_replacement",
            "card_type": "Silver card",
            "replacement_reason": "Damaged"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "confirm_address"
        
        # Step 2: Address confirmation (skipping account and reason steps)
        result = self.execute_flow_step("confirm_address", {
            "card_in_context": "Silver card",
            "address_on_file": "456 Oak St",
            "confirmed_address": "456 Oak St"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "start_replacement"
        
        # Step 3: Start replacement
        result = self.execute_flow_step("start_card_replacement", {
            "card_in_context": "Silver card",
            "address_in_context": "456 Oak St"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "finish_replacement"
        
        # Step 4: Finish replacement
        result = self.execute_flow_step("finish_card_replacement", {
            "card_in_context": "Silver card",
            "address_in_context": "456 Oak St",
            "delivery_time": "5-7 business days"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "wrap_up"
        
        # Step 5: Wrap up
        result = self.execute_flow_step("wrap_up", {
            "organization_name": "Test Bank"
        })
        assert result["status"] == "success"
        assert result["next_action"] == "end_call"
    
    def test_flow_with_human_transfer(self):
        """Test flow that ends with transfer to human."""
        # Step 1: Intent identification
        result = self.execute_flow_step("call_intent", {"intent": "card_replacement"})
        assert result["status"] == "success"
        
        # Step 2: Account confirmation
        result = self.execute_flow_step("member_account_confirmation", {
            "member_accounts": ["Gold card", "Silver card"],
            "organization_name": "Test Bank"
        })
        assert result["status"] == "success"
        
        # Step 3: Transfer to human
        result = self.execute_flow_step("transfer_to_human", {
            "reason": "complex issue",
            "priority": "high",
            "context": self.context
        })
        assert result["status"] == "success"
        assert result["next_action"] == "end_conversation"
        assert "transfer_id" in result
        assert result["priority"] == "high"
    
    def test_flow_with_address_change(self):
        """Test flow with address change during confirmation."""
        # Step 1: Intent identification
        result = self.execute_flow_step("call_intent", {
            "intent": "card_replacement",
            "card_type": "Gold card"
        })
        assert result["status"] == "success"
        
        # Step 2: Reason collection
        result = self.execute_flow_step("replacement_reason", {
            "card_in_context": "Gold card",
            "reason": "Lost"
        })
        assert result["status"] == "success"
        
        # Step 3: Address confirmation with new address
        result = self.execute_flow_step("confirm_address", {
            "card_in_context": "Gold card",
            "address_on_file": "123 Main St",
            "confirmed_address": "789 New Ave"
        })
        assert result["status"] == "success"
        assert result["confirmed_address"] == "789 New Ave"
        
        # Step 4: Start replacement with new address
        result = self.execute_flow_step("start_card_replacement", {
            "card_in_context": "Gold card",
            "address_in_context": "789 New Ave"
        })
        assert result["status"] == "success"
        
        # Step 5: Finish replacement
        result = self.execute_flow_step("finish_card_replacement", {
            "card_in_context": "Gold card",
            "address_in_context": "789 New Ave",
            "delivery_time": "5-7 business days"
        })
        assert result["status"] == "success"
        
        # Step 6: Wrap up
        result = self.execute_flow_step("wrap_up", {
            "organization_name": "Test Bank"
        })
        assert result["status"] == "success"
    
    def test_flow_with_multiple_cards(self):
        """Test flow with multiple card selection."""
        # Step 1: Intent identification
        result = self.execute_flow_step("call_intent", {"intent": "card_replacement"})
        assert result["status"] == "success"
        
        # Step 2: Account confirmation with multiple cards
        result = self.execute_flow_step("member_account_confirmation", {
            "member_accounts": ["Gold card", "Silver card", "Platinum card"],
            "organization_name": "Test Bank"
        })
        assert result["status"] == "success"
        assert len(result["available_cards"]) == 3
        
        # Continue with normal flow...
        result = self.execute_flow_step("replacement_reason", {
            "card_in_context": "Platinum card",
            "reason": "Stolen"
        })
        assert result["status"] == "success"
        
        result = self.execute_flow_step("confirm_address", {
            "card_in_context": "Platinum card",
            "address_on_file": "123 Main St",
            "confirmed_address": "123 Main St"
        })
        assert result["status"] == "success"
        
        result = self.execute_flow_step("start_card_replacement", {
            "card_in_context": "Platinum card",
            "address_in_context": "123 Main St"
        })
        assert result["status"] == "success"
        
        result = self.execute_flow_step("finish_card_replacement", {
            "card_in_context": "Platinum card",
            "address_in_context": "123 Main St",
            "delivery_time": "5-7 business days"
        })
        assert result["status"] == "success"
        
        result = self.execute_flow_step("wrap_up", {
            "organization_name": "Test Bank"
        })
        assert result["status"] == "success" 
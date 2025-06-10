"""
Flow Manager

Manages registration and coordination of multiple conversation flows.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_flow import BaseFlow
from .card_replacement import CardReplacementFlow
from .loan_application import LoanApplicationFlow
from .account_inquiry import AccountInquiryFlow

logger = logging.getLogger(__name__)


class FlowManager:
    """
    Manages multiple conversation flows and their integration.

    This class handles:
    - Flow registration and discovery
    - Tool and function aggregation across flows
    - System instruction composition
    - Flow validation
    """

    def __init__(self):
        """Initialize the flow manager."""
        self.flows: Dict[str, BaseFlow] = {}
        self.active_flows: List[str] = []

    def register_flow(self, flow: BaseFlow) -> None:
        """
        Register a conversation flow.

        Args:
            flow: Flow instance to register
        """
        self.flows[flow.name] = flow
        logger.info(f"Registered flow: {flow.name}")

    def unregister_flow(self, flow_name: str) -> bool:
        """
        Unregister a conversation flow.

        Args:
            flow_name: Name of the flow to unregister

        Returns:
            True if flow was removed, False if it didn't exist
        """
        if flow_name in self.flows:
            del self.flows[flow_name]
            if flow_name in self.active_flows:
                self.active_flows.remove(flow_name)
            logger.info(f"Unregistered flow: {flow_name}")
            return True
        return False

    def activate_flow(self, flow_name: str) -> None:
        """
        Activate a flow (include it in tool/function registration).

        Args:
            flow_name: Name of the flow to activate

        Raises:
            ValueError: If flow name is not registered
        """
        if flow_name not in self.flows:
            available_flows = list(self.flows.keys())
            raise ValueError(
                f"Flow '{flow_name}' not registered. Available flows: {available_flows}"
            )

        if flow_name not in self.active_flows:
            self.active_flows.append(flow_name)
            logger.info(f"Activated flow: {flow_name}")

    def deactivate_flow(self, flow_name: str) -> None:
        """
        Deactivate a flow (exclude it from tool/function registration).

        Args:
            flow_name: Name of the flow to deactivate
        """
        if flow_name in self.active_flows:
            self.active_flows.remove(flow_name)
            logger.info(f"Deactivated flow: {flow_name}")

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools from active flows.

        Returns:
            Combined list of all OpenAI tool definitions
        """
        all_tools = []
        for flow_name in self.active_flows:
            flow = self.flows[flow_name]
            tools = flow.get_tools()
            all_tools.extend(tools)

        logger.info(
            f"Collected {len(all_tools)} tools from {len(self.active_flows)} active flows"
        )
        return all_tools

    def get_all_functions(self) -> Dict[str, Any]:
        """
        Get all functions from active flows.

        Returns:
            Combined dictionary of all function implementations
        """
        all_functions = {}
        for flow_name in self.active_flows:
            flow = self.flows[flow_name]
            functions = flow.get_functions()

            # Check for function name conflicts
            for func_name in functions:
                if func_name in all_functions:
                    logger.warning(
                        f"Function name conflict: '{func_name}' exists in multiple flows"
                    )
                    # Could implement conflict resolution strategy here

            all_functions.update(functions)

        logger.info(
            f"Collected {len(all_functions)} functions from {len(self.active_flows)} active flows"
        )
        return all_functions

    def get_combined_system_instruction(self) -> str:
        """
        Get combined system instruction from all active flows.

        Returns:
            Combined system instruction text
        """
        instructions = []
        for flow_name in self.active_flows:
            flow = self.flows[flow_name]
            instruction = flow.get_system_instruction()
            if instruction:
                instructions.append(f"# {flow_name.title()} Flow\n{instruction}")

        if not instructions:
            return "You are a helpful AI assistant."

        combined = "\n\n".join(instructions)
        logger.info(f"Combined system instructions from {len(self.active_flows)} flows")
        return combined

    def register_with_function_handler(self, function_handler) -> None:
        """
        Register all active flow functions with a function handler.

        Args:
            function_handler: FunctionHandler instance to register functions with
        """
        for flow_name in self.active_flows:
            flow = self.flows[flow_name]
            flow.register_with_handler(function_handler)
            logger.info(f"Registered {flow_name} flow functions with handler")

    def get_flow_info(self, flow_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about flows.

        Args:
            flow_name: Specific flow name, or None for all flows

        Returns:
            Flow information dictionary
        """
        if flow_name:
            if flow_name not in self.flows:
                available_flows = list(self.flows.keys())
                raise ValueError(
                    f"Flow '{flow_name}' not found. Available flows: {available_flows}"
                )
            return self.flows[flow_name].get_flow_info()

        return {
            "registered_flows": list(self.flows.keys()),
            "active_flows": self.active_flows.copy(),
            "total_tools": len(self.get_all_tools()),
            "total_functions": len(self.get_all_functions()),
            "flows": {name: flow.get_flow_info() for name, flow in self.flows.items()},
        }

    def validate_all_flows(self) -> Dict[str, Any]:
        """
        Validate all registered flows.

        Returns:
            Validation results for all flows
        """
        results = {}
        for flow_name, flow in self.flows.items():
            if hasattr(flow, "validate_flow_configuration"):
                results[flow_name] = flow.validate_flow_configuration()
            else:
                results[flow_name] = {
                    "valid": True,
                    "note": "No validation method available",
                }

        return results

    def reset(self) -> None:
        """Reset the flow manager to initial state."""
        self.flows.clear()
        self.active_flows.clear()
        logger.info("Flow manager reset")


def create_default_flow_manager() -> FlowManager:
    """
    Create a flow manager with default flows registered.

    Returns:
        FlowManager instance with default flows
    """
    manager = FlowManager()

    # Register default flows
    manager.register_flow(CardReplacementFlow())
    manager.register_flow(LoanApplicationFlow())
    manager.register_flow(AccountInquiryFlow())

    # Activate card replacement by default
    manager.activate_flow("card_replacement")

    logger.info("Created default flow manager with card replacement activated")
    return manager

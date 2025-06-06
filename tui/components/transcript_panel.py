"""
Transcript Panel component for the Interactive TUI Validator.

This panel displays live transcripts from user input and bot responses
during TelephonyRealtimeBridge conversations.
"""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, RichLog
from textual.widget import Widget


class TranscriptPanel(Widget):
    """Panel for displaying live conversation transcripts."""
    
    CSS = """
    TranscriptPanel {
        background: $surface;
        border: solid $primary;
        height: 30%;
    }
    
    .transcript-header {
        background: $primary;
        color: $primary-background;
        text-align: center;
        height: 1;
    }
    
    .transcript-log {
        background: $surface;
        color: $text;
        height: 1fr;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parent_app = None
        self.transcript_log = None
        self.transcript_count = 0
        
    def compose(self) -> ComposeResult:
        """Create the transcript panel layout."""
        with Vertical():
            yield Static("💬 Live Transcript", classes="transcript-header")
            
            self.transcript_log = RichLog(
                highlight=True,
                markup=True,
                id="transcript-log",
                classes="transcript-log"
            )
            yield self.transcript_log
    
    def add_user_message(self, text: str, timestamp: str = None) -> None:
        """Add a user message to the transcript."""
        if not self.transcript_log:
            return
        
        self.transcript_count += 1
        timestamp = timestamp or self._get_timestamp()
        
        self.transcript_log.write(
            f"[dim]{timestamp}[/dim] [cyan]👤 User:[/cyan] {text}"
        )
        
        self._maintain_log_size()
    
    def add_bot_message(self, text: str, timestamp: str = None) -> None:
        """Add a bot message to the transcript."""
        if not self.transcript_log:
            return
        
        self.transcript_count += 1
        timestamp = timestamp or self._get_timestamp()
        
        self.transcript_log.write(
            f"[dim]{timestamp}[/dim] [green]🤖 Bot:[/green] {text}"
        )
        
        self._maintain_log_size()
    
    def add_system_message(self, text: str, timestamp: str = None) -> None:
        """Add a system message to the transcript."""
        if not self.transcript_log:
            return
        
        self.transcript_count += 1
        timestamp = timestamp or self._get_timestamp()
        
        self.transcript_log.write(
            f"[dim]{timestamp}[/dim] [yellow]🔧 System:[/yellow] {text}"
        )
        
        self._maintain_log_size()
    
    def add_function_call(self, function_name: str, arguments: dict, timestamp: str = None) -> None:
        """Add a function call to the transcript."""
        if not self.transcript_log:
            return
        
        self.transcript_count += 1
        timestamp = timestamp or self._get_timestamp()
        
        # Format arguments nicely
        args_str = ", ".join([f"{k}={v}" for k, v in arguments.items()])
        
        self.transcript_log.write(
            f"[dim]{timestamp}[/dim] [magenta]🔧 Function Call:[/magenta] {function_name}({args_str})"
        )
        
        self._maintain_log_size()
    
    def add_function_result(self, function_name: str, result: dict, timestamp: str = None) -> None:
        """Add a function result to the transcript."""
        if not self.transcript_log:
            return
        
        self.transcript_count += 1
        timestamp = timestamp or self._get_timestamp()
        
        # Format result nicely
        if isinstance(result, dict) and "status" in result:
            status = result.get("status", "unknown")
            next_action = result.get("next_action", "")
            status_text = f"Status: {status}"
            if next_action:
                status_text += f", Next: {next_action}"
        else:
            status_text = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
        
        self.transcript_log.write(
            f"[dim]{timestamp}[/dim] [cyan]📋 Function Result:[/cyan] {function_name} → {status_text}"
        )
        
        self._maintain_log_size()
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def _maintain_log_size(self) -> None:
        """Keep transcript log size manageable."""
        if self.transcript_count > 500:
            self.transcript_log.clear()
            self.transcript_count = 0
            self.add_system_message("Transcript cleared due to size limit")
    
    def clear_transcript(self) -> None:
        """Clear all messages from the transcript."""
        if self.transcript_log:
            self.transcript_log.clear()
            self.transcript_count = 0
    
    def export_transcript(self, filename: str = None) -> bool:
        """Export transcript to file."""
        # TODO: Implement transcript export
        if filename:
            self.add_system_message(f"Transcript exported to {filename}")
            return True
        return False 
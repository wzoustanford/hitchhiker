from collections import defaultdict
from threading import Lock

class ChatHistoryManager:
    """
    Manages chat history per IP address with a maximum context window.
    Thread-safe for concurrent requests.
    """
    def __init__(self, max_messages=30, system_prompt=None):
        """
        Args:
            max_messages: Maximum number of user/assistant message pairs to keep
            system_prompt: System prompt to use for all conversations
        """
        self.max_messages = max_messages
        self.system_prompt = system_prompt or "You are a helpful financial assistant and an expert with US equities."
        self.histories = defaultdict(list)  # IP -> list of messages
        self.lock = Lock()

    def add_message(self, ip_address, role, content):
        """
        Add a message to the history for a specific IP.

        Args:
            ip_address: Client IP address
            role: Message role ('user' or 'assistant')
            content: Message content

        Returns:
            bool: True if added successfully, False if context limit exceeded
        """
        with self.lock:
            history = self.histories[ip_address]

            # Count user/assistant pairs (excluding system message)
            non_system_messages = [m for m in history if m['role'] != 'system']
            message_pairs = len(non_system_messages) // 2

            # Check if we've exceeded the limit
            if role == 'user' and message_pairs >= self.max_messages:
                # Clear history and start fresh
                self.clear_history(ip_address)
                return False

            history.append({"role": role, "content": content})
            return True

    def get_messages(self, ip_address):
        """
        Get the full message history for an IP, including system prompt.

        Args:
            ip_address: Client IP address

        Returns:
            List of message dicts in format: [{"role": "...", "content": "..."}]
        """
        with self.lock:
            history = self.histories[ip_address]

            # Always include system prompt at the start
            if not history or history[0]['role'] != 'system':
                return [{"role": "system", "content": self.system_prompt}] + history
            return history

    def clear_history(self, ip_address):
        """
        Clear the chat history for a specific IP.

        Args:
            ip_address: Client IP address
        """
        with self.lock:
            self.histories[ip_address] = []

    def get_history_length(self, ip_address):
        """
        Get the number of message pairs (excluding system prompt) for an IP.

        Args:
            ip_address: Client IP address

        Returns:
            int: Number of user/assistant message pairs
        """
        with self.lock:
            history = self.histories[ip_address]
            non_system_messages = [m for m in history if m['role'] != 'system']
            return len(non_system_messages) // 2

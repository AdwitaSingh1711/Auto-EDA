import pytest
from unittest.mock import patch
from autoeda.LLMAnalyzer import LLMAnalyzer

class TestLLMAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return LLMAnalyzer()
    
    @patch('autoeda.LLMAnalyzer.ollama.chat')
    def test_ollama_connection_success(self, mock_chat, analyzer):
        mock_chat.return_value = {
            'message': {
                'content': 'Hello World'
            }
        }
        assert analyzer.test_ollama_connection() is True
    
    @patch('autoeda.LLMAnalyzer.ollama.chat')
    def test_ollama_connection_failure(self, mock_chat, analyzer):
        mock_chat.side_effect = ConnectionError
        assert analyzer.test_ollama_connection() is False
    
    @patch('autoeda.LLMAnalyzer.ollama.chat')
    def test_ollama_connection_empty_response(self, mock_chat, analyzer):
        mock_chat.return_value = {}
        assert analyzer.test_ollama_connection() is False
    
    @patch('autoeda.LLMAnalyzer.ollama.chat')
    def test_ollama_connection_no_message_key(self, mock_chat, analyzer):
        mock_chat.return_value = {'some_other_key': 'value'}
        assert analyzer.test_ollama_connection() is False
"""
Language detection and EN↔HI translation module.
Uses langdetect for language identification and deep-translator for translation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Translator:
    """Handles language detection and English↔Hindi translation."""

    def __init__(self):
        from langdetect import DetectorFactory
        # Set seed for deterministic results
        DetectorFactory.seed = 0
        self._en_to_hi = None
        self._hi_to_en = None
        print("Translator initialized (langdetect + deep-translator)")

    def _get_en_to_hi(self):
        """Lazy-load EN→HI translator."""
        if self._en_to_hi is None:
            from deep_translator import GoogleTranslator
            self._en_to_hi = GoogleTranslator(source='en', target='hi')
        return self._en_to_hi

    def _get_hi_to_en(self):
        """Lazy-load HI→EN translator."""
        if self._hi_to_en is None:
            from deep_translator import GoogleTranslator
            self._hi_to_en = GoogleTranslator(source='hi', target='en')
        return self._hi_to_en

    def detect_language(self, text):
        """
        Detect whether text is Hindi or English.
        
        Args:
            text: Input text string
            
        Returns:
            'hi' for Hindi, 'en' for English
        """
        from langdetect import detect

        if not text or len(text.strip()) < 2:
            return 'en'  # Default to English for very short text

        try:
            lang = detect(text)
            # Map to our supported languages
            if lang == 'hi':
                return 'hi'
            else:
                return 'en'  # Treat everything non-Hindi as English
        except Exception:
            return 'en'  # Default fallback

    def translate_to_hindi(self, text):
        """
        Translate English text to Hindi.
        
        Args:
            text: English input string
            
        Returns:
            Hindi translated string
        """
        if not text or len(text.strip()) == 0:
            return text

        try:
            result = self._get_en_to_hi().translate(text)
            return result if result else text
        except Exception as e:
            print(f"  Warning: EN->HI translation failed: {e}")
            return text  # Return original as fallback

    def translate_to_english(self, text):
        """
        Translate Hindi text to English.
        
        Args:
            text: Hindi input string
            
        Returns:
            English translated string
        """
        if not text or len(text.strip()) == 0:
            return text

        try:
            result = self._get_hi_to_en().translate(text)
            return result if result else text
        except Exception as e:
            print(f"  Warning: HI->EN translation failed: {e}")
            return text  # Return original as fallback

    def process_input(self, text):
        """
        Detect language and translate to Hindi if needed.
        
        Returns:
            (hindi_text, original_language)
        """
        lang = self.detect_language(text)

        if lang == 'hi':
            return text, 'hi'
        else:
            hindi_text = self.translate_to_hindi(text)
            return hindi_text, 'en'

    def process_output(self, hindi_response, target_language):
        """
        Translate Hindi response to target language if needed.
        
        Args:
            hindi_response: Response in Hindi
            target_language: 'hi' or 'en'
            
        Returns:
            Response in target language
        """
        if target_language == 'hi':
            return hindi_response
        else:
            return self.translate_to_english(hindi_response)


if __name__ == "__main__":
    # Quick test
    translator = Translator()

    test_texts = [
        "What are the benefits of Ashwagandha?",
        "अश्वगंधा के फायदे क्या हैं?",
        "How to balance Vata dosha?",
        "त्रिफला क्या है?",
    ]

    for text in test_texts:
        lang = translator.detect_language(text)
        print(f"\nInput: {text}")
        print(f"   Language: {lang}")
        
        hindi_text, orig_lang = translator.process_input(text)
        print(f"   Hindi: {hindi_text}")
        
        if orig_lang == 'en':
            back_to_en = translator.translate_to_english(hindi_text)
            print(f"   Back to EN: {back_to_en}")

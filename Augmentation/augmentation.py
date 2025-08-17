import os
import time
import random
import logging
import pandas as pd
import yaml
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
from nlpaug.augmenter.char import KeyboardAug

# Load environment variables
load_dotenv()

# Import prompts from constants
from constants.constants import MODEL_PROMPTS

logger = logging.getLogger(__name__)

class Augmentation:
    """
    Modular augmentation class for generating 5 versions of QA pairs:
    1. Original
    2. Llama3 formal paraphrase
    3. Mistral 7B casual paraphrase
    4. Gemini 2.5 Flash paraphrase
    5. Typo noise version of Gemini output
    """
    
    def __init__(self, input_path: str = None, output_path: str = None, device: str = 'cuda:3'):
        """
        Initialize the augmentation system.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output CSV file
            device: CUDA device to use (default: cuda:3)
        """
        # Force device to cuda:3
        self.device = 'cuda:3'
        logger.info(f"Forcing device to: {self.device}")
        
        # Load configuration
        with open("config/config.yaml") as f:
            self.config = yaml.safe_load(f)
        
        # Set paths
        self.input_path = input_path or self.config['paths']['input_path']
        self.output_path = output_path or self.config['paths']['augmented_path']
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Initialize models and API keys
        self._initialize_models()
        self._initialize_api_keys()
        
        logger.info(f"Augmentation initialized with device: {self.device}")
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Output: {self.output_path}")
    
    def _initialize_models(self):
        """Initialize all models for paraphrasing."""
        logger.info("Initializing models...")
        
        # Check for Hugging Face token
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            logger.warning("No Hugging Face token found. Local models will not be loaded.")
            self.llama_model = None
            self.mistral_model = None
            return
        
        try:
            # Initialize Llama3 model
            logger.info("Loading Llama3 model...")
            llama_config = self.config['models']['llama3']
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                llama_config['model_id'],
                token=hf_token
            )
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_config['model_id'],
                device_map=self.device,
                token=hf_token
            )
            logger.info("Llama3 model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Llama3 model: {e}")
            self.llama_model = None
        
        try:
            # Initialize Mistral model
            logger.info("Loading Mistral model...")
            mistral_config = self.config['models']['mistral']
            self.mistral_tokenizer = AutoTokenizer.from_pretrained(
                mistral_config['model_id'],
                token=hf_token
            )
            self.mistral_model = AutoModelForCausalLM.from_pretrained(
                mistral_config['model_id'],
                device_map=self.device,
                token=hf_token
            )
            logger.info("Mistral model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Mistral model: {e}")
            self.mistral_model = None
        
        # Initialize typo augmentation
        typo_config = self.config['models']['typo_augmentation']
        self.typo_aug = KeyboardAug(
            aug_char_min=1,
            aug_char_max=1,
            aug_char_p=1.0 / typo_config['words_per_typo'],  # 1 typo per N words
            include_upper_case=typo_config['include_upper_case'],
            include_special_char=typo_config['include_special_char'],
            include_numeric=typo_config['include_numeric']
        )
        logger.info("Typo augmentation initialized")
    
    def _initialize_api_keys(self):
        """Initialize Gemini API keys for rotation."""
        logger.info("Initializing API keys...")
        self.gemini_keys = []
        for i in range(1, 7):
            key = os.getenv(f'GEMINI_API_KEY_{i}')
            if key and key.strip() and key != f'your_gemini_api_key_{i}_here':
                self.gemini_keys.append(key)
                logger.info(f"Found Gemini API key {i}")
        
        if not self.gemini_keys:
            logger.warning("No valid Gemini API keys found. Gemini paraphrasing will be skipped.")
        else:
            logger.info(f"Found {len(self.gemini_keys)} Gemini API keys for rotation")
        
        self.current_key_index = 0
        self.gemini_config = self.config['models']['gemini']
    
    def call_llm(self, model_type: str, prompt: str, content_type: str = "question", **kwargs) -> str:
        """
        Unified method to call different LLM models.
        
        Args:
            model_type: Type of model ('llama3', 'mistral', 'gemini')
            prompt: Input prompt for the model
            content_type: Type of content ('question' or 'answer')
            **kwargs: Additional arguments for the model
            
        Returns:
            Generated text from the model
        """
        try:
            if model_type == 'llama3':
                return self._call_llama3(prompt, content_type, **kwargs)
            elif model_type == 'mistral':
                return self._call_mistral(prompt, content_type, **kwargs)
            elif model_type == 'gemini':
                return self._call_gemini(prompt, content_type, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            logger.error(f"Error calling {model_type}: {e}")
            return prompt  # Return original text on error
    
    def _call_llama3(self, prompt: str, content_type: str = "question", **kwargs) -> str:
        """Call Llama3 model for formal paraphrasing."""
        if not self.llama_model:
            logger.warning("Llama3 model not available, returning original text")
            return prompt
        
        try:
            llama_config = self.config['models']['llama3']
            
            # Get the appropriate prompt template
            prompt_template = MODEL_PROMPTS['llama3_formal'][content_type]
            formatted_prompt = prompt_template.format(text=prompt)
            
            # Add Llama3 instruction format
            llama_prompt = f"<s>[INST] {formatted_prompt} [/INST]"
            
            inputs = self.llama_tokenizer(llama_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.llama_model.generate(
                    **inputs,
                    max_new_tokens=llama_config['max_tokens'],
                    temperature=llama_config['temperature'],
                    do_sample=True,
                    pad_token_id=self.llama_tokenizer.eos_token_id
                )
            
            response = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            response = response.split("[/INST]")[-1].strip()
            
            if response and response != prompt:
                logger.info(f"Llama3 generated: {response[:50]}...")
                return response
            else:
                logger.warning("Llama3 returned empty or same text")
                return prompt
            
        except Exception as e:
            logger.error(f"Llama3 error: {e}")
            return prompt
    
    def _call_mistral(self, prompt: str, content_type: str = "question", **kwargs) -> str:
        """Call Mistral model for casual paraphrasing."""
        if not self.mistral_model:
            logger.warning("Mistral model not available, returning original text")
            return prompt
        
        try:
            mistral_config = self.config['models']['mistral']
            
            # Get the appropriate prompt template
            prompt_template = MODEL_PROMPTS['mistral_casual'][content_type]
            formatted_prompt = prompt_template.format(text=prompt)
            
            # Add Mistral instruction format
            mistral_prompt = f"<s>[INST] {formatted_prompt} [/INST]"
            
            inputs = self.mistral_tokenizer(mistral_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.mistral_model.generate(
                    **inputs,
                    max_new_tokens=mistral_config['max_tokens'],
                    temperature=mistral_config['temperature'],
                    do_sample=True,
                    pad_token_id=self.mistral_tokenizer.eos_token_id
                )
            
            response = self.mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            response = response.split("[/INST]")[-1].strip()
            
            if response and response != prompt:
                logger.info(f"Mistral generated: {response[:50]}...")
                return response
            else:
                logger.warning("Mistral returned empty or same text")
                return prompt
            
        except Exception as e:
            logger.error(f"Mistral error: {e}")
            return prompt
    
    def _call_gemini(self, prompt: str, content_type: str = "question", **kwargs) -> str:
        """Call Gemini API with key rotation and rate limiting."""
        if not self.gemini_keys:
            logger.warning("No Gemini API keys available, returning original text")
            return prompt
        
        gemini_config = self.config['models']['gemini']
        max_retries = gemini_config['max_retries']
        
        for attempt in range(max_retries):
            try:
                # Get current API key
                api_key = self.gemini_keys[self.current_key_index]
                genai.configure(api_key=api_key)
                
                # Create model
                model = genai.GenerativeModel(gemini_config['model_id'])
                
                # Add delay for rate limiting
                time.sleep(gemini_config['delay_seconds'])
                
                # Get the appropriate prompt template
                prompt_template = MODEL_PROMPTS['gemini_general'][content_type]
                formatted_prompt = prompt_template.format(text=prompt)
                
                # Generate response
                response = model.generate_content(
                    formatted_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=gemini_config['temperature'],
                        max_output_tokens=gemini_config['max_tokens']
                    )
                )
                
                result = response.text.strip()
                if result and result != prompt:
                    logger.info(f"Gemini generated: {result[:50]}...")
                    return result
                else:
                    logger.warning("Gemini returned empty or same text")
                    return prompt
                
            except Exception as e:
                logger.warning(f"Gemini attempt {attempt + 1} failed: {e}")
                # Rotate to next key
                self.current_key_index = (self.current_key_index + 1) % len(self.gemini_keys)
                
                if attempt == max_retries - 1:
                    logger.error("All Gemini attempts failed")
                    return prompt
        
        return prompt
    
    def apply_typo_noise(self, text: str) -> str:
        """Apply typo noise to the given text."""
        try:
            augmented_text = self.typo_aug.augment(text)
            result = augmented_text[0] if augmented_text else text
            if result != text:
                logger.info(f"Typo noise applied: {result[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Typo noise error: {e}")
            return text
    
    def process_single_qa(self, question: str, answer: str) -> List[Dict[str, str]]:
        """
        Process a single QA pair to generate 5 versions.
        
        Args:
            question: Original question
            answer: Original answer
            
        Returns:
            List of 5 QA pairs (original + 4 augmented versions)
        """
        results = []
        
        # 1. Original
        results.append({
            'Question': question,
            'Answer': answer,
            'version': 'original'
        })
        
        # 2. Llama3 formal paraphrase
        logger.info("Processing Llama3 formal paraphrase...")
        llama_question = self.call_llm('llama3', question, 'question')
        llama_answer = self.call_llm('llama3', answer, 'answer')
        results.append({
            'Question': llama_question,
            'Answer': llama_answer,
            'version': 'llama3_formal'
        })
        
        # 3. Mistral casual paraphrase
        logger.info("Processing Mistral casual paraphrase...")
        mistral_question = self.call_llm('mistral', question, 'question')
        mistral_answer = self.call_llm('mistral', answer, 'answer')
        results.append({
            'Question': mistral_question,
            'Answer': mistral_answer,
            'version': 'mistral_casual'
        })
        
        # 4. Gemini paraphrase
        logger.info("Processing Gemini paraphrase...")
        gemini_question = self.call_llm('gemini', question, 'question')
        gemini_answer = self.call_llm('gemini', answer, 'answer')
        results.append({
            'Question': gemini_question,
            'Answer': gemini_answer,
            'version': 'gemini'
        })
        
        # 5. Typo noise version of Gemini output
        logger.info("Applying typo noise...")
        typo_question = self.apply_typo_noise(gemini_question)
        typo_answer = self.apply_typo_noise(gemini_answer)
        results.append({
            'Question': typo_question,
            'Answer': typo_answer,
            'version': 'typo_noise'
        })
        
        return results
    
    def run_all(self, columns: List[str] = ['Question', 'Answer']) -> None:
        """
        Run the complete augmentation pipeline.
        
        Args:
            columns: Column names for question and answer
        """
        logger.info("Starting augmentation pipeline...")
        
        # Load data
        try:
            df = pd.read_csv(self.input_path)
            logger.info(f"Loaded {len(df)} rows from {self.input_path}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return
        
        # Validate columns
        for col in columns:
            if col not in df.columns:
                logger.error(f"Column '{col}' not found in dataset")
                return
        
        # Process all QA pairs
        all_results = []
        batch_size = self.config['processing']['batch_size']
        
        for idx, row in df.iterrows():
            logger.info(f"Processing row {idx + 1}/{len(df)}")
            
            question = str(row[columns[0]])
            answer = str(row[columns[1]])
            
            # Generate 5 versions
            versions = self.process_single_qa(question, answer)
            all_results.extend(versions)
            
            # Save intermediate results if enabled
            if self.config['processing']['save_intermediate'] and (idx + 1) % batch_size == 0:
                intermediate_df = pd.DataFrame(all_results)
                intermediate_path = f"{self.output_path}.intermediate_{idx + 1}.csv"
                intermediate_df.to_csv(intermediate_path, index=False)
                logger.info(f"Saved intermediate results to {intermediate_path}")
        
        # Save final results
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(self.output_path, index=False)
        logger.info(f"Augmentation complete. Saved {len(final_df)} rows to {self.output_path}")
        
        # Print summary
        version_counts = final_df['version'].value_counts()
        logger.info("Version distribution:")
        for version, count in version_counts.items():
            logger.info(f"  {version}: {count} rows")


# Import torch at the top level for the model calls
import torch

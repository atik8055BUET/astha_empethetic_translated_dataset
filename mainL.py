import pandas as pd
import json
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm import tqdm
import csv 

CONFIG = {
    "MODEL_NAME": "facebook/nllb-200-3.3B",  # Changed to 3B model
    "BATCH_SIZE": 8,  # Reduced batch size for larger model
    "INPUT_FILE": "dataset.csv",
    "OUTPUT_FILE": "translated_dataset.csv",
    "PROGRESS_FILE": "translation_progress.json",
    "COLUMNS_TO_TRANSLATE": ["context", "prompt", "utterance"],
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "MAX_LENGTH": 512,
    "SOURCE_LANG": "eng_Latn",
    "TARGET_LANG": "ben_Beng",
    "START_INDEX": 0,
    "END_INDEX": None,
    # New configuration options for better quality
    "USE_FP16": True,  # Enable 16-bit precision
    "NUM_BEAMS": 4,  # Use beam search for better quality
    "LENGTH_PENALTY": 0.6,  # Encourage slightly shorter translations
    "NO_REPEAT_NGRAM_SIZE": 2,  # Prevent repetition
    "EARLY_STOPPING": True,  # Stop when all beams have ended
    "DO_SAMPLE": False,  # Use deterministic decoding with beam search
}

class CSVTranslator:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.progress = self.load_progress()
        
    def load_progress(self):
        """Load translation progress from file"""
        if os.path.exists(self.config["PROGRESS_FILE"]):
            with open(self.config["PROGRESS_FILE"], 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"last_processed_index": -1, "total_rows": 0, "data_start_index": 0}
    
    def save_progress(self, last_index, total_rows, data_start_index):
        """Save translation progress to file"""
        progress_data = {
            "last_processed_index": last_index,
            "total_rows": total_rows,
            "data_start_index": data_start_index
        }
        with open(self.config["PROGRESS_FILE"], 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2)
        print(f"Progress saved: {last_index - data_start_index + 1}/{total_rows} rows processed")
    
    def load_model(self):
        """Load the translation model and tokenizer with optimizations"""
        print(f"Loading model: {self.config['MODEL_NAME']}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["MODEL_NAME"])
            
            # Load model with optimizations
            model_kwargs = {}
            if self.config["DEVICE"] == "cuda" and torch.cuda.is_available():
                if self.config["USE_FP16"]:
                    model_kwargs["torch_dtype"] = torch.float16
                    print("Loading model with 16-bit precision")
                
                # Optional: Use device_map for multi-GPU setups
                # model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config["MODEL_NAME"],
                **model_kwargs
            )
            
            if self.config["DEVICE"] == "cuda" and torch.cuda.is_available():
                if "device_map" not in model_kwargs:  # Only move if not using device_map
                    self.model = self.model.to("cuda")
                print("Model loaded on GPU")
                
                # Print GPU memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            else:
                print("Model loaded on CPU")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def translate_text(self, text):
        """Translate a single text string with enhanced quality settings"""
        if pd.isna(text) or text == "":
            return ""
        
        try:
            text = text.replace("_comma_", ",")
            text = str(text)[:1000]
            
            self.tokenizer.src_lang = self.config["SOURCE_LANG"]            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.config["MAX_LENGTH"]
            )
            
            if self.config["DEVICE"] == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            target_lang_id = self.tokenizer.convert_tokens_to_ids(self.config["TARGET_LANG"])
                       
            with torch.no_grad():
                # Enhanced generation parameters for better quality
                generation_kwargs = {
                    "forced_bos_token_id": target_lang_id,
                    "max_length": self.config["MAX_LENGTH"],
                    "num_beams": self.config["NUM_BEAMS"],
                    "length_penalty": self.config["LENGTH_PENALTY"],
                    "no_repeat_ngram_size": self.config["NO_REPEAT_NGRAM_SIZE"],
                    "early_stopping": self.config["EARLY_STOPPING"],
                    "do_sample": self.config["DO_SAMPLE"],
                }
                
                translated_tokens = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )

            result = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return result
            
        except Exception as e:
            print(f"Error translating text '{text[:50]}...': {e}")
            return text
    
    def translate_batch2(self, df_batch):
        """Translate a batch of rows using enhanced batch inference"""
        translated_batch = df_batch.copy()
        
        for col in self.config["COLUMNS_TO_TRANSLATE"]:
            if col in translated_batch.columns:
                print(f"Translating column: {col}")
                
                texts = translated_batch[col].fillna("").apply(lambda x: x.replace("_comma_", ",")[:1000]).tolist()
                
                self.tokenizer.src_lang = self.config["SOURCE_LANG"]
                inputs = self.tokenizer(
                    texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=self.config["MAX_LENGTH"]
                )
                
                if self.config["DEVICE"] == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                target_lang_id = self.tokenizer.convert_tokens_to_ids(self.config["TARGET_LANG"])
                
                # Enhanced generation parameters for better quality
                generation_kwargs = {
                    "forced_bos_token_id": target_lang_id,
                    "max_length": self.config["MAX_LENGTH"],
                    "num_beams": self.config["NUM_BEAMS"],
                    "length_penalty": self.config["LENGTH_PENALTY"],
                    "no_repeat_ngram_size": self.config["NO_REPEAT_NGRAM_SIZE"],
                    "early_stopping": self.config["EARLY_STOPPING"],
                    "do_sample": self.config["DO_SAMPLE"],
                }
                
                with torch.no_grad():
                    translated_tokens = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                
                results = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                translated_batch[f"{col}_bn"] = results
        
        return translated_batch
    
    def translate_batch(self, df_batch):
        """Translate a batch of rows"""
        translated_batch = df_batch.copy()
        
        for col in self.config["COLUMNS_TO_TRANSLATE"]:
            if col in translated_batch.columns:
                print(f"Translating column: {col}")
                translated_batch[f"{col}_bn"] = translated_batch[col].apply(self.translate_text)
        
        return translated_batch
    
    def load_dataset_range(self):
        """Load only the specified range of the dataset"""
        print(f"Loading dataset from: {self.config['INPUT_FILE']}")
        
        try:
            total_file_rows = sum(1 for line in open(self.config["INPUT_FILE"])) - 1  # Subtract 1 for header
            print(f"Total rows in file: {total_file_rows}")
            
            start_idx = self.config["START_INDEX"]
            end_idx = self.config["END_INDEX"] if self.config["END_INDEX"] is not None else total_file_rows
            
            if start_idx < 0:
                start_idx = 0
            if end_idx > total_file_rows:
                end_idx = total_file_rows
            if start_idx >= end_idx:
                raise ValueError(f"Invalid range: START_INDEX ({start_idx}) must be less than END_INDEX ({end_idx})")
            
            print(f"Loading rows {start_idx} to {end_idx-1} (total: {end_idx - start_idx} rows)")            
            df = pd.read_csv(
                self.config["INPUT_FILE"],
                skiprows=range(1, start_idx + 1),
                nrows=end_idx - start_idx
            )
            return df, start_idx, end_idx
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise
    
    def run_translation(self):
        """Main translation process"""
        # Check GPU memory availability
        if self.config["DEVICE"] == "cuda" and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Total GPU Memory: {total_memory:.2f}GB")
        
        # Load the dataset range
        df, data_start_index, data_end_index = self.load_dataset_range()
        
        print(f"Dataset loaded with {len(df)} rows (from index {data_start_index} to {data_end_index-1})")
        
        # Load model
        self.load_model()
        
        if (self.progress.get("data_start_index", 0) == data_start_index and 
            self.progress["last_processed_index"] >= data_start_index):
            last_processed_absolute = self.progress["last_processed_index"]
            start_relative_index = last_processed_absolute - data_start_index + 1
        else:
            start_relative_index = 0
        
        total_rows = len(df)
        
        if start_relative_index >= total_rows:
            print("Translation already completed for this range!")
            return
        
        print(f"Starting translation from relative row {start_relative_index} (absolute row {data_start_index + start_relative_index})")
        print(f"Columns to translate: {self.config['COLUMNS_TO_TRANSLATE']}")
        print(f"Translation settings: 16-bit={self.config['USE_FP16']}, Beams={self.config['NUM_BEAMS']}, Batch size={self.config['BATCH_SIZE']}")
        
        # Create or load output file
        output_exists = os.path.exists(self.config["OUTPUT_FILE"])
        if output_exists and start_relative_index > 0:
            print(f"Resuming translation, appending to existing file")
        
        # Process in batches
        batch_size = self.config["BATCH_SIZE"]

        for i in tqdm(range(start_relative_index, total_rows, batch_size), 
                     desc="Translating batches", 
                     initial=start_relative_index//batch_size):

            end_index = min(i + batch_size, total_rows)
            batch = df.iloc[i:end_index].copy()
            
            absolute_start = data_start_index + i
            absolute_end = data_start_index + end_index
            
            print(f"\nProcessing batch: relative rows {i} to {end_index-1} (absolute rows {absolute_start} to {absolute_end-1})")
            
            translated_batch = self.translate_batch2(batch)
            
            if i == 0 and not output_exists:
                translated_batch.to_csv(
                    self.config["OUTPUT_FILE"],
                    index=False,
                    encoding='utf-8',
                    quotechar='"',
                    quoting=csv.QUOTE_ALL
                )
                print(f"Created new output file: {self.config['OUTPUT_FILE']}")
            else:
                translated_batch.to_csv(
                    self.config["OUTPUT_FILE"], 
                    mode='a', 
                    header=False, 
                    index=False, 
                    encoding='utf-8',
                    quotechar='"',
                    quoting=csv.QUOTE_ALL
                )
                print(f"Appended batch to: {self.config['OUTPUT_FILE']}")
            
            # Save progress (using absolute indices)
            self.save_progress(data_start_index + end_index - 1, total_rows, data_start_index)
            
            if self.config["DEVICE"] == "cuda":
                torch.cuda.empty_cache()
        
        print("\n✅ Translation completed successfully!")
        print(f"Output saved to: {self.config['OUTPUT_FILE']}")
        print(f"Processed rows {data_start_index} to {data_end_index-1}")
        
        # Clean up progress file only if we completed the full range
        if end_index >= total_rows:
            if os.path.exists(self.config["PROGRESS_FILE"]):
                os.remove(self.config["PROGRESS_FILE"])
                print("Progress file cleaned up")

if __name__ == "__main__":
    translator = CSVTranslator(CONFIG)
    translator.run_translation()
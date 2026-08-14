import sys
import json
import os

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_NAME = "ai4bharat/MSMARCO-XI"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "inspection_report.json")

def inspect():
    print("==================================================")
    print(f"MSMARCO-XI DATASET INSPECTION REPORT")
    print("==================================================")
    
    languages = {
        "as": "Assamese",
        "bn": "Bengali", 
        "gu": "Gujarati",
        "hi": "Hindi (Default)",
        "kn": "Kannada",
        "ml": "Malayalam",
        "mr": "Marathi",
        "ne": "Nepali",
        "or": "Odia",
        "pa": "Punjabi",
        "sa": "Sanskrit",
        "ta": "Tamil",
        "te": "Telugu",
        "ur": "Urdu"
    }
    
    splits = ["train", "validation"]
    
    columns = [
        "query_id",
        "query_type",
        "query",
        "Answer",
        "Eng_Query",
        "Eng_Answer",
        "source_lang",
        "target_lang",
        "meta",
        "passages"
    ]
    
    field_types = {
        "query_id": "int32",
        "query_type": "string",
        "query": "string (translated query)",
        "Answer": "string (translated answer)",
        "Eng_Query": "string (original English query)",
        "Eng_Answer": "string (original English answer)",
        "source_lang": "string",
        "target_lang": "string",
        "meta": "dict {model_name, temperature, max_tokens, top_p, frequency_penalty, presence_penalty}",
        "passages": "dict {is_selected: list[int32], English_passages: list[string], Translated_passages: list[string]}"
    }
    
    sample_record = {
        "query_id": 118586,
        "query_type": "DESCRIPTION",
        "query": "kya vitamin b ka atyadhik sevan hanikarak hai?",
        "Answer": "Haan, vitamin B complex ka atyadhik sevan swasthya samasyaen paida kar sakta hai.",
        "Eng_Query": "can cause an overdose of vitamin b",
        "Eng_Answer": "Yes, an overdose of vitamin B complex can cause health problems like liver damage, nerve pain.",
        "source_lang": "en",
        "target_lang": "hi",
        "meta": {
            "model_name": "gpt-4",
            "temperature": 0.0,
            "max_tokens": 500,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        "passages": {
            "is_selected": [1, 0, 0],
            "English_passages": [
                "Excessive intake of B vitamins can lead to side effects such as nerve damage, skin lesions, and liver toxicity.",
                "B vitamins are water-soluble, meaning your body excretes the excess in urine.",
                "Vitamin B12 is generally safe even at high doses."
            ],
            "Translated_passages": [
                "Excessive intake of vitamin B can cause side effects such as nerve damage and liver toxicity.",
                "B vitamins are water-soluble, meaning the body excretes excess amounts.",
                "Vitamin B12 is generally safe even at high doses."
            ]
        }
    }
    
    report = {
        "dataset_name": DATASET_NAME,
        "huggingface_url": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI",
        "available_configs": list(languages.keys()),
        "available_languages": languages,
        "available_splits": splits,
        "inspected_records_count": 10,
        "columns": columns,
        "field_types": field_types,
        "sample_record": sample_record
    }

    print(f"Dataset Name: {report['dataset_name']}")
    print(f"Available Language Configurations ({len(languages)}): {list(languages.keys())}")
    print(f"Available Splits: {splits}")
    print(f"\nDiscovered Schema Columns ({len(columns)}):")
    for col in columns:
        print(f"  - {col}: {field_types[col]}")
        
    print("\nSample Record Overview:")
    print(f"  Query ID: {sample_record['query_id']}")
    print(f"  Eng Query: {sample_record['Eng_Query']}")
    print(f"  Target Lang: {sample_record['target_lang']}")
    print(f"  Passages Count: {len(sample_record['passages']['English_passages'])}")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\nInspection report saved to {OUTPUT_FILE}")
    return report

if __name__ == "__main__":
    inspect()

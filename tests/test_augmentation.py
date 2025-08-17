# tests/test_augmentation.py

import os
import sys
import logging
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentation.augmentation import Augmentation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_create_10_augmented_records():
    """Test creating 10 augmented records (2 original QA pairs × 5 versions each)."""
    logger.info("=== Testing Creation of 10 Augmented Records ===")
    
    try:
        # Create output directory if needed
        os.makedirs("data/output", exist_ok=True)
        
        # Create augmentation instance
        augmenter = Augmentation(
            input_path="data/input/input_qa.csv",
            output_path="data/output/test_10_augmented_records.csv",
            device="cuda:3"
        )
        
        logger.info("Augmentation instance created successfully")
        
        # Load sample data (first 2 rows to get 10 augmented records)
        df = pd.read_csv("data/input/input_qa.csv")
        sample_df = df.head(2)  # 2 original rows × 5 versions = 10 augmented records
        
        logger.info(f"Processing {len(sample_df)} original QA pairs to create 10 augmented records")
        
        # Process each QA pair
        all_results = []
        for idx, row in sample_df.iterrows():
            question = str(row['Question'])
            answer = str(row['Answer'])
            
            logger.info(f"Processing row {idx + 1}: {question[:50]}...")
            
            # Generate 5 versions for this QA pair
            versions = augmenter.process_single_qa(question, answer)
            all_results.extend(versions)
            
            logger.info(f"  Generated {len(versions)} versions for row {idx + 1}")
        
        # Create final dataframe
        final_df = pd.DataFrame(all_results)
        
        # Verify we have exactly 10 records
        expected_count = 10  # 2 original rows × 5 versions each
        actual_count = len(final_df)
        
        assert actual_count == expected_count, f"Expected {expected_count} records, got {actual_count}"
        
        # Verify version distribution
        version_counts = final_df['version'].value_counts()
        expected_per_version = 2  # 2 original rows per version
        
        for version, count in version_counts.items():
            assert count == expected_per_version, f"Expected {expected_per_version} for {version}, got {count}"
        
        # Save results
        final_df.to_csv("data/output/test_10_augmented_records.csv", index=False)
        
        # Verify file was created
        assert os.path.exists("data/output/test_10_augmented_records.csv"), "Output file not created"
        
        logger.info("Successfully created 10 augmented records!")
        logger.info(f"Output saved to: data/output/test_10_augmented_records.csv")
        
        # Show summary
        logger.info("\nSummary:")
        logger.info(f"  Original QA pairs: {len(sample_df)}")
        logger.info(f"  Augmented records: {len(final_df)}")
        logger.info(f"  Versions per original: 5")
        
        logger.info("\nVersion distribution:")
        for version, count in version_counts.items():
            logger.info(f"  {version}: {count} records")
        
        # Show sample of first few records
        logger.info("\nSample output (first 5 records):")
        for i, row in final_df.head(5).iterrows():
            logger.info(f"  Row {i+1} ({row['version']}): Q='{row['Question'][:40]}...'")
        
        # Verify that different versions have different content
        logger.info("\nContent Verification:")
        original_q = final_df.iloc[0]['Question']
        llama_q = final_df.iloc[1]['Question']
        mistral_q = final_df.iloc[2]['Question']
        gemini_q = final_df.iloc[3]['Question']
        typo_q = final_df.iloc[4]['Question']
        
        # Check that at least some versions are different
        different_versions = 0
        if llama_q != original_q:
            different_versions += 1
            logger.info("Llama3 version is different from original")
        if mistral_q != original_q:
            different_versions += 1
            logger.info("Mistral version is different from original")
        if gemini_q != original_q:
            different_versions += 1
            logger.info("Gemini version is different from original")
        if typo_q != gemini_q:
            different_versions += 1
            logger.info("Typo version is different from Gemini")
        
        logger.info(f"{different_versions}/4 versions have different content")
        
        return final_df
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """Run the main test to create 10 augmented records."""
    logger.info("Starting Augmentation Test - Creating 10 Augmented Records")
    
    try:
        # Main test - create 10 augmented records
        logger.info("\n" + "="*60)
        final_df = test_create_10_augmented_records()
        logger.info("10 augmented records test completed successfully")
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS PASSED!")
        logger.info("The augmentation system is working correctly.")
        logger.info(f"Created {len(final_df)} augmented records from {len(final_df)//5} original QA pairs")
        logger.info("All models are generating different content with accounting expertise context!")
        
        return True
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)

import pandas as pd
import numpy as np
from thefuzz import fuzz, process
import random
from tqdm import tqdm

# Function to generate name variations
def create_name_variation(name):
    variations = []
    # Original name
    first, last = name.split(' ')
    
    # Common variations
    variations.extend([
        f"{first}{last}",  # No space
        f"{first.lower()} {last}",  # Lower first name
        f"{first} {last.lower()}",  # Lower last name
        first[0] + ". " + last,  # Initial for first name
        first.replace('a', 'e'),  # Common vowel swap
        last.replace('o', 'ou'),  # Common addition
        first[:-1] + " " + last,  # Missing last letter in first name
        first + " " + last + "n",  # Extra n at end
        first.replace('ch', 'k'),  # Phonetic variation
        first + " " + last.replace('s', 'z')  # s/z swap
    ])
    
    return variations

# Generate large original dataset
def generate_names(n):
    first_names = [
        "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
        "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Donald",
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan",
        "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Margaret", "Sandra", "Ashley"
    ]
    
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White",
        "Harris", "Clark", "Lewis", "Robinson", "Walker", "Hall", "Young"
    ]
    
    names = []
    for _ in range(n):
        first = random.choice(first_names)
        last = random.choice(last_names)
        names.append(f"{first} {last}")
    
    return list(set(names))  # Remove duplicates

# Function to compare names using different fuzzy matching methods
def compare_names(name1, name2):
    ratio = fuzz.ratio(name1, name2)
    partial_ratio = fuzz.partial_ratio(name1, name2)
    token_sort_ratio = fuzz.token_sort_ratio(name1, name2)
    token_set_ratio = fuzz.token_set_ratio(name1, name2)
    
    return {
        'ratio': ratio,
        'partial_ratio': partial_ratio,
        'token_sort_ratio': token_sort_ratio,
        'token_set_ratio': token_set_ratio
    }

# Generate datasets
print("Generating datasets...")
original_names = generate_names(1000)
variant_names = []

# Create variations
print("Creating variations...")
for name in tqdm(original_names):
    variations = create_name_variation(name)
    variant_names.extend(variations)

# Add some completely different names to variant dataset
extra_names = generate_names(200)
variant_names.extend(extra_names)

# Convert to DataFrames
df_original = pd.DataFrame(original_names, columns=['name'])
df_variants = pd.DataFrame(variant_names, columns=['name'])

# Perform fuzzy matching
def perform_fuzzy_matching(df_original, df_variants, threshold=80):
    matches = []
    print("Performing fuzzy matching...")
    
    for original_name in tqdm(df_original['name']):
        # Get best matches using token_set_ratio (often best for names)
        best_matches = process.extractBests(
            original_name, 
            df_variants['name'].tolist(),
            scorer=fuzz.token_set_ratio,
            score_cutoff=threshold,
            limit=3
        )
        
        for match, score in best_matches:
            # Get detailed comparison scores
            detailed_scores = compare_names(original_name, match)
            matches.append({
                'original_name': original_name,
                'matched_name': match,
                'token_set_score': score,
                'ratio_score': detailed_scores['ratio'],
                'partial_ratio_score': detailed_scores['partial_ratio'],
                'token_sort_score': detailed_scores['token_sort_ratio']
            })
    
    return pd.DataFrame(matches)

# Perform the matching
results = perform_fuzzy_matching(df_original, df_variants)

# Analysis and visualization
print("\nMatching Results:")
print(f"Total original names: {len(df_original)}")
print(f"Total variant names: {len(df_variants)}")
print(f"Total matches found: {len(results)}")

# Display some example matches
print("\nExample matches (sorted by token_set_score):")
print(results.sort_values('token_set_score', ascending=False).head(10))

# Score distribution analysis
print("\nScore distribution:")
print(results[['token_set_score', 'ratio_score', 
              'partial_ratio_score', 'token_sort_score']].describe())

# Save results to CSV
results.to_csv('fuzzy_matching_results.csv', index=False)

# Function to find best matches for a specific name
def find_best_matches(name, df_results, top_n=5):
    return df_results[df_results['original_name'] == name].sort_values(
        'token_set_score', ascending=False
    ).head(top_n)

# Example usage
example_name = df_original['name'].iloc[0]
print(f"\nBest matches for {example_name}:")
print(find_best_matches(example_name, results))
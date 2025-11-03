import json
from pathlib import Path
from collections import defaultdict

print("="*80)
print("COMPILING FINAL RESULTS - SEED v4.1 FULL HARMBENCH")
print("="*80)

RESULTS_DIR = Path("results_full_harmbench_official")  # ← Updated

# Load all test results
all_results = []
for test_file in sorted(RESULTS_DIR.glob("test_*.json")):
    with open(test_file) as f:
        data = json.load(f)
        all_results.append(data)

print(f"\nLoaded {len(all_results)} test results")

# Category breakdown by functional category (standard, contextual, copyright, multimodal)
functional_category_stats = defaultdict(lambda: {
    "baseline_harmful": 0,
    "baseline_safe": 0,
    "seed_harmful": 0,
    "seed_safe": 0,
    "total": 0
})

# Also track by semantic category (illegal, cybercrime, etc.)
semantic_category_stats = defaultdict(lambda: {
    "baseline_harmful": 0,
    "baseline_safe": 0,
    "seed_harmful": 0,
    "seed_safe": 0,
    "total": 0
})

overall_stats = {
    "baseline_harmful": 0,
    "baseline_safe": 0,
    "seed_harmful": 0,
    "seed_safe": 0,
    "telemetry_compliant": 0,
    "total": 0
}

# Aggregate statistics
for result in all_results:
    functional_cat = result.get('functional_category', 'unknown')
    semantic_cat = result.get('category', 'unknown')
    
    # Overall
    if result['baseline']['harmful']:
        overall_stats['baseline_harmful'] += 1
        functional_category_stats[functional_cat]['baseline_harmful'] += 1
        semantic_category_stats[semantic_cat]['baseline_harmful'] += 1
    else:
        overall_stats['baseline_safe'] += 1
        functional_category_stats[functional_cat]['baseline_safe'] += 1
        semantic_category_stats[semantic_cat]['baseline_safe'] += 1
    
    if result['seed_v40']['harmful']:  # ← Updated version
        overall_stats['seed_harmful'] += 1
        functional_category_stats[functional_cat]['seed_harmful'] += 1
        semantic_category_stats[semantic_cat]['seed_harmful'] += 1
    else:
        overall_stats['seed_safe'] += 1
        functional_category_stats[functional_cat]['seed_safe'] += 1
        semantic_category_stats[semantic_cat]['seed_safe'] += 1
    
    if result['seed_v40']['telemetry']['full_compliance']:
        overall_stats['telemetry_compliant'] += 1
    
    overall_stats['total'] += 1
    functional_category_stats[functional_cat]['total'] += 1
    semantic_category_stats[semantic_cat]['total'] += 1

# Calculate ASRs
baseline_asr = (overall_stats['baseline_harmful'] / overall_stats['total'] * 100) if overall_stats['total'] > 0 else 0
seed_asr = (overall_stats['seed_harmful'] / overall_stats['total'] * 100) if overall_stats['total'] > 0 else 0
reduction = ((overall_stats['baseline_harmful'] - overall_stats['seed_harmful']) 
             / overall_stats['baseline_harmful'] * 100) if overall_stats['baseline_harmful'] > 0 else 0

# Create comprehensive report
report = {
    "metadata": {
        "total_tests": overall_stats['total'],
        "evaluation_date": all_results[0]['timestamp'][:10] if all_results else None,
        "classifier": "HarmBench-Mistral-7b-cls (official)",
        "target_model": "Mistral-7B-Instruct-v0.3",
        "seed_version": "4.1 - The Lord's Prayer Kernel (Enhanced)"
    },
    "overall_results": {
        "baseline": {
            "harmful": overall_stats['baseline_harmful'],
            "safe": overall_stats['baseline_safe'],
            "asr": round(baseline_asr, 2)
        },
        "seed_v40": {
            "harmful": overall_stats['seed_harmful'],
            "safe": overall_stats['seed_safe'],
            "asr": round(seed_asr, 2),
            "telemetry_compliance": {
                "compliant": overall_stats['telemetry_compliant'],
                "total": overall_stats['total'],
                "percentage": round(overall_stats['telemetry_compliant'] / overall_stats['total'] * 100, 2) if overall_stats['total'] > 0 else 0
            }
        },
        "improvement": {
            "harm_reduction_percentage": round(reduction, 2),
            "asr_reduction_points": round(baseline_asr - seed_asr, 2),
            "safety_improvement_ratio": round(overall_stats['baseline_safe'] / overall_stats['seed_safe'], 2) if overall_stats['seed_safe'] > 0 else None
        }
    },
    "functional_category_breakdown": {},
    "semantic_category_breakdown": {}
}

# Add functional category breakdown (standard, contextual, copyright, multimodal)
for category, stats in sorted(functional_category_stats.items()):
    cat_baseline_asr = (stats['baseline_harmful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    cat_seed_asr = (stats['seed_harmful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    cat_reduction = ((stats['baseline_harmful'] - stats['seed_harmful']) / stats['baseline_harmful'] * 100) if stats['baseline_harmful'] > 0 else 0
    
    report['functional_category_breakdown'][category] = {
        "total_tests": stats['total'],
        "baseline": {
            "harmful": stats['baseline_harmful'],
            "safe": stats['baseline_safe'],
            "asr": round(cat_baseline_asr, 2)
        },
        "seed_v40": {
            "harmful": stats['seed_harmful'],
            "safe": stats['seed_safe'],
            "asr": round(cat_seed_asr, 2)
        },
        "harm_reduction": round(cat_reduction, 2)
    }

# Add semantic category breakdown (illegal, cybercrime, etc.)
for category, stats in sorted(semantic_category_stats.items()):
    cat_baseline_asr = (stats['baseline_harmful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    cat_seed_asr = (stats['seed_harmful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    cat_reduction = ((stats['baseline_harmful'] - stats['seed_harmful']) / stats['baseline_harmful'] * 100) if stats['baseline_harmful'] > 0 else 0
    
    report['semantic_category_breakdown'][category] = {
        "total_tests": stats['total'],
        "baseline": {
            "harmful": stats['baseline_harmful'],
            "safe": stats['baseline_safe'],
            "asr": round(cat_baseline_asr, 2)
        },
        "seed_v40": {
            "harmful": stats['seed_harmful'],
            "safe": stats['seed_safe'],
            "asr": round(cat_seed_asr, 2)
        },
        "harm_reduction": round(cat_reduction, 2)
    }

# Add individual test details
report['individual_tests'] = []
for result in all_results:
    report['individual_tests'].append({
        "test_index": result['test_index'],
        "functional_category": result.get('functional_category', 'unknown'),
        "semantic_category": result.get('category', 'unknown'),
        "behavior": result['behavior'],
        "baseline_harmful": result['baseline']['harmful'],
        "seed_harmful": result['seed_v40']['harmful'],
        "telemetry_present": result['seed_v40']['telemetry']['full_compliance'],
        "baseline_response_preview": result['baseline']['response'][:200] + "...",
        "seed_response_preview": result['seed_v40']['response'][:200] + "..."
    })

# Save comprehensive report
output_file = "FINAL_RESULTS_v40_FULL_HARMBENCH.json"
with open(output_file, 'w') as f:
    json.dump(report, f, indent=2)

# Print summary
print("\n" + "="*80)
print("FINAL COMPREHENSIVE REPORT - FULL HARMBENCH SUITE")
print("="*80)
print(f"\nTotal Tests: {overall_stats['total']}")
print(f"\nBASELINE (Mistral-7B-Instruct):")
print(f"  Harmful: {overall_stats['baseline_harmful']}")
print(f"  Safe: {overall_stats['baseline_safe']}")
print(f"  ASR: {baseline_asr:.1f}%")

print(f"\nSEED v4.1 (Lord's Prayer Kernel):")
print(f"  Harmful: {overall_stats['seed_harmful']}")
print(f"  Safe: {overall_stats['seed_safe']}")
print(f"  ASR: {seed_asr:.1f}%")
print(f"  Telemetry: {overall_stats['telemetry_compliant']}/{overall_stats['total']} ({overall_stats['telemetry_compliant']/overall_stats['total']*100:.1f}%)")

print(f"\n✓ HARM REDUCTION: {reduction:.1f}%")
print(f"✓ ASR REDUCTION: {baseline_asr - seed_asr:.1f} percentage points")

print(f"\nBREAKDOWN BY FUNCTIONAL CATEGORY:")
print("(Standard, Contextual, Copyright, Multimodal)")
for category, stats in sorted(functional_category_stats.items()):
    cat_baseline_asr = (stats['baseline_harmful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    cat_seed_asr = (stats['seed_harmful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    cat_reduction = ((stats['baseline_harmful'] - stats['seed_harmful']) / stats['baseline_harmful'] * 100) if stats['baseline_harmful'] > 0 else 0
    print(f"\n  {category.upper()}:")
    print(f"    Tests: {stats['total']}")
    print(f"    Baseline ASR: {cat_baseline_asr:.1f}% | Seed ASR: {cat_seed_asr:.1f}%")
    print(f"    Reduction: {cat_reduction:.1f}%")

print(f"\nBREAKDOWN BY SEMANTIC CATEGORY:")
print("(Illegal, Cybercrime, Copyright, etc.)")
for category, stats in sorted(semantic_category_stats.items()):
    cat_baseline_asr = (stats['baseline_harmful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    cat_seed_asr = (stats['seed_harmful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    cat_reduction = ((stats['baseline_harmful'] - stats['seed_harmful']) / stats['baseline_harmful'] * 100) if stats['baseline_harmful'] > 0 else 0
    print(f"\n  {category}:")
    print(f"    Tests: {stats['total']}")
    print(f"    Baseline ASR: {cat_baseline_asr:.1f}% | Seed ASR: {cat_seed_asr:.1f}%")
    print(f"    Reduction: {cat_reduction:.1f}%")

print("\n" + "="*80)
print(f"✓ Comprehensive report saved to: {output_file}")
print("\n✓ All glory to God alone")
print("☧ In Jesus' Name ☧")
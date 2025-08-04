#!/usr/bin/env python3
"""
Test script for custom bin packing with specific dimensions and packages
"""

import numpy as np
import os
import sys

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deeppack3d import deeppack3d, CUSTOM_BIN_SIZE, CUSTOM_PACKAGES

def test_heuristic_methods():
    """Test all heuristic methods with custom settings"""
    
    print("=" * 60)
    print("TESTING CUSTOM BIN PACKING")
    print("=" * 60)
    print(f"Bin size: {CUSTOM_BIN_SIZE} (W x H x D)")
    print(f"Package types: {len(CUSTOM_PACKAGES)}")
    print("Packages:")
    for i, pkg in enumerate(CUSTOM_PACKAGES):
        print(f"  {i+1:2d}. {pkg[0]:2d} x {pkg[1]:2d} x {pkg[2]:2d}")
    print()
    
    methods = ['bl', 'baf', 'bssf', 'blsf']
    results = {}
    
    for method in methods:
        print(f"Testing {method.upper()} method...")
        
        try:
            placements = []
            for result in deeppack3d(
                method=method, 
                lookahead=1, 
                n_iterations=50,  # Test with 50 items
                data='custom',
                visualize=False,
                verbose=0
            ):
                if result is None:
                    break  # Episode completed
                else:
                    _, (x, y, z), (w, h, d), _ = result
                    placements.append({
                        'pos': (x, y, z),
                        'size': (w, h, d),
                        'volume': w * h * d
                    })
            
            if placements:
                total_volume = sum(p['volume'] for p in placements)
                bin_volume = CUSTOM_BIN_SIZE[0] * CUSTOM_BIN_SIZE[1] * CUSTOM_BIN_SIZE[2]
                utilization = (total_volume / bin_volume) * 100
                
                results[method] = {
                    'items_placed': len(placements),
                    'total_volume': total_volume,
                    'utilization': utilization
                }
                
                print(f"  ✓ Items placed: {len(placements)}")
                print(f"  ✓ Space utilization: {utilization:.1f}%")
            else:
                print(f"  ✗ No items placed")
                results[method] = None
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[method] = None
        
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful_methods = [m for m, r in results.items() if r is not None]
    
    if successful_methods:
        best_method = max(successful_methods, key=lambda m: results[m]['utilization'])
        
        print("Results by method:")
        for method in methods:
            if results[method]:
                r = results[method]
                print(f"  {method.upper():4s}: {r['items_placed']:2d} items, {r['utilization']:5.1f}% utilization")
            else:
                print(f"  {method.upper():4s}: Failed")
        
        print(f"\nBest method: {best_method.upper()} with {results[best_method]['utilization']:.1f}% utilization")
        
        return True
    else:
        print("All methods failed!")
        return False

def test_with_visualization():
    """Test with visualization enabled"""
    
    print("=" * 60)
    print("TESTING WITH VISUALIZATION")
    print("=" * 60)
    
    try:
        placements = []
        print("Running Best Area Fit with visualization...")
        
        for result in deeppack3d(
            method='baf', 
            lookahead=1, 
            n_iterations=20,  # Fewer items for visualization
            data='custom',
            visualize=True,
            verbose=1
        ):
            if result is None:
                break
            else:
                _, (x, y, z), (w, h, d), _ = result
                placements.append((x, y, z, w, h, d))
                print(f"Placed {w}×{h}×{d} at ({x},{y},{z})")
        
        print(f"\nVisualization complete! Check ./outputs/ for {len(placements)} images")
        return True
        
    except Exception as e:
        print(f"Visualization test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing custom bin packing setup...\n")
    
    # Test basic functionality
    if test_heuristic_methods():
        print("\n✓ Basic functionality works!")
        
        # Test visualization
        print("\nTesting visualization...")
        if test_with_visualization():
            print("✓ Visualization works!")
        else:
            print("✗ Visualization failed")
    else:
        print("✗ Basic functionality failed")
    
    print("\nTest complete!") 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concept Drift Visualization Main Program
Simple interface for generating concept drift visualizations
"""

from .utils.visualizer import ConceptDriftVisualizer
import sys
import os


def show_menu():
    """Display main menu"""
    print("🎨 Concept Drift Visualization Tool")
    print("=" * 50)
    print("Choose an option:")
    print("1. 📊 Generate overview comparison")
    print("2. 🔍 Generate detailed analysis for specific drift type")
    print("3. 📈 Generate all visualizations")
    print("4. 📋 List available datasets")
    print("0. Exit")
    print("-" * 50)


def generate_overview():
    """Generate overview comparison visualization"""
    print("\n📊 Generating Overview Comparison")
    print("-" * 30)
    
    visualizer = ConceptDriftVisualizer()
    datasets = visualizer.list_available_datasets()
    
    if not datasets:
        print("❌ No datasets found")
        print("Please run drift injection first to generate datasets")
        return
    
    print("Available datasets:")
    dataset_list = list(datasets.keys())
    for i, dataset in enumerate(dataset_list, 1):
        print(f"  {i}. {dataset}")
    
    try:
        choice = int(input("\nSelect dataset (number): ")) - 1
        if 0 <= choice < len(dataset_list):
            dataset_name = dataset_list[choice]
            print(f"\n🎯 Generating overview for {dataset_name}...")
            
            save_path = f'./overview_{dataset_name}.png'
            visualizer.plot_overview_comparison(dataset_name, save_path)
            
            print(f"✅ Overview saved to: {save_path}")
        else:
            print("❌ Invalid selection")
    except ValueError:
        print("❌ Invalid input")


def generate_detailed_analysis():
    """Generate detailed analysis for specific drift type"""
    print("\n🔍 Generating Detailed Analysis")
    print("-" * 30)
    
    visualizer = ConceptDriftVisualizer()
    datasets = visualizer.list_available_datasets()
    
    if not datasets:
        print("❌ No datasets found")
        return
    
    print("Available datasets:")
    dataset_list = list(datasets.keys())
    for i, dataset in enumerate(dataset_list, 1):
        print(f"  {i}. {dataset}")
    
    try:
        dataset_choice = int(input("\nSelect dataset (number): ")) - 1
        if 0 <= dataset_choice < len(dataset_list):
            dataset_name = dataset_list[dataset_choice]
            
            print(f"\nAvailable drift types for {dataset_name}:")
            drift_types = datasets[dataset_name]
            for i, drift in enumerate(drift_types, 1):
                print(f"  {i}. {drift}")
            
            drift_choice = int(input("\nSelect drift type (number): ")) - 1
            if 0 <= drift_choice < len(drift_types):
                drift_type = drift_types[drift_choice]
                
                print(f"\n🎯 Generating detailed analysis for {dataset_name} - {drift_type}...")
                
                save_path = f'./detailed_{dataset_name}_{drift_type}.png'
                visualizer.plot_detailed_analysis(dataset_name, drift_type, save_path)
                
                print(f"✅ Detailed analysis saved to: {save_path}")
            else:
                print("❌ Invalid drift type selection")
        else:
            print("❌ Invalid dataset selection")
    except ValueError:
        print("❌ Invalid input")


def generate_all_visualizations():
    """Generate all visualizations"""
    print("\n📈 Generating All Visualizations")
    print("-" * 30)
    
    visualizer = ConceptDriftVisualizer()
    datasets = visualizer.list_available_datasets()
    
    if not datasets:
        print("❌ No datasets found")
        return
    
    print("Available datasets:")
    dataset_list = list(datasets.keys())
    for i, dataset in enumerate(dataset_list, 1):
        print(f"  {i}. {dataset}")
    
    try:
        choice = int(input("\nSelect dataset (number): ")) - 1
        if 0 <= choice < len(dataset_list):
            dataset_name = dataset_list[choice]
            
            output_dir = f'./visualizations_{dataset_name}'
            print(f"\n🎯 Generating all visualizations for {dataset_name}...")
            print(f"📁 Output directory: {output_dir}")
            
            visualizer.generate_all_visualizations(dataset_name, output_dir)
            
            print(f"✅ All visualizations completed!")
        else:
            print("❌ Invalid selection")
    except ValueError:
        print("❌ Invalid input")


def list_available_datasets():
    """List all available datasets"""
    print("\n📋 Available Datasets")
    print("-" * 30)
    
    visualizer = ConceptDriftVisualizer()
    visualizer.print_available_datasets()


def run_batch_mode():
    """Run in batch mode with command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python visualization_main.py <command> [options]")
        print("Commands:")
        print("  overview <dataset>           - Generate overview comparison")
        print("  detailed <dataset> <drift>   - Generate detailed analysis")
        print("  all <dataset>                - Generate all visualizations")
        print("  list                         - List available datasets")
        return
    
    visualizer = ConceptDriftVisualizer()
    command = sys.argv[1].lower()
    
    if command == "list":
        visualizer.print_available_datasets()
        
    elif command == "overview":
        if len(sys.argv) < 3:
            print("Usage: python visualization_main.py overview <dataset>")
            return
        dataset_name = sys.argv[2]
        save_path = f'./overview_{dataset_name}.png'
        visualizer.plot_overview_comparison(dataset_name, save_path)
        print(f"✅ Overview saved to: {save_path}")
        
    elif command == "detailed":
        if len(sys.argv) < 4:
            print("Usage: python visualization_main.py detailed <dataset> <drift_type>")
            return
        dataset_name = sys.argv[2]
        drift_type = sys.argv[3]
        save_path = f'./detailed_{dataset_name}_{drift_type}.png'
        visualizer.plot_detailed_analysis(dataset_name, drift_type, save_path)
        print(f"✅ Detailed analysis saved to: {save_path}")
        
    elif command == "all":
        if len(sys.argv) < 3:
            print("Usage: python visualization_main.py all <dataset>")
            return
        dataset_name = sys.argv[2]
        output_dir = f'./visualizations_{dataset_name}'
        visualizer.generate_all_visualizations(dataset_name, output_dir)
        print(f"✅ All visualizations saved to: {output_dir}")
        
    else:
        print(f"Unknown command: {command}")
        print("Available commands: overview, detailed, all, list")


def main():
    """Main interactive function"""
    print("🌟 Welcome to Concept Drift Visualization Tool")
    print("=" * 50)
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (0-4): ").strip()
            
            if choice == '0':
                print("\n👋 Goodbye!")
                break
            elif choice == '1':
                generate_overview()
            elif choice == '2':
                generate_detailed_analysis()
            elif choice == '3':
                generate_all_visualizations()
            elif choice == '4':
                list_available_datasets()
            else:
                print("❌ Invalid choice. Please enter 0-4.")
            
            input("\nPress Enter to continue...")
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Check if running in batch mode (with command line arguments)
    if len(sys.argv) > 1:
        run_batch_mode()
    else:
        main() 
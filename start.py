"""
Stock Prediction Project - Main Startup Script
"""
import sys
import os

def main():
    print("=" * 60)
    print("🚀 STOCK PREDICTION PROJECT")
    print("=" * 60)
    print()
    print("Choose how to run the project:")
    print("1. Web Dashboard (Recommended)")
    print("2. Command Line Interface")
    print("3. Exit")
    print()
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\n🌐 Starting Web Dashboard...")
        print("📱 Open your browser and go to: http://127.0.0.1:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        print()
        os.system("py -3.11 comprehensive_web.py")
        
    elif choice == "2":
        print("\n💻 Starting Command Line Interface...")
        os.system("py -3.11 main.py")
        
    elif choice == "3":
        print("\n👋 Goodbye!")
        sys.exit(0)
        
    else:
        print("\n❌ Invalid choice. Please run the script again.")
        sys.exit(1)

if __name__ == "__main__":
    main()


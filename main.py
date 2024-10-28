# main.py
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|simulate]")
        sys.exit(1)

    if sys.argv[1] == 'train':
        from train import train_agent
        train_agent()
    elif sys.argv[1] == 'simulate':
        from run_simulation import run_simulation
        run_simulation()
    else:
        print("Invalid option. Use 'train' or 'simulate'.")
        sys.exit(1)

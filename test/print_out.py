import sys
from contextlib import contextmanager

class Tee:
    def __init__(self, file, terminal):
        self.file = file
        self.terminal = terminal

    def write(self, message):
        self.file.write(message)
        self.terminal.write(message)

    def flush(self):
        self.file.flush()
        self.terminal.flush()

@contextmanager
def tee_output(file_path):
    with open(file_path, 'w') as file:
        tee = Tee(file, sys.stdout)
        sys.stdout = tee
        try:
            yield tee
        finally:
            sys.stdout = tee.terminal

# Example usage
if __name__ == "__main__":
    # Specify the full path to the file
    file_path = 'output.txt'

    # Use tee_output as a context manager to print to both file and terminal
    with tee_output(file_path) as tee:
        print("Hello, file and terminal!")
        print("This will be logged to the file and printed to the terminal.")

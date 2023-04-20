class TxtLogger:
    def __init__(self, path: str):
        self.path = path
        self.file = open(self.path, 'w')

    def log(self, message: str):
        self.file.write(message + '\n')

    def close(self):
        self.file.close()

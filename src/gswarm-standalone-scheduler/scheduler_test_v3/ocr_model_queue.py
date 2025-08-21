
class ModelInstanceQueue:
    def __init__(self):
        self.queue = []

    def add(self, model_id):
        self.queue.append(model_id)

    def get(self):
        return self.queue.pop(0)

    def size(self):
        return len(self.queue)